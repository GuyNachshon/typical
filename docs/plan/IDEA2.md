# PCDM — Parallel Calibrated Decision Model

## 1. Idea

Large language models are trained to generate sequences:

$$
P(t_k\mid x,t_{<k})
$$

but many real software tasks do not require text generation. They require a bounded decision:

* classify an event,
* choose among alternatives,
* determine whether a proposition is supported,
* assign a risk level,
* route a request,
* decide whether evidence is sufficient.

Today these tasks are often implemented by asking an autoregressive language model to generate JSON or prose and then interpreting that generation as a decision.

PCDM investigates a different computational primitive:

$$
\boxed{
F_\theta(x,q,A)\rightarrow
P(y\mid x,q,A)
}
$$

where:

* \(x\) is arbitrary unstructured state;
* \(q\) is a natural-language decision/query;
* \(A=\{a_1,\ldots,a_K\}\) is a runtime-defined answer space;
* \(y\in A\cup\{\varnothing\}\);
* the output is a numerical probability distribution, not generated text.

For many decisions over the same state:

$$
F_\theta
\left(
x,
\{q_i,A_i\}_{i=1}^{M}
\right)
\rightarrow
\{P_i\}_{i=1}^{M}
$$

the expensive representation of the state should be computed once and reused by all queries.

The core research question is:

> **Can a pretrained language model be transformed from a generative sequence model into a reusable probabilistic decision engine, retaining general semantic reasoning while eliminating autoregressive answer generation and amortizing state computation across many independent decisions?**

---

# 2. What PCDM is trying to achieve

A successful PCDM should have six properties.

### 2.1 General decisions

The model should not have a fixed classifier vocabulary.

At inference time it should accept arbitrary candidate spaces such as:

```python
Choice[
    "approve",
    "deny",
    "send to manual review"
]
```

or:

```python
Choice[
    "engine failure",
    "sensor failure",
    "operator error",
    "insufficient evidence"
]
```

without retraining a classifier head.

---

### 2.2 Direct probabilities

The native neural output is:

$$
P(a_j\mid x,q,A)
$$

rather than a generated string such as:

> `"answer": "approve", "confidence": "82%"`

The distribution itself is the model output.

---

### 2.3 Reusable state

For a large state \(x\), state processing should be amortized:

$$
\boxed{
\text{state computation once}
+
\text{many inexpensive query computations}
}
$$

rather than:

$$
M\times
\text{full model evaluation of state + question}.
$$

---

### 2.4 Query isolation

Questions sharing a state should not influence each other:

$$
P_i=P(y_i\mid x,q_i,A_i)
$$

not:

$$
P(y_i\mid x,q_1,\ldots,q_M).
$$

Queries can therefore be scheduled in parallel without creating conversational dependencies.

---

### 2.5 Explicit unknown

The model must be able to represent:

$$
\varnothing
$$

meaning that the supplied candidate set does not adequately explain the evidence.

This should be a learned property of:

$$
(x,q,A)
$$

rather than a global OOD score or a literal text candidate called `"unknown"`.

---

### 2.6 Calibrated uncertainty

Ultimately:

$$
P(Y=\hat Y\mid \hat p\approx p)\approx p
$$

so downstream software can use probabilities as decision variables rather than decorative confidence scores.

---

# 3. Current architectural conclusion

The original PCDM hypothesis assumed state and query could be encoded independently and that a small fresh transformer could learn their interaction.

Experiments falsified that strong version.

The current evidence indicates that **pretrained state–query interaction is load-bearing**.

Therefore PCDM should use a causal shared-prefix architecture:

```text
                    STATE
                      │
                      ▼
              pretrained backbone
                      │
                reusable state
                cache / memory
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
         q1          q2          q3
          │           │           │
    conditioned  conditioned  conditioned
      suffix        suffix        suffix
          │           │           │
          ▼           ▼           ▼
       decision    decision    decision
       readout     readout     readout
```

This is compatible with external black-box analysis of Jev, which finds evidence consistent with shared state computation, isolated question branches, a causal backbone and direct probability readouts. That analysis is reverse engineering rather than confirmed TypeSafe architecture.

---

# 4. PCDM architecture

The proposed architecture has four conceptual stages:

$$
\boxed{
\text{Shared-State Backbone}
\rightarrow
\text{Query-Conditioned Decision Representation}
\rightarrow
\text{Listwise Candidate Model}
\rightarrow
\text{Energy Readout}
}
$$

---

## 4.1 Shared-state backbone

Given:

$$
x=(x_1,\ldots,x_{L_s})
$$

a pretrained causal model computes reusable state:

$$
S=B_\theta(x).
$$

With the current Qwen implementation, \(S\) consists primarily of:

* hidden-state memory;
* per-layer KV cache.

Importantly, because \(x\) is a causal prefix, adding a query suffix does not change the state-token representations.

Therefore:

$$
S
$$

can be computed once.

---

## 4.2 Query-conditioned representation

For each query \(q_i\):

$$
Q_i
=
B_\theta(q_i\mid S)
$$

using the cached state prefix.

This gives the pretrained backbone an opportunity to perform semantic alignment:

> Which parts of the state matter for this question?

This was the missing capability in the original separate-encoder architecture.

The query branches remain independent:

$$
Q_i\perp Q_j\mid S.
$$

They may therefore be batched while remaining semantically isolated.

---

## 4.3 Decision tower

The current decision tower maps:

$$
(S,Q_i)\rightarrow h_i
$$

where:

$$
h_i\in\mathbb{R}^{d_h}
$$

is one decision representation.

Conceptually:

```text
state memory S ──────────────┐
                              │ cross-attention
conditioned query Q ──────────┤
                              │
learned decision slot ────────┘
                              │
                              ▼
                         decision h
```

The tower is **not responsible for understanding the query from scratch**.

That occurs largely in the pretrained backbone.

Its role is narrower:

> Extract a compact representation suitable for evaluating this decision.

An important ablation remains:

$$
\text{joint backbone}\rightarrow\text{scorer}
$$

versus:

$$
\text{joint backbone}\rightarrow\text{tower}\rightarrow\text{scorer}.
$$

If the tower contributes little after joint encoding, it should be simplified or removed.

---

# 5. Candidate model

Each runtime-defined answer \(a_j\) receives a semantic representation:

$$
c_j=C_\theta(a_j).
$$

These candidate embeddings may be cached independently of the state/query when their encoder is frozen.

The original scorer uses:

$$
s_j=f_\theta(h,c_j)
$$

which is equivalent to defining a conditional energy:

$$
E_\theta(x,q,a_j)=-s_j.
$$

The predictive distribution is:

$$
P(a_j\mid x,q,A)
=
\frac{\exp(-E_j/T)}
{\sum_k\exp(-E_k/T)+\exp(-E_\varnothing/T)}.
$$

Thus PCDM is not primarily “an EBM architecture,” but its decision readout is naturally an **energy-based conditional model**.

---

# 6. Proposed change: set-conditioned / listwise candidates

The current independent scorer has an important structural limitation.

If:

$$
s_j=f(x,q,a_j),
$$

then adding another candidate changes the softmax denominator but cannot alter the relative odds between two existing candidates:

$$
\frac{P(a_i)}{P(a_j)}
=
e^{s_i-s_j}.
$$

However, answer choices often define one another.

For example:

```text
payments
account
other
```

defines a different decision boundary from:

```text
bank
payment provider
merchant
customer
```

Black-box experiments against Jev found that adding an irrelevant fifth option changed the relative odds between existing options, which is evidence that Jev's answer computation has access to the whole option set rather than assigning immutable independent logits. This does not reveal the exact implementation, but it motivates a listwise candidate model.

PCDM should therefore test:

$$
\boxed{
E_\theta(x,q,A,a_j)
}
$$

rather than only:

$$
E_\theta(x,q,a_j).
$$

A candidate-set mixer can implement this:

```text
                decision h
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
       c1           c2           c3
       │            │            │
       └──── candidate set ──────┘
                    │
                    ▼
          permutation-equivariant
              set transformer
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
      c'1          c'2          c'3
       │            │            │
       ▼            ▼            ▼
      E1           E2           E3
```

There should be:

* no candidate positional embeddings;
* shared candidate processing;
* permutation-equivariant self-attention.

Then reordering candidates only reorders the probabilities.

---

# 7. Architectural null

The null should similarly be defined over the complete candidate set:

$$
E_\varnothing
=
E_\theta(x,q,A,\varnothing).
$$

Its semantic interpretation is:

> **Given the state, question and available alternatives, none of these choices is sufficiently supported.**

This is stronger than generic OOD detection.

The null model should distinguish:

1. correct answer absent from \(A\);
2. insufficient evidence in \(x\);
3. irrelevant question;
4. semantically close but incorrect alternatives;
5. genuinely unfamiliar state.

These should eventually become distinct evaluation slices.

---

# 8. Training stack

PCDM should separate three training problems that are easy to conflate.

---

## Stage A — Decision adaptation

Goal:

> Teach the model to map arbitrary `(state, query, candidate-set)` tuples into useful decision distributions.

Use:

* deterministic hard labels;
* human distributions such as ChaosNLI;
* continuous probability labels such as UNLI;
* synthetic null examples;
* randomized candidate wording;
* heterogeneous task families.

Objective:

$$
L_{\text{decision}}
=
-\sum_j p_j^*\log p_\theta(j).
$$

This is the stage currently implemented.

It is best described as:

> **probabilistic decision SFT**

rather than RLCD.

---

# 9. Stage B — Calibrated decision optimization

Accuracy does not guarantee that:

$$
0.8
$$

means “correct approximately 80% of the time.”

The next stage should explicitly train probability quality.

Training examples should emphasize:

* varying difficulty;
* missing evidence;
* contradictory evidence;
* ambiguity;
* OOD inputs;
* near-decision-boundary cases;
* evidence addition/removal;
* counterfactual interventions;
* distribution shift.

Use proper scoring rules such as log loss and Brier score.

For outcome \(y\):

$$
L_{\log}
=
-\log p_\theta(y)
$$

and:

$$
L_{\text{Brier}}
=
\sum_j
(p_j-\mathbf 1[j=y])^2.
$$

Both reward truthful probability estimates in expectation.

A broader objective can include:

$$
L_{\text{CD}}
=
\alpha L_{\log}
+
\beta L_{\text{Brier}}
+
\gamma L_{\text{invariance}}
+
\delta L_{\text{evidence}}
+
\eta L_{\text{null}}.
$$

This should initially remain directly differentiable.

There is no reason to introduce policy-gradient RL merely because the desired output is probabilistic.

---

# 10. Where RLCD fits

TypeSafe describes **Reinforcement Learning for Calibrated Decisions** as the training counterpart to its System One architecture: rather than optimizing generated text preference, the objective is calibrated decision probability. TypeSafe has not publicly disclosed the actual RLCD algorithm.

Therefore PCDM should not claim to reproduce RLCD.

Instead, we should test the underlying idea.

The architecture produces a policy:

$$
\pi_\theta(y\mid x,q,A).
$$

The question becomes:

> What post-training objective makes this distribution useful for downstream decision-making?

---

## 10.1 Why simple proper scoring is not really RL

If:

$$
R=\log p_\theta(y),
$$

maximizing expected reward is equivalent to minimizing cross-entropy.

Likewise Brier score is directly differentiable.

Therefore calling this reinforcement learning adds no useful machinery.

We should only introduce RL where the reward depends on **consequences of decisions**, rather than simply the correct answer.

---

# 11. Stage C — workflow-level RLCD-like training

Consider a probability:

$$
P_\theta(\text{fraud})=0.73.
$$

A software policy might implement:

```text
p < .30      → allow
.30–.85      → manual review
p > .85      → block
```

with asymmetric costs:

```text
missed fraud      -100
false block        -20
manual review       -2
correct automatic   +1
```

Now the relevant objective is:

$$
J(\theta)
=
\mathbb E[
U(\text{downstream action},y)
].
$$

The distribution influences an external, possibly nondifferentiable process.

This is where genuine reinforcement learning becomes useful.

A practical objective might be:

$$
J=
R_{\text{workflow}}
+
\lambda R_{\text{proper}}
-
\beta KL(
\pi_\theta
\Vert
\pi_{\text{calibrated-base}}
).
$$

The proper-scoring component prevents the model from learning strategically useful but dishonest probability values.

---

# 12. Sequential RLCD

The stronger case is a branching workflow:

```text
decision 1
    │
    ▼
software action
    │
    ▼
new state
    │
    ▼
decision 2
    │
    ▼
new action
    │
    ▼
eventual outcome
```

Now:

$$
s_t\rightarrow p_t\rightarrow a_t\rightarrow s_{t+1}
$$

and delayed reward may depend on multiple decisions.

At this point PCDM is genuinely a policy inside an environment, and PPO/GRPO/direct policy-gradient approaches become scientifically justified.

This is the PCDM analogue of RLCD that is worth studying.

---

# 13. Architecture versus RLCD

The distinction should remain explicit:

### Architecture asks:

$$
\boxed{
\text{How can we cheaply compute }P(y\mid x,q,A)?
}
$$

### Calibrated post-training asks:

$$
\boxed{
\text{What should those probabilities mean?}
}
$$

### Workflow RL asks:

$$
\boxed{
\text{How should those probabilities behave when software acts on them?}
}
$$

These are separate research questions.

---

# 14. KDA — Kimi Delta Attention

KDA is highly relevant, but for a completely different reason.

Kimi Delta Attention is a recurrent linear-attention mechanism introduced in Kimi Linear. Instead of preserving the complete KV history at every attention layer, it maintains a finite recurrent matrix state with fine-grained per-channel forgetting. Kimi Linear combines KDA layers with occasional global MLA layers; Moonshot reports up to 75% KV-cache reduction and up to 6× decoding throughput at very long context.

For ordinary generation, this primarily improves long-context decoding.

For PCDM, it may be even more structurally interesting.

---

# 15. The current PCDM cost

With full attention, we encode the state once:

$$
KV(x).
$$

But every query suffix still performs attention against that state.

Ignoring constants, query work contains a term resembling:

$$
O(L_qL_s).
$$

For \(M\) independent queries:

$$
O(ML_qL_s).
$$

So state **representation computation** is amortized, but state **attention** is not free.

This is likely to become PCDM's dominant cost as:

$$
L_s\rightarrow\text{large}
$$

and:

$$
M\rightarrow\text{large}.
$$

---

# 16. PCDM with KDA

A KDA layer reduces the entire processed prefix into recurrent state:

$$
R_x.
$$

After processing the shared state once:

```text
STATE
   │
   ▼
KDA backbone
   │
   ▼
recurrent layer states R(x)
```

each query can fork that state:

```text
                     ┌── copy R(x) ── q1
                     ├── copy R(x) ── q2
state → R(x) ────────┼── copy R(x) ── q3
                     └── ...
```

The query does not need to attend explicitly over all \(L_s\) state tokens in every KDA layer.

Conceptually:

$$
\boxed{
\text{state tokens}
\rightarrow
\text{fixed-size neural memory}
\rightarrow
\text{many independent query branches}
}
$$

This is exceptionally well matched to PCDM.

---

# 17. Why a hybrid is preferable

Pure linear attention may lose exact long-range retrieval.

Kimi Linear itself therefore uses a hybrid architecture rather than replacing every attention layer: most layers use KDA and periodic layers retain global attention.

A PCDM-specific backbone could therefore resemble:

```text
KDA
KDA
KDA
Global Attention
KDA
KDA
KDA
Global Attention
...
```

For each shared state:

* KDA layers cache compact recurrent state;
* global layers retain a smaller set of conventional KV caches;
* each question forks the recurrent states and shares the global-prefix cache.

That gives PCDM a continuum between:

$$
\text{perfect prefix memory / expensive querying}
$$

and:

$$
\text{compressed prefix memory / extremely cheap querying}.
$$

---

# 18. Why KDA should not be introduced yet

KDA should be treated as a **scaling experiment**, not a prerequisite for proving PCDM.

The current `joint_v1` architecture already tests the central semantic hypothesis.

Changing the backbone simultaneously would make it difficult to know whether failures come from:

* decision architecture;
* calibration;
* candidate representation;
* backbone conversion;
* recurrent-memory capacity.

There is also concrete evidence that converting an existing Qwen model from full attention to KDA can preserve perplexity-like measures while damaging downstream multiple-choice behavior. A 2026 study converting most Qwen3-0.6B attention layers to KDA found that hidden-state alignment and KL distillation were insufficient to preserve the answer interface without additional targeted repair.

Therefore:

> **prove PCDM with the native Qwen backbone first; introduce KDA only once the semantic architecture is stable.**

---

# 19. KDA hypothesis

KDA introduces a separate hypothesis:

### H6 — Compressed shared-state memory

A hybrid recurrent-attention backbone can preserve PCDM decision quality while materially reducing:

$$
\text{memory/query}
$$

and:

$$
\frac{\partial\,Latency}{\partial L_s}
$$

for many-query workloads.

The experiment should compare:

```text
A. Qwen full-attention PCDM

B. Qwen → partial-KDA converted PCDM

C. native/pretrained Kimi-Linear backbone + PCDM head
```

across:

$$
L_s=
256,\ 1k,\ 4k,\ 16k,\ 64k
$$

and:

$$
M=
1,\ 8,\ 32,\ 128,\ 512.
$$

Primary measurements:

* task accuracy;
* NLL/Brier;
* null AUROC;
* latency;
* state-cache memory;
* marginal cost/query;
* retrieval sensitivity as evidence moves deeper into the state.

---

# 20. Full research hypothesis stack

PCDM can now be expressed as a series of increasingly ambitious hypotheses.

### H1 — Decision retention

A direct decision architecture can retain most of the task performance available in its pretrained backbone without answer generation.

---

### H2 — Shared-state economics

For \(M\) queries:

$$
Latency(M)
\ll
M\times Latency(1)
$$

because state computation is amortized.

This is already strongly supported by v1.

---

### H3 — Probability quality

Distributional decision training produces better calibrated probabilities than prompted language-model confidence or candidate-token probabilities.

Current experiments support this, but broader held-out calibration remains necessary.

---

### H4 — Candidate/generalization

The model can reason over candidate meanings not seen during training and identify when the supplied candidate set is insufficient.

Current results support null detection more strongly than fully unseen candidate spaces.

---

### H5 — Listwise decision semantics

Conditioning each candidate on the complete runtime answer set improves arbitrary-choice reasoning, null behavior and unseen candidate generalization.

---

### H6 — Compressed shared-state memory

KDA/hybrid recurrent attention can further amortize state access and reduce many-query scaling costs without unacceptable loss of semantic reasoning.

---

### H7 — Calibrated workflow utility

Outcome/workflow post-training produces probabilities that yield better downstream software decisions than models trained only for classification accuracy.

This is the PCDM research analogue of RLCD.

---

# 21. Recommended experimental roadmap

## Phase 1 — Finish the current architecture

Complete:

* `joint_v1`;
* second seed;
* latency curves;
* direct-head vs decision-tower ablation.

Do not change the backbone.

---

## Phase 2 — Listwise candidates

Add a small permutation-equivariant candidate mixer.

Compare:

$$
E(x,q,a)
$$

against:

$$
E(x,q,A,a).
$$

Run targeted tests:

* add irrelevant candidate;
* remove correct candidate;
* paraphrase candidate;
* duplicate candidate;
* reorder candidates;
* near-duplicate candidate;
* “none of the above” semantics.

---

## Phase 3 — Calibrated Decision Optimization

Create controlled uncertainty training:

* evidence removal;
* evidence addition;
* counterfactual evidence;
* ambiguity;
* OOD;
* difficulty stratification.

Compare:

```text
soft CE only

vs

CE + Brier / proper scoring

vs

proper scoring + uncertainty shaping
```

Evaluate calibration on held-out task families.

---

## Phase 4 — Workflow RL

Construct a small synthetic decision environment with known asymmetric costs.

For example:

```text
fraud / routing / triage simulation
```

Compare:

```text
accuracy-trained model

vs

calibration-trained model

vs

workflow-RL model
```

Measure actual expected utility and calibration simultaneously.

Only here should we introduce policy-gradient RL.

---

## Phase 5 — KDA scaling

Once behavior is understood, replace portions of the backbone with KDA or attach PCDM to a pretrained Kimi-Linear checkpoint.

The scientific question is:

> Can PCDM's reusable state representation become a **fixed-size recurrent memory** rather than an ever-growing KV cache?

If yes, the final architecture becomes considerably more interesting:

$$
\boxed{
\text{large unstructured state}
\rightarrow
\text{compact reusable neural memory}
\rightarrow
\text{many independent questions}
\rightarrow
\text{typed calibrated distributions}
}
$$

---

# 22. End-state architecture

The ambitious final system is:

```text
                       UNSTRUCTURED STATE
                               │
                               ▼
                    hybrid KDA/global backbone
                               │
                     reusable neural memory
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
       query 1              query 2              query M
          │                    │                    │
          ▼                    ▼                    ▼
   pretrained semantic  pretrained semantic  pretrained semantic
       conditioning         conditioning         conditioning
          │                    │                    │
          ▼                    ▼                    ▼
       decision h           decision h           decision h
          │                    │                    │
          ▼                    ▼                    ▼
       runtime A1           runtime A2           runtime AM
          │                    │                    │
          ▼                    ▼                    ▼
       listwise             listwise             listwise
    candidate mixer      candidate mixer      candidate mixer
          │                    │                    │
          ▼                    ▼                    ▼
      energy + null        energy + null        energy + null
          │                    │                    │
          ▼                    ▼                    ▼
          P1                   P2                   PM

               trained with calibrated decision objectives
                    + eventual workflow-level RL
```

The conceptual endpoint is therefore not simply a faster classifier.

It is:

> **A reusable neural decision substrate: encode the world/state once, ask many arbitrary typed questions about it, and receive probability distributions that software can act on directly.**

RLCD-like training makes those probabilities trustworthy.

KDA potentially makes the underlying state reusable at much larger scales.

The dynamic energy/listwise head makes the answer spaces arbitrary.

These are complementary pieces of the same system rather than competing architectural ideas.
