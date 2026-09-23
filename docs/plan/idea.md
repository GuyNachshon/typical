# Parallel Calibrated Decision Model — v0

## 1. Core thesis

Current LLMs model:

$$
P(\text{next token}\mid x,t_{<k})
$$

and obtain decisions indirectly through text generation.

We instead want to directly model:

$$
\boxed{
P(y\mid x,q,A)
}
$$

where:

* \(x\) = arbitrary unstructured state
* \(q\) = a question / decision to make about that state
* \(A=\{a_1,\ldots,a_K\}\) = runtime-defined set of possible answers
* \(y\in A\cup\{\varnothing\}\)
* \(\varnothing\) = optional “none / insufficient evidence / unknown” outcome

For multiple decisions:

$$
\boxed{
F_\theta
\left(
x,
\{(q_i,A_i)\}_{i=1}^M
\right)
\rightarrow
\{P_i\}_{i=1}^M
}
$$

All \(M\) decisions should be evaluated **in parallel**, sharing the computation required to understand \(x\).

The model does not generate explanations, JSON, reasoning traces, or arbitrary strings.

Its native output is a probability distribution over a predefined typed space.

---

# 2. Desired properties

The model should satisfy five properties.

### A. Dynamic decision spaces

The answer classes must not be fixed during training.

This should work:

```python
Choice(
    "approve",
    "reject",
    "send_to_manual_review"
)
```

as should:

```python
Choice(
    "dog",
    "cat",
    "horse",
    "none"
)
```

without adding new classifier heads.

### B. Probabilistic output

The primitive output is:

$$
P(a_j|x,q,A)
$$

rather than:

$$
\operatorname{argmax}_j P(a_j)
$$

### C. Calibration

Across predictions assigned confidence \(p\):

$$
P(Y=\hat Y\mid C=p)\approx p
$$

A model saying 0.8 should actually be correct approximately 80% of the time on the relevant population.

### D. Parallelism

For:

$$
q_1,q_2,\ldots,q_{100}
$$

we should not run 100 autoregressive inference loops.

The state is understood once and queried many times.

### E. Closed output space

If the supplied type is:

```python
Choice["approve", "reject"]
```

the model physically cannot output:

```text
"Maybe ask Sarah."
```

Type safety comes from the computational interface, not instruction following.

---

# 3. Architecture

The conceptual architecture is:

```text
                     STATE x
                        │
                        ▼
              ┌──────────────────┐
              │   State Encoder  │
              │      fθ(x)       │
              └────────┬─────────┘
                       │
                 State Memory H
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
      q₁              q₂               q₃
       │               │                │
    choices         choices          choices
       │               │                │
       ▼               ▼                ▼
 Decision Slot     Decision Slot     Decision Slot
       │               │                │
       ▼               ▼                ▼
   energies          energies         energies
       │               │                │
       ▼               ▼                ▼
      P₁              P₂               P₃
```

The crucial decomposition is:

$$
H=f_\theta(x)
$$

followed by many cheap queries against \(H\).

---

# 4. State encoder

Given state tokens:

$$
x=(x_1,\ldots,x_L)
$$

produce:

$$
H=f_\theta(x)\in\mathbb R^{L\times d}
$$

Optionally compress this into a smaller latent memory:

$$
Z=C(H)\in\mathbb R^{N\times d},\qquad N\ll L
$$

so that decision computation scales against \(N\), not the complete context length.

This makes a Perceiver-like architecture attractive later:

```text
20k state tokens
       ↓
Transformer
       ↓
128–512 state latents
       ↓
1000 parallel queries
```

For v0, compression is unnecessary.

---

# 5. Decision slots

For every query \(q_i\), construct a decision representation:

$$
h_i =
g_\theta(q_i,H)
$$

where \(g\) is primarily cross-attention:

$$
h_i =
\operatorname{CrossAttention}
(Q=q_i,K=H,V=H)
$$

All queries can be stacked:

$$
Q=
[q_1,\ldots,q_M]
$$

and evaluated as one tensor operation.

Queries should either:

1. not attend to each other, or
2. have explicitly controlled interaction.

For the first model I would choose **no query-to-query attention**.

This ensures:

$$
P_i=P(y_i|x,q_i,A_i)
$$

does not depend on query ordering.

---

# 6. Dynamic candidate scoring

Each candidate is itself semantic input:

$$
c_{ij}=e_\theta(a_{ij})
$$

The model evaluates:

$$
s_{ij} =
S_\theta(h_i,c_{ij})
$$

A simple initial scorer:

$$
s_{ij}
=
MLP(
[h_i;
c_{ij};
h_i\odot c_{ij};
h_i^\top c_{ij}]
)
$$

A more interesting formulation is explicitly energy-based:

$$
E_\theta(x,q_i,a_{ij})
$$

with:

$$
s_{ij}=-E_\theta(x,q_i,a_{ij})
$$

and:

$$
P(a_{ij}|x,q_i,A_i)
=
\frac{\exp(-E_{ij}/T)}
{\sum_k\exp(-E_{ik}/T)}
$$

This gives us the basic abstraction:

$$
\boxed{
\text{language-defined symbolic space}
+
\text{neural energy function}
}
$$

which I think is the most useful way of thinking about the model.

---

# 7. The null state

Pure softmax has an important defect.

Given:

```text
Question:
What animal is described?

Choices:
dog
cat
horse
```

and input:

```text
The document discusses semiconductor manufacturing.
```

softmax still requires:

$$
P(dog)+P(cat)+P(horse)=1
$$

Therefore introduce:

$$
E_\varnothing
$$

giving:

$$
P(y)\quad
y\in
\{\varnothing,a_1,\ldots,a_K\}
$$

\(\varnothing\) can mean:

* none of the above,
* unsupported,
* insufficient information,
* out-of-distribution,
* malformed question,

depending on the type definition.

This should be an architectural primitive rather than a candidate literally called `"unknown"`.

---

# 8. Output types

The initial system only needs three primitives.

### `Choice`

$$
Y\sim Categorical(P_1,\ldots,P_K,P_\varnothing)
$$

Example:

```python
Choice[
    "approve",
    "deny",
    "manual_review"
]
```

### `Bool / Proposition`

For proposition \(r\):

$$
P(r|x)
=
\sigma(s_r)
$$

Example:

```python
Proposition("The customer is likely to churn")
→ 0.71
```

This avoids forcing true and false through separate natural-language candidates.

### `Score`

For ordered values:

$$
Y\in\{1,\ldots,K\}
$$

predict a complete distribution:

$$
P(Y=k)
$$

and expose:

$$
E[Y]=\sum_k kP(Y=k)
$$

rather than directly regressing a scalar.

Later we can add continuous distributions, multi-label outputs, extraction spans, etc.

---

# 9. Confidence

I would **not initially train a separate confidence head**.

Confidence should derive from the predictive distribution itself:

Top probability:

$$
C_{\max}=\max_iP_i
$$

Entropy:

$$
H(P)=-\sum_iP_i\log P_i
$$

Margin:

$$
C_{\text{margin}}=P_{(1)}-P_{(2)}
$$

If we later find that correctness cannot be inferred adequately from the distribution, then add a learned correctness predictor.

But first we should force the primary probabilities themselves to carry the epistemic signal.

---

# 10. Training representation

Normalize every dataset into:

```python
DecisionExample(
    state=x,

    query=q,

    candidates=[
        a1,
        a2,
        ...
    ],

    target_distribution=[
        p1,
        p2,
        ...
    ],

    null_probability=p_null,

    supervision_type=
        "hard_label"
        | "human_distribution"
        | "teacher_distribution"
        | "resolved_outcome",

    group_id=...,

    perturbation_type=...,

    metadata=...
)
```

This is important.

SNLI, UNLI, CLINC, forecasting, QA and synthetic tasks stop being different tasks.

They all become observations of:

$$
P(y|x,q,A)
$$

UNLI is particularly useful because it directly supplies subjective probabilistic judgments rather than categorical labels.

ChaosNLI goes further by providing 100 human annotations per example, allowing the empirical distribution of judgments to be learned directly.

---

# 11. Training Stage I — Decision pretraining

Start from a pretrained language model rather than learning semantic knowledge from scratch.

For v0:

```text
pretrained 1–4B LM
        ↓
shared state representation
        ↓
small bidirectional decision tower
        ↓
dynamic energy head
```

Initially preserve most pretrained weights.

Train primarily:

* decision tower,
* candidate encoder/adapters,
* energy scorer,
* upper backbone layers.

For deterministic examples:

$$
L_{\text{hard}}
=
-\log P_\theta(y^*)
$$

For soft labels:

$$
L_{\text{soft}}
=
-\sum_j
P^*(a_j)
\log P_\theta(a_j)
$$

equivalently minimizing:

$$
D_{KL}(P^*||P_\theta)
$$

This phase primarily teaches:

> Convert semantic understanding into typed decision distributions.

---

# 12. Training Stage II — uncertainty shaping

Now add examples where probability itself matters.

### Evidence deletion

$$
x \rightarrow x^{-}
$$

Remove relevant evidence.

Desired behavior:

$$
H(P(x^-)) > H(P(x))
$$

### Evidence addition

$$
x\rightarrow x^{+}
$$

Add decisive evidence.

Desired:

$$
P(y^*|x^+)>P(y^*|x)
$$

### Semantic invariance

For meaning-preserving perturbation \(T\):

$$
P(y|x)\approx P(y|T(x))
$$

Use:

$$
L_{\text{inv}}
=
JS(P_\theta(x),P_\theta(T(x)))
$$

### Counterfactual sensitivity

For meaning-changing intervention \(I\):

$$
P(y|x)
\not\approx
P(y|I(x))
$$

with a known direction of change.

### OOD/null

Train explicitly:

$$
P(\varnothing|x,q,A)
\rightarrow1
$$

when the candidate set cannot explain the input.

---

# 13. Objective

A reasonable v0 objective:

$$
L =
L_{\text{decision}}
+
\lambda_sL_{\text{soft}}
+
\lambda_iL_{\text{invariance}}
+
\lambda_cL_{\text{counterfactual}}
+
\lambda_nL_{\text{null}}
$$

I would **not directly optimize ECE** during the first stage.

Log-loss, Brier score and related proper scoring rules have the important property that, in expectation, the true predictive distribution is optimal. Proper scoring rules are therefore a much cleaner foundation for probabilistic training than simply minimizing a histogram-based calibration metric.

We evaluate calibration separately.

---

# 14. Training Stage III — RLCD-like post-training

This is where I would experimentally investigate whether RL adds something.

Define an episode/workflow:

$$
W=(x,q_1,\ldots,q_M,\text{program})
$$

The model produces:

$$
P_1,\ldots,P_M
$$

The surrounding program uses these probabilities to make downstream decisions.

Define:

$$
R(W)
=
R_{\text{task}}
+
\alpha R_{\text{proper-score}}
-
\beta R_{\text{miscalibration}}
-
\gamma R_{\text{inconsistency}}
$$

Then:

$$
J(\theta)
=
E_{W}
[
E_{a\sim P_\theta}[R(W,a)]
]
$$

For tiny discrete action spaces, we may often calculate the expected reward exactly rather than sampling.

For example:

$$
E[R]
=
\sum_a P_\theta(a)R(a)
$$

This means **RL may not initially be necessary at all**.

RL becomes genuinely useful when:

* decisions affect later states,
* rewards arrive after several decisions,
* the workflow is nondifferentiable,
* we optimize downstream business/agent utility,
* several probabilistic decisions interact.

That distinction is important.

The real question is not:

> “How do we copy RLCD?”

It is:

> **Is calibration mostly solved by proper probabilistic supervision, or does workflow-level reinforcement learning materially improve it?**

That is experimentally answerable.

---

# 15. Parallel inference

Inference should conceptually be:

```text
x
│
├── expensive state encoding ─────────────── once
│
└── state memory H
       │
       ├── q1 × candidates ── energy
       ├── q2 × candidates ── energy
       ├── q3 × candidates ── energy
       │
       └── qM × candidates ── energy

              all vectorized
```

There is no:

```text
token → token → token → token
```

loop.

And importantly, this isn't ordinary **batch inference**.

Batching parallelizes:

$$
x_1,x_2,x_3
$$

Our architecture additionally parallelizes:

$$
q_1,q_2,\ldots,q_M
$$

**inside the same state**.

That is likely where much of the interesting serving economics come from.

TypeSafe explicitly says Jev produces all probabilities in parallel, and notes that very high-cardinality choice problems use a two-stage procedure of independent scoring followed by explicit choice. That makes a candidate-energy/scoring architecture at least directionally compatible with their publicly described behavior, though it does not establish that Jev uses this architecture.

---

# 16. High-cardinality decisions

For:

$$
K=1,000,000
$$

full normalized scoring becomes expensive.

Use:

### Stage 1: retrieval / independent scoring

$$
s_j=f(x,q,a_j)
$$

for candidates.

Retrieve top \(k\):

$$
A'=\operatorname{TopK}(s)
$$

### Stage 2: normalized choice

$$
P(a|x,q,A')
$$

This also creates an interesting distinction:

$$
\text{relevance score}
\neq
\text{choice probability}
$$

which is probably desirable.

---

# 17. First falsifiable hypotheses

## H1 — Decision compression

A non-autoregressive decision model can retain a meaningful fraction of a much larger reasoning model's performance on bounded decision tasks.

This is the fundamental hypothesis.

If false, nothing else matters.

---

## H2 — Parallel advantage

For \(M\) questions over the same state:

$$
Latency(M)
\ll
M\times Latency(1)
$$

The advantage should increase as \(M\) grows.

---

## H3 — Probability quality

Soft-distribution + uncertainty training produces lower:

$$
NLL,\quad Brier,\quad calibration\ error
$$

than:

* LLM verbal confidence,
* LLM choice-token logits,
* hard-label classifier.

---

## H4 — Unknown detection

Explicit null modeling substantially lowers confident errors on OOD / insufficient-information examples.

---

## H5 — Workflow reliability

Calibrated probabilities produce better downstream decisions than equally accurate but poorly calibrated predictors.

This matters more than raw classification accuracy.

---

# 18. Critical baselines

The model should be compared against:

```text
A. Autoregressive LLM prompted for JSON + confidence

B. Autoregressive LLM using candidate token/log probabilities

C. Pretrained LM + fixed classification head

D. Dynamic candidate scorer

E. Dynamic energy model + null

F. E + soft-label training

G. F + uncertainty/counterfactual training

H. G + RLCD-like workflow post-training
```

If D–H do not outperform B meaningfully in either quality or efficiency, the new architecture isn't justified.

---

# 19. Evaluation

Primary:

$$
Accuracy
$$

$$
NLL
$$

$$
Brier
$$

$$
ECE / calibration curves
$$

$$
AUROC_{\text{unknown}}
$$

$$
Risk@Coverage
$$

Then architectural metrics:

$$
Latency
$$

$$
Queries/sec
$$

$$
Cost/query
$$

$$
Latency\ as\ function\ of\ M
$$

And behavioral:

$$
\text{paraphrase consistency}
$$

$$
\text{confidence after evidence deletion}
$$

$$
\text{response to counterfactual evidence}
$$

$$
\text{OOD confidence}
$$

---

# 20. v0 experiment

I would deliberately make the first model small.

### Model

$$
\sim1B
$$

pretrained backbone.

Small decision tower + dynamic energy scorer.

### Data

Approximately:

$$
500k-1M
$$

decision examples.

Main sources:

```text
SNLI / MNLI / ANLI       → hard semantic decisions
UNLI                     → scalar probabilities
ChaosNLI                 → human distributions
CLINC150 + OOS           → null / unknown
Natural Questions        → answerability
CalibratedMath           → controlled confidence
synthetic perturbations  → invariance + evidence effects
```

### Most important experiment

Compare:

$$
\boxed{
\text{AR LLM}
\quad vs\quad
\text{dynamic decision model}
\quad vs\quad
\text{calibrated decision model}
}
$$

under identical state/query/choice inputs.

---

# 21. What would constitute success?

Not “we recreated Jev.”

The interesting result would be:

> **A pretrained language model can be converted into a general-purpose non-autoregressive decision engine whose runtime answer space is dynamically specified in natural language, whose outputs are meaningfully calibrated probabilities, and whose cost scales primarily with understanding the state rather than generating an answer.**

That would already be a distinct research result.

The stronger version is:

> **General reasoning can be partially amortized into a shared state representation and queried cheaply through parallel probabilistic decision heads.**

And that has implications substantially beyond TypeSafe.

It starts looking like a general interface between learned representations and software:

$$
\boxed{
\text{unstructured world state}
\rightarrow
\text{shared latent state}
\rightarrow
\text{arbitrary probabilistic queries}
}
$$

which is also why the architecture begins to overlap with **world models** rather than merely classification.

