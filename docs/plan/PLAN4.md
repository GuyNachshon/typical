# PLAN4 — PCDM Redirect: Closed-Set System-One Decision Model

## 0. Why we are redirecting

PCDM was originally pushed toward a stronger constraint than the actual product/research goal:

> expensive reasoning should happen before the candidate set is known.

This led to the candidate-blind decision-state hypothesis:

$$
Z(x,q)
$$

followed by cheap scoring of arbitrary candidates:

$$
F(Z,A)\rightarrow P(A).
$$

E3 tested whether candidate-conditioned reasoning could be compiled into this representation.

So far, the result is negative on the capability that matters most for becoming a general System-One model:

* the option-aware 1.7B teacher reaches `.310` MMLU-Pro among-K;
* choices-only reaches `.222`;
* therefore only approximately `.088` is genuinely question-dependent;
* `z1`, `zr`, and `zr_set` transfer candidate priors and calibration but currently show approximately zero question-dependent closed-book knowledge.

E3-ms remains useful because it tests whether multi-set supervision changes this result.

However:

> **Candidate blindness is not a requirement for a Jev-like model.**

The actual requirement is that each decision has a **closed runtime-defined output space**.

The model is allowed to know:

$$
A=\{a_1,\ldots,a_K\}
$$

when it computes the decision.

Therefore the mainline architecture should be redirected toward:

$$
\boxed{
P(y\mid x,q,A)
}
$$

where the complete choice set participates in one non-generative decision computation.

---

# 1. Revised goal

Build an open System-One decision model that:

1. accepts arbitrary unstructured state \(x\);
2. accepts an arbitrary natural-language query \(q\);
3. accepts a closed runtime-defined answer set \(A\);
4. returns a direct probability distribution over that set;
5. optionally returns an explicit unsupported / none state;
6. does not generate an answer string;
7. reuses state computation across many independent queries;
8. retains the pretrained model's evidence reasoning and parametric knowledge;
9. remains calibrated enough for software to consume probabilities directly;
10. has a scalable path for very large candidate sets.

Formally:

$$
\boxed{
F_\theta(x,q,A)
\rightarrow
P(A,\varnothing)
}
$$

rather than:

$$
F_\theta(x,q)\rightarrow\text{text}.
$$

The closed answer set is part of the decision.

It is **not** part of the global model vocabulary.

---

# 2. What remains from current PCDM

Most of the current work remains relevant.

### Keep

* Qwen pretrained causal backbone;
* state-as-prefix formulation;
* state KV caching;
* independent query branches;
* LoRA adaptation;
* direct probability outputs;
* semantic candidate representations;
* learned null/reject behavior;
* calibrated probabilistic training;
* current energy model for high-cardinality scoring;
* evidence-grounded training/evaluation suite;
* latency infrastructure;
* candidate-set and null diagnostics.

### Remove as a requirement

Do **not** require:

$$
C_{\mathrm{LM}}\perp A
$$

for ordinary `Choice`.

That becomes an optional optimization.

The required systems property is instead:

> **The closed choice set is processed once as part of a single decision pass, not through K independent language-model calls and not through autoregressive answer generation.**

---

# 3. Native Choice architecture

For ordinary bounded choices, process the entire choice set jointly.

Conceptually:

```text
STATE x
   │
   │  expensive once
   ▼
Qwen prefix
   │
   ▼
reusable KV(x)
   │
   │
   ├──────────────────────────────────────┐
   │                                      │
   ▼                                      │
QUERY q                                   │
CHOICES                                   │
  A. a1                                   │
  B. a2                                   │
  C. a3                                   │
  ...                                     │
  K. ak                                   │
DECISION TOKEN                            │
   │                                      │
   ▼                                      │
one Qwen suffix pass against KV(x) ◄──────┘
   │
   ▼
global decision representation h_D
   │
   ├──────── candidate representation c1
   ├──────── candidate representation c2
   ├──────── ...
   └──────── candidate representation cK
                  │
                  ▼
             direct scorer
                  │
        s1 ... sK + s_null
                  │
                  ▼
             softmax / calibrated
                  │
                  ▼
         P(a1 ... aK, null)
```

There is:

* one state prefix computation;
* one suffix computation per query;
* no generated answer;
* no per-candidate LM pass.

---

# 4. Why this should recover parametric knowledge

The current PCDM computes:

$$
h(x,q)
$$

before the answer candidates participate in pretrained reasoning.

The scorer then has to infer:

$$
(h,c_j)\rightarrow s_j.
$$

This works extremely well for evidence/candidate compatibility but fails on problems where the answer must be derived from knowledge stored in the backbone.

The option-aware teacher demonstrates the missing mechanism.

For MMLU-Pro:

$$
\text{candidate-blind interface}\approx .13
$$

while:

$$
\text{same 1.7B backbone with options in context}\approx .31.
$$

Therefore the next architecture should deliberately preserve:

$$
\boxed{
\text{pretrained question ↔ option interaction}
}
$$

instead of trying to reconstruct it in a small head.

---

# 5. Direct readout — do not generate a letter

The simplest option-aware baseline is:

```text
question
options
→ LM
→ next-token probability of A/B/C/...
```

That is useful as a control, but it is not the desired architecture.

PCDM should instead produce an internal decision representation after the complete choice set has been processed.

For example:

$$
h_D =
H_{\text{terminal decision token}}.
$$

Each answer also has a representation:

$$
c_j.
$$

Then:

$$
s_j=
S_\phi(h_D,c_j).
$$

Possible first implementation:

$$
s_j =
MLP(
[
h_D;
c_j;
h_D\odot c_j;
|h_D-c_j|
]
).
$$

A better low-cost form may be:

$$
u=W_hh_D,
\qquad
v_j=W_cc_j,
$$

$$
s_j=
u^\top v_j
+
w^\top(u\odot v_j).
$$

The important property is:

$$
h_D=h_D(x,q,A).
$$

Therefore:

$$
s_i-s_j
$$

can change when the candidate set changes.

The model no longer obeys IIA by construction.

---

# 6. Candidate representations

Three candidate representations should be compared.

### A. Contextual option representation

Pool each option's hidden states from the joint suffix.

Advantage:

* candidate representation uses the same LM;
* very cheap because the tokens have already been processed.

Disadvantage:

* causal asymmetry;
* earlier options do not see later options directly.

The global \(h_D\) does see all options, so this may not matter.

### B. Semantic candidate representation

Use the existing Qwen3-Embedding representation:

$$
c_j=C_{\text{emb}}(a_j).
$$

Advantages:

* excellent unseen-label transfer;
* already demonstrated;
* cacheable for stable vocabularies.

### C. Hybrid

Use:

$$
[c_j^{context};c_j^{semantic}]
$$

or a projected combination.

This may preserve both:

* pretrained joint reasoning;
* strong semantic label transfer.

Start with A+B as an ablation rather than assuming one is correct.

---

# 7. Option-order invariance

The model sees the choices sequentially inside a causal transformer, so order artifacts are possible.

The output semantics must nevertheless be approximately permutation-equivariant:

$$
F(x,q,\pi(A))
\approx
\pi(F(x,q,A)).
$$

Training should therefore use:

* randomized option order;
* candidate identities tracked independently of letters;
* optional consistency loss across option permutations.

Evaluation must report:

$$
\Delta_{\text{reorder}}
$$

in both accuracy and log-odds.

Do not train the model to reproduce arbitrary positional biases of the teacher.

---

# 8. Null / reject

Retain an explicit null:

$$
P(\varnothing\mid x,q,A).
$$

However, null and choice should continue to be evaluated separately.

Report:

* among-K accuracy;
* overall accuracy;
* false abstention;
* true-null recall;
* null AUROC;
* coverage / selective risk.

Do not allow better null behavior to hide weak candidate reasoning or vice versa.

The existing K-sweep result remains important:

$$
K\uparrow
$$

makes null detection intrinsically harder through both:

* extreme-value effects;
* harder near-miss distractors.

This is now treated as a property to characterize rather than something that a particular null functional form must eliminate.

---

# 9. Two computational regimes

One architecture does not need to serve every value of \(K\).

## Regime A — Native Choice

For moderate \(K\):

$$
A=\{a_1,\ldots,a_K\}
$$

goes directly into one option-aware suffix.

```text
state KV
   +
query + all candidates
   ↓
one LM suffix pass
   ↓
direct probability readout
```

This should be the default System-One decision primitive.

The native \(K\) limit should be empirical rather than hard-coded initially.

Measure:

$$
K=2,4,10,32,64,128,256.
$$

---

## Regime B — Large Choice

For very large \(K\), reuse the current PCDM energy model.

```text
state + query
      │
      ▼
current fast energy model
      │
score K candidates cheaply
      │
      ▼
top-r
      │
      ▼
Native Choice over r candidates
      │
      ▼
final calibrated distribution
```

For example:

$$
K=10,000
\rightarrow
r=16\text{ or }32.
$$

Then:

$$
C(K)
=
C_{\text{query}}
+
O(Kd)
+
C_{\text{native-choice}}(r).
$$

Expensive choice-set reasoning is bounded by \(r\), not \(K\).

This preserves one of PCDM's strongest existing results: extremely cheap large-\(K\) candidate reduction.

---

# 10. Revised role of E3

E3/E3-ms is **not cancelled**.

It becomes a separate research question:

> Can option-conditioned reasoning be compiled out of the online LM path?

If successful, it gives an even faster Native Choice implementation.

If unsuccessful, it does not falsify the Jev-alternative thesis.

Therefore:

### Mainline

$$
(x,q,A)
\rightarrow
\text{one candidate-aware decision pass}
\rightarrow
P(A).
$$

### Optimization branch

$$
(x,q)
\rightarrow Z
$$

followed by cheap candidate interaction.

E3 determines whether some or all Native Choice computation can eventually be replaced by this cheaper path.

---

# 11. Immediate experiment: `native_choice_v1`

Build the smallest possible option-aware direct-decision model.

### Input

```text
[state prefix]

[query]
Question text

[choices]
A. candidate 1
B. candidate 2
...
J. candidate K

[decision]
<terminal token>
```

The terminal token's hidden state:

$$
h_D
$$

is the global decision state.

It has causal access to:

$$
x,q,A.
$$

### Backbone

Use the complete Qwen3-1.7B backbone for the suffix.

The state prefix remains KV-cached.

Do **not** reuse the earlier conclusion that layer 28 is bad.

That experiment read final-layer features through the candidate-blind PCDM interface.

Here the final layer is being used for the behavior it was pretrained for:

> making a decision after reading all relevant context.

This is a different hypothesis.

---

# 12. `native_choice_v1` readouts

Run three cheap readout ablations.

### N1 — Letter logits

The existing MCQ-LoRA behavior.

This is the capability control.

### N2 — Direct candidate head

$$
h_D + c_j
\rightarrow s_j.
$$

No generated letter.

This is the actual PCDM architecture.

### N3 — Contextual pointer

$$
h_D
$$

scores contextualized option representations taken from the same suffix.

Potentially:

$$
s_j=h_D^\top W h_{a_j}.
$$

N2/N3 must approach N1 performance.

If they do, the generative letter interface is unnecessary.

---

# 13. Training mixture

Train the Native Choice model on both existing regimes.

### Evidence-grounded

Existing PCDM datasets:

* NLI;
* BoolQ;
* SQuAD/null;
* intents;
* classification tasks;
* null/near-miss examples;
* held-out label-space framework.

### Closed-book

Current `data_kb`:

* MMLU auxiliary data after deduplication;
* ARC;
* OpenBookQA;
* CommonsenseQA;
* SciQ;
* QASC;
* LogiQA;
* AQuA;
* MedMCQA subset;
* etc.

MMLU/MMLU-Pro remain held out.

This produces one model that must support:

$$
\text{reason from state}
$$

and:

$$
\text{reason from pretrained knowledge}.
$$

That is the central capability requirement for a Jev alternative.

---

# 14. Training objective

Start simple.

For choice distribution:

$$
L_{\text{choice}}
=
-\sum_jp_j^*\log p_j.
$$

For teacher distributions where useful:

$$
L_{KD}
=
T^2KL(P_T^T\Vert P_S^T).
$$

For null:

$$
L_{\text{null}}.
$$

Total:

$$
L=
L_{\text{choice}}
+
\lambda_{KD}L_{KD}
+
\lambda_NL_{\text{null}}.
$$

No RL yet.

Do not introduce RLCD-like training until the architecture retains both evidence and parametric capability.

---

# 15. Success criteria

The model should be judged on four independent axes.

## A. Evidence retention

Native Choice should retain current PCDM capability within approximately the established seed/noise floor on:

* SNLI;
* MNLI;
* ANLI;
* BoolQ;
* CLINC;
* HWU64.

A model that solves MMLU but loses evidence reasoning is not the target.

---

## B. Parametric knowledge

Measure:

$$
\Delta_q
=
Acc(q,A)
-
Acc(q_{\text{shuffled}},A).
$$

Native Choice should reproduce most of the option-aware teacher's question-dependent component.

Current teacher:

$$
\Delta_q\approx0.088.
$$

The desired result is not merely high raw MCQ accuracy; it is substantial:

$$
\Delta_q>0.
$$

Ideally:

$$
\Delta_q^{student}
\approx
\Delta_q^{teacher}.
$$

---

## C. Decision quality

Report:

* accuracy;
* among-K accuracy;
* NLL;
* Brier;
* ECE;
* ChaosNLI;
* null AUROC;
* false abstention;
* true-null recall.

Native Choice must not buy parametric knowledge by destroying the probability-quality result.

---

## D. Systems

Measure:

$$
L_{\text{native}}(K)
$$

for:

$$
K=2,4,10,32,64,128,256.
$$

Compare to:

1. current PCDM energy path;
2. query-batched per-candidate log-prob baseline;
3. MCQ-LoRA / letter-logit baseline.

The important requirement is:

$$
\boxed{
\text{one LM suffix per query, not one per candidate}
}
$$

and:

$$
\boxed{
\text{no autoregressive answer generation}.
}
$$

There is no requirement that latency be independent of \(K\).

It should scale primarily with total option-token count.

---

# 16. Decision rules

### If N2/N3 match N1

Adopt direct Native Choice.

This establishes:

> candidate-conditioned pretrained reasoning can be exposed directly as typed probabilities without using the generative answer interface.

Proceed to calibration training.

### If N1 works but N2/N3 fail

The candidate-aware backbone has the capability, but the direct readout still cannot expose it.

Focus the next phase entirely on the readout—not on backbone scale.

### If all N1/N2/N3 fail

Then the current 1.7B backbone/training is inadequate.

Only then test larger models.

### If Native Choice is accurate but too slow beyond some \(K^*\)

Define:

$$
K_{\text{native}}=K^*
$$

and route larger sets through:

$$
\text{energy prefilter}\rightarrow\text{Native Choice}.
$$

This is an architectural feature, not a failure.

---

# 17. Revised PCDM architecture

The resulting system is:

```text
                           STATE
                             │
                             ▼
                       cached Qwen KV
                             │
             ┌───────────────┴────────────────┐
             │                                │
             │                                │
        moderate K                         huge K
             │                                │
             ▼                                ▼
     query + all choices                query representation
             │                                │
             ▼                                ▼
      ONE Qwen suffix               cheap semantic energy scorer
             │                                │
             ▼                                ▼
    global decision state                  top-r
             │                                │
             │                    ┌───────────┘
             ▼                    ▼
          candidate-aware Native Choice
                    │
                    ▼
             direct probabilities
                    │
                    ▼
             P(A, null)
```

This keeps:

* shared-state computation;
* direct typed probabilities;
* no autoregressive answer generation;
* arbitrary runtime answer sets;
* pretrained option-conditioned reasoning;
* scalable large-K operation.

---

# 18. What becomes novel

Do not claim novelty for:

* multiple choice;
* KV caching;
* candidate embeddings;
* reranking;
* energy scoring;
* direct classification heads individually.

The research claim should remain narrower.

The immediate engineering/research contribution is:

> **A pretrained causal language model can be converted into a shared-state, non-generative closed-set decision engine that directly returns calibrated probabilities while retaining both evidence-conditioned reasoning and parametric knowledge.**

The more distinctive extension remains E3:

> **How much of its candidate-conditioned reasoning can be compiled into a reusable candidate-blind state without loss of capability?**

If E3 succeeds, that becomes a stronger architectural contribution.

If it fails, the main System-One architecture remains intact.

---

# 19. Revised roadmap

## Phase A — finish E3-ms

Finish the already-running pilot.

Record it as the candidate-blind compilation experiment.

Do not let its result gate the main project.

---

## Phase B — `native_choice_v1`

Immediately build the candidate-aware single-pass architecture.

Start from the existing MCQ-LoRA infrastructure.

Compare:

* N1 letter logits;
* N2 global decision state + semantic candidates;
* N3 global decision state + contextual option representations.

Use the complete 1.7B backbone.

---

## Phase C — unify evidence + knowledge

Train Native Choice on:

$$
\text{existing decision mix}
+
\text{closed-book corpus}.
$$

Test whether a single model retains both.

This is the critical Jev-alternative experiment.

---

## Phase D — determine the native-K boundary

Benchmark quality and latency over increasing \(K\).

Choose the boundary empirically.

Above it:

$$
\text{current energy scorer}
\rightarrow
\text{top-r}
\rightarrow
\text{Native Choice}.
$$

---

## Phase E — calibration / RLCD-like work

Only after evidence + knowledge retention are demonstrated.

Investigate:

* human disagreement;
* ambiguity;
* missing evidence;
* OOD;
* counterfactual evidence;
* workflow utility;
* proper scoring;
* eventual RL where downstream rewards justify it.

---

# 20. Revised thesis

The project is no longer:

> Can all reasoning be compressed into a candidate-independent embedding?

The project is:

> **Can we turn a pretrained generative language model into a fast, reusable, closed-set probabilistic decision model?**

The model should reason over the complete decision specification:

$$
(x,q,A)
$$

but output only:

$$
P(A,\varnothing).
$$

No explanation is required.

No answer text is generated.

The state is reusable.

Questions are independent.

Choices are dynamic but closed for each invocation.

Large answer spaces use a cheap first-stage scorer.

And calibration training determines whether the returned probabilities are trustworthy enough to become a software primitive.

That is the mainline PCDM / open System-One direction.

The candidate-blind \(Z(x,q)\) architecture remains an optional stronger result—not the definition of success.

---

# 21. Execution notes (lead, 2026-09-19)

- **What already exists:** `pcdm/mcq.py` (`MCQHead`: options rendered in the suffix, LoRA on the top 8 of 28 layers, letter readout with
  hierarchical chunking above 51 options; `run_batch_mcq`, `collate_mcq`; `--readout mcq`) *is* N1. `teacher_kb` (N1 on data_kb) and
  `mcq_lora` (N1 on evidence data v3) are the existing capability controls: MMLU-Pro among-K .310 / Δ_q +.088, and evidence-level
  .853/.727/.400/.768 respectively. Cold/warm latency, counterfactual battery, choices-only/shuffled probes, disaggregated
  abstention metrics all carry over unchanged.
- **Build (`--readout native`, `--nc_head {n2,n3,n2n3}`):** terminal decision token appended after the rendered options; h_D = its
  last-layer hidden state (z-scored); N2 candidates = Qwen3-Embedding vectors (VecCache; cold path embeds); N3 candidates = mean of
  each option's own hidden states in the suffix (spans recorded at render time); N2N3 = concat. Scorer = factorized bilinear
  `u = W_h h_D, v_j = W_c c_j, s_j = u·v_j + w·(u⊙v_j)` + small residual MLP; null from h_D + set statistics (reuse the factored-null
  gate); loss = the existing soft-CE over [candidates, ∅] (+ optional KD); options shuffled every training step with candidate
  identities tracked independently of letters. Letters are kept in the rendering (the LM was pretrained on them) but never read out.
- **Runs (Phase B/C together, one training mix = data_v5 + data_kb hard labels, 12k steps, same seed):** `nc_n1` (letter readout,
  control), `nc_n2`, `nc_n3`. ~$3–4 each on the A100. Eval = full suite + MMLU-Pro + `mmlu_cf` + probes (Δ_q) + kb val + TruthfulQA.
- **Success (per §15):** evidence within seed floor; Δ_q(student) ≈ Δ_q(teacher) (+.088); NLL/ECE/Brier/ChaosNLI/null no worse than
  `joint_emb_lw_v5` beyond noise; one suffix per query, no generation; latency reported for K = 2…256 against the energy path and the
  query-batched log-prob baseline (Phase D).
- **E3-ms** finishes as Phase A and is recorded as the candidate-blind compilation experiment; it does not gate this.
- **Budget cut (2026-09-19 18:40, ~$22 left):** native runs cost 0.72 s/step on the H100 (≈ $8.4 per 12k-step run). Phase B runs only
  `nc_n2` and `nc_n1` (the §16 decision needs N2 vs N1 first); `nc_n3`/`nc_n2n3` are skipped placeholders. `e3d_zr_emb` (candidate-blind
  head with the semantic embedder) is cancelled — under PLAN4 the semantic candidates are native to N2. If N2 < N1, N3 is the next
  spend; if N2 ≈ N1, N3 is optional.
- **Outcome (2026-09-20 00:30; balance at the time was actually $120.89 — the "exhausted" estimate was wrong):** §16 first branch. `nc_n3` (h_D × contextual option states) ≈ `nc_n1`
  on knowledge — MMLU-Pro among-K .314 vs .325, Δ_q .111 vs .112 (teacher .088), TruthfulQA .461 vs .465, kb val .580 vs .547,
  CLINC-heldout .916 / Banking77-77 .497 / 20NG .559 — with a quarter of N1's IIA (.13 vs .46) and better evidence retention.
  `nc_n2` (semantic candidates) retains 87% of the teacher's Δ_q but loses raw score for lack of slot identity. Native choice is
  adopted for Regime A; the energy path stays for evidence / null / large K. Open, in order: a K-robust native null (abstention error
  .18 on MMLU-Pro, P(∅|absent) .99 at K = 150), the 5–13-pt evidence gap of the options-in-suffix formulation, native-path latency vs K
  (Phase D, needs a KV-cached suffix in pcdm/bench.py), an 8B fine-tuned teacher as scale reference. Full numbers: REPORT.md §3l;
  `uv run pcdm/report_native.py teacher_kb nc_n1 nc_n2 nc_n3`.
