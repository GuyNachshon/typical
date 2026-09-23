I’d stop broad exploration now and switch to a much tighter **Phase 5: turn the discovered pieces into one coherent System-One model**.

The architecture search has answered most of the foundational questions. `nc_n3` showed that candidate-conditioned pretrained reasoning can be exposed directly through a non-generative contextual readout, essentially matching the letter-based knowledge signal while reducing its order dependence.  Meanwhile, the energy path remains clearly superior for evidence tasks, null detection, and large-\(K\) efficiency. 

So I’d proceed in this order:

## 0. Freeze and reproduce what you have

Before another training run:

* commit everything;
* pin exact configs/dataset manifests;
* upload N3 and all critical checkpoints;
* save the MMLU frozen slice + leak exclusions;
* save generated teacher labels;
* make `pcdm/report_native.py` regenerate the headline tables from checkpoints;
* delete the volume only after this works from a clean checkout.

At this point the experiments are valuable enough that reproducibility is higher priority than another +2%.

---

## 1. Measure Native Choice latency

This is the biggest missing result and requires **no new model training**.

Benchmark N3 with proper state KV reuse at:

$$
K=2,4,10,32,64,128,256
$$

for several state lengths:

$$
L_s=256,1k,2k
$$

and preferably \(M=1,32,256\) queries per state.

Compare:

1. **Energy PCDM**
2. **Native N3**
3. **batched candidate-logprob baseline**

We need to learn the actual Native Choice operating region.

The result should determine an empirical crossover \(K^*\):

$$
K\le K^*
\Rightarrow \text{Native Choice}
$$

$$
K>K^*
\Rightarrow \text{energy retrieval}\rightarrow top-r\rightarrow\text{Native Choice}.
$$

That gives you the real serving architecture rather than guessing at 32/255/etc.

---

# 2. Fix the Native Choice formulation, not its head

I wouldn't run N4/N5/N6 heads.

N1/N2/N3 have already told you the readout story.

The remaining evidence regression is common to the entire native family: N3 is .863/.752/.431 on SNLI/MNLI/ANLI versus .909/.880/.555 for the energy model. 

So attack the **input representation**.

### `native_v2`: remove letters

Instead of:

```text
A. entailment
B. neutral
C. contradiction
```

use structural option delimiters:

```text
<choice>
entailment
</choice>

<choice>
neutral
</choice>

<choice>
contradiction
</choice>
```

Each candidate's contextual representation comes from its `<choice>` span.

Candidate identity is external.

Randomizing candidate order should therefore teach:

$$
F(x,q,\pi(A))
\approx \pi(F(x,q,A)).
$$

The goals for this experiment are very explicit:

* retain N3 MMLU \(\Delta_q\);
* reorder \(\Delta p\): `.10 → ideally < .03`;
* recover meaningful evidence accuracy.

If this works, it's likely the canonical `Choice` representation.

---

# 3. Stop asking Native Choice to solve abstention

This is now one of the clearest conclusions in the data.

N3 discrimination works, but native null is broken:

* MMLU among-K `.314` vs accuracy `.133`;
* OOS recall `.204`;
* \(P(\varnothing|absent,K=150)\approx.99\). 

Meanwhile the energy model has excellent support/null discrimination.

So explicitly factor:

$$
r=P(\text{answerable}\mid x,q,A)
$$

and:

$$
P_N(a_j)
=
P(a_j\mid x,q,A,\text{answerable}).
$$

Then:

$$
P(\varnothing)=1-r
$$

$$
P(a_j)=rP_N(a_j).
$$

Initially, **reuse the existing energy model for \(r\)**.

This isn't a hack. It separates two semantically different questions:

> Does the provided answer space contain a supported answer?

from:

> If so, which answer is correct?

That should be tested mostly as an eval/composition experiment before retraining anything.

---

# 4. Then try to unify the two experts

Once the support gate works, I'd investigate whether the energy and native paths can become complementary rather than separate.

My first attempt would be a residual:

$$
s_j
=
s_j^{energy}
+
g\,\Delta s_j^{native}.
$$

Or learned fusion:

$$
s_j =
F(
s_j^{energy},
s_j^{native},
h_E,
h_N
).
$$

The conceptual split is compelling:

* energy path: **what does the supplied evidence support?**
* native path: **what does pretrained knowledge + joint choice reasoning imply?**

Test whether fusion can approach:

$$
\text{energy evidence performance}
+
\text{native knowledge performance}.
$$

That's probably the most important architecture experiment left.

I would not spend much time on it if it immediately causes interference. A two-expert model is perfectly legitimate.

---

# 5. Implement the actual three typed primitives

Once Choice is stable, stop measuring everything through generic multiclass classification.

Make the API correspond to the actual model abstraction.

### `Noul`

First-class proposition:

$$
P(\text{true}|x,q)
$$

with support separately if required.

Don't encode false as `∅`.

### `Choice`

Your N3 architecture:

$$
P(a_1,\ldots,a_K|x,q,A,\text{supported}).
$$

### `Score`

Ordered levels:

$$
P(Y=s_k|x,q,S).
$$

Prefer an ordinal formulation rather than ordinary categorical CE.

That gives you a real:

$$
\boxed{
\text{state}
\rightarrow
\text{typed probabilistic decision}
}
$$

system rather than an MCQ research prototype.

---

# 6. Only then start the calibration / RLCD phase

This is where I'd return to the original TypeSafe question.

Right now you have good evidence that the architecture can produce decisions.

Next ask whether its probabilities actually mean something.

Build Stage-B data around:

* ambiguous examples;
* annotator disagreement;
* evidence deletion;
* evidence addition;
* contradiction;
* OOD;
* candidate absence;
* parametric-knowledge uncertainty.

Train first with:

$$
L=
L_{\log}
+
\lambda_B L_{\text{Brier}}
+
\lambda_E L_{\text{evidence}}
+
\lambda_I L_{\text{invariance}}.
$$

No PPO/GRPO yet.

Evaluate:

$$
\text{Brier},\ NLL,\ ECE,\ \text{risk/coverage}
$$

on both:

* evidence-grounded tasks;
* closed-book tasks.

And especially ChaosNLI, because your current architecture repeatedly becomes too sharp there.

---

# 7. Add workflow RL only after that

Once probability semantics are good, create one small decision environment with asymmetric consequences.

For example:

```text
allow
review
block
```

with different false-positive/false-negative costs.

Then optimize actual expected utility while retaining a proper-scoring term.

That's where the **RLCD-like** question becomes real rather than branding normal cross-entropy as RL.

---

# 8. Scale last

I would absolutely **not** spend the next money on an 8B model yet.

You now know 1.7B has enough parametric capability to prove the interface.

Scale once this equation holds:

$$
\text{NativeChoice evidence}
\approx
\text{Energy evidence}
$$

or you've deliberately settled on the two-expert architecture.

Then the interesting scaling experiment becomes:

$$
1.7B\rightarrow 8B
$$

and ask:

> Does direct System-One decision capability scale with the underlying pretrained model?

That's an actual result.

An 8B fine-tuned Native Choice teacher/reference would also finally give you a meaningful Jev-comparison ceiling.

---

## I’d structure the whole roadmap as

**Phase 5A — Native Choice closure**

* N3 latency vs K.
* remove letter/slot artifacts.
* settle \(K^*\).

**Phase 5B — Support + Choice composition**

* energy/null gate + N3 conditional choice.
* test whether this fixes native abstention essentially for free.

**Phase 5C — Evidence/knowledge fusion**

* residual or gated fusion.
* kill quickly if interference remains.

**Phase 6 — Typed model**

* Noul.
* Choice.
* Score.

**Phase 7 — Calibration**

* uncertainty shaping.
* proper scores.
* OOD/ambiguity/evidence intervention.

**Phase 8 — RLCD-like workflow training**

* downstream utility.

**Phase 9 — Scale**

* 8B / MoE / eventually KDA if long-context economics matter.

---

### The critical next two experiments

If I had to reduce everything to just **two**:

**1. Benchmark N3 Native Choice latency with proper shared-state KV caching.**

You need to know whether the architecture is actually fast enough to be the normal `Choice` primitive.

**2. Train one letter-free, typed Native Choice model and combine its conditional probabilities with the existing energy null/support signal.**

If that model retains roughly:

$$
\text{N3 knowledge}
+
\text{energy evidence/null}
$$

then I think you have the first coherent **PCDM v2 architecture** rather than a collection of successful experimental branches.

After that, the problem stops being architecture discovery and becomes **training a System-One model well**.

---

# Execution notes (lead, 2026-09-20)

- **§0 done:** commit `fbd737e` pushed (`gpu-runpod-full-experiment`); `frozen/` holds the MMLU-Pro slice, counterfactual battery, Δ_q
  probes, sibling-excluded K-sweep, leak exclusions and the corpus manifest; `uv.lock` pins the env; checkpoints/results/labels on HF;
  `pcdm/report_native.py` / `pcdm/report.py` regenerate the tables from `runs/*/results.json` (`RESULTS.md`). Clean-checkout reproduction of
  the tables = `git clone … && uv sync && uv run pcdm/report_native.py …`; regenerating a *number* from a checkpoint = `pcdm/train.py --eval_only`
  after `hf download guychuk/pcdm-runs --include <run>/best.pt`. Volume `pcdm-vol` can go once that has been exercised once.
- **Critical experiment 1 — native latency with state KV reuse:** `pcdm/bench.py --native` (KV-cached state, one suffix per query,
  K = 2…256, L_s = 256/1k/2k, M = 1/32/256) vs the energy path and the batched log-prob baseline → empirical K*. Code built locally
  (tests); needs ~$2 of GPU.
- **Critical experiment 2 — `native_v2` (letter-free `<choice>` spans) + support gate composition:** rendering change in `pcdm/native.py`
  (`--nc_render tags`), the composition `P(∅) = 1 − r_energy`, `P(a_j) = r_energy · P_N3(a_j | answerable)` as an eval-time script over
  two checkpoints (`scripts/compose_support.py`), then one 12k-step run (~$9) with the goals: keep N3's Δ_q (.111), reorder Δp
  .10 → < .03, recover evidence. Budget: actual RunPod balance queried 2026-09-20 = **$120.89** (earlier "≈$5 left" was an unverified running estimate); both experiments are funded.
