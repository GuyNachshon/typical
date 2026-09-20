This Phase 5 pass is a very good stopping point for architecture exploration. The main thing I’d change in the project plan now is **what gets the next dollar**.

The new JevBench result is especially informative because its tasks are exactly the shape Jev is meant for: request routing, answer adequacy, policy yes/no checks, intent classification, ordinal severity scoring, and enum extraction. Each item supplies the state, instructions/rubric, and exact label set, and native decision models are evaluated on their actual probability distributions rather than verbalized confidence. ([GitHub][1]) That is much closer to the target than MMLU-Pro.

So I’d interpret the current situation as:

$$
\boxed{\text{architecture mostly works; training distribution is now the bottleneck}}
$$

with one caveat: JevBench alone doesn’t prove that *all* of the gap is task-family training. Backbone scale/capacity and calibration training can still contribute. But the fact that PCDM is already strong on easy items and weak on standard/hard rubric-conditioned decisions strongly points there.

## What I would spend the remaining ~$94 on

In this order:

1. **Commit + reproduce first — $0.** Do not start another pod until §3m–§3q, `native_v3`, the JevBench adapter, fusion scripts, configs, checkpoints, and manifests are reproducible from a clean checkout. You now have enough valuable results that an uncommitted working tree is the highest-risk part of the project.

2. **Replicate the central Native Choice result — ~$18.** I would fund the seed pair. The claim that matters is not “N3 gets .314 MMLU,” but that a direct contextual readout preserves essentially the same question-dependent signal as the letter interface while reducing some of its artifacts. That is central enough that it should not rest on one seed. Ideally replicate `nc_v3`/N3 against the matched letter control, not every old branch.

3. **Learned per-input expert gate — ~$5.** Very high information-per-dollar. Phase 5C already says the energy and native experts are complementary, but a global mixture is the wrong abstraction. Train a tiny gate over signals available at inference—query/state representation, K, energy margins/entropy, native margins/entropy, maybe task/type embedding—and ask whether it can approach the envelope of the two experts without task IDs. Keep it tiny enough that a negative result is cheap.

4. **Phase 6 should be “typed workflow training,” not merely “implement Noul and Score” — ~$25–35.** The three primitives are necessary, but the key missing ingredient is the **rubric-conditioned decision distribution**. TypeSafe’s own workflow evals decompose policies into narrow independent Noul, Choice, and Score questions, with code handling deterministic rules. ([Typesafe Evaluations][2]) Build a training mixture modeled on that *shape* without training on JevBench itself:

   * `Noul`: policy applicability, adequacy, support, safety/guardrail checks.
   * `Choice`: routing, action selection, enum extraction, categorical judgments with rich criteria per option.
   * `Score`: ordered severity/quality/relevance/urgency rubrics.
   * Vary rubrics and label vocabularies aggressively; hold out whole workflow families and whole rubric styles.

   This is much more likely to move JevBench than another readout ablation.

5. **Calibration phase — ~$10–15.** Once the model has learned the right workflow family, optimize probability quality with proper scoring + ambiguity/evidence-shaping. Jev’s public positioning explicitly treats calibrated probabilities as core, and its workflow design assumes software will threshold those probabilities. ([TypeSafe AI][3]) I still would not touch PPO/GRPO yet.

6. **Keep ~$20–30 unspent.** Use it only after Phase 6 tells you what the bottleneck is. If workflow accuracy rises sharply, spend it on a larger/native backbone. If accuracy rises but Brier/ECE remain bad, spend it on calibration. If Noul/Score lag Choice, spend it on type-specific data/objectives.

### I would *not* spend next on

More N1/N2/N3 readouts, candidate-blind distillation, another null functional form, KDA, or an 8B model immediately. Those questions are either already answered or downstream of the task-family issue.

There is also a useful product/architecture conclusion from Phase 5A: **Native Choice is cheap enough to be the default for ordinary closed sets.** At \(K=10\), ~1.6× the energy path is a very reasonable price for recovering candidate-conditioned reasoning; the energy path is then best understood as the high-\(K\) front end and evidence-specialized expert, not as the only runtime primitive. TypeSafe publicly does something conceptually similar at high cardinality: native `Choice` supports up to 255 options and higher-cardinality cases use independent scoring followed by explicit choice. ([TypeSafe AI][3])

One important wording change: I would **not yet say “JevBench proves the gap is task family.”** Say:

> “JevBench shifts the leading hypothesis from architecture to training distribution: PCDM performs well on easy bounded decisions but degrades sharply on harder rubric-conditioned workflow judgments, the family Jev is explicitly optimized for.”

That is defensible and gives Phase 6 a falsifiable target.

If Phase 6 workflow training takes standard JevBench from ~.40–.47 toward, say, .65–.75 without changing the architecture, you’ve learned something very important: the remaining gap was largely **what System-One models are trained to decide**, not how they compute the decision. If it barely moves, then scaling/backbone capability becomes the next suspect.

So my next milestone would be:

$$
\boxed{
\text{PCDM Phase 6: one model, Noul + Choice + Score, trained on held-out-rubric workflow decisions}
}
$$

and the pass condition should be **improvement on held-out workflow families**, not simply training accuracy or one benchmark headline.

[1]: https://github.com/fstandhartinger/jevbench?utm_source=chatgpt.com "GitHub - fstandhartinger/jevbench: JevBench v1 - a benchmark for Jev-class typed decision models: smart, cheap, fast, reliable, open. · GitHub"
[2]: https://evals.typesafe.ai/?utm_source=chatgpt.com "Workflow evals"
[3]: https://typesafe.ai/blog/introducing-system-one-models-and-jev?utm_source=chatgpt.com "Introducing System One Models & Jev - TypeSafe AI Blog"

---

# Dataset triage for Phase 6 (PI, 2026-09-20 11:37) — adopted as the working plan pending verification (survey agent)

| Dataset                                                     | What I’d do                                                  | Why                                                                                                                                                                                                                                                                                                                                               |
| ----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`cua-ai/cua-s1-forms`**                                   | **TRAIN + held-out eval**                                    | Highest-value addition. 234k decisions, 7–33 runtime options, synthetic train with form-signature-disjoint test plus a small real demo set. It is explicitly a Jev-like one-pass option scorer dataset. Great `Choice` data and candidate-confuser training. ([Hugging Face][1])                                                                  |
| **`LocalLLaMA/typed-decisions`** from the `typesafe` search | **EVAL first; optionally train only with workflow holdouts** | Very close to the target abstraction: 400 cases / 2,000 decisions, all three primitives, full instructions + criteria + probabilistic gold. Crucially, criteria descriptions are part of the input. Fine-tuning Laya on it moves accuracy from ~.36 to ~.77, which strongly supports your “training distribution” hypothesis. ([Hugging Face][2]) |
| **`INSTRUCT_JEV`**                                          | **Template/generator seed**                                  | Appears to be only ~119 examples, so far too small to matter as raw training data. But very useful for extracting question/rubric patterns for your synthetic workflow generator. ([GitHub][3])                                                                                                                                                   |
| **`jevlogs-log-triage-benchmark`**                          | **HOLD-OUT Noul/calibration eval**                           | Real System-One use case, thousands of sanitized log decisions. But the card itself warns that Loghub ground truth is the wrong granularity for line-level triage, and it contains Jev outputs. Don't learn from Jev's probabilities as gold. ([Hugging Face][4])                                                                                 |
| **`jev-luna-pagerduty-trigger`**                            | **HOLD-OUT Noul + threshold eval**                           | Excellent for testing whether a calibrated `Noul` probability can drive application thresholds. The accompanying experiment shows exactly that use: a single boolean probability + code threshold outperformed discrete urgency classification. ([DEV Community][5])                                                                              |
| **`jev-tree-choice-cap`**                                   | **High-K eval only**                                         | Perfect for your energy → top-r → Native Choice path. It exists specifically to handle Jev's ≤255 choice limit using a taxonomy, with a 320-leaf synthetic catalog and 180 labeled tickets. Don't need it to teach basic Choice. ([Socket][6])                                                                                                    |
| **`typed-decisions-repo-governance`**                       | **DO NOT TRAIN**                                             | Explicitly n=1, unresolved, all gold values null; the Jev response is stored as a prediction and the card explicitly says not to treat it as gold. Useful only as a realistic schema/example. ([Hugging Face][7])                                                                                                                                 |
| **`typed-decisions-code-holes`**                            | **Potentially valuable, but verify provenance first**        | I couldn't verify the indexed card. If its labels come from runnable tests/compilers, that's exactly the kind of externally grounded workflow data we want. If they're model predictions, keep it eval-only.                                                                                                                                      |
| **`jev-stage2-image-beans-pilot`**                          | **DEFER**                                                    | Current PCDM is text-only. This becomes useful when you deliberately test whether the decision primitive extends to multimodal state; adding it now muddies Phase 6.                                                                                                                                                                              |
| **`DGUI_HYPERMEM-JEV`**                                     | **DEFER / inspect manually**                                 | I couldn't verify enough of the card from indexed sources. Don't ingest it until provenance, gold construction, and overlap are clear.                                                                                                                                                                                                            |
| **`open-jev-laya-bench` / `laya-jev-benchmark`**            | **EVAL ONLY**                                                | These exist specifically to compare Jev/open replicas and calibration. Training on them would destroy their value. The Laya benchmark is especially useful because it demonstrates that fine-tuning on the target workflow can produce huge gains while zero-shot behavior remains weak. ([Hugging Face][8])                                      |

[1]: https://huggingface.co/datasets/cua-ai/cua-s1-forms?utm_source=chatgpt.com "cua-ai/cua-s1-forms · Datasets at Hugging Face"
[2]: https://huggingface.co/datasets/LocalLLaMA/typed-decisions?duplicate=true&utm_source=chatgpt.com "LocalLLaMA/typed-decisions · Datasets at Hugging Face"
[3]: https://github.com/24601/Augustus?utm_source=chatgpt.com "GitHub - 24601/Augustus: Agent skill: design judgment-assisted systems with TypeSafe Jev (System One). Maps Choice/Score/Noul onto decision theory, reranking, and routing. Composition algebra, question design, validation gates. MIT. · GitHub"
[4]: https://huggingface.co/datasets/reachjalil/jevlogs-log-triage-benchmark?utm_source=chatgpt.com "reachjalil/jevlogs-log-triage-benchmark · Datasets at Hugging Face"
[5]: https://dev.to/reachjalil/how-we-tuned-typesafe-jev-for-log-triage-without-alert-storms-1ei0?utm_source=chatgpt.com "How we tuned TypeSafe Jev for log triage without alert storms - DEV Community"
[6]: https://socket.dev/npm/package/jev-tree?utm_source=chatgpt.com "jev-tree - npm Package Security Analysis - Socket"
[7]: https://huggingface.co/datasets/com-kotobalabs/typed-decisions-repo-governance?utm_source=chatgpt.com "com-kotobalabs/typed-decisions-repo-governance · Datasets at Hugging Face"
[8]: https://huggingface.co/datasets/Luni/laya-jev-benchmark?utm_source=chatgpt.com "Luni/laya-jev-benchmark · Datasets at Hugging Face"

---

# Queue review + Phase 6 tightening (PI, 2026-09-20 11:38) — adopted

The queue is mostly right. I would **not reshuffle the two running pods**, but I would modify what happens immediately after them and tighten Phase 6 before launching it.

### Keep both running jobs

`e3b_zr_tap28` is worth finishing even though the candidate-blind branch is already negative. Its value is no longer architectural improvement; it's a **closure experiment**. If full-depth \(Z\) still shows \(\Delta_q\approx0\), you can make the much stronger statement that the negative result was not an artifact of tapping layer 20. That materially strengthens contribution #2.

`nc_n3_s1` is essential. N3 is now one of the load-bearing positive results: contextual direct readout preserves the question-dependent knowledge of the letter interface. It needs a seed.

Likewise `nc_n1_s1` is useful because the comparison:

$$
\text{N1 letters} \quad\text{vs}\quad \text{N3 direct contextual readout}
$$

should be paired, not “two single runs happened to be close.”

And I like `nc_v3_tap20`. The Phase 5 diagnosis says two things independently hurt native choice: using all 28 layers costs evidence quality, while the rendered `none of the above` line caused much of the abstention pathology. So “retain the option-aware interface but recover the strong layer-20 evidence representation and remove the textual null artifact” is exactly the next unified-model attempt.

I would **not add anything else to the existing pod chains**.

---

# I would change the Phase 6 launch slightly

The current plan says:

> `nc_v3_tap20` + `data_v5,data_kb,data_wf`

That's directionally right, but **the mixture itself is now an experimental variable**.

You've already learned twice that dataset weighting changes model behavior substantially. Data-v5 improved unseen vocabularies but moved null behavior; kb-heavy native training improved knowledge while evidence regressed.

So don't concatenate the corpora and sample uniformly.

Use an explicit family-balanced sampler, something like conceptually:

$$
P(\text{batch family})
=
\begin{cases}
0.35 & \text{existing evidence / classification}\\
0.25 & \text{closed-book knowledge}\\
0.40 & \text{workflow/rubric decisions}
\end{cases}
$$

The exact percentages aren't sacred. The key is **family-level sampling independent of raw corpus size**.

Otherwise a 180k forms dataset or 147k knowledge corpus can silently determine the model.

And log performance separately on all three axes during training:

$$
E=\text{evidence}
$$

$$
K=\text{knowledge}
$$

$$
W=\text{workflow/rubric}.
$$

Phase 6 succeeds only if \(W\) rises without unacceptable collapse in \(E\) or \(K\).

---

## More importantly, `data_wf` should explicitly teach rubric dependence

Don't let Phase 6 become “add more Jev-looking classifiers.”

The decisive training unit should often be a **group**:

```text
same state
same candidate labels

rubric A -> answer 1
rubric B -> answer 2
rubric C -> answer 3
```

That prevents the model from solving the task from state/candidate priors.

I would literally add a training/eval statistic:

$$
\Delta_r
=
Acc(x,r,A)
-
Acc(x,r_{\text{shuffled}},A)
$$

analogous to your MMLU \(\Delta_q\).

And probably a stronger counterfactual metric:

$$
Acc\left[
f(x,r_1,A)\neq f(x,r_2,A)
\mid y_{r_1}\neq y_{r_2}
\right].
$$

Call it **rubric flip accuracy**.

That should be a primary Phase 6 result.

If JevBench moves but rubric-flip stays poor, you've probably just learned benchmark-adjacent priors again.

---

# How I'd use the external datasets within this queue

I would update `data_wf` now, before the Phase 6 pod starts.

Use **CUA-S1 Forms as a training family**, but cap it. It gives you a big structurally different Choice family with runtime options. Don't let its raw volume dominate.

Use `INSTRUCT_JEV` primarily as **rubric/template inspiration and augmentation seeds**, not as meaningful training volume.

I would **keep `LocalLLaMA/typed-decisions` out of the main Phase 6 training run**. It is too valuable as a clean external Noul/Choice/Score workflow evaluation. Once Phase 6 is done, you can optionally run a second “target adaptation” experiment on part of it, but don't destroy the zero-shot generalization test.

Likewise keep the PagerDuty/log triage datasets, tree-choice benchmark, Laya/Open-Jev benchmarks and JevBench **evaluation-only** for now.

That gives Phase 6 a nice hierarchy:

```text
TRAIN
your data_wf generator
+ CUA forms
+ externally grounded workflow datasets where provenance is clean

HELD-OUT INTERNAL
entire workflow families
entire rubric styles

EXTERNAL
typed-decisions
JevBench
log/PagerDuty
tree-choice
Laya/Open-Jev
```

That's much more credible than training on everything you can find.

---

# I would postpone the learned expert gate slightly

Building it locally now is fine. Running it is nearly free.

But I would **not promote it into the architecture before seeing `nc_v3_tap20` and Phase 6**.

Why?

Because if `nc_v3_tap20 + workflow training` substantially closes the evidence/knowledge/workflow gap, a gate becomes unnecessary complexity.

The gate should answer:

> after a serious unified-model attempt, is there still irreducible complementarity?

So:

1. build it now;
2. evaluate the existing expert oracle/gate now;
3. keep results;
4. don't make it PCDM v2 unless Phase 6 still leaves a clear two-expert frontier.

One useful number is the **oracle envelope**:

$$
Acc_{\text{oracle}}
=
P(\text{either expert is correct}).
$$

Then compare:

$$
\text{learned gate}
$$

to:

$$
\text{best single expert}
$$

and:

$$
\text{oracle}.
$$

If the oracle is only +2 points, don't bother with routing.

If it's +15 and the learned gate captures +10, that's real.

---

# Your Phase 6 pass rule needs one addition

Current:

> improvement on held-out workflow families and per-item JevBench.

Good, but I'd require all of:

1. **Held-out workflow-family improvement**
2. **Held-out rubric-style improvement**
3. **positive rubric dependence / flip accuracy**
4. evidence and knowledge retention within a predefined regression budget
5. improvement on untouched external workflow benchmarks

I would *not* make raw JevBench the sole gate.

Something like:

$$
W_{\text{heldout}} \uparrow \ge 10\text{ pts}
$$

with:

$$
E\text{ regression}\le3\text{ pts}
$$

$$
K\text{ regression}\le3\text{ pts}
$$

would be a much stronger success criterion than “JevBench went from .44 to .60.”

---

# One change I would make to the ordering after Phase 6

You currently have:

1. Phase 6
2. report/review
3. calibration if needed
4. hold budget

I'd make it:

**Phase 6A — workflow Choice/Noul/Score mixture**

Then **Phase 6B — typed primitive diagnostic**, probably cheap.

Before calibration, answer:

> Is Score actually benefiting from ordinal structure?
> Is Noul actually better as Bernoulli than as two-way Choice?

Because right now if all three types are serialized into generic choice labels, you've tested **workflow data**, but not yet the full Jev-like typed architecture.

So after Phase 6, run cheap matched comparisons:

$$
Noul_{\text{Bernoulli}}
\quad vs\quad
Choice[\text{true,false}]
$$

and:

$$
Score_{\text{ordinal}}
\quad vs\quad
Choice[\text{level}_1,\ldots,\text{level}_K].
$$

Only train dedicated heads if they clearly help.

---

## Budget-wise

With ~$94 before the current ~$40 spend, you should end around ~$54.

I'd allocate approximately:

* Current two pods: **already committed ~$40**
* Phase 6 main run: **$10–15**
* Phase 6 seed/ablation only if it passes: **$8–10**
* Calibration: **$0 initially**, then **$10–15 only if accuracy rises and calibration doesn't**
* Leave **$15–25 reserve**

I would not spend the reserve on 8B unless Phase 6 produces a strong model and the remaining errors clearly correlate with reasoning difficulty rather than rubric-following failure.

---

### So my verdict on the queue

**Don't cancel or reorder anything currently running.**

The only material change I would make is:

> **upgrade `data_wf` before the Phase 6 launch and make rubric dependence—not just workflow accuracy—the central training/evaluation object.**

Then the queue becomes logically very clean:

```text
close old hypotheses
    ↓
replicate Native Choice
    ↓
test unified tap20 native model
    ↓
teach runtime-defined rubrics + Noul/Choice/Score
    ↓
measure held-out rubric generalization
    ↓
only then decide:
calibration?
expert routing?
scale?
```

That is exactly where I'd take the project.

---

# Dataset survey verdicts (lead, 2026-09-20 12:20; `scripts/jev_hf_datasets.py` → `data_wf_hf/`, HF `wf_hf/`)

| source | verdict | notes |
|---|---|---|
| cua-ai/cua-s1-forms | **TRAIN** (60k, skip ≤ 35%) + eval (test 24,370 signature-disjoint; demo 196 real) | MIT; procedural synth; K 7–33; 0 JevBench overlap |
| MagaBitmex/jev-4b-distill-data | **TRAIN** (7.5k, programmatic gold) + eval (120 adversarial) | Apache-2.0; closest shape to JevBench |
| dwidlee/systemone-lite-general | **TRAIN** (cap 20k) + eval (`test_hard` shift) | MIT; rule-labeled typed decisions |
| LocalLLaMA/typed-decisions | **EVAL only** (test 2,000; train 6,000 held aside) | soft teacher gold → report NLL/Brier/agreement, not accuracy vs truth |
| reachjalil/jev-luna-pagerduty-trigger | **EVAL** (6,000; 854 trap rows flagged) | MIT; Noul + K=3 Choice; threshold eval |
| reachjalil/jev-tree-choice-cap | **EVAL** (720; flat K=320 + step levels) | MIT; the energy → top-r → native path test |
| reachjalil/jevlogs-log-triage-benchmark | caveated sanity eval (5,080) | Loghub research license; block-level labels; never train, never in commercial claims |
| AndeyTait/JevForge-Mind2Web | **EVAL** (website-disjoint test) | CC-BY-4.0; GUI choice |
| Laya benches (pngwn, Luni) | no data (recipes/logs) | nothing to ingest |
| code-holes / repo-governance / INSTRUCT_JEV / DGUI / image-beans / Praveenrajus jev-bench | drop | modality/length, n=1 with null gold, doc snippets, image, re-packaged public sets already in v5 |

**Execution note (2026-09-20, `data_wf_hf/` built + uploaded as `wf_hf/` on `guychuk/pcdm-data`, `scripts/jev_hf_datasets.py`).** Train 86,600 rows, all `fam_bucket=W`: cua 60,000 (skip 35.0%, fill 32,518, click 3,469, check 3,013) / systemone-lite 20,000 (of 32.4k) / jev-4b 6,600 (card says 7,500; 900 K=1 `record` rows dropped; Score `low/medium/high` → ordinal `0..2`; Jev teacher kept in `meta.teacher`, never a target). Val: cua 3,000. Eval (never trained): cua test 24,370 + demo 196; typed-decisions test 2,000 (+6,000 train held aside, `meta.family` for whole-workflow holdout); jevlogs 5,080 (block-level label flag); pagerduty 6,000 (traps flagged); tree-choice 720 (K up to 320); jev-4b adversarial 600; systemone-lite hard 5,400; Mind2Web 1,600 (choice 800 / noul 800; 28 rows soft multi-positive). 0 JevBench leaks per file. Still pending before Phase 6A: `data_wf` with rubric groups (w-workflow2) and `--family_weights` (w-sampler).
