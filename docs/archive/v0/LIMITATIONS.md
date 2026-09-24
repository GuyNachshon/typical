> **Archived — typical-v0.** Describes the v0 model (run `joint_emb_lw_v5`), not the shipped
> `typical-small` (`ts1b`) / `typical-medium` (`tm1b`) checkpoints. Every number below is
> superseded by `releases/typical-small.md`, `releases/typical-medium.md` and
> `releases/typical-small-preview.md`. Kept for the writing, not the measurements.

# Typical v0 — where it breaks

Every item: what happens, the number, the run and eval set it comes from, what would fix it. Numbers are from
`runs/<run>/results.json` unless a section of `REPORT.md` is cited. Default run = `joint_emb_lw_v5` (the shipped
`typical-v0`), seed 0. "not measured" means exactly that.

Seed floor for reading any of these (REPORT §3d(3), two seed pairs): NLI ± 0.7, trained large-K sets ± 3, unseen
label spaces and OOS ± 5 points.

## 1. No parametric knowledge

- What happens: questions whose answer is not in the state come back near chance, and often as ∅.
- Number: MMLU-Pro 1,200-item slice (`frozen/mmlu_pro.jsonl`, K = 10): acc **.092**, among-K **.123**, ECE .395, NLL
  3.94 (`runs/mmlu_joint_emb_lw_v5`). Chance is .10. Pointwise `joint_emb`: .082 / .132. The same 1.7B backbone with
  options rendered in context and a letter readout (`mcq_lora`) gets .235 / .307; Jev reports 84.6 % (third party).
- Why: every training task is "the answer is in the state" (NLI, BoolQ, SQuAD, intents, topics); the head learned
  evidence ↔ candidate fit. The backbone is cut at layer 20 of 28, before answer formation. Novelty ⇒ null fires on
  unfamiliar option strings (among-K > acc). REPORT §3f.
- Research preview: `nc_n3` (options in the suffix through all 28 layers, knowledge-MCQ corpus added) reaches among-K
  **.314** / acc .133 (REPORT §3l) but its null is broken and it is not shipped — see item 20.
- Fix: a native-choice readout with a K-robust null (PLAN4 §16), or route knowledge questions to an LLM. Not a
  calibration fix.

## 2. State truncated at 256 tokens

- What happens: `decide()` truncates the state to 256 backbone tokens silently (`typical/__init__.py STATE_TOKENS`).
  Anything past that is not read.
- Number: no eval set exceeds it; latency was never benchmarked past ≈ 212 state tokens (`bench.py` default
  paragraph; `runs/bench_a100/bench.json` records `state_tokens: 0` = default). Behaviour at 1k–30k tokens: **not
  measured**. IDEA2 §15 predicts O(M · L_q · L_s) growth.
- Fix: raise the cap and re-run `bench.py --state_tokens N`; long-state training data.

## 3. Unseen label vocabularies

- What happens: accuracy drops on option sets that look unlike any training vocabulary, and part of the drop is
  false abstention.
- Numbers (`joint_emb_lw_v5`): Banking77-77 **.527** (among-K .542) vs supervised BERT .937 (Casanueva et al. 2020);
  TREC-coarse **.356** / TREC-fine **.308**; 20 Newsgroups **.336** (among-K .357); CLINC 30 held-out intents .762 vs
  among-K .876 — the 11-point gap is abstention on a familiar task with unfamiliar strings ("novelty ⇒ null",
  REPORT §3d). Pointwise `joint_emb` (data v4) is worse: Banking77-77 .440, TREC-coarse .210, 20NG .273.
- Data v5 (30 vocabularies instead of 16) moved every one of these 1.5–3× the seed floor (REPORT §3e); the gap to
  options-in-context (`mcq_lora`: .568 / .430 / .534) is now 4–20 points.
- Fix: more label vocabularies at lower rows each (60+, cap 1–2k) — a data-build cost, not a model change; or the
  native-choice readout.

## 4. Null dilutes with K

- What happens: with the gold option absent, P(∅) falls as the option set grows.
- Number: `ksweep_clinc`: P(∅ | absent) **.936 → .805 → .560 → .413** at K = 2 / 10 / 50 / 150 (range .523);
  P(∅ | present) stays .067–.104; null AUROC by K .989 / .963 / .901 / .850. Banking77 (unseen): .778 at K = 2.
- Sibling-excluded sweep (`runs/ks_joint_emb_lw_v5`, `frozen/ksweep_clinc_nosib.jsonl`): **.945 → .689 → .588** at
  K = 2 / 50 / 150, AUROC .988 / .939 / .896. So about half of the drop is near-miss difficulty (siblings of the
  gold left in the set) and half is genuine K-dependence — an extreme-value effect over 149 unrelated distractors
  costing ≈ 9 AUROC points on its own (REPORT §3h).
- Fix attempts that did not work: post-hoc `α·log K + β` bias (REPORT §3d(1)); factored null (`factnull_v5`: range
  .48, +4.5 OOS, killed). Fix: fine-grained discrimination at large K, not the null's functional form.

## 5. Calibration out of distribution

- What happens: probabilities are over-confident on sets far from training.
- Numbers (raw / T-scaled ECE, `joint_emb_lw_v5`): Banking77-77 **.213 / .157**; Banking77-K .238 / .202;
  near-miss Banking77 .292 / .255; TREC-coarse .264 / .205; TREC-fine .178 / .100; 20NG .171 / .105; CLINC-heldout
  .130 / .102; ANLI .209 / .170; MMLU-Pro .395. In-distribution for contrast: SNLI .009, MNLI .026, BoolQ .057,
  CLINC .065, HWU64 .040 raw.
- Also: one global T = 1.215 fitted on a classification-heavy val set over-softens NLI (SNLI raw .009 → scaled .020;
  REPORT §3e).
- Fix: per-family temperature; label-diverse training. Do not read the in-distribution ECE as a shift guarantee.

## 6. Listwise IIA is not exact (by design)

- What happens: adding an irrelevant option shifts the log-odds between the top two real options.
- Number: `cse_*` add_irr/dlo_top2 = **.204** (clinc) / **.243** (banking77) / .205 (hwu64) / .035 (snli). The
  irrelevant option itself gets P ≈ .002. Reorder (max Δp < 1e-5) and duplicate slot gap (0.0) are exact for every
  energy checkpoint.
- The pointwise `joint_emb` has add_irr Δ = **0.000** (4e-8) but OOS recall .649 vs .735 and Banking77-77 .440.
- Fix: use `typical-v0-pointwise` when IIA matters more than the null.

## 7. Single seed

- What happens: every number on this page is one training run.
- Number: seed floor from `joint_v1`/`joint_v1_s1` and `joint_emb`/`joint_emb_s1` (REPORT §3d(3)): NLI ± 0.7, trained
  large-K ± 3, unseen label spaces / OOS ± 5. `joint_emb_lw_v5` itself has no second seed.
- Fix: a second seed ≈ $4 (LAUNCH.md nice-to-have).

## 8. Near-duplicate inflation on BoolQ and HWU64

- What happens: some eval states have near-duplicates in train (Wikipedia-revision twins, reused utterances).
- Number: `runs/leak_audit.json` (MinHash char-5-gram): BoolQ **3.9 %** of eval states, HWU64 **2.6 %**; SNLI/MNLI/
  ANLI pair-level near-dups 1 / 3 / 0 rows. Treat BoolQ .834 and HWU64 .818 as ≤ 1–2 points inflated (REVIEW §6).
- Fix: drop the colliding eval rows; the audit script already lists them.

## 9. Single-question latency is worse than a log-prob baseline

- What happens: the win comes from amortising the state and batching suffixes; for one question the overhead
  dominates.
- Number (`runs/bench_a100/bench.json`, K = 4): M = 1 ours **131 ms** vs plain log-prob baseline **43 ms**; M = 8
  ours 138 ms vs 341 ms (baseline) / 104 ms (query-batched baseline). Break-even with the query-batched baseline is
  between M = 8 and M = 64 (ours 149 ms vs 319 ms at M = 64). At M = 256 per-decision marginal is 1.29 ms vs 4.33 ms
  (3.3×); the 69× / 267× ratios are at K = 150 / 1000.
- Cold candidates (not yet embedded) add 35–50 % per decision (`runs/bench_cold_emb`: 1.65 → 2.23 ms at K = 4).
- Fix: none planned; use it for many questions per state.

## 10. Candidate strings only, ≤ 32 tokens, descriptions ignored

- What happens: options are label strings embedded by a frozen encoder (`CAND_TOKENS = 32`). The Jev-wire adapter
  (`typical/serve.py`) takes the `criteria` keys and drops the descriptions; Score levels are used as strings.
- Number: effect of dropping descriptions vs a model that reads them: **not measured**. The closest datapoint is
  options-in-context `mcq_lora` beating this head by +27 (Banking77-77) / +28 (CLINC-heldout) / +26 (20NG) on
  unseen vocabularies (COMPARE §3, data v4 numbers).
- Fix: candidate encoder that reads descriptions, or the native readout.

## 11. English only

- Training data is English; multilingual sources (MASSIVE, MTOP) are used in their `en` splits only. Non-English
  accuracy: **not measured**.

## 12. JSON / structured states untested

- `serve.py` `json.dumps` a non-string state and feeds it as text. Accuracy on JSON states: **not measured**.
  The 256-token cap makes most JSON payloads truncate.

## 13. Score and yes/no are derived, not trained

- `typical.score` and `typical.noul` are Choice over a fixed option list plus post-processing (`typical/__init__.py`).
  No Score training data exists; the ordinal structure of levels is not modelled. Calibration of the expected-value
  `score` and of `noul` on statement-style questions: **not measured**. BoolQ (.834, ECE .057) is the only yes/no
  number and its questions are interrogative, not statements.

## 14. ∅ on statement-style yes/no questions is base-rate sensitive

- What happens: for `["yes", "no"]` the model was trained on BoolQ questions and NLI hypotheses; a statement whose
  truth is not decidable from the state should go to ∅, but the split between "no" and ∅ depends on phrasing.
- Number: **not measured** as a separate eval. Related: SNLI hypothesis-only .410 (`snli_test_hyponly`) shows the
  NLI head is not artefact-driven; irrelevant-question null AUROC .972 (intent) / .984 (squad).
- Fix: an explicit statement-style yes/no/undecidable eval set.

## 15. No adversarial robustness testing

- Prompt injection inside the state, adversarial option strings, unicode tricks: **not measured**. The near-miss
  battery (`null_nearmiss_*`) is the only hard-negative test: null AUROC .906 (clinc) / .942 (hwu64) / .772
  (banking77).

## 16. No fine-tuning API

- `train.py` is the only path (full data pipeline, hand-rolled LoRA, not PEFT). No `Typical.finetune()`, no PEFT
  adapter format, no HF `Trainer` integration.

## 17. Not a supervised classifier replacement

- CLINC-150 .784 vs BERT fine-tuned .969; HWU64 .818 vs .921; Banking77 .527 vs .937 (COMPARE §2a). The model
  trades accuracy for runtime-defined label sets and a null.

## 18. Eval sets are capped

- Most eval sets are capped at ~3–5k items (`runs/leak_audit.json` n: SNLI 4,989, MNLI 2,947, ANLI 3,200, BoolQ
  3,270, CLINC 4,449, Banking77 3,076, TREC 500 each, 20NG 7,315). TREC at n = 500 has a wide interval.

## 19. Hyperparameters were not tuned on held-out sets, but T was

- Reported "scaled" ECE uses a temperature fitted on val. Raw ECE is reported alongside everywhere (MODEL_CARD.md).
  Jev's third-party ECE .031 is raw API output; compare against our raw column only.

## 20. `nc_n3` (research preview) is not usable as shipped

- MMLU-Pro among-K .314 but acc .133 (abstention error .18), ECE .459; P(∅ | absent) .99 at K = 150 while CLINC-OOS
  recall is .20 (`runs/nc_n3/results.json` raw .194); SNLI .863 / MNLI .752 / BoolQ .737 (−5 to −13 vs v0); IIA
  Δ .126; no shared-state inference path, so no latency numbers apply (REPORT §3l).
