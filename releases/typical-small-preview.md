# typical-small-preview

Release 0 (PLAN7 Track 0): a frozen copy of checkpoint `nc_v3_tap20_wf`, uploaded verbatim to
`guychuk/pcdm-runs` under `typical-small-preview/`. Nothing was retrained for this release — this
is a freeze-and-document step.

## What this model is

- **Backbone:** `Qwen/Qwen3-1.7B-Base`, truncated at layer 20 of 28 (tapped mid-depth, not at the
  final layer — REPORT.md §3r found depth, not the option-in-suffix formulation, was what had been
  costing evidence reasoning at earlier taps).
- **Adaptation:** LoRA, rank 16, on the top 8 layers (13–20) of the truncated backbone. The rest of
  the backbone is frozen.
- **Readout:** native contextual decision head (`nc_head=n3`) — candidates are rendered in the
  prompt suffix and the head reads the state after them, rather than scoring pooled embeddings
  independently. Candidate rendering is `letters_nonull`: options get letter labels but there is no
  rendered "none of the above" line.
- **Null handling:** *factored* null — abstention (∅) is a separate head decision, not a candidate
  competing for softmax mass with the real options. REPORT.md §3t showed the rendered ∅ line was
  itself the main null-calibration pathology (K=150 always-abstain, K=5 never-abstain); factoring it
  out fixed the K-sweep and cost ~nothing on accuracy.
- **Output:** no generated answer token. The model produces a probability distribution over
  whatever candidate set is passed at inference time (K runtime options) plus ∅ — it does not
  memorize a fixed label vocabulary.

## Training args (verbatim from `results.json["args"]`, run `nc_v3_tap20_wf`)

```
backbone: Qwen/Qwen3-1.7B-Base
lora_layers: 8            lora_r: 16            lora_lr: 0.0001
tap_layer: 20             zscore: true
readout: native           nc_head: n3           nc_render: letters_nonull
null: factored
steps: 12000               bs: 64                lr: 0.0003
eval_every: 6000           val_every: 1000       ckpt_every: 2000
data: data_v5              extra_data: data_kb,data_wf,data_wf_hf
family_weights: E:0.35,K:0.25,W:0.40
tower_d: 512               tower_layers: 2       tower_heads: 8
seed: 0
```

Final val NLL 0.384, temperature T = 1.040 (fit post-hoc, applied at eval).

## Data

Family-balanced sampler over three buckets (E .35 / K .25 / W .40 of every training batch):

- **`data_v5`** (E/K evidence + knowledge-label mix): NLI/BoolQ-style evidence tasks plus a wide
  span of classification label vocabularies at low rows-per-source (`cls_cap`), specifically so the
  model can't memorize ~16 fixed label sets — it has to read the options.
- **`data_kb`** (`scripts/distill_corpus.py`, 147,446 rows): MMLU-aux, AQuA, MedMCQA, LogiQA-style
  knowledge MCQ, distilled from a `mcq_lora` teacher warm-started on the same corpus.
- **`data_wf`** (`scripts/workflow_corpus.py`): our own rubric-conditioned typed-workflow corpus in
  JevBench's shape (Noul/Choice/Score), 72k rows / 12 families, 37% of rows in rubric groups — the
  same state and labels rendered under 2–3 different rubrics with different gold, specifically to
  train and measure rubric dependence rather than pattern-matching a fixed decision. One rubric
  style per family is held out of train/val entirely.
- **`data_wf_hf`** (`scripts/jev_hf_datasets.py`, 86.6k rows, all bucket W): `cua-s1-forms` (60k,
  procedural GUI-form synth, MIT), `systemone-lite-general` (20k, rule-labeled typed decisions,
  MIT), `jev-4b-distill-data` (6.6k, programmatic gold, Apache-2.0). See PLAN6.md's dataset-survey
  table for the full per-source verdicts and held-out eval splits (cua test 24,370 signature-disjoint
  + demo 196, typed-decisions 2,000, pagerduty 6,000, tree-choice 720, jevlogs 5,080 caveated,
  Mind2Web 1,600, jev-4b adversarial 600, systemone-lite hard 5,400 — none of these were trained on).

## Results

### §3t — this readout vs its ∅-line ablation and the energy baseline (same v5+kb mix, no workflow data yet)

| | `nc_n3_tap20` (∅ line) | **`nc_v3_tap20`** (no ∅ line, = this model's base) | energy `joint_emb_lw_v5` |
|---|---|---|---|
| SNLI / MNLI / ANLI / BoolQ | .906 / .865 / .519 / .834 | .904 / .865 / .507 / .834 | .909 / .880 / .555 / .834 |
| CLINC-150 / HWU64 / 20NG / TREC-fine | .862 / .801 / .589 / .430 | .845 / .757 / .515 / .468 | .784 / .818 / .336 / .308 |
| Δ_q shuffled (primary) / choices-only | .118 / .100 | .122 / .098 | ≈0 |
| MMLU-Pro among-K / acc / false-abstain | .363 / .133 / .692 | .354 / .353 / .003 | .123 / .092 / – |
| Banking77-77 acc / false-abstain | .063 / .924 | .579 / .135 | .527 / – |
| CLINC-OOS acc / CLINC-K null AUROC | .639 / .840 | .798 / .973 | .735 / .961 |
| best val NLL | .341 | .347 | .371 |

JevBench (same 231 public ids, §3q protocol): `nc_v3_tap20` standard **.694** / easy **1.00** / hard
.378 (Brier .47 / .02 / .74), vs .472 / .812 / .297 for `nc_v3` at 28 layers and .417 / .854 / .306
for `nc_n3`.

### §3w — Phase 6A: adding the workflow data (this release, `nc_v3_tap20_wf`, vs its matched baseline `nc_v3_tap20`)

**W — held-out workflow (same 500 rows, full length):**

| | `nc_v3_tap20` | **`nc_v3_tap20_wf`** | Δ | NLL |
|---|---|---|---|---|
| held-out family: choice / noul / score | .890 / .548 / .400 | .994 / .682 / .514 | +10.4 / +13.4 / +11.4 | .42→.02 / .83→.85 / 1.24→2.07 |
| held-out rubric styles | .578 | .904 | +32.6 | .98→.22 |
| rubric-flip: flip rate / pair acc / both correct | .312 / .592 / – | .504 / .736 / .488 | +.19 / +14 | |
| Δ_r = own rubric − shuffled rubric | .285 | .344 | ↑ | shuffled NLL 1.5→4.6 |
| trained-source shift: systemone hard / jev-4b adversarial | .458 / .652 | .888 / .940 | +43 / +29 | |
| cua-s1-forms test (signature-disjoint) / real demo | .314 / .173 | .998 / 1.000 | +68 / +83 | |

**External, never trained (same rows):** Mind2Web .254 → .438 (+18), jevlogs .162 → .832 (+67,
block-level labels, caveated), PagerDuty .736 → .762 (+3), tree-choice .452 → .494 (+4; 125/500 rows
overflow the native suffix and score as chance), **typed-decisions .332 → .276 (−6), NLL 1.40 → 2.40,
Brier .27 → .54** (soft probabilistic gold — the model got sharper, not righter).

**JevBench (same 231 public ids, §3q protocol):** standard .694 → **.750** (Brier .47 → .40; ordinal
5 → 10/12, policy 8 → 9, adequacy 6 → 5, intent 9 → 8), easy 1.00 → 1.00 (Brier .02 → .001), hard
.378 → .387 (flat; Brier .74 → .88, ECE .21 → .37).

**E / K retention (full sets, train-time pass):** SNLI −1.0, MNLI −1.6, BoolQ −0.3, ANLI −2.6, HWU64
−2.2, 20NG +0.7, CLINC-OOS +10.5; **CLINC-150 −11.2 (.845 → .734) and TREC-fine −9.6 (.468 → .372)**
— both from increased false-abstain, not lost discrimination (CLINC-K acc holds at .892, paired-null
AUROC unchanged). MMLU-Pro among-K .354 → .331, kb val +2.0, TruthfulQA −0.9. Val NLL .347 → .384.

**Verdict (REPORT.md §3w):** rubric-conditioned training on this frozen backbone moves every
held-out workflow axis (+10 to +33) and JevBench standard (+.056), and the rubric-flip / shuffled-rubric
numbers show it is reading the rubric rather than memorizing priors. JevBench hard stays at chance.
Two known regressions, both with identified (not yet fixed) causes: over-abstention on two label-space
sets (CLINC-150, TREC-fine — a null-weighting/sampling interaction with W's own catch-all options, not
a discrimination loss), and worse probability quality wherever the gold is soft or ordinal (held-out
score NLL, typed-decisions, JevBench-hard Brier/ECE) — accuracy held or rose while calibration fell.

### JevBench disclosures (REPORT.md §3q — apply to all numbers above)

This is a **public-subset run, not a ranked entry** (JevBench's own leaderboard requires ≥95%
coverage including the 146 non-public judge items; we ran the 231 public ids only, all disclosures
below carried over from the harness's `per-task.json`). Reported probabilities are the head's
softmax **conditioned on non-∅** (P(∅) is dropped and the rest renormalized; mean p_null ≈ .03,
native). Checkpoints trained at 256-token states / 64-token queries are run at up to 4096 / 256
tokens (out-of-training-length but not truncated). Latency is in-process on one H100, one decision
at a time, model load excluded. Cost is null (no tariff — this is not a hosted-API run). Option
order is the harness's own label order; a reversed-order control on a related checkpoint swung
standard accuracy ±4 points, larger than the differences between our own model variants, so no fine
ranking among our own checkpoints is supported by these numbers alone. Majority/chance baselines on
this split are .311 standard / .284 easy / .336 hard (n=72 standard, SE ≈ .058) — this model's hard
tier is within 1 SE of chance.

## Known limitations

- Hard multi-step workflow reasoning is weak (JevBench hard stays at .38, i.e. chance, after
  workflow training).
- Probability quality (NLL/Brier/ECE) degrades on soft or ambiguous targets even where accuracy
  holds or improves — the workflow training data is almost all hard-labeled.
- `Score` is currently treated as a categorical `Choice`, not as an ordinal type; Phase 6B is
  expected to test an ordinal head.
- `Noul` (yes/no) is a 2-way `Choice`, not a specialized Bernoulli primitive.
- Workflow-mix training induces over-abstention on some intent/topic classification tasks (CLINC-150,
  TREC-fine) that share label space with the workflow data's own catch-all options — a known,
  diagnosed, not-yet-fixed sampling/null-weighting interaction.
- Non-zero order sensitivity remains (reorder Δp ≈ .12; this is a letter-interface artifact, not
  eliminated by any rendering tried so far).
- Very large candidate sets (K in the hundreds to thousands) need the energy → top-r → native path,
  not this native head directly — the native suffix has a practical token budget.

## License

- Backbone (`Qwen/Qwen3-1.7B-Base`): Apache-2.0.
- Training data: per-source licenses triaged in PLAN6.md's dataset-survey table (mostly MIT /
  Apache-2.0 / CC-BY-4.0 for the trained sources; `jevlogs` is research-licensed and used only as a
  caveated held-out sanity eval, never trained on, never used in commercial claims).
- JevBench numbers in this card are a **public-subset run** against `fstandhartinger/jevbench`
  v1.2.1 (72 standard / 48 easy / 111 hard public ids) — not a submitted or ranked leaderboard entry.
  The 72 MIT-licensed original JevBench items are the only public JevBench material that is
  training-eligible, and none of it was trained on for this checkpoint.

## How to run

```python
from pcdm_jev.decider import PCDMDecider

decider = PCDMDecider("runs/nc_v3_tap20_wf", mode="native")
question = {"instructions": "...", "type": "choice", "criteria": {"a": "option a", "b": "option b"}}
probs, runtime = decider.decide(state="...", question=question, labels=["a", "b"])
# probs: dict label -> probability; runtime["p_null"] is the abstention mass
```

Held-out workflow / external eval reproduction (post-hoc, full state length):

```bash
uv run --no-sync python scripts/eval_wf.py --run runs/nc_v3_tap20_wf --mode native \
    --files data_wf/eval/*.jsonl data_wf_hf/eval/*.jsonl \
    --limit 500 --out runs/nc_v3_tap20_wf/eval_wf.json
```

JevBench: `scripts/jevbench_run.py` against the checkpoint in `mode="native"` (see `pcdm_jev/` for
the harness adapter and REPORT.md §3q for the exact protocol and disclosures reproduced above).

## Files in `guychuk/pcdm-runs/typical-small-preview/`

- `best.pt` — this checkpoint (`nc_v3_tap20_wf`).
- `results.json` — training args + full eval suite (train-time pass).
- `eval_wf.json` — post-hoc held-out workflow / external eval (full state length, `scripts/eval_wf.py`).
- `probe_results.json` — Δ_q probe results.
- `jevbench_summary.json` — JevBench public-subset run summary.
- `baseline_nc_v3_tap20/` — the matched pre-workflow baseline (identical config minus `data_wf`,
  `data_wf_hf`), same five files, for the Δ comparisons above.
