# typical-small

**Release line: https://huggingface.co/OzLabs/typical-small** — this card describes checkpoint
`ts1b_semif`, shipped **2026-09-24** as v3 (`best.pt` sha256 `dc183b39…`). It superseded `ts1c` (v2)
one day after v2 shipped, and `ts1c` in turn superseded `ts1b`. Parent release:
`typical-small-preview`. Sibling release at 4B: `typical-medium` (`tm2`, `releases/typical-medium.md`).
Sibling at 14B: `typical-large-preview` (`tl1b_semif`).

**What changed in v3, in one line:** the state render. v3 is `ts1b`'s recipe with `nc_render` set to
`semif` instead of `letters_nonull` — same backbone, same tap, same tower, same corpus, same 12k
steps at `max_state 1024`. It is not a longer-context retrain; `ts1c` was that, and v3 is not it.

| | `ts1b` (v1) | `ts1c` (v2) | **`ts1b_semif` (v3)** |
|---|---|---|---|
| render | `letters_nonull` | `letters_nonull` | **`semif`** |
| `max_state` / steps | 1,024 / 12k | 2,048 / 8k | 1,024 / 12k |
| `--drop_truncated` | off | on | off |
| JevBench standard / hard | .694 / .432 | .708 / .432 | **.792 / .441** |
| long states, facts-first / facts-last | .598 / .798 | .947 / .790 | **.952 / .846** |
| held-out yes/no (full-file) | .715 | — | **.705** |

v2 fixed the long-state trap by moving the facts and widening the window. v3 gets the same result
from the render alone, and is the first checkpoint in this line that is strong in **both** orders —
which is what retires the deployment caveat that v1 and v2 shipped with (see below).

## What this model is

- **Backbone:** `Qwen/Qwen3-1.7B-Base`, truncated at layer 20 of 28 — same tap as the preview.
- **Adaptation:** LoRA, rank 16, on the top 8 layers (13–20) of the truncated backbone.
- **Readout:** native contextual decision head (`nc_head=n3`), **`semif` rendering** (the change in
  this version), *factored* null (∅ is a separate head decision, not a rendered candidate). The head
  shape is identical to the preview and to `ts1b`; only the state layout differs.
- **Tap normalisation:** z-scored tap activations (`zscore: true`).
- **Typed primitives** (PLAN7 Track C):
  - **Choice** — the N3 contextual readout above, unchanged.
  - **Noul** (yes/no) — a Bernoulli head, `P(yes) = σ(w·h_D)`, on the decision state with a
    query-only suffix (no rendered candidates). Routed **per row** (commit `7b9d520`): a row only
    goes through the Bernoulli head if its candidate set is exactly `{yes, no}`; every other 2-way
    label set (e.g. `{true, false}`, `{approve, reject}`) is scored as ordinary Choice. The first
    `ts1` run collapsed to chance because this routing was global instead of per-row; `ts1b` was the
    fix, gated by 140 tests, and every checkpoint since inherits it unchanged.
  - **Score** — K-way Choice trained with ordinal-smoothed targets (τ = 0.7, §3ac's adopted recipe).

## Training args

```
backbone: Qwen/Qwen3-1.7B-Base
lora_layers: 8            lora_r: 16            lr: 0.0003            lora_lr: 0.0001
tap_layer: 20             zscore: true
readout: native           nc_head: n3            nc_render: semif
null: factored            noul_head: bern        score_head: choice     ordinal_smooth: 0.7
steps: 12000              bs: 64                 grad_accum: 8
eval_every: 6000          val_every: 1000        ckpt_every: 2000
data: data_v5             extra_data: data_kb,data_wf,data_wf_hf,data_wf_long,data_wh,data_u
family_weights: E:0.40,K:0.15,W:0.35,U:0.10
bucket_map: data_wf_long=W,data_wh=W,data_u=U
null_aug: W:0.20          max_state: 1024        drop_truncated: false
best_on: (unset)
tower_d: 512              tower_layers: 2        tower_heads: 8
seed: 0
```

Effective batch 64 via `--grad_accum 8`. Three flags differ from `ts1c` and all three revert to
`ts1b`'s settings: `max_state` 1,024 rather than 2,048, `--drop_truncated` off rather than on, and no
`--best_on`, so selection is overall val loss rather than the uncertainty + hard-curriculum split.
The only forward change is `nc_render: semif`. That is deliberate — v3 exists to measure the render,
so everything else is held at `ts1b`'s values and the two are a matched pair.

Final val NLL 0.370, temperature T = 1.124, null_offset 0.0 (fit post-hoc on the `val` split, applied
at eval). `ts1c`'s val NLL was 0.439.

## Data

Family-balanced sampler, E .40 / K .15 / W .35 / U .10 of every training batch, plus `--null_aug
W:0.20` (15–20% of W rows get their gold candidate removed so the target is ∅) and `--max_state 2048`
(up from the preview's 256):

- **`data_v5` / `data_kb` / `data_wf` / `data_wf_hf`** — unchanged from the preview (see
  `releases/typical-small-preview.md` for the per-source breakdown: evidence/knowledge mix, MMLU-aux/
  AQuA/MedMCQA/LogiQA-style distilled MCQ, the rubric-conditioned typed-workflow corpus, and the three
  HF-sourced workflow sets).
- **`data_wf_long`** — 1,024-token-state variant of the workflow corpus (PLAN7 §3z/§3aa): the same
  rubric-conditioned generator at longer state length, added because long-state families (policy
  documents, multi-hop) were untrained at 256 tokens.
- **`data_wh`** (`scripts/decisionmix_v2.py`, DecisionMix v2, PLAN7 Track D): 60,605 train / 3,000
  val / 6 eval files, a programmatic rule engine (12 domains × 6 boolean conditions, levels 1–7),
  100% of train rows in counterfactual rubric groups (same state+candidates, ≥2 distinct golds). Held
  out: 1 domain/level (`wh_heldout_grammar`), 3 whole domains (`wh_heldout_family`), 2 rubric styles
  (`wh_heldout_style`), and level 7 — temporal/numeric/probability/trade-off composition — which is
  eval-only, never trained on. 0/231 JevBench leak hits.
- **`data_u`** (same script, DecisionMix v2): 31,029 train / 2,000 val / 3 eval files, 100%
  soft-target rows. Real sources: UNLI's *validation* split (train/test already consumed elsewhere)
  and `metaeval/ambient`'s ambiguous rows (uniform target over listed labels); `metaeval/chaos-mnli-
  ambiguity` is eval-only (MNLI-only mirror of `chaos-nli`, which 404s). The remaining 92% is four
  synthetic generators with exact closed-form targets (partial evidence, noisy-sensor Bayes
  posterior, ordinal confusion, conflicting-sources log-odds), each with a held-out parameter range.
  0/231 JevBench leak hits.

See PLAN7.md's "Track D — DecisionMix v2 build notes" for the full corpus notes.

## Results

### `ts1b_semif` at 1.7B, against the constant-prediction floor and the 4B sibling

Evidence/knowledge rows are the train-time eval pass (`results.json`). Held-out workflow, uncertainty
and external rows for v3 are the **full-file** post-hoc pass (`eval_wf.json`, `mode native`,
`max_state 4096`, `limit 0`), and "floor" is that file's `baselines.majority_acc` for the same set.
JevBench rows are `jev_native_<ck>/summary.json`.

**The `tm2` column is a different protocol and is marked as such.** `tm2` has no full-file pass, so
its held-out / uncertainty / external rows are its capped training-pass numbers (`eval_cap 1500`,
`max_state 2048`). On `ts1b` the two protocols disagree by up to 34 points on `pagerduty_trigger`, so
**do not difference the marked rows across columns.** The unmarked rows (evidence/knowledge, MMLU,
JevBench) are the same protocol in both columns and are comparable.

| | floor | **`ts1b_semif`** (v3) | `tm2` (4B sibling) |
|---|---|---|---|
| CLINC-150 / TREC-fine / HWU64 / 20NG | – | .830 / .494 / .784 / .549 | .795 / .480 / .717 / .579 |
| SNLI / MNLI / BoolQ / ANLI | – | .903 / .867 / .817 / .512 | .900 / .855 / .840 / .554 |
| MMLU-Pro among-K / shuffled-q among-K (Δ_q_sh) / false-abstain | – | .316 / .198 (Δ .118) / .003 | .429 / .235 (Δ .194) / .005 |
| held-out noul / score / style / flip both-correct (full rows) | .583 / .355 / .364 / – | .705 / .514 / .854 / .741 | *capped:* .858 / .586 / .842 / – |
| held-out score NLL / typed-decisions NLL (full rows) | – | 1.15 / 1.29 | – |
| wh held-out family / grammar / style / level 7 / flip | .384 / .385 / .427 / .460 / .400 | .832 / .898 / .896 / .500 / .718 | *capped:* .889 / .886 / .896 / .540 / .835 |
| u ChaosNLI / real held-out / synthetic | .464 / .507 / .575 | .498 / .614 / .926 | *capped:* .525 / .694 / .909 |
| external: PagerDuty / jevlogs / Mind2Web / tree-choice / typed-decisions | .792 / .697 / .427 / .114 / .290 | .857 / .648 / .399 / .408 / .478 | *capped:* .757 / .518 / .497 / .585 / .515 |
| JevBench std / easy / hard (Brier std / hard; ECE std) | – | .792 / 1.000 / .441 (.370 / .746; .163) | .861 / 1.00 / .495 (.22 / .72; .07) |

**Read.** v3 clears the constant-prediction floor on every held-out workflow, DecisionMix-v2 and
uncertainty set, and on three of the five external sets. PagerDuty is .857 against a .792 floor.
Rubric sensitivity is intact: flip accuracy .718 on `wh` and .741 both-correct on `wf`, against a
.400 floor.

Against `ts1b`, on the matched full-file pass, held-out yes/no is **.705 vs .715** — level, within a
point. The v2 card reported a 7-point decrease on this primitive; that comparison put `ts1c`'s
*capped* .6427 against `ts1b`'s *full-file* .7147, which are two different protocols. Matched, the
regression was never that large, and at v3 it is gone.

Where v3 pays for the render: held-out family .832 vs `ts1b`'s .837, level-7 .500 vs .496, and
320-way tree choice .408 vs .474 — the last is the one real loss on the full-file pass. JevBench
standard is .792 (57/72) with ECE .163.

### JevBench disclosures (apply to the numbers above)

This is a **public-subset run, not a ranked entry** (JevBench's own leaderboard requires ≥95%
coverage including the non-public judge items; we ran the public ids only). Harness version
`jevbench-v1`; probability source `native` for all three tiers, with 0 renormalised items and schema
validity 1.00. Reported probabilities are the head's softmax **conditioned on non-∅** (P(∅) is
dropped and the rest renormalized); the mean p_null for this run is
`_not measured for this checkpoint_` — the summary does not record it. The checkpoint was trained at
up to 1,024-token states (`args.max_state`, reverted from `ts1c`'s 2,048) and the held-out pass runs at up to 4,096
(`eval_wf.json` `max_state`) — out-of-training-length but not truncated. Latency is in-process, one
decision at a time, model load excluded; the benchmark record does not name the GPU. Cost is null
(`ledger_charged_usd: null`, `cost_basis: local_gpu_no_provider_tariff` — this is not a hosted-API
run). Option order is the harness's own label order; a reversed-order control is
`_not measured for this checkpoint_`, so no fine ranking among our own checkpoints is supported by
these numbers alone. Split sizes are 72 standard / 48 easy / 111 hard scorable ids; the standard tier
has 36 paraphrase pairs (`paraphrase_consistency.pairs`), so n_eff = 36 and SE ≈ .06 at p = .5. The
majority/chance baselines for the three JevBench tiers are `_not measured for this checkpoint_` —
the summary carries no baseline field, and the previous card's values are not re-derivable from
these artefacts.

## Latency (`bench_ts1b_semif/bench.json`, L_s = 256, native path)

| 1.7B | single decision, K = 2 / 32 / 128 (ms) | marginal per query, M = 32, K = 2 / 32 / 128 (ms) |
|---|---|---|
| `ts1b_semif` | 59 / 64 / 106 | 3.5 / 6.4 / 24.4 |

This bench is **partial**: it covers L_s = 256 only and carries no peak-memory figures, so the
`6.2 → 9.1 GB` this card used to quote is not restated. The shape is unchanged from earlier
checkpoints of this size — a single decision costs ~59-64 ms across most of K, because the KV-cached
state and a short suffix dominate, and only climbs at large K.

**Two different latency numbers exist for this line and they measure different things. Do not
present either without its protocol.** The site publishes 45 ms for `typical-small`, which is the
apples-to-apples ladder in REPORT §3ab: one pod, one torch build, for comparing rungs of the size
ladder against each other. The 59 ms above is this checkpoint's own bench on a different pod. Neither
is comparable to a leaderboard latency measured serially over a network, and `tm2` has no benchmark
on either path, so the two release checkpoints still cannot be placed on a common latency axis.

## Deployment caveat (v1 and v2 only): **retired in v3**

Earlier checkpoints in this line carried a positional trap, and it was the headline caveat on both
previous cards: `data_wf_long` rendered `Case: <facts>` *after* the policy body, training
right-truncated at `--max_state 1024`, and 98.8% of those rows lost their facts before the model saw
them (REPORT §3ag). v1 therefore learned to answer long policies from where its training data put the
evidence, and callers had to render the policy first and the case facts last to avoid a 20-point
drop.

Scored on 605 held-out long states (`scripts/make_long_eval.py`), identical items, differing only in
where the `Case:` block sits, no truncation at eval (majority-class floor .413):

| checkpoint | facts **before** the policy body | facts **after** it | spread |
|---|---:|---:|---:|
| `typical-small` v1 (`ts1b`) | .598 | .798 | −20.0 pts |
| `typical-small` v2 (`ts1c`) | .947 | .790 | +15.7 pts |
| **`typical-small` v3 (`ts1b_semif`)** | **.952** | **.846** | **+10.6 pts** |

**v3 needs no ordering instruction.** It is the first checkpoint in this line above .84 in both
orders, so you can hand it a document without telling the caller how to arrange it. Note that v2 did
not remove the asymmetry so much as invert it — strong facts-first, still .790 the other way — while
v3 lifts its own weaker order (.846) 4.8 points above v1's better one (.798).

What v3 does *not* fix is the same capability as JevBench measures it: the `long_policy` family on
the hard tier reads .263 on 19 items, where this 605-item pass reads .952. That disagreement is a
property of a 19-item sample, not a second result; see the limitations below.

## Known limitations

- Soft-target quality is the weak axis at this size: held-out score NLL 1.15 at .514 accuracy, and
  typed-decisions NLL 1.29 at .478 accuracy (floor .290).
- **320-way tree choice is the one clear regression against `ts1b`**: .408 vs .474 on the matched
  full-file pass. Large flat candidate sets are what the render change cost.
- Mind2Web stays below its constant-prediction floor (.399 vs floor .427) — general web-agent
  candidate sets remain outside what this training distribution covers. jevlogs also falls below its
  floor at this version (.648 vs .697), where `ts1b` cleared it at .710.
- Level-7 composition (temporal / unit / expected-value / trade-off reasoning, `wh_level7`) sits at
  .500 against a .460 floor. That is marginally above constant prediction and 30-odd points below the
  same checkpoint's held-out family / grammar / style scores. The gap to the in-distribution families
  is the finding, not the level. Nothing trained so far moves it, because the generator never
  produces this composition.
- JevBench hard is .441 with Brier .746, ECE .240 and ordinal MAE .658. Within that tier
  `long_policy` is still the weakest family at .263 (n = 19), then `temporal_numeric` .333 (n = 15)
  and `tradeoff` .333 (n = 6). Read those n's before reading the accuracies: the same long-policy
  capability measured on 605 items reads .952. Probability quality on soft/ordinal gold is the
  clearest open defect at this size.
- The shuffled-rubric controls sit at their floors — `wf_rubric_shuffled` .370 (floor .360),
  `wh_rubric_shuffled` .402 (floor .410) — i.e. with the rubric scrambled the model is no better than
  guessing, which is the intended behaviour but leaves no margin.
- Very large candidate sets (K in the hundreds to thousands) still need the energy → top-r → native
  path, not this native head directly (unchanged from the preview).
- The render gain is a point estimate, not a resolved result: `ts1b_semif` − `ts1b` on JevBench
  standard is +.097 with a 95% interval of [−.056, +.250], p = .23 (buffalo:RESULTS.md §5a). The
  long-state and held-out numbers above are where the evidence for v3 actually sits.

## License

- Backbone (`Qwen/Qwen3-1.7B-Base`): Apache-2.0.
- Training data: per-source licenses triaged in PLAN6.md's dataset-survey table. **Two trained
  sources are non-commercial and were previously mis-summarised here as permissive:
  ANLI (`facebook/anli`, in the E mix) is CC BY-NC 4.0, and SciQ (11.7k rows of `data_kb`) is
  CC BY-NC 3.0.** The rest of the trained mix is MIT / Apache-2.0 / CC-BY-4.0 or generated in-repo.
  `jevlogs` is research-licensed and used only as a caveated held-out sanity eval, never trained
  on. Whether non-commercially-licensed training data constrains use of the Apache-2.0 weights is
  unsettled; this card states what went in and does not assert a conclusion. LogiQA2, MedMCQA and
  AQuA have not yet been re-triaged to this standard.
- New in this release (PLAN7 Track D, `data_wh` / `data_u`): `data_wh` is a programmatic rule-engine
  corpus generated in-repo (`scripts/decisionmix_v2.py`), no external license constraints. `data_u`'s
  UNLI validation split is MIT; `metaeval/ambient` (trained on, ambiguous rows) and
  `metaeval/chaos-mnli-ambiguity` (eval-only, never trained on) do not declare a license on their HF
  cards — flagged, not asserted.
- JevBench numbers in this card are a **public-subset run** against `fstandhartinger/jevbench`
  (harness version `jevbench-v1`; 72 standard / 48 easy / 111 hard public ids) — not a submitted or
  ranked leaderboard entry.

## How to run

**Inference code included under `inference/`; training code release to follow.** The public repo
(`OzLabs/typical-small`) ships a minimal, self-contained inference package (own `Typical` class —
no dependency on this training repo, just `torch`, `transformers`, `safetensors`, `huggingface_hub`,
`numpy`). The repo carries `ts1b_semif`, so the API below downloads v3.

```bash
pip install -r inference/requirements.txt
```

```python
from typical import Typical  # inference/typical/, downloaded alongside your code

m = Typical.from_pretrained("OzLabs/typical-small", device="auto")

# K-way choice over a fixed label set -> {label: p, ...} + p_null
m.choice(state, "What does the customer want?", ["refund", "replacement", "repair"])

# Yes/no -> P(yes)
m.noul(state, "Is the order still under warranty?")

# Ordinal levels -> {level: p, ...} + p_null + expected (E[index])
m.score(state, "How urgent is this ticket?", ["0", "1", "2", "3"])

# Full JevBench-style decide(): (probs, runtime) -- mirrors PCDMDecider.decide exactly
probs, runtime = m.decide(
    state,
    {"type": "choice", "instructions": "What does the customer want?",
     "criteria": {"refund": "money back", "replacement": "a new item shipped"}},
    ["refund", "replacement"],
)
```

`state` is a string or JSON-serialisable dict. Parity between the packaged `Typical` class and the
internal `PCDMDecider(mode="native")` is `_not measured for this checkpoint_` — the max-abs-diff
parity check (and the reversed-label invariance check that accompanied it) has not been re-run
against `ts1b_semif`; the reversed-label invariance reported in the table above comes from the training-run
eval (`results.json`, `.eval.wf_rubric_flip.noul_reversed`), not from the packaged class.

Internal call shape (this training repo, not yet public):

```python
from pcdm_jev.decider import PCDMDecider

decider = PCDMDecider("runs/ts1b_semif", mode="native")
question = {"instructions": "...", "type": "choice", "criteria": {"a": "option a", "b": "option b"}}
probs, runtime = decider.decide(state="...", question=question, labels=["a", "b"])
# probs: dict label -> probability; runtime["p_null"] is the abstention mass
```

(`PCDMDecider.__init__(run_dir=None, device="auto", mode="energy", energy_run=None, max_state=4096,
max_query=256, backbone=None, tap_layer=0, shots=0)` — pass `mode="native"` and the checkpoint dir.)

Held-out workflow / external eval reproduction (post-hoc, full state length; requires this private
training repo):

```bash
uv run --no-sync python scripts/eval_wf.py --run runs/ts1b_semif --mode native \
    --files data_wf/eval/*.jsonl data_wf_hf/eval/*.jsonl data_wh/eval/*.jsonl data_u/eval/*.jsonl \
    --limit 0 --out runs/ts1b_semif/eval_wf.json
```

JevBench: `scripts/jevbench_run.py` against the checkpoint in `mode="native"` (see `pcdm_jev/` for the
harness adapter and the protocol disclosures reproduced above).

## Artefacts backing this card

Published on the hub under `guychuk/pcdm-runs`; `best.pt` is mirrored into the public repo and its
sha256 (`dc183b39f90b0485d88ea4c659ad5248101e378a99428f123042fbd24cb60f22`) was verified against the
source path after upload.

- `ts1b_semif/results.json` — training args + train-time eval suite.
- `ts1b_semif/eval_wf.json` — post-hoc held-out workflow / external eval (full file, `max_state 4096`,
  `limit 0`).
- `ts1b_semif/eval_wf_long_policy.json` — long-policy held-out pass (`wf_long_policy`, acc .952 vs
  floor .413, n = 605), and `…_lastrender.json` for the facts-last arm.
- `probe_ts1b_semif/results.json` — Δ_q probe results.
- `jev_native_ts1b_semif/{original,easy,hard}/summary.json` — JevBench public-subset run, harness
  version `jevbench-v1`.
- `bench_ts1b_semif/bench.json` — latency benchmark (partial: L_s = 256, no peak memory).

## Links

- Release line: https://huggingface.co/OzLabs/typical-small (carries `ts1b_semif`, v3)
- Parent release: https://huggingface.co/OzLabs/typical-small-preview
- Sibling release (4B): https://huggingface.co/OzLabs/typical-medium
- Sibling release (14B, preview): https://huggingface.co/OzLabs/typical-large-preview
