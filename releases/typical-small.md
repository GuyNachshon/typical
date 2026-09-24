# typical-small

**Release line: https://huggingface.co/OzLabs/typical-small** — this card describes checkpoint
`ts1c`, which has **not** been uploaded yet; the public repo still carries the previous checkpoint.

A clean retrain of the 1.7B checkpoint at `runs/ts1c`, superseding `ts1b` (the previous
`typical-small`). Still the successor line to `typical-small-preview` (`nc_v3_tap20_wf`). Parent
release: `typical-small-preview`. Sibling release at 4B: `typical-medium` (`tm2`,
`releases/typical-medium.md`).

## What this model is

- **Backbone:** `Qwen/Qwen3-1.7B-Base`, truncated at layer 20 of 28 — same tap as the preview.
- **Adaptation:** LoRA, rank 16, on the top 8 layers (13–20) of the truncated backbone.
- **Readout:** native contextual decision head (`nc_head=n3`), `letters_nonull` rendering, *factored*
  null (∅ is a separate head decision, not a rendered candidate) — identical shape to the preview.
- **Tap normalisation:** z-scored tap activations (`zscore: true`), new in this checkpoint — `ts1b`
  trained without it.
- **Typed primitives** (PLAN7 Track C):
  - **Choice** — the N3 contextual readout above, unchanged.
  - **Noul** (yes/no) — a Bernoulli head, `P(yes) = σ(w·h_D)`, on the decision state with a
    query-only suffix (no rendered candidates). Routed **per row** (commit `7b9d520`): a row only
    goes through the Bernoulli head if its candidate set is exactly `{yes, no}`; every other 2-way
    label set (e.g. `{true, false}`, `{approve, reject}`) is scored as ordinary Choice. The first
    `ts1` run collapsed to chance because this routing was global instead of per-row; `ts1b` was the
    fix, gated by 140 tests, and `ts1c` inherits it unchanged.
  - **Score** — K-way Choice trained with ordinal-smoothed targets (τ = 0.7, §3ac's adopted recipe).

## Training args

```
backbone: Qwen/Qwen3-1.7B-Base
lora_layers: 8            lora_r: 16            lr: 0.0003            lora_lr: 0.0001
tap_layer: 20             zscore: true
readout: native           nc_head: n3            nc_render: letters_nonull
null: factored            noul_head: bern        score_head: choice     ordinal_smooth: 0.7
steps: 8000               bs: 64                 grad_accum: 8
eval_every: 8000          val_every: 2000        ckpt_every: 2000
data: data_v5             extra_data: data_kb,data_wf,data_wf_hf,data_wf_long,data_wh,data_u
family_weights: E:0.40,K:0.15,W:0.35,U:0.10
bucket_map: data_wf_long=W,data_wh=W,data_u=U
null_aug: W:0.20          max_state: 2048        drop_truncated: true
best_on: data_u_val,data_wh_val
tower_d: 512              tower_layers: 2        tower_heads: 8
seed: 0
```

Effective batch 64 via `--grad_accum 8`. Checkpoint selection is `--best_on data_u_val,data_wh_val`
(uncertainty + hard-curriculum validation), not overall val loss.

Final val NLL 0.439, temperature T = 1.040, null_offset 0.0 (fit post-hoc on the `val` split, applied
at eval).

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

### `ts1c` at 1.7B, against the constant-prediction floor and the 4B sibling

Evidence/knowledge rows are the train-time eval pass (`runs/<ck>/results.json`). Held-out workflow,
uncertainty and external rows are the **full-file** post-hoc pass (`runs/<ck>/eval_wf.json`, `mode
native`, `max_state 4096`, `limit 0`), and "floor" is that file's `baselines.majority_acc` for the
same set. JevBench rows are `runs/jev_native_<ck>/summary.json`.

| | floor | **`ts1c`** | `tm2` (4B sibling) |
|---|---|---|---|
| CLINC-150 / TREC-fine / HWU64 / 20NG | – | .797 / .500 / .749 / .553 | .795 / .486 / .720 / .580 |
| SNLI / MNLI / BoolQ / ANLI | – | .889 / .853 / .776 / .492 | .900 / .855 / .840 / .554 |
| MMLU-Pro among-K / shuffled-q among-K (Δ_q_sh) / false-abstain | – | .338 / .212 (Δ .127) / .008 | .429 / .235 (Δ .194) / .005 |
| held-out noul / score / style / flip both-correct (full rows) | .583 / .355 / .364 / – | .643 / .488 / .830 / .543 | .858 / .585 / .841 / .718 |
| held-out score NLL / typed-decisions NLL (full rows) | – | 1.09 / 1.50 | 0.97 / 1.13 |
| wh held-out family / grammar / style / level 7 / flip | .384 / .385 / .427 / .460 / .400 | .853 / .901 / .883 / .524 / .730 | .889 / .887 / .896 / .539 / .740 |
| u ChaosNLI / real held-out / synthetic (acc, NLL) | .464 / .507 / .575 | .514 (1.03) / .612 (.67) / .924 (.66) | .525 (1.04) / .694 (.66) / .909 (.67) |
| external: PagerDuty / jevlogs / Mind2Web / tree-choice / typed-decisions acc | .792 / .697 / .427 / .114 / .291 | .857 / .766 / .364 / .471 / .464 | .834 / .612 / .496 / .588 / .517 |
| JevBench std / easy / hard (Brier std / hard; ECE std) | – | .708 / 1.00 / .432 (.42 / .72; .15) | .861 / 1.00 / .495 (.22 / .72; .07) |
| Noul reversed-label \|ΔP(yes)\| max (mean) | – | .007 (.00005) | .002 (.00008) |

The comparison columns the previous card carried (`typical-small-preview`, `r1_dmv2`, the untrained
1.7B base) are dropped: none of those checkpoints has an eval artefact in this release, so their
numbers cannot be re-read and are not reproduced here.

**Read.** `ts1c` clears the constant-prediction floor on every held-out workflow, DecisionMix-v2 and
uncertainty set, and on four of the five external sets — including both floors that mattered at this
size, PagerDuty (.857 vs floor .792) and jevlogs (.766 vs .697). Rubric sensitivity is intact: flip
accuracy .730 on `wh` and .543 both-correct on `wf`, against a .400 floor. The Bernoulli Noul head is
order-invariant to within .007 max |ΔP(yes)|.

Against `ts1b` it gives back held-out yes/no: **.6427 here vs `ts1b`'s .7147** — a 7-point decrease
on the primitive this line was built to fix, and the clearest regression in this checkpoint.
(`ts1b`'s other eval numbers are not restated here: that checkpoint has no artefact in this release,
so only the yes/no figure above is a verified comparison.) JevBench standard is .708 (51/72; extraction 11 of 12) with ECE .152. Mind2Web remains below
its floor (.364 vs .427) and level-7 composition sits just above its (.524 vs .460).

### JevBench disclosures (apply to the numbers above)

This is a **public-subset run, not a ranked entry** (JevBench's own leaderboard requires ≥95%
coverage including the non-public judge items; we ran the public ids only). Harness version
`jevbench-v1`; probability source `native` for all three tiers, with 0 renormalised items and schema
validity 1.00. Reported probabilities are the head's softmax **conditioned on non-∅** (P(∅) is
dropped and the rest renormalized); the mean p_null for this run is
`_not measured for this checkpoint_` — the summary does not record it. The checkpoint was trained at
up to 2,048-token states (`args.max_state`) and the held-out pass runs at up to 4,096
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

## Latency (`runs/bench_ts1c/bench.json`, L_s = 256, native path)

| 1.7B | single decision, K = 2 / 32 / 256 (ms) | marginal per query, M = 32, K = 2 / 32 / 256 (ms) | peak memory, K = 2 → 256 (GB) |
|---|---|---|---|
| `ts1c` | 61 / 58 / 118 | 3.4 / 3.8 / 25.8 | 6.2 → 9.1 |

**Two different latency numbers exist for this checkpoint and they measure different things. Do not
present either without its protocol.** The table above is the apples-to-apples ladder: one pod, one
torch build, L_s = 256, state encode included, on the pre-optimisation serving path. It is for
comparing rungs of the size ladder against each other, not for quoting as the cost of a decision in
production. Neither is comparable to a leaderboard latency measured serially over a network.

On the ladder protocol a single decision costs ~58-61 ms across most of K (the KV-cached state and a
short suffix dominate) and roughly doubles to 118 ms only at K = 256; batched marginal cost is
3.4 ms/query at K = 2, rising to 25.8 ms/query at K = 256. The cross-size version of that ladder
(1.7B vs 4B vs 14B) is `_not measured for this checkpoint_`: `tm2` has no latency benchmark, so the
two release checkpoints cannot be placed on a common latency axis.

## Deployment caveat: put case facts at the END of a long state

**Read this before using the checkpoint on anything longer than a paragraph.** This checkpoint was
trained before the facts-first corpus fix (REPORT §3ag), so it carries that corpus's positional
bias: `data_wf_long` rendered `Case: <facts>` *after* the policy body, training right-truncates at
`--max_state 1024`, and 98.8% of those rows therefore lost their facts before the model saw them.
The model learned to answer long policies from where its training data put the evidence.

Scored on 605 held-out long states (`scripts/make_long_eval.py`), identical items, differing only in
where the `Case:` block sits, no truncation at eval:

| | facts **before** the policy body | facts **after** it | cost |
|---|---:|---:|---:|
| `typical-small` | .598 | .798 | **-20.0 pts** |

Rendered facts-first, this checkpoint sits on the majority-class floor for the yes/no family -- it
has stopped reading the evidence rather than reading it poorly.

Callers write their own state text, so this is under your control: **render the policy or document
first and the case facts last.** Retrained checkpoints on the fixed corpus do not show the bias
(REPORT §3ak-a/§3ak-c; the facts-first 14B reaches .997 on the same set), and retraining this
checkpoint on it is the next planned release work.

## Known limitations

- Held-out yes/no (Noul) is **.6427 against a .5833 floor, down from `ts1b`'s .7147** — a real
  decrease, not noise, on the primitive the per-row Bernoulli routing was introduced to fix.
- Soft-target quality is the weak axis at this size: held-out score NLL 1.09 at .488 accuracy, and
  typed-decisions NLL 1.50 at .464 accuracy (floor .291).
- Mind2Web stays below its constant-prediction floor (.364 vs floor .427) — general web-agent
  candidate sets remain outside what this training distribution covers.
- Level-7 composition (temporal / unit / expected-value / trade-off reasoning, `wh_level7`) sits at
  .524 against a .460 floor. That is marginally above constant prediction and 30-odd points below
  the same checkpoint's held-out family / grammar / style scores. Earlier versions of this card said
  "chance"; the gap to the in-distribution families is the finding, not the level. Nothing trained
  so far moves it, because the generator never produces this composition.
- JevBench hard is .432 with Brier .722, ECE .241 and ordinal MAE .817; within that tier
  `long_policy` is the weakest family at .211 (n = 19), then `temporal_numeric` .333 (n = 15) and
  `tradeoff` .333 (n = 6). Probability quality on soft/ordinal gold is still the clearest open defect
  at this size.
- The shuffled-rubric controls sit at their floors — `wf_rubric_shuffled` .365 (floor .360),
  `wh_rubric_shuffled` .394 (floor .410) — i.e. with the rubric scrambled the model is no better than
  guessing, which is the intended behaviour but leaves no margin.
- Very large candidate sets (K in the hundreds to thousands) still need the energy → top-r → native
  path, not this native head directly (unchanged from the preview).

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
`numpy`). Note that the repo does not yet carry `ts1c`; the API below is the interface, and the
weights it downloads are still the previous checkpoint until this one is published.

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
against `ts1c`; the reversed-label invariance reported in the table above comes from the training-run
eval (`results.json`, `.eval.wf_rubric_flip.noul_reversed`), not from the packaged class.

Internal call shape (this training repo, not yet public):

```python
from pcdm_jev.decider import PCDMDecider

decider = PCDMDecider("runs/ts1c", mode="native")
question = {"instructions": "...", "type": "choice", "criteria": {"a": "option a", "b": "option b"}}
probs, runtime = decider.decide(state="...", question=question, labels=["a", "b"])
# probs: dict label -> probability; runtime["p_null"] is the abstention mass
```

(`PCDMDecider.__init__(run_dir=None, device="auto", mode="energy", energy_run=None, max_state=4096,
max_query=256, backbone=None, tap_layer=0, shots=0)` — pass `mode="native"` and the checkpoint dir.)

Held-out workflow / external eval reproduction (post-hoc, full state length; requires this private
training repo):

```bash
uv run --no-sync python scripts/eval_wf.py --run runs/ts1c --mode native \
    --files data_wf/eval/*.jsonl data_wf_hf/eval/*.jsonl data_wh/eval/*.jsonl data_u/eval/*.jsonl \
    --limit 0 --out runs/ts1c/eval_wf.json
```

JevBench: `scripts/jevbench_run.py` against the checkpoint in `mode="native"` (see `pcdm_jev/` for the
harness adapter and the protocol disclosures reproduced above).

## Artefacts backing this card

All local to this training repo; nothing here has been uploaded or hashed for publication yet.

- `runs/ts1c/results.json` — training args + train-time eval suite.
- `runs/ts1c/eval_wf.json` — post-hoc held-out workflow / external eval (full file, `max_state 4096`).
- `runs/ts1c/eval_wf_long_policy.json` — long-policy held-out pass (`wf_long_policy`, acc .947 vs
  floor .413).
- `runs/probe_ts1c/results.json` — Δ_q probe results.
- `runs/jev_native_ts1c/summary.json` — JevBench public-subset run summary.
- `runs/bench_ts1c/bench.json` — latency / peak-memory benchmark.

## Links

- Release line: https://huggingface.co/OzLabs/typical-small (does not yet carry `ts1c`)
- Parent release: https://huggingface.co/OzLabs/typical-small-preview
- Sibling release (4B): https://huggingface.co/OzLabs/typical-medium
