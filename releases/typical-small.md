# typical-small

**Public release: https://huggingface.co/OzLabs/typical-small**

Release 1 (PLAN7 Tracks A–D): a clean retrain of the 1.7B checkpoint, `ts1b`, uploaded verbatim to
`guychuk/pcdm-runs` under `typical-small/`. This is the frozen successor to `typical-small-preview`
(`nc_v3_tap20_wf`) — REPORT.md §3ae's verdict, unchanged. Parent release: `typical-small-preview`.
Sibling release at 4B: `typical-medium` (`tm1b`, `releases/typical-medium.md`, REPORT.md §3af).

## What this model is

- **Backbone:** `Qwen/Qwen3-1.7B-Base`, truncated at layer 20 of 28 — same tap as the preview.
- **Adaptation:** LoRA, rank 16, on the top 8 layers (13–20) of the truncated backbone.
- **Readout:** native contextual decision head (`nc_head=n3`), `letters_nonull` rendering, *factored*
  null (∅ is a separate head decision, not a rendered candidate) — identical shape to the preview.
- **Typed primitives** (PLAN7 Track C, new in this release):
  - **Choice** — the N3 contextual readout above, unchanged.
  - **Noul** (yes/no) — a Bernoulli head, `P(yes) = σ(w·h_D)`, on the decision state with a
    query-only suffix (no rendered candidates). Routed **per row** (commit `7b9d520`): a row only
    goes through the Bernoulli head if its candidate set is exactly `{yes, no}`; every other 2-way
    label set (e.g. `{true, false}`, `{approve, reject}`) is scored as ordinary Choice. The first
    `ts1` run collapsed to chance because this routing was global instead of per-row (REPORT.md
    §3ae) — `ts1b` is the fix, gated by 140 tests.
  - **Score** — K-way Choice trained with ordinal-smoothed targets (τ = 0.7, §3ac's adopted recipe).

## Training args (`best.pt`'s args are authoritative; `results.json["args"]` is from an `--eval_only` rerun to restore missing MMLU sets and records `ordinal_smooth: 0.0` for that reason — training used 0.7, see the flag comment below)

```
backbone: Qwen/Qwen3-1.7B-Base
lora_layers: 8            lora_r: 16            lora_lr: 0.0001
tap_layer: 20             zscore: false
readout: native           nc_head: n3            nc_render: letters_nonull
null: factored            noul_head: bern        score_head: choice     ordinal_smooth: 0.0   # eval-only rerun flag; training used 0.7 (see above)
steps: 12000              bs: 64                 grad_accum: 1
eval_every: 4000          val_every: 1000        ckpt_every: 2000
data: data_v5             extra_data: data_kb,data_wf,data_wf_hf,data_wf_long,data_wh,data_u
family_weights: E:0.40,K:0.15,W:0.35,U:0.10
bucket_map: data_wf_long=W,data_wh=W,data_u=U
null_aug: W:0.20          max_state: 1024
tower_d: 512              tower_layers: 2        tower_heads: 8
seed: 0
```

Final val NLL 0.376, temperature T = 1.124, null_offset 0.0 (fit post-hoc on the `val` split, applied
at eval).

## Data

Family-balanced sampler, E .40 / K .15 / W .35 / U .10 of every training batch, plus `--null_aug
W:0.20` (15–20% of W rows get their gold candidate removed so the target is ∅) and `--max_state 1024`
(vs. the preview's 256):

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

See PLAN7.md's "Track D — DecisionMix v2 build notes" for the full corpus notes, and REPORT.md
§3z–§3ae for the mixture-sweep and Release-1-candidate history.

## Results

### §3ae — Release-1 candidate at 1.7B (`ts1b`) vs the preview and the intermediate `r1_dmv2`

| | `typical-small-preview` (§3t/§3w) | `r1_dmv2` (§3ad) | **`ts1b`** | 1.7B base |
|---|---|---|---|---|
| CLINC-150 / TREC-fine / HWU64 / 20NG | .734 / .372 / .735 / .522 | .820 / .474 / .726 / .501 | .804 / **.508** / .761 / .540 | .845 / .468 / .757 / .515 |
| SNLI / MNLI / BoolQ / ANLI | .894 / .849 / .831 / .481 | .894 / .852 / .834 / .501 | .894 / .859 / .824 / .497 | .904 / .865 / .834 / .507 |
| MMLU-Pro among-K / Δ_q_sh / false-abstain | .330 / .119 / .006 | .333 / .110 / – | .343 / .127 / .005 | .353 / .122 / .003 |
| held-out noul / score / style / flip both-correct (full rows) | .699 / .498 / .901 / .565 | .658 / .501 / .876 / .519 | **.715** / **.520** / .859 / **.581** | – |
| held-out score NLL / typed-decisions NLL (full rows) | 2.03 / 2.06 | 1.56 / 1.71 | **1.01 / 1.25** | – / 1.22 |
| wh held-out family / grammar / style / level 7 / flip | – | .827 / .881 / .888 / .498 / .710 | **.836 / .898 / .896** / .495 / **.718** | – |
| u ChaosNLI / real held-out / synthetic (acc, NLL) | – | .537 (1.00) / .623 (.67) / .909 (.67) | .547 (1.00) / **.649** (.66) / .916 (.67) | – |
| external: PagerDuty (floor .792) / jevlogs (.697) / Mind2Web (.427) / tree-choice / typed-decisions acc | .779 / .522 / .427 / .539 / .457 | .651† / .500† / .346† / – / .421† | **.817 / .710** / .357 / .474 / **.510** | .776 / .697 / .300 / .506 / .487 |
| JevBench std / easy / hard (Brier std / hard; ECE std) | .750 / 1.00 / .387 (.40 / .88; .15) | .750 / 1.00 / .441 (.42 / .79) | .694 / 1.00 / .432 (.40 / .79; **.11**) | .694 / 1.00 / .378 (.47 / .74) |
| Noul reversed-label |ΔP(yes)| max | up to .55 | – | **0** (10/11 sets; .009 on one) | – |

† `r1_dmv2`'s external numbers are a train-time 1,500-row/256-token pass, not re-run at full length.

**Read (REPORT.md §3ae).** Against the frozen preview, `ts1b` keeps E and K inside budget (CLINC −4 vs
the untrained base is the one edge; TREC/20NG land above base), raises held-out noul/score/flip,
halves the soft-target NLLs (score 2.03 → 1.01, typed-decisions 2.06 → 1.25), beats or ties `r1_dmv2`
on every curriculum and uncertainty set, is exactly order-invariant on Noul, and is the first 1.7B
model above the constant-prediction floor on PagerDuty (.817) and jevlogs (.710, marginal). It gives
back held-out styles (−4), Mind2Web (below floor again), tree-choice (−6), and JevBench standard
(.750 → .694, ~1 SE at n_eff = 36; extraction 12 → 9 of 12) while JevBench hard rises (.387 → .432,
Brier .88 → .79). **Decision: freeze `ts1b` as `typical-small` (Release 1 at 1.7B)**, with the preview
kept as the reference.

### JevBench disclosures (REPORT.md §3q — apply to the numbers above)

This is a **public-subset run, not a ranked entry** (JevBench's own leaderboard requires ≥95%
coverage including the 146 non-public judge items; we ran the 231 public ids only). Reported
probabilities are the head's softmax **conditioned on non-∅** (P(∅) is dropped and the rest
renormalized; mean p_null ≈ .03, native). Checkpoints trained at up to 1,024-token states / 64-token
queries are run at up to 4096 / 256 tokens (out-of-training-length but not truncated). Latency is
in-process on one H100, one decision at a time, model load excluded. Cost is null (no tariff — this
is not a hosted-API run). Option order is the harness's own label order; a reversed-order control on
a related checkpoint swung standard accuracy ±4 points, larger than the differences between our own
model variants, so no fine ranking among our own checkpoints is supported by these numbers alone.
Majority/chance baselines on this split are .311 standard / .284 easy / .336 hard (n=72 standard,
SE ≈ .058; n_eff = 36) — `ts1b`'s hard tier (.432) is within 1 SE of chance, standard (.694) is ~1 SE
below the preview.

## Latency (REPORT.md §3ab, apples-to-apples ladder, one pod, one torch build, L_s = 256)

| 1.7B | single decision, K = 2 / 32 / 256 (ms) | marginal per query, M = 32, K = 2 / 32 / 256 (ms) | peak memory, K = 2 → 256 (GB) |
|---|---|---|---|
| `ts1b`-shape recipe | 45 / 46 / 106 | 2.7 / 3.7 / 28.0 | 6.4 → 9.3 |

**Two different latency numbers exist for this checkpoint and they measure different things. Do not
present either without its protocol.** The table above is the §3ab *apples-to-apples ladder*: one
pod, one torch build, L_s = 256, state encode included, on the pre-optimisation serving path. It is
for comparing rungs of the size ladder against each other, not for quoting as the cost of a
decision in production. The number the blog posts and the site quote is **warm p50 per decision,
15.5–17 ms** (`runs/serve_bench2/results.json`, K = 2, 256-token state, prefix KV already cached,
one stream, in process, model load excluded) — measured after the KV-cache deep copy was replaced
with a stride-0 view (§3ag). Neither is comparable to a leaderboard latency measured serially over
a network.

A single decision on the ladder protocol costs 45 ms regardless of most of K (the KV-cached state and a short suffix
dominate); batched marginal cost is 2.7 ms/query at K = 2, rising to 28 ms/query at K = 256. Capability
per millisecond (JevBench standard / single-decision ms) puts the 4B at the knee (.833/56 vs 1.7B's
.750/45 vs 14B's .875/60) — this checkpoint is the cheapest point on that ladder, not the highest-
capability one.

## Known limitations

- JevBench standard is ~1 SE below the preview (.694 vs .750, n_eff = 36) — a real trade, not noise
  in the other direction: `ts1b` buys held-out noul/score/flip and both external floors (PagerDuty,
  jevlogs) at that cost (§3ae).
- Mind2Web stays below its constant-prediction floor (.357 vs floor .427) — general web-agent
  candidate sets remain outside what this training distribution covers.
- Level-7 composition (temporal / unit / expected-value / trade-off reasoning, `wh_level7`) sits at
  .495. The uniform-guess rate on that set is **.441** (598 items at K = 2, 224 at 3, 58 at 4;
  majority-class .250), so this is marginally above guessing and 34-40 points below the same
  checkpoint's held-out family / grammar / style scores (.836 / .898 / .896). Earlier versions of
  this card said "chance"; the gap to the in-distribution families is the finding, not the level.
  Nothing trained so far moves it, because the generator never produces this composition (§3ad).
- Hard tier is .432 with Brier .79 — within 1 SE of the .336 chance baseline; probability quality on
  soft/ordinal gold is still the clearest open defect at this size (§3aa, §3ae).
- **Do not describe this model as "calibrated", and don't threshold on hard-tier confidence.** The
  supported claim is that probability quality is trainable and that good calibration is conditional
  on decision type and family — not that the output is calibrated (paper §6.4; a single global
  temperature/offset moves in opposite directions for the E/K and W regimes). Concretely, under
  JevBench's sum-of-squares Brier a *uniform* predictor over the hard tier's candidate-set mix
  scores .664, and this checkpoint scores .79 — worse than guessing evenly. On the standard tier
  the same comparison is .689 uniform against .40 here, which is where P(∅) thresholding is worth
  using. Any card, tag or post copy claiming calibrated probabilities without that tier split is
  overstating this checkpoint.
- Held-out rubric styles regress −4 vs the preview (.859 vs .901).
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
- JevBench numbers in this card are a **public-subset run** against `fstandhartinger/jevbench` v1.2.1
  (72 standard / 48 easy / 111 hard public ids) — not a submitted or ranked leaderboard entry.

## How to run

**Inference code included under `inference/`; training code release to follow.** The public repo
(`OzLabs/typical-small`) ships a minimal, self-contained inference package (own `Typical` class —
no dependency on this training repo, just `torch`, `transformers`, `safetensors`, `huggingface_hub`,
`numpy`). Download the `inference/` folder from that repo, then:

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

`state` is a string or JSON-serialisable dict. Parity with the internal `PCDMDecider(mode="native")`
was verified at max abs probability diff = 0.0 on this checkpoint (CPU and MPS), including the
dedicated Bernoulli Noul head's reversed-label invariance check.

Internal call shape (this training repo, not yet public):

```python
from pcdm_jev.decider import PCDMDecider

decider = PCDMDecider("runs/ts1b", mode="native")
question = {"instructions": "...", "type": "choice", "criteria": {"a": "option a", "b": "option b"}}
probs, runtime = decider.decide(state="...", question=question, labels=["a", "b"])
# probs: dict label -> probability; runtime["p_null"] is the abstention mass
```

(`PCDMDecider.__init__(run_dir=None, device="auto", mode="energy", energy_run=None, max_state=4096,
max_query=256, backbone=None, tap_layer=0, shots=0)` — pass `mode="native"` and the checkpoint dir.)

Held-out workflow / external eval reproduction (post-hoc, full state length; requires this private
training repo):

```bash
uv run --no-sync python scripts/eval_wf.py --run runs/ts1b --mode native \
    --files data_wf/eval/*.jsonl data_wf_hf/eval/*.jsonl data_wh/eval/*.jsonl data_u/eval/*.jsonl \
    --limit 0 --out runs/ts1b/eval_wf_full.json
```

JevBench: `scripts/jevbench_run.py` against the checkpoint in `mode="native"` (see `pcdm_jev/` for the
harness adapter and REPORT.md §3q for the exact protocol and disclosures reproduced above).

## Files in `guychuk/pcdm-runs/typical-small/`

- `best.pt` — this checkpoint (`ts1b`).
- `results.json` — training args + full eval suite (train-time pass).
- `eval_wf_full.json` — post-hoc held-out workflow / external eval (full state length, `scripts/eval_wf.py`).
- `probe_results.json` — Δ_q probe results.
- `jevbench_summary.json` — JevBench public-subset run summary.

## Links

- Public release: https://huggingface.co/OzLabs/typical-small
- Parent release: https://huggingface.co/OzLabs/typical-small-preview
- Sibling release (4B): https://huggingface.co/OzLabs/typical-medium
