# typical-medium

**Public release: https://huggingface.co/OzLabs/typical-medium**

Release 1 at 4B (PLAN7 Tracks A–D): the same recipe as `typical-small`'s `ts1b` — r1 mix +
DecisionMix v2 + ordinal-smoothed Score + per-row Bernoulli Noul + 1,024-token states — retrained on
`Qwen/Qwen3-4B-Base`, checkpoint `tm1b`, uploaded verbatim to `guychuk/pcdm-runs` under
`typical-medium/`. REPORT.md §3af's verdict: freeze `tm1b` as `typical-medium`. Parent release:
`typical-small-preview`. Sibling release at 1.7B: `typical-small` (`releases/typical-small.md`,
REPORT.md §3ae).

## What this model is

- **Backbone:** `Qwen/Qwen3-4B-Base`, truncated at layer 26 of 36 (≈71% depth, same fraction as the
  1.7B's tap at 20/28).
- **Adaptation:** LoRA, rank 16, on the top 8 kept layers of the truncated backbone.
- **Readout:** native contextual decision head (`nc_head=n3`), `letters_nonull` rendering, *factored*
  null — identical shape to `typical-small`.
- **Typed primitives** (same as `typical-small`, PLAN7 Track C):
  - **Choice** — the N3 contextual readout above, unchanged.
  - **Noul** (yes/no) — a Bernoulli head, `P(yes) = σ(w·h_D)`, routed per row: only rows whose
    candidate set is exactly `{yes, no}` use it; every other 2-way label set scores as ordinary
    Choice.
  - **Score** — K-way Choice trained with ordinal-smoothed targets (τ = 0.7, §3ac's adopted recipe).

## Training args

```
backbone: Qwen/Qwen3-4B-Base
lora_layers: 8            lora_r: 16            lora_lr: 0.0001
tap_layer: 26             zscore: false
readout: native           nc_head: n3            nc_render: letters_nonull
null: factored            noul_head: bern        score_head: choice     ordinal_smooth: 0.7
steps: 12000              bs: 64                 grad_accum: 4
data: data_v5             extra_data: data_kb,data_wf,data_wf_hf,data_wf_long,data_wh,data_u
family_weights: E:0.40,K:0.15,W:0.35,U:0.10
bucket_map: data_wf_long=W,data_wh=W,data_u=U
null_aug: W:0.20          max_state: 1024
tower_d: 512              tower_layers: 2        tower_heads: 8
seed: 0
```

Effective batch 64 via `--grad_accum 4` (58 GB). Matched control = `ladder_4b` (§3ab: same backbone,
the 6A recipe — no typed heads, no DecisionMix v2, 256-token states).

## Data

Identical mixture and sources to `typical-small` — see `releases/typical-small.md`'s Data section
for the full per-source breakdown (`data_v5`/`data_kb` evidence+knowledge, `data_wf`/`data_wf_hf`/
`data_wf_long` rubric-conditioned workflow, `data_wh` DecisionMix v2 hard curriculum, `data_u`
soft-target uncertainty corpus). Same held-out splits, same 0/231 JevBench leak hits.

## Results

### §3af — Release-1 candidate at 4B (`tm1b`) vs its matched untyped baseline and the 1.7B release

| | `ladder_4b` | **`tm1b`** | `ts1b` (1.7B) |
|---|---|---|---|
| CLINC-150 / TREC-fine / HWU64 / 20NG | .850 / .516 / .787 / .577 | .847 / .414 / .769 / .588 | .804 / .508 / .761 / .540 |
| SNLI / MNLI / BoolQ / ANLI | .909 / .868 / .865 / .536 | .909 / .861 / .843 / .544 | .894 / .859 / .824 / .497 |
| MMLU-Pro among-K / Δ_q_sh / TruthfulQA | .457 / .193 / .386 | .458 / .193 / **.408** | .343 / .127 / .257 |
| held-out noul / score / style / flip (full rows) | .841 / .535 / .923 / .785 | .811 / .528 / .859 / .786 | .715 / .520 / .859 / .581 |
| held-out score NLL / typed-decisions acc, NLL | 2.37 / .465, 1.88 | **1.02** / **.532, 1.18** | 1.01 / .510, 1.25 |
| wh family / grammar / style / **level 7** / flip / shuffled | – | .874 / .893 / .891 / **.544** / .743 / .378 (NLL 3.4) | .836 / .898 / .896 / .495 / .718 / .410 |
| u ChaosNLI / real / synthetic | – | .513 / .658 / .907 | .547 / .649 / .916 |
| external: PagerDuty (floor .792) / jevlogs (.697) / Mind2Web (.427) / tree-choice | .790 / .487 / .559 / .713 | **.838** / .673 / .544 / .690 | .817 / .710 / .357 / .474 |
| JevBench std / easy / hard (Brier std / hard; ECE std) | .833 / 1.00 / .432 (.29 / .83; .12) | .806 / 1.00 / .423 (.30 / .77; **.09**) | .694 / 1.00 / .432 (.40 / .79; .11) |
| single decision ms K = 2 / 32 / 256 (same pod as `ladder_4b`) | 56 / 56 / 118 | 57 / 58 / 96 | 45 / 46 / 106 |

**Read (REPORT.md §3af).** The Release-1 recipe on 4B keeps knowledge (MMLU .458, Δ_q .193,
TruthfulQA +2), keeps NLI, halves the soft-target NLLs again (score 2.37 → 1.02, typed-decisions
1.88 → 1.18 with accuracy +7), is the first model above .50 on level-7 composition (.544), clears
the PagerDuty floor (.838), and is the best-calibrated model on JevBench standard (ECE .086) — at
unchanged latency. It gives back TREC-fine (−10, the one clear regression; K = 50 fine-grained
topics), held-out styles (−6), noul (−3) and ~2 JevBench items on standard and 1 on hard (within
n_eff = 36 noise). **Decision: freeze `tm1b` as `typical-medium`.** The 4B remains the knee of
capability per millisecond: it beats `typical-small` by +4–11 on evidence, +11 on MMLU among-K, +10
on held-out noul, +5 on level 7 and +11 on JevBench standard for 1.25x the per-decision latency. Open
at 4B, same as at 1.7B: the hard tier (.42, long_policy .21, temporal/trade-off ≤ .2) and TREC-fine.

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
SE ≈ .058; n_eff = 36).

## Latency (REPORT.md §3ab, apples-to-apples ladder, one pod, one torch build, L_s = 256)

| 4B | single decision, K = 2 / 32 / 256 (ms) | marginal per query, M = 32, K = 2 / 32 / 256 (ms) | peak memory, K = 2 → 256 (GB) |
|---|---|---|---|
| `tm1b`-shape recipe | 56 / 56 / 118 | 3.3 / 6.1 / 50.6 | 14.6 → 18.6 |

Capability per millisecond (JevBench standard / single-decision ms): 1.7B .694/45, 4B .806/57, 14B
frozen-with-shots .559 hard (not part of the release ladder) — the 4B is the knee, +11 JevBench
standard and +11 MMLU among-K over the 1.7B for 1.25x the latency.

## Known limitations

- Hard tier is .423 with Brier .77 — still within range of the .336 chance baseline; long_policy
  specifically is .21, the weakest single family, and temporal/trade-off composition stays ≤ .2.
- TREC-fine regresses 10 points vs. `ladder_4b` (the untyped, non-DecisionMix-v2 4B baseline on the
  same backbone) — the one clear regression at this size (K = 50 fine-grained topics).
- Held-out rubric styles (−6) and Noul (−3) give back a few points vs. `ladder_4b`, in exchange for
  the calibration and external-floor gains above.
- Very large candidate sets (K in the hundreds to thousands) still need the energy → top-r → native
  path, not this native head directly.

## License

- Backbone (`Qwen/Qwen3-4B-Base`): Apache-2.0.
- Training data: identical sources and licenses to `typical-small` (see `releases/typical-small.md`),
  including the two "license undeclared" flags on `metaeval/ambient` and
  `metaeval/chaos-mnli-ambiguity`.
- JevBench numbers in this card are a **public-subset run** against `fstandhartinger/jevbench` v1.2.1
  (72 standard / 48 easy / 111 hard public ids) — not a submitted or ranked leaderboard entry.

## How to run

**Inference code included under `inference/`; training code release to follow.** The public repo
(`OzLabs/typical-medium`) ships the same minimal, self-contained inference package as
`typical-small`/`typical-small-preview` (own `Typical` class — no dependency on this training repo,
just `torch`, `transformers`, `safetensors`, `huggingface_hub`, `numpy`). Download the `inference/`
folder from that repo, then:

```bash
pip install -r inference/requirements.txt
```

```python
from typical import Typical  # inference/typical/, downloaded alongside your code

m = Typical.from_pretrained("OzLabs/typical-medium", device="auto")

m.choice(state, "What does the customer want?", ["refund", "replacement", "repair"])
m.noul(state, "Is the order still under warranty?")
m.score(state, "How urgent is this ticket?", ["0", "1", "2", "3"])
```

`state` is a string or JSON-serialisable dict. See `inference/README.md` for the full API
(`choice`/`noul`/`score`/`decide`).

Held-out workflow / external eval reproduction (post-hoc, full state length; requires this private
training repo):

```bash
uv run --no-sync python scripts/eval_wf.py --run runs/tm1b --mode native \
    --files data_wf/eval/*.jsonl data_wf_hf/eval/*.jsonl data_wh/eval/*.jsonl data_u/eval/*.jsonl \
    --limit 0 --out runs/tm1b/eval_wf_full.json
```

JevBench: `scripts/jevbench_run.py` against the checkpoint in `mode="native"` (see `pcdm_jev/` for
the harness adapter and REPORT.md §3q for the exact protocol and disclosures reproduced above).

## Files in `guychuk/pcdm-runs/typical-medium/`

- `best.pt` — this checkpoint (`tm1b`).
- `results.json` — training args + full eval suite (train-time pass).
- `eval_wf_full.json` — post-hoc held-out workflow / external eval (full state length, `scripts/eval_wf.py`).
- `probe_results.json` — Δ_q probe results.
- `jevbench_summary.json` — JevBench public-subset run summary.

## Links

- Public release: https://huggingface.co/OzLabs/typical-medium
- Parent release: https://huggingface.co/OzLabs/typical-small-preview
- Sibling release (1.7B): https://huggingface.co/OzLabs/typical-small
