# typical-medium

**Release line: https://huggingface.co/OzLabs/typical-medium** — this card describes checkpoint
`tm2`, which has **not** been uploaded yet; the public repo still carries the previous checkpoint.

The 4B release, superseding `tm1b` (the previous `typical-medium`). Same recipe family as
`typical-small`'s `ts1c` — r1 mix + DecisionMix v2 + ordinal-smoothed Score + per-row Bernoulli Noul
— but retrained on a **different backbone generation**, `Qwen/Qwen3.5-4B-Base`, at `runs/tm2`.
Parent release: `typical-small-preview`. Sibling release at 1.7B: `typical-small` (`ts1c`,
`releases/typical-small.md`).

## What this model is

- **Backbone:** `Qwen/Qwen3.5-4B-Base`, truncated at layer 23 of 32. This is **not** the same
  backbone as `tm1b`'s `Qwen/Qwen3-4B-Base` (which was tapped at 26 of 36): Qwen3.5 is a hybrid
  Gated-DeltaNet stack, a different architecture generation. The depth fractions happen to land close
  (23/32 ≈ 72%, against the 1.7B's 20/28 ≈ 71%), but equal depth fraction across two different
  backbone generations does not imply an equivalent tap — treat this as a new tap that was selected
  for this stack, not as a port of the 1.7B's.
- **Adaptation:** LoRA, rank 16, on the top 8 kept layers of the truncated backbone.
- **Readout:** native contextual decision head (`nc_head=n3`), `letters_nonull` rendering, *factored*
  null — identical shape to `typical-small`.
- **Tap normalisation:** z-scored tap activations (`zscore: true`), new in this checkpoint — `tm1b`
  trained without it.
- **Typed primitives** (same as `typical-small`, PLAN7 Track C):
  - **Choice** — the N3 contextual readout above, unchanged.
  - **Noul** (yes/no) — a Bernoulli head, `P(yes) = σ(w·h_D)`, routed per row: only rows whose
    candidate set is exactly `{yes, no}` use it; every other 2-way label set scores as ordinary
    Choice.
  - **Score** — K-way Choice trained with ordinal-smoothed targets (τ = 0.7, §3ac's adopted recipe).

## Training args

```
backbone: Qwen/Qwen3.5-4B-Base
lora_layers: 8            lora_r: 16            lr: 0.0003            lora_lr: 0.0001
tap_layer: 23             zscore: true
readout: native           nc_head: n3            nc_render: letters_nonull
null: factored            noul_head: bern        score_head: choice     ordinal_smooth: 0.7
steps: 8000               bs: 64                 grad_accum: 16
eval_every: 4000          val_every: 1000        ckpt_every: 2000
data: data_v5             extra_data: data_kb,data_wf,data_wf_hf,data_wf_long,data_wh,data_u
family_weights: E:0.40,K:0.15,W:0.35,U:0.10
bucket_map: data_wf_long=W,data_wh=W,data_u=U
null_aug: W:0.20          max_state: 2048        drop_truncated: true
best_on: data_u_val,data_wh_val
tower_d: 512              tower_layers: 2        tower_heads: 8
seed: 0
```

Effective batch 64 via `--grad_accum 16`; training-time peak memory is
`_not measured for this checkpoint_`. Checkpoint selection is `--best_on data_u_val,data_wh_val`
(uncertainty + hard-curriculum validation), not overall val loss. There is **no matched untyped
control on this backbone**: `ladder_4b`, the control `tm1b` was read against, is a `Qwen3-4B-Base`
run and is not comparable to a Qwen3.5 tap. The comparison column below is the 1.7B sibling `ts1c`
instead.

Final val NLL 0.429, temperature T = 1.040, null_offset 0.0 (fit post-hoc on the `val` split, applied
at eval).

## Data

Identical mixture and sources to `typical-small` — see `releases/typical-small.md`'s Data section
for the full per-source breakdown (`data_v5`/`data_kb` evidence+knowledge, `data_wf`/`data_wf_hf`/
`data_wf_long` rubric-conditioned workflow, `data_wh` DecisionMix v2 hard curriculum, `data_u`
soft-target uncertainty corpus). Same held-out splits, same 0/231 JevBench leak hits, same
`--max_state 2048`.

## Results

### `tm2` at 4B, against the constant-prediction floor and the 1.7B sibling

Evidence/knowledge rows are the train-time eval pass (`runs/<ck>/results.json`). Held-out workflow,
uncertainty and external rows are the **full-file** post-hoc pass (`runs/<ck>/eval_wf.json`, `mode
native`, `max_state 4096`, `limit 0`), and "floor" is that file's `baselines.majority_acc` for the
same set. JevBench rows are `runs/jev_native_<ck>/summary.json`.

| | floor | **`tm2`** | `ts1c` (1.7B sibling) |
|---|---|---|---|
| CLINC-150 / TREC-fine / HWU64 / 20NG | – | .795 / .486 / .720 / .580 | .797 / .500 / .749 / .553 |
| SNLI / MNLI / BoolQ / ANLI | – | .900 / .855 / .840 / .554 | .889 / .853 / .776 / .492 |
| MMLU-Pro among-K / shuffled-q among-K (Δ_q_sh) / TruthfulQA-MC1 | – | .429 / .235 (Δ .194) / .414 | .338 / .212 (Δ .127) / .267 |
| held-out noul / score / style / flip both-correct (full rows) | .583 / .355 / .364 / – | .858 / .585 / .841 / .718 | .643 / .488 / .830 / .543 |
| held-out score NLL / typed-decisions acc, NLL (full rows) | – | 0.97 / .517, 1.13 | 1.09 / .464, 1.50 |
| wh family / grammar / style / **level 7** / flip / shuffled | .384 / .385 / .427 / .460 / .400 / .410 | .889 / .887 / .896 / **.539** / .740 / .392 (NLL 3.18) | .853 / .901 / .883 / .524 / .730 / .394 (NLL 3.06) |
| u ChaosNLI / real / synthetic | .464 / .507 / .575 | .525 / .694 / .909 | .514 / .612 / .924 |
| external: PagerDuty / jevlogs / Mind2Web / tree-choice / typed-decisions acc | .792 / .697 / .427 / .114 / .291 | .834 / .612 / .496 / .588 / .517 | .857 / .766 / .364 / .471 / .464 |
| JevBench std / easy / hard (Brier std / hard; ECE std) | – | **.861** / 1.00 / .495 (.22 / .72; **.07**) | .708 / 1.00 / .432 (.42 / .72; .15) |
| Noul reversed-label \|ΔP(yes)\| max (mean) | – | .002 (.00008) | .007 (.00005) |
| single decision ms, K = 2 / 32 / 256 | – | `_not measured for this checkpoint_` | 61 / 58 / 118 |

The comparison columns the previous card carried (`ladder_4b`, `tm1b`) are dropped: neither
checkpoint has an eval artefact in this release, so their numbers cannot be re-read and are not
reproduced here. Note in particular that `tm1b`'s rubric-flip figure (.7430) is **not** `tm2`'s —
`tm2`'s `wh_rubric_flip` is .7403 on the full-file pass.

**Read.** `tm2` is the stronger of the two release checkpoints nearly everywhere it can be compared:
+9 on MMLU-Pro among-K with a wider shuffled-question gap (Δ .194 vs .127), +15 TruthfulQA, +6 BoolQ,
+6 ANLI, +21 on held-out yes/no, +10 on held-out score, +18 on `wf` flip both-correct, and +15 on
JevBench standard (.861, 62/72, extraction 12 of 12) at roughly half the standard-tier ECE (.071 vs
.152) and half the Brier (.224 vs .423). It is the first checkpoint on this line above .50 on level-7
composition against a .460 floor (.539), and it clears the PagerDuty floor (.834 vs .792).

The regressions are real and not confined to noise-sized sets. **jevlogs falls to .612 against a
.697 constant-prediction floor** — below the floor, and 15 points below `ts1c`'s .766; the 1.7B
sibling clears that floor and this one does not. TREC-fine is .486 (vs `ts1c`'s .500) and HWU64 .720
(vs .749), so the fine-grained intent sets do not benefit from the extra size. The shuffled-rubric
control sits at .392 against a .410 floor, as intended. JevBench hard rises to .495 but its Brier
(.721) and ECE (.252) are no better than the 1.7B's, and its ordinal MAE is worse (.957 vs .817) —
the extra hard-tier accuracy does not come with better probabilities.

### JevBench disclosures (apply to the numbers above)

This is a **public-subset run, not a ranked entry** (JevBench's own leaderboard requires ≥95%
coverage including the non-public judge items; we ran the public ids only). Harness version
`jevbench-v1`; probability source `native` for all three tiers, with 0 renormalised items and schema
validity 1.00. Reported probabilities are the head's softmax **conditioned on non-∅** (P(∅) is
dropped and the rest renormalized); the mean p_null for this run is
`_not measured for this checkpoint_` — the summary does not record it. The checkpoint was trained at
up to 2,048-token states (`args.max_state`) and the held-out pass runs at up to 4,096
(`eval_wf.json` `max_state`) — out-of-training-length but not truncated. Cost is null
(`ledger_charged_usd: null`, `cost_basis: local_gpu_no_provider_tariff` — this is not a hosted-API
run). Option order is the harness's own label order; a reversed-order control is
`_not measured for this checkpoint_`, so no fine ranking among our own checkpoints is supported by
these numbers alone. Split sizes are 72 standard / 48 easy / 111 hard scorable ids; the standard tier
has 36 paraphrase pairs (`paraphrase_consistency.pairs`), so n_eff = 36 and SE ≈ .06 at p = .5. The
majority/chance baselines for the three JevBench tiers are `_not measured for this checkpoint_` —
the summary carries no baseline field, and the previous card's values are not re-derivable from
these artefacts.

## Latency

**No latency or memory benchmark was run for `tm2`.** There is no `runs/bench_tm2/`, so every figure
in this section is `_not measured for this checkpoint_`. The numbers the previous card carried belong
to `tm1b` on a different backbone and are not reused.

| 4B | single decision, K = 2 / 32 / 256 (ms) | marginal per query, M = 32, K = 2 / 32 / 256 (ms) | peak memory, K = 2 → 256 (GB) |
|---|---|---|---|
| `tm2` | `_not measured for this checkpoint_` | `_not measured for this checkpoint_` | `_not measured for this checkpoint_` |

Capability per millisecond is therefore also `_not measured for this checkpoint_`: with no per-decision
timing for `tm2`, this checkpoint cannot be placed on the same latency axis as `typical-small`
(`ts1c`: JevBench standard .708 at 61 ms per single decision, K = 2, from `runs/bench_ts1c/bench.json`),
and no "knee of capability per millisecond" claim is supported here.

## Known limitations

- **jevlogs is below its constant-prediction floor: .612 vs floor .697**, and 15 points below the
  1.7B sibling's .766. This is the clearest regression at this size, and the 4B is the only one of
  the two release checkpoints that fails this external floor.
- JevBench hard is .495 with Brier .721, ECE .252 and ordinal MAE .957 (worse than the 1.7B's .817).
  Within that tier `temporal_numeric` is the weakest family at .267 (n = 15), then `probability` .300
  (n = 10) and `long_policy` .316 (n = 19).
- TREC-fine (.486) and HWU64 (.720) are both below the 1.7B sibling's (.500 / .749) — the
  fine-grained, large-K intent sets get nothing from the extra size.
- The shuffled-rubric control sits at .392 against a .410 floor — with the rubric scrambled the model
  is no better than guessing, which is the intended behaviour but leaves no margin.
- No latency, throughput or memory characterisation exists for this checkpoint (see above).
- Very large candidate sets (K in the hundreds to thousands) still need the energy → top-r → native
  path, not this native head directly.

## License

- Backbone (`Qwen/Qwen3.5-4B-Base`): **licence not verified for this release.** The previous card's
  Apache-2.0 claim applied to `Qwen/Qwen3-4B-Base`, a different model; check the `Qwen3.5-4B-Base`
  model card before redistributing weights or derivatives.
- Training data: identical sources and licenses to `typical-small` (see `releases/typical-small.md`),
  including the two "license undeclared" flags on `metaeval/ambient` and
  `metaeval/chaos-mnli-ambiguity`.
- JevBench numbers in this card are a **public-subset run** against `fstandhartinger/jevbench`
  (harness version `jevbench-v1`; 72 standard / 48 easy / 111 hard public ids) — not a submitted or
  ranked leaderboard entry.

## How to run

**Inference code included under `inference/`; training code release to follow.** The public repo
(`OzLabs/typical-medium`) ships the same minimal, self-contained inference package as
`typical-small`/`typical-small-preview` (own `Typical` class — no dependency on this training repo,
just `torch`, `transformers`, `safetensors`, `huggingface_hub`, `numpy`). Note that the repo does not
yet carry `tm2`; the API below is the interface, and the weights it downloads are still the previous
checkpoint until this one is published.

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
(`choice`/`noul`/`score`/`decide`). Parity between the packaged `Typical` class and the internal
`PCDMDecider(mode="native")` is `_not measured for this checkpoint_` — that check has not been re-run
against `tm2`.

Held-out workflow / external eval reproduction (post-hoc, full state length; requires this private
training repo):

```bash
uv run --no-sync python scripts/eval_wf.py --run runs/tm2 --mode native \
    --files data_wf/eval/*.jsonl data_wf_hf/eval/*.jsonl data_wh/eval/*.jsonl data_u/eval/*.jsonl \
    --limit 0 --out runs/tm2/eval_wf.json
```

JevBench: `scripts/jevbench_run.py` against the checkpoint in `mode="native"` (see `pcdm_jev/` for
the harness adapter and the protocol disclosures reproduced above).

## Artefacts backing this card

All local to this training repo; nothing here has been uploaded or hashed for publication yet.

- `runs/tm2/results.json` — training args + train-time eval suite.
- `runs/tm2/eval_wf.json` — post-hoc held-out workflow / external eval (full file, `max_state 4096`).
- `runs/tm2/eval_wf_long_policy.json` — long-policy held-out pass (`wf_long_policy`, acc .950 vs
  floor .413).
- `runs/probe_tm2/results.json` — Δ_q probe results.
- `runs/jev_native_tm2/summary.json` — JevBench public-subset run summary.
- No latency benchmark (`runs/bench_tm2/` does not exist).

## Links

- Release line: https://huggingface.co/OzLabs/typical-medium (does not yet carry `tm2`)
- Parent release: https://huggingface.co/OzLabs/typical-small-preview
- Sibling release (1.7B): https://huggingface.co/OzLabs/typical-small
