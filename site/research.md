# Research

Typical reads the decision out of the model before it emits a letter, instead of generating an
answer and parsing it back. The negative result matters just as much: a candidate-blind state
carries priors and calibration, not question-conditioned knowledge (§3j, §3s). Below is the full
architecture, data, pipeline, and result set, including the numbers that didn't move.

Section references (§…) point to the project's internal `REPORT.md`, which isn't public yet. The
Hugging Face model cards reproduce the tables cited here.

## Architecture

A frozen Qwen3 trunk, cut at roughly 71% depth, feeds a small contextual readout with LoRA on the
top 8 kept layers and a factored ∅ head. The state is read once and cached. Each query appends a
short rendered suffix to that cache. Nothing is generated: the label, the yes/no, and the level all
come out of the same terminal decision state.

```text
 state tokens (≤1,024 tok)
                               │
                               ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ frozen Qwen3 trunk, truncated at the tap layer               │
 │   typical-small (1.7B): layer 20 / 28                        │
 │   typical-medium (4B):  layer 26 / 36           (≈71% depth) │
 │ LoRA r16 on the top 8 kept layers, rest frozen               │
 └──────────────────────────────────────────────│───────────────────────────────────┘
                               │
                               │   KV cache — the state is computed once, reused per query
                               │
                               │   per query: a short rendered suffix is appended to the cache
                               │   "<question>  <letters_nonull candidates>"
                               ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │ contextual readout (nc_head = n3)                             │
 │ terminal decision state h_D + per-candidate contextual states │
 └───────────────────────────────────────────────────────────────────┘
                               │
         ┌────────────────────│─────────────────────────────────┐
         ▼                     ▼                         ▼
  ┌───────────┐      ┌────────────────┐      ┌─────────────────┐
  │ Choice     │      │ Noul            │      │ Score            │
  │ softmax    │      │ Bernoulli       │      │ K-way softmax,   │
  │ over K     │      │ P(yes)=σ(w·h_D) │      │ ordinal-smoothed │
  │ candidates │      │ query-only      │      │ targets (τ=0.7), │
  │ + factored │      │ suffix, no      │      │ reports E[index] │
  │ ∅ (a head  │      │ rendered        │      │ + factored ∅     │
  │ decision,  │      │ candidates —    │      │                  │
  │ not a      │      │ exactly order-  │      │                  │
  │ rendered   │      │ invariant by    │      │                  │
  │ line)      │      │ construction    │      │                  │
  └───────────┘      └────────────────┘      └─────────────────┘
```

*Fig. 1. Typical forward pass, state to probabilities.*

### a) Why mid-depth, not the last layer

The project's first version (a candidate-blind decision vector, frozen 0.6B backbone) read the
trunk's *last* layer and built a hypothesis-only model: SNLI accuracy landed at 58.6%, exactly what
a model gets from the hypothesis alone, ignoring the premise (REPORT §2.4). Moving the tap to layer
20 of 28 and adding LoRA fixed it: 65.3% frozen, 68.2% with LoRA. The premise finally mattered.

The same question came back for the native, candidate-aware readout: is full depth (28 of 28)
better once the candidates are inside the computation too? It isn't. Matched runs at layer 28 vs.
layer 20 (same N3 readout, same data, same steps) put tap 20 ahead on every evidence task: SNLI .906
vs. .863, MNLI .865 vs. .752, ANLI .519 vs. .431, BoolQ .834 vs. .737 (REPORT §3r), while keeping the
question-dependent signal (Δ_q .118 vs. .117 at 28 layers). Depth isn't free capacity here: for a
candidate-aware readout, the last third of the trunk loses evidence reasoning instead of
accumulating it.

### b) Why there's no rendered "none of the above" line

An earlier version rendered ∅ as a literal candidate in the suffix, the way a multiple-choice prompt
adds "E. none of the above." That line was the failure mode: null probability collapsed to
always-abstain at large label sets (K=150 CLINC, K=77 Banking77) and never-abstain at small ones
(K=5), with MMLU-Pro false-abstain at .692.

Making ∅ a separate head decision instead (`letters_nonull` plus a factored null) fixed the K-sweep:
smooth and monotone at every K. MMLU-Pro false-abstain dropped to .003, Banking77-77 accuracy went
from .063 (unusable) to .579, Δ_q held at .122, and NLI/BoolQ cost nothing (REPORT §3t). The
remaining cost: topic/intent label spaces lose 2–7 points (HWU64 .801 → .757, 20NG .589 → .515), and
choice-set null probes lose some AUROC. A real trade, not a free win.

### c) Why the candidates have to be inside the computation

A separate line of experiments tried to compile a listwise teacher's knowledge into a
*candidate-blind* decision state Z(x, q), computed before the candidate set is known, so one state
could in principle serve any question. A choices-only control (state = ".", labels only) showed why
that fails: every student architecture tested (a single vector, 8 probes, 8 probes with
set-conditioning, under both single-set and same-Z multi-set distillation with an explicit
Δ-log-odds objective) scored the same or better with the real question replaced by a shuffled one.
None of the above-chance accuracy was question-dependent. All of it was candidate-set priors and
calibration, learned uniformly across every head shape (REPORT §3j).

Re-running the same student at full depth (28 layers, not just tap 20) closed the question: Δ_q
stayed at −0.008, within noise of zero, while evidence tasks regressed exactly as the depth result
above predicts (REPORT §3s). A candidate-blind state compiles priors and calibration.
Question-conditioned knowledge doesn't, at any tap. That's why Typical renders its candidates
(Choice's label set, Score's levels) into the suffix and reads them in the same forward pass as the
question, instead of looking them up afterward against a fixed state.

## Data

E .40 / K .15 / W .35 / U .10. The workflow corpora are ours and programmatic. Held-out families,
grammars, styles, and an untrained level-7 tier measure transfer, not memorisation.

```chart mix-donut
```

```chart mix-bars
```

Family weights and corpus row counts (releases/typical-small.md, Training args + Data section). kb
147,446 rows (MMLU-aux/AQuA/MedMCQA/LogiQA-style distillation) · wf 72,000 rows / 12 families, our
rubric-conditioned workflow corpus · wf_hf 86,600 rows from three HF-sourced workflow sets · wh
60,605 rows, DecisionMix v2 · u 31,029 rows, soft-target uncertainty corpus.

### DecisionMix v2 (bucket W, `data_wh`)

A programmatic rule engine (`scripts/decisionmix_v2.py`): 12 domains × 6 boolean conditions,
rendered at levels 1–7, with 100% of training rows drawn in counterfactual rubric groups (same state
and candidates, at least two distinct golds across the group: same facts, different rule, different
answer). Held out entirely from training: one domain/level pair (`wh_heldout_grammar`), three whole
domains (`wh_heldout_family`), two rubric styles (`wh_heldout_style`), and level 7 (temporal,
numeric, probability, and trade-off composition), which is eval-only and never trained on. A leak
audit against JevBench's 231 public items found 0/231 hits.

### The uncertainty corpus (bucket U, `data_u`)

31,029 soft-target training rows, same generator family. Real sources: UNLI's *validation* split
(train/test are consumed elsewhere in the mix) and the ambiguous rows of `metaeval/ambient` (uniform
target over the listed labels). `metaeval/chaos-mnli-ambiguity` is eval-only, never trained on. The
remaining ~92% comes from four synthetic generators with exact closed-form targets: partial evidence,
a noisy-sensor Bayes posterior, ordinal confusion, and conflicting-sources log-odds, each with a
held-out parameter range. Also 0/231 JevBench leak hits.

### Null augmentation

`--null_aug W:0.20`: 15–20% of workflow rows have their gold candidate removed at train time, so the
target becomes ∅. This teaches abstention on rows that would otherwise always have an answer.
Combined with the factored-∅ head design (see Architecture above), it keeps the null calibrated
across K instead of collapsing at the extremes.

## Pipeline

```text
backbone: Qwen/Qwen3-1.7B-Base
lora_layers: 8            lora_r: 16            lora_lr: 0.0001
tap_layer: 20             zscore: false
readout: native           nc_head: n3            nc_render: letters_nonull
null: factored            noul_head: bern        score_head: choice     ordinal_smooth: 0.0
                          # ^ eval-only rerun flag; training used 0.7 (see above)
steps: 12000              bs: 64                 grad_accum: 1
eval_every: 4000          val_every: 1000        ckpt_every: 2000
data: data_v5             extra_data: data_kb,data_wf,data_wf_hf,data_wf_long,data_wh,data_u
family_weights: E:0.40,K:0.15,W:0.35,U:0.10
bucket_map: data_wf_long=W,data_wh=W,data_u=U
null_aug: W:0.20          max_state: 1024
tower_d: 512              tower_layers: 2        tower_heads: 8
seed: 0
```

*Specimen: typical-small training args.*

88 training, baseline, and ablation run directories sit under `runs/` on the source branch, across
six days (2026-09-16 → 09-21), most of them well under $20 of H100 time. The line moves because of
bugs found and controls run, not because of scale. See the decision log below.

typical-medium (4B) uses the same recipe on `Qwen3-4B-Base`, tap 26/36, effective batch 64 via
`--grad_accum 4` (releases/typical-medium.md). Final val NLL 0.376, temperature T = 1.124,
null_offset 0.0 (typical-small; fit post-hoc on the val split, applied at eval).

### Cost, where REPORT states it

typical-small (`ts1b`): ~$12 of H100 time, plus ~$10 lost to a routing bug caught and fixed mid-run
([REPORT §3ae](https://huggingface.co/OzLabs/typical-small)). typical-medium (`tm1b`): ~$15, plus ~$8
lost to the same bug class ([REPORT §3af](https://huggingface.co/OzLabs/typical-medium)). The
DecisionMix v2 dry run that preceded them (`r1_dmv2`) was ~$12 (REPORT §3ad). Wall-clock and per-run
cost for the 4B/8B/14B scaling ladder aren't itemized separately in REPORT §3ab, so they're left out
here rather than estimated.

### Eval suite

Evidence: SNLI, MNLI, ANLI, BoolQ. Topic/intent: CLINC-150, HWU64, 20 Newsgroups, TREC-fine.
Knowledge: MMLU-Pro among-K, Δ_q (shuffled-question control), TruthfulQA-MC1. Typed/workflow,
held-out: noul, score, style, and DecisionMix v2's family/grammar/style/level-7/rubric-flip.
Uncertainty: ChaosNLI, real and synthetic held-out sets from the U corpus. External, never trained
on: PagerDuty, jevlogs, Mind2Web, tree-choice (K=320), typed-decisions. Public benchmark: JevBench
(231-item public subset; see the D1 disclosure in Findings below).

## Timeline

The chart plots JevBench-standard accuracy for the lineage that became typical-small and
typical-medium, from the first energy checkpoint to the frozen releases, annotated with the bugs
found along the way and the verdict each phase closed on. Hover any marker for its exact number and
REPORT section. Six days, <span id="timeline-run-count">88</span> runs.

D1 — JevBench: public subset (231 ids), unranked, n = 72 standard (SE ≈ .058); probabilities
conditioned on non-∅; hard tier at chance for both models.

```chart timeline
```

The accuracy line isn't monotone: .750 → .694 at the typical-small step is a real trade
([§3ae](https://huggingface.co/OzLabs/typical-small)). DecisionMix v2 and the typed heads raise
held-out noul/score/flip and clear two external floors (PagerDuty, jevlogs), at the cost of about 1
SE of JevBench standard. The run count (88) is the source branch's `runs/` directories, minus
probe/jev/bench/zero-shot/smoke/dump/leak/kb-audit subdirectories. It includes baseline and ablation
runs, not just the release lineage above.

## Findings

Eight established results. Each has a matched control and, where it mattered, a seed pair or a
full-row re-evaluation (REPORT §6).

D1 — JevBench: public subset (231 ids), unranked, n = 72 standard (SE ≈ .058); probabilities
conditioned on non-∅; hard tier at chance for both models.

1. **Candidate-blind states carry priors and calibration, not question-dependent knowledge.** Holds
   at tap 20 and at full depth. A candidate-aware suffix computation keeps the knowledge. A direct
   contextual readout keeps it too, with a quarter of the letter interface's order fragility.
   (REPORT §3j, §3l, §3s, §3v)

2. **The evidence-vs-knowledge trade-off was two artifacts: last-layer tap and a rendered "none of
   the above" line.** Removing both gives one model (`nc_v3_tap20`, research checkpoint, not
   released): evidence at energy level, Δ_q kept, null monotone in K, JevBench standard .694 with no
   workflow training. (REPORT §3r, §3t)

3. **Rubric-conditioned workflow data teaches the decision shapes it contains, not general external
   abstraction.** Held-out rubric styles +34, held-out families +9–14, rubric-flip both-correct .43
   → .57. jevlogs, PagerDuty, and Mind2Web stayed at their constant-prediction floor for K-way
   Choice until the typed heads and DecisionMix v2 arrived (findings 6–7 below). (REPORT §3w, §3x)

4. **Mixing is a first-class variable, not a detail.** E .45–.50 restores evidence. ∅-augmented W
   rows fix abstention and MMLU, where a single eval-time null threshold can't serve both E and W at
   once. The exact W fraction in [.1, .2] doesn't matter once null augmentation is on.
   (REPORT §3y, §3z, §3aa)

5. **Small batch = under-fitting = better hard-tier and soft-target numbers, at the cost of
   everything in-distribution.** At 1.7B, the JevBench hard tier is a probability-quality problem,
   not a data-volume one. (REPORT §3aa)

6. **Typed primitives: ordinal-smoothed Score and a Bernoulli Noul head both win on probability
   quality, at zero or negative decision-accuracy cost.** Ordinal smoothing (τ=0.7) changes no
   JevBench decision while held-out score NLL goes 2.07 → 1.23 (intermediate checkpoints, §3ac;
   released models' held-out score NLL: 1.01 / 1.02). The Bernoulli Noul head is exactly
   order-invariant by construction and is the first thing to clear an untouched external floor by a
   margin: PagerDuty .602 → .886 (intermediate checkpoint, §3ac; released models .817 / .838 vs floor
   .792). (REPORT §3ac)

7. **DecisionMix v2's hard curriculum transfers within its own rule grammar, not beyond it.**
   Held-out families/grammars +34–37, rubric-flip .48 → .71, JevBench adversarial .33 → .83
   (intermediate checkpoints r1_cand → r1_dmv2, §3ad, not released; the released models' equivalent
   wh_heldout numbers are in the held-out chart below: typical-small .836/.898/.896, typical-medium
   .874/.893/.891), while level-7 composition (temporal, unit, expected-value, trade-off) sits at
   ~.50, chance, for every checkpoint on the ladder. The U corpus buys likelihood, not top-1
   accuracy. (REPORT §3ad)

8. **Same recipe at scale: 1.7B → 4B → 14B lifts standard-tier accuracy, knowledge, and
   in-distribution workflow decisions monotonically, at roughly constant latency.** JevBench standard
   .750 → .833 → .875, MMLU-Pro among-K .330 → .457 → .514, held-out noul .70 → .84 → .89, at 45 → 56
   → 60 ms per decision (scaling-ladder checkpoints, §3ab, same pre-DecisionMix-v2 recipe at every
   size, none released; typical-small/typical-medium use a later recipe and score .694 / .806 on
   JevBench standard). 8B is a checkpoint outlier in both frozen and trained form, not part of the
   trend. Long-state families and soft-target calibration don't scale: the frozen 14B with three
   shots beats the trained 14B on the hard tier (.559 vs. .468), so the hard tier is a data/objective
   problem at every size, not a capacity one. (REPORT §3ab)

### Held-out curriculum vs. external, never-trained sets

The sets below split into two kinds. One is inside DecisionMix v2's own grammar (held out by family,
style, or grammar, plus level 7, kept in on purpose as an untrained-composition control; it sits at
chance for every checkpoint). The other is never trained on at all: real logging/ticketing/web-agent
corpora, scored against a constant-prediction floor.

```chart heldout
```

```chart external
```

### JevBench, per family (typical-small)

```chart jevfamily
```

> **D1 — JevBench disclosure ([REPORT §3q](https://huggingface.co/OzLabs/typical-small)).**
> JevBench numbers are a public-subset run against `fstandhartinger/jevbench` v1.2.1 (72 standard /
> 48 easy / 111 hard public ids), not a submitted or ranked leaderboard entry (the leaderboard
> requires ≥95% coverage including 146 non-public judge items). Probabilities are the head's softmax
> conditioned on non-∅ (mean p_null ≈ .03). Latency is in-process on one H100, one decision at a
> time, model load excluded. Option order is the harness's label order; a reversed-order control on
> a related checkpoint moved standard accuracy ±4 points, larger than the differences between our
> own checkpoints. Chance/majority baselines are .311 standard / .284 easy / .336 hard (n = 72
> standard, SE ≈ .058) — the hard tier of both released models is within 1 SE of chance.

### Knowledge retained from the backbone

MMLU-Pro among-K, Δ_q (shuffled-question control), and TruthfulQA-MC1 measure how much of the frozen
backbone's knowledge survives the fine-tune. They aren't a selling point. typical-small: among-K
.343, Δ_q .127, TruthfulQA .257. typical-medium: among-K .458, Δ_q .193, TruthfulQA .408 ([REPORT
§3af table](https://huggingface.co/OzLabs/typical-medium)). Compare to level-7 composition,
discussed above, which the training data never touches and which stays at chance regardless of size.

## The long-state bug and what comes next

The hard-tier collapse at scale had a specific cause, not a capacity limit, and the fix is training
now (REPORT §3ag, 2026-09-21/22; runs in flight).

### a) The data bug

`data_wf_long` (the 23k long-policy rows added in §3z) renders `Case: <facts> <request>` with the
facts last. Measured state lengths are p10/p50/p90 = 1,190 / 1,845 / 2,490 tokens, and training
right-truncates at `--max_state`. At 1,024 tokens (Release 1) that cut the facts off 98.8% of those
rows; at 256 tokens (the scaling ladder) it cut all of them off. Every model since §3z was trained to
answer long policies confidently from states that no longer contained the facts, which is exactly
the long_policy score: .05 for the 14B (§3ab), .21 for typical-medium.

Fixes (commit `2fad326`): facts-first regeneration (`wf/train_long_v2.jsonl`, case position under 8%
of the text), `--drop_truncated` (rows longer than the window are dropped, never cut), `--grad_ckpt`,
`--best_on` (checkpoint selection on the uncertainty and curriculum val NLL), `--brier_lambda`, and
frozen-backbone teacher labels via `scripts/teacher_label.py --zero_shot --shots 3`.

### b) The mask bug

A second bug (commit `6d0a7e3`): the SDPA padding mask was built as `long`, which forces PyTorch's
O(L²) math kernel. That caused the 14B OOMs at 3,072-token states, and the scaling ladder's memory
pain generally. Building it as `bool` fixed it.

### c) Frozen controls

Frozen backbones, scored directly (letter logits over the rendered options, 3 exemplars, the same
public 231 JevBench ids):

| frozen backbone | standard | easy | hard | Brier hard |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B-Base | .528 | 1.00 | .369 | .71 |
| Qwen3-4B-Base | .778 | 1.00 | .441 | .67 |
| Qwen3-4B (instruct) | .778 | 1.00 | .423 | .95 |
| Qwen3-8B-Base | .556 | .958 | .369 | .74 |
| Qwen3-14B-Base | .819 | 1.00 | .559 | .60 |
| **Qwen3.5-4B-Base** | .764 | 1.00 | **.495** | **.60** |

Instruct-tuning of Qwen3 changes nothing on this task. The Qwen3.5 generation buys +5 hard and gives
a 4B model hard-tier calibration matching the frozen 14B's, but not standard accuracy. The
leaderboard's .83–.99 standard on the same checkpoint (SemIf, open-alternative-jev) comes from their
rendering and method, which we're now replicating as a probe.

### d) The Qwen3.5 port

Qwen3.5 port (commit `9b1ffac`). Base checkpoints exist at 0.8B / 2B / 4B / 9B / 27B / 35B-A3B, a
complete replacement ladder. Differences handled: the VL wrapper config (`text_config`,
`.language_model` unwrapped), the hybrid stack (3× Gated-DeltaNet + 1× full attention, tap at 71%
depth keeps 4 of 6 full-attention layers at 0.8B), LoRA targets (`in_proj_qkv/z/b/a` and `out_proj`
on DeltaNet layers, q/k/v/o on attention layers, MLP everywhere), and a transformers-5.17 cache gap
(`LinearAttentionLayer` lacks `batch_repeat_interleave`) patched in `native_kv_decide`. Cached-vs-full
parity is 1e-7, Qwen3 behavior is unchanged (153 tests), and the public `inference/` package mirrors
the port.

### e) In flight

`tl1b` (Qwen3-14B, facts-first long rows, 3,072-token states, frozen-14B knowledge distillation, 8k
steps, best-on calibration val) and its matched control `tl1b_nokd`; `tm2` (the Release-1 recipe on
Qwen3.5-4B-Base, 2,048-token states); and the SemIf rendering probe. Pass rule for typical-large:
hard ≥ .559 or hard Brier ≤ .65, long_policy ≥ .35, and CLINC-150 / MMLU among-K within 2 points of
`ladder_14b`.

## Limitations

Re-derived for typical-small and typical-medium specifically, not inherited from any prior write-up.

- We do not claim to be faster, better, or cheaper than Jev or any hosted system. There's no
  apples-to-apples run, and JevBench is unranked for us.
- We do not do vision, audio, real-time control, game-playing skill, or planning/lookahead.
  Text-only, one forward pass, no search.
- We are not a knowledge model and do not reason: MMLU-Pro sits at .34 / .46 among-K, and level-7
  composition (temporal, units, expected value, trade-off) is at chance (.495 / .544, floor .46). We
  do not do arithmetic, timezones, or trade-offs.
- We do not handle long documents. Trained at ≤1,024-token states, JevBench long_policy is .26 /
  .21, and the preview checkpoint's hard tier goes from .64 at ≤256 tokens to .23 above 1,024. The
  cause is known: `data_wf_long` rendered facts last, and training right-truncated them out at
  `--max_state`, so the model learned to answer long policies from states that had already lost the
  facts (REPORT §3ag). A fix (facts-first regeneration, `--drop_truncated`) is training now, not yet
  in a release.
- We do not claim "any label set, any domain." CLINC is −4 versus the untrained base; TREC-fine is
  .508 / .414 (typical-medium is −10 versus its own untyped control); held-out rubric styles regress
  −4 / −6.
- We do not claim exact order-invariance for Choice, or exact independence of irrelevant
  alternatives. Choice is order-robust (mean-max Δp .01–.08), not order-invariant; only Noul is
  exact. Adding an irrelevant candidate moves Δlog-odds .09–.22.
- We do not claim out-of-distribution calibration or hard-tier competence. Both released models are
  at chance on the JevBench hard tier, with Brier .79 / .77.
- We do not ship 3 open sizes. There are two backbone sizes (1.7B, 4B) and three checkpoints
  (typical-small-preview is also 1.7B). There is no typical-large; the 14B point is a scaling-ladder
  measurement, not a release.
- Training code is not released. Only `inference/` is public today; training code release to
  follow.
- We do not claim JevBench standard improved from typical-small to the preview. It went .750 →
  .694 (~1 SE), traded for calibration and the typed heads.
- We do not natively serve very large candidate sets (hundreds to thousands) at this head; that
  needs an energy → top-r → native path, not shipped yet. The native suffix budget is 1,024 tokens by
  default.
- We do not claim to beat supervised classifiers on CLINC, Banking77, or TREC.
- jevlogs numbers are not for commercial use (research-licensed, block-level labels, held-out sanity
  eval only). The licenses of `metaeval/ambient` and `metaeval/chaos-mnli-ambiguity` are undeclared
  on their HF cards; flagged here, not asserted.

### License

The model card declares Apache-2.0 for the released artifact. See the [card's license
section](https://huggingface.co/OzLabs/typical-small#license) for the authoritative statement, which
this page doesn't restate beyond that link. The training mix includes sources with non-commercial or
undeclared licenses: `facebook/anli` (CC BY-NC 4.0, loaded for train_r1–r3) is in the mix and was
flagged as a non-commercial blocker in an earlier launch memo. `metaeval/ambient` and
`metaeval/chaos-mnli-ambiguity` don't declare a license on their HF cards. Only `inference/` is
public today; training code release to follow.

## Reproduce

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
```

Inference code is public under `inference/`; training code release to follow. The commands below are
verbatim from the release cards.

### Held-out workflow / external eval reproduction

Post-hoc, full state length; requires the private training repo.

```bash
uv run --no-sync python scripts/eval_wf.py --run runs/ts1b --mode native \
    --files data_wf/eval/*.jsonl data_wf_hf/eval/*.jsonl data_wh/eval/*.jsonl data_u/eval/*.jsonl \
    --limit 0 --out runs/ts1b/eval_wf_full.json
```

JevBench: `scripts/jevbench_run.py` against the checkpoint in `mode="native"`. See `pcdm_jev/` for
the harness adapter, and REPORT.md §3q for the exact protocol and disclosures, reproduced above in
Findings.

### Checkpoints

- [OzLabs/typical-small](https://huggingface.co/OzLabs/typical-small) — 1.7B, released.
- [OzLabs/typical-medium](https://huggingface.co/OzLabs/typical-medium) — 4B, released.
- [OzLabs/typical-small-preview](https://huggingface.co/OzLabs/typical-small-preview) — 1.7B, the
  frozen parent of typical-small.

### JevBench context ([REPORT §3q](https://huggingface.co/OzLabs/typical-small), same 231 public ids)

Shown as neutral context only: a public-subset run against an unranked leaderboard, not a comparison
we're making a claim from (see D1 above).

D1 — JevBench: public subset (231 ids), unranked, n = 72 standard (SE ≈ .058); probabilities
conditioned on non-∅; hard tier at chance for both models.

| entry | standard | easy | hard |
| --- | --- | --- | --- |
| open-jev-deberta-v3-large (classifier, local CPU) | .431 | 1.00 | .378 |
| GLiNER2 / jeff (GLiFormer 400M) / Laya (ModernBERT) | .639 / .750 / .694 | 1.00 | .369 / .387 / .351 |
| **typical-small** (`ts1b`) | .694 | 1.00 | .432 |
| **typical-medium** (`tm1b`) | .806 | 1.00 | .423 |
| 14B ladder point (not released) | .875 | 1.00 | .468 |
| open-alternative-jev (Qwen3.5-4B) | .833 | 1.00 | .568 |
| system-one-open (Gemma E2B LoRA) / system-one (Qwen3-8B) | .931 / – | 1.00 | .486 / .486 |
| SemIf (Qwen3.5-4B) / OpenJev (26B-A4B) / djev | .986 / .972 / .986 | 1.00 | .613 / .640 / .676 |
| Jev 1.13.0 (closed) | .986 | 1.00 | .730 |

typical-small/-medium rows use the released checkpoints' own JevBench runs ([REPORT
§3ae](https://huggingface.co/OzLabs/typical-small)/[§3af](https://huggingface.co/OzLabs/typical-medium));
the rest of the table is [REPORT §3q](https://huggingface.co/OzLabs/typical-small)'s original
comparison and predates workflow training on our side. Chance/majority baselines are .311 standard /
.284 easy / .336 hard.
