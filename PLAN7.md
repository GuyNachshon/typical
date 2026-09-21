# PLAN7 — from architecture discovery to training + scaling a typed decision model (2026-09-20)

Synthesis of the lead's "Next Plan" memo (attached 19:11; architecture frozen; data → objective → scale; releases
`typical-small-preview` → `typical-small` → medium → large) with the Phase 6A verdict (REPORT §3w) and the JevBench
leaderboard survey (below). Two changes to the memo's order, both evidence-driven:

1. **Scale moves up from last to parallel-first as a *diagnostic*.** Every open entry ≥ .84 JevBench-standard is a
   *frozen* 4B–26B model reading raw option logits zero-shot (SemIf/OpenJev/system-one/open-alternative-jev); no
   workflow training. The only sub-4B entry above .40 hard is Gemma-E2B + LoRA + trained native head (our recipe shape).
   So "is hard = capacity?" is answerable this week for ~$60 by running the *same frozen recipe* at 4B / 8B / 14B and by
   zero-shot frozen-logit controls at each size — and it is exactly the memo's Phase-11 scaling curve, just earlier.
   Efficiency is benchmarked at every size (single-decision ms, L(K), memory, $/1k) so the curve is capability *per ms*.
2. **The mixture sweep and typed primitives run in parallel with it, not before it** — they are independent of size,
   and the memo's Release-1 recipe needs all three anyway. Budget is not the constraint this round (lead, 19:11).

The memo is otherwise adopted as written: DecisionMix v2 metadata schema, WH hard curriculum with counterfactual rubric
groups (levels 1–7), U soft-target corpus, Noul/Score typed heads (Score first), calibration last, external evals stay
untouched (JevBench, typed-decisions, PagerDuty, tree-choice, Laya benches), energy path = large-K front end, closed list
of reopened-only-with-evidence architecture items.

## Tracks (all start now; pods in parallel)

| track | what | pods | owner |
|---|---|---|---|
| 0 | **Release 0**: freeze `nc_v3_tap20_wf` as `typical-small-preview` on HF (weights + card with §3t/§3w numbers + known limitations from the memo), git tag | – | fast-worker |
| A | **Scaling ladder, same recipe** (native N3, `letters_nonull`, factored null, LoRA r16 top-8, tap ≈ 71% depth, v5+kb+wf+wf_hf, E .35/K .25/W .40, 12k steps): Qwen3-4B-Base, Qwen3-8B-Base, Qwen3-14B-Base. Each: probe (Δ_q), `eval_wf`, JevBench, `bench.py --native` latency + memory. Plus **zero-shot frozen-logit controls** (letter logits over options, no training) at 1.7B/4B/8B/14B on JevBench + the evidence/knowledge suite — separates backbone capacity from our training. | 3 | fast-worker (+ me for pod ops) |
| B | **Mixture sweep (memo 7B)** at 1.7B: E/K/W ∈ {.50/.20/.30, .45/.20/.35, .40/.25/.35, .40/.20/.40} + null control: W rows with gold removed → ∅ at 15% (`--null_aug`) at the .40/.25/.35 point. Scorecards E/K/W/U tracked separately; pick the Pareto point (E, K within 3; W ≥ 6A; false-abstain ≤ baseline). | 2–3 | fast-worker |
| C | **Typed primitives (memo 8)** on the native head at 1.7B: Score — K-way Choice vs ordinal threshold (cumulative link) vs distributional ordinal; Noul — Bernoulli head vs 2-way Choice. Matched runs on the 6A mix; metrics acc / ordinal MAE / NLL / Brier / ECE / threshold utility. Design from deep-reasoner review, then implement. | 1–2 | deep-reasoner → fast-worker |
| D | **DecisionMix v2 (memo 7A) + hard curriculum (9) + U corpus**: extend `scripts/workflow_corpus.py` with the memo's metadata schema, rule-depth levels 1–7 with counterfactual rubric groups, long-policy distractors, temporal/numeric/probability/trade-off families; U from soft-label sources (ChaosNLI, UNLI, ambiguity sets, synthetic known-distribution). CPU only. | – | fast-worker ×2 |
| E | **Calibration (memo 10)**: after C + D — `L = log + λ_B Brier + λ_O ordinal` on U; per-type/per-tier calibration reporting; no single global T. | 1 | later |
| F | **Large-K path (memo 12)**: energy → top-r → native; recall@r, final acc, null, K to 10k, cold/warm latency. Eval-time. | shared | fast-worker, after A |

Release 1 (`typical-small`) = clean retrain of 1.7B with B's mix + C's heads + D's data + E; then the same recipe at the
ladder sizes = `typical-medium` / `typical-large` (14B or a MoE with ~3B active if it serves faster — decided by track A's
capability-per-ms curve, not by parameter count).

## Pass/decision rules
- Track A: if 4B/8B raise JevBench-hard well above .40 with the same recipe → hard = capacity; the release ladder is
  justified and the small model's job is efficiency. If not → reopen rubric/instruction interaction (memo's caveat).
- Track B: Pareto point as above; no aggregate-loss selection.
- Track C: adopt a typed head only if probability quality (NLL/Brier/ECE) or threshold utility improves at equal accuracy.
- Every claim gets a matched baseline on identical rows; JevBench stays per-item on the public ids and unranked.

## Execution notes (2026-09-20 20:45, after the two reviews)
- $0 first: null-logit offset `b` fitted jointly with T on held-aside soft rows (typed_decisions_train + wf val) →
  re-evaluate both checkpoints (predicted: CLINC-150 false-abstain .168 → ~.06, held-out score NLL 2.07 → ~1.3); CLINC
  `other` probe; rubric-shuffle on the 36 short JevBench-hard items (does .64 fall to chance?). Worker `w-calib`.
- `eval_wf` rebuilt: batched, stratified `--limit` (pairs adjacent), per-family/qtype/class breakdowns, majority
  baselines; eval files re-written interleaved; full-file re-eval of both checkpoints → REPORT §3x supersedes §3w's
  W/external numbers. Worker `w-evalfix`, one pod.
- Track B runs (6): four ratios, `mix_e45_null` (`--null_aug W:0.20`), `long_e45` (`--max_state 1024` + 25% long
  rows) — the long-state run is the capacity-vs-length test at 1.7B. Worker `w-mix`, three pods.
- Track C arms (5) from the 6A checkpoint, 3k steps each: score K-way / ordinal-smoothed targets (τ = .7) /
  cumulative-link head; noul 2-way / Bernoulli (reversed-label control must be exactly 0). Worker `w-typed`, one pod.
- Track A ladder (4B / 8B / 14B, tap ≈ 71%) + zero-shot frozen-logit controls at 1.7B/4B/8B/14B + latency/memory
  bench at every size. Worker `w-ladder`, three pods.
- Track D DecisionMix v2: `data_wh` (levels 1–7, counterfactual rubric groups, held-out grammars/families/styles/level 7)
  and `data_u` (ChaosNLI / UNLI / ambiguity / synthetic known distributions). Worker `w-data2`, CPU.
- Cost guard now reads pods from /tmp/PODS_ACTIVE (workers append `<id>|<sshfile>|<waiter>`).

## Track D — DecisionMix v2 build notes (2026-09-20, `scripts/decisionmix_v2.py`, `guychuk/pcdm-data` `wh/`+`u/`)

**data_wh**: 60,605 train / 3,000 val / 6 eval files, 0/231 JevBench leak hits. Programmatic rule engine
(12 domains x 6 boolean conditions); levels 1-6 -> train/val, level 7 (temporal/numeric/probability/trade-off)
is eval-only. 100% of train rows are counterfactual rubric groups (same state+candidates, >=2 distinct golds;
`meta.rubric_group`). Mix noul/choice/score 40/41/19 (target ~40/40/20); 9.7% true-null rows; 12% catch-all.
Holdouts: 1 domain/level as `wh_heldout_grammar`, 3 whole domains (`wh_heldout_family`), 2 rubric styles
(terse/audit, `wh_heldout_style`), plus `wh_level7`, `wh_rubric_flip`, `wh_rubric_shuffled` -- reusing
workflow_corpus.py's `flip_pairs`/`shuffled_rubric`/`leak_check` unmodified.

**data_u**: 31,029 train / 2,000 val / 3 eval files, 100% soft_target, 0/231 leak hits. Real sources: UNLI's
*validation* split only (train/test already fully consumed by data.py's `unli`/`unli_test`) and metaeval/ambient's
ambiguous rows (uniform target over listed labels, no per-annotator counts on the HF mirror). chaos-mnli-ambiguity
is eval-only (`u_chaosnli`) since data.py's `chaos_mnli` eval already claims the whole file. Remaining 92% is 4
synthetic generators with exact closed-form targets (partial evidence, noisy-sensor Bayes posterior, ordinal
confusion via inverted confusion matrix, conflicting-sources log-odds), each with a held-out parameter range.

**Not sourced**: `metaeval/chaos-nli` 404s (used `metaeval/chaos-mnli-ambiguity`, MNLI portion only); `nyu-mll/
multi_nli`'s HF parquet has no per-annotator label columns (AmbiEnt covers the ambiguity requirement instead).
Neither ambient nor chaos-mnli-ambiguity declares a license on its HF card (flagged, not asserted); UNLI is MIT.

Test suite caught a real bug pre-upload: `finish()` re-shuffled state per row, so rubric-group members meant to
share one state string diverged (order, or independently-redrawn L6 padding). Fixed to render once per group;
both corpora regenerated after. 10/10 tests pass (`tests/test_decisionmix_v2.py`); `--limit` smoke works for
`--corpus wh|u|both`.

## Execution notes (2026-09-22)
- Release 1 shipped publicly: OzLabs/typical-small-preview, typical-small (1.7B, `ts1b`), typical-medium (4B, `tm1b`),
  each with the `inference/` package; local demo `demo/app.py`. `ts1` collapsed on a global Bernoulli-routing bug (fixed
  per row, 7b9d520).
- Long-state data bug found and fixed (§3ag); frozen controls incl. Qwen3.5-4B-Base; Qwen3.5 port (9b1ffac).
- Running: tl1b + tl1b_nokd (14B), tm2 (Qwen3.5-4B), SemIf rendering probe. Next: decide the release family (Qwen3 vs
  Qwen3.5 ladder), back-port the fixes to small/medium, Phase 10 calibration objective, level-7 generator work.
