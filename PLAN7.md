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
