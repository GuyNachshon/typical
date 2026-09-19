# PCDM v0 — Parallel Calibrated Decision Model (PoC)

PoC of `idea.md`: encode a state once with a frozen LM, then answer many natural-language-defined
questions with runtime-defined candidate sets in parallel, as calibrated probabilities with an
architectural "null" (none of the above). Trained on real data, locally on an M4 Pro (MPS).

See `PLAN.md` for the design, data mix, interfaces and pre-registered decision rules.

## Run

```bash
uv sync
uv run data.py          # HF datasets -> data/{train,val}.jsonl + data/eval/*.jsonl  (~1 min)
uv run encode.py        # frozen Qwen3-0.6B-Base (20 layers) -> data/cache.pt        (~10 min, 3 GB)
uv run train.py --name F                 # soft labels + null      (the model)
uv run train.py --name D --no_null       # ablation: no null
uv run train.py --name E --hard_only     # ablation: hard labels only
uv run train.py --name G --cand_null     # variant: candidate-aware null
uv run baselines.py Clate                # pooled late-interaction MLP (is the cross-attn tower needed?)
uv run baselines.py C                    # frozen cross-encoder + per-task linear heads (accuracy ceiling)
uv run baselines.py B                    # AR baseline: prompted candidate log-probs, full 28-layer LM
uv run report.py                         # side-by-side table of runs/*/results.json
uv run bench.py --model runs/F/model.pt  # H2: latency vs number of queries on one state
```

## What was found while building it

- **Position-0 attention sink.** In Qwen3 (no BOS), the hidden state at position 0 is content-independent
  (cosine 0.9998 between `neutral`, `yes`, `transfer`). Every single-token candidate collapsed to one vector.
  Fix: prepend `<|endoftext|>` at encode time and drop it (`encode.py`).
- **Fixed classifier in disguise.** With three fixed NLI label strings, the scorer memorised three vectors:
  accuracy on paraphrased labels was 11% (below chance). Randomly paraphrasing label names during training
  (`data.py: TRAIN_NAMES`) fixed the collapse, but transfer is shallow: over 5 never-seen wordings F gets 43%
  (chance 33%), from 64% on wordings close to a training set down to chance on distant ones. B (prompted LM)
  picks its literal `"none of the above"` on every paraphrased item (0%) — the §7 argument against a null string.
- **State-only null vs unseen label spaces.** On 30 never-seen CLINC intents the model ranks the right
  candidate 62% of the time (chance 10%) but answers null 93% of the time, because the null energy only sees
  the state. Adding Banking77 (label diversity) helps somewhat; variant G makes the null candidate-aware.

## Results (single seed unless noted; scaled = after temperature scaling on val)

Runs: **F** = the model (cross-attn slot + energy scorer, soft labels, state-only null). Ablations: **D** no null,
**E** hard labels. Variant **G** = null energy sees the score-weighted best candidate. Baselines: **Clate** pooled
late-interaction MLP (no cross-attention), **C** frozen cross-encoder + per-task linear head (frozen-feature ceiling;
cannot do dynamic candidates), **B** full 28-layer LM, 3-shot prompt, length-normalised candidate log-probs.

`uv run report.py` prints the table; `RESULTS.md` has the final snapshot.

### Verdicts against the pre-registered rules in PLAN.md

| H | Rule | Outcome |
|---|---|---|
| H1 accuracy retention | F ≥ C − 2 pts on SNLI | **Holds (barely)**: 66.6 vs 68.4. F ties C on MNLI and beats it on CLINC (72 vs 45). Both sit at the frozen-layer-20 ceiling — the number is a backbone limit, not a decision-architecture limit. LoRA is the upgrade path. |
| H3 probability quality | F beats E and C on ChaosNLI NLL and SNLI ECE | **Fails as stated.** F's ChaosNLI NLL (1.33) is worse than D (1.17) and C (1.27). Cause: a state-only null learns the null-synthesis base rate and leaks a constant ~13% onto ∅ on clean 3-way items (≈0.15 nats). Renormalising ∅ out, F ≈ D. **G removes the leak** (P(∅) ≈ 1%) and has the best NLL/ECE of all runs (1.13 / 0.046). Soft labels alone (E vs F) buy nothing measurable. |
| H4 null detection | AUROC > 0.85 on CLINC-OOS and SNLI-null; ≥ 0.8 on held-out intents | **Partial.** CLINC in-scope vs OOS 0.90 ✓, SNLI-null 0.82 ✗ (narrow), held-out intents 0.20 ✗. Versus a no-null model's best signal (1 − max P): +8 to +17 AUROC. Null probability is K-dependent (P(∅ | gold absent) 0.98 at K=2 → 0.61 at K=50). |
| H2 parallel latency | latency(M) ≪ M·latency(1) | **Holds**: 256 queries on one state = 11.8× the cost of 1 (1.58 s) vs ~70× at M=64 for the prompted LM (17.5 s); ~6 ms marginal per query. |

### Other findings
- **Tower vs pooled scorer.** Cross-attention is worth +8 pts on SNLI/MNLI and is what makes paraphrased labels work
  (Clate: 29%). But Clate generalises better to *never-seen* intent names (76% vs 62% among-K): scoring in the frozen
  space (`u ⊙ c`) is a zero-shot similarity, while the tower's learned 512-d space partly memorises the seen label set.
- **Unsolved:** at K=150 with mixed seen/unseen candidates, utterances of unseen intents are confidently mapped to a
  *seen* intent (12% among-K, null fires 27%) by every variant. This is the confident-wrong case H4 is about.
- Noise floor: ±0.7 pt accuracy at n=5000, ±0.02–0.03 NLL on ChaosNLI, ±0.01–0.02 ECE. D/E/F/G are indistinguishable on
  in-scope accuracy; F-vs-D NLL, G-vs-F NLL, and tower-vs-Clate gaps are real.

## Cut from v0 / next
SQuAD, `Score` type, uncertainty-shaping losses (§12), RL (§14), latent compression, LoRA.
Next by information-per-hour: (1) 2 more seeds of F and G (queued); (2) hybrid scorer — add the frozen-space
similarity `u ⊙ c` to the tower's scorer input to get Clate's unseen-label transfer without losing the NLI edge.
