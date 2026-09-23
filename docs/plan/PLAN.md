# PCDM v0 — PoC plan

Goal: test H1 (accuracy retention), H3 (calibration), H4 (null detection), H2 (parallel latency) of `idea.md`
with a frozen `Qwen/Qwen3-0.6B-Base` backbone and a small trainable decision tower, locally on MPS.

Cut from v0 (deliberately): SQuAD, `Score` type, uncertainty-shaping losses (§12), RL (§14), latent compression, LoRA.
Add LoRA only if ours < baseline C − 8 pts on SNLI.

## Files

| file | role |
|---|---|
| `data.py` | HF datasets → `DecisionExample` JSONL (train/val/eval sets). `uv run data.py` |
| `encode.py` | frozen backbone; `FeatureCache` of token features for every unique string |
| `model.py` | `DecisionModel` (cross-attn slot + energy scorer + null) and `decision_loss` |
| `metrics.py` | nll, brier, ece, auroc_null, selective acc, confident-wrong |
| `train.py` | train tower on cache, evaluate all eval sets, temperature-scale on val, write `runs/<name>/results.json` |
| `baselines.py` | B (prompted log-prob), C (cross-encoder linear head), C-late (pooled MLP) |
| `bench.py` | H2: latency vs M queries on one state, ours vs B |

## Data schema (JSONL, one per line)

```python
{"state": str, "query": str, "candidates": [str],   # K >= 1
 "target": [float],   # len K, sums to 1 (distribution over candidates, ignoring null)
 "p_null": float,     # mass on the architectural null
 "task": str,         # "snli" | "mnli" | "snli_soft" | "unli" | "clinc"
 "label": int|null}   # argmax index for hard-label metrics; -1 means null is correct; null for soft-only sets
```

`data/train.jsonl`, `data/val.jsonl` (2k from train mix), `data/eval/<name>.jsonl`.

### Train mix (~40k), seed 0
- `snli` — `stanfordnlp/snli` train, label≠-1, 12k balanced. state=premise, query=`"Relation of the text to: {hyp}"`,
  candidates `["entailment","neutral","contradiction"]` (HF 0/1/2). Null synthesis per example: 0.70 keep K=3;
  0.15 drop a non-gold (K=2, p_null=0); 0.15 drop gold (K=2, p_null=1, label=-1).
- `mnli` — `nyu-mll/multi_nli` train, 6k, same mapping + null synthesis.
- `snli_soft` — `snli_1.0.zip` dev `annotator_labels` (5 votes) → vote distribution, ~9k. Same null synthesis
  (drop-gold uses argmax; renormalise remaining).
- `unli` — `Zhengping/UNLI` train, 6k. Proposition: query=`"Is this true given the text? {hyp}"`,
  candidates `["yes"]`, target `[1.0]`, p_null=1−p.
- `clinc` — `clinc/clinc_oos` config `plus` train. Hold out 30 intents (seeded) entirely. 8k in-scope from the 120
  visible intents + all 250 oos. query=`"What is the user's intent?"`, candidates = intent names with `_`→space.
  In-scope: 0.5 → K=10 random incl. gold; 0.2 → all 120; 0.3 → K=10 excl. gold (p_null=1). OOS: K=10 or K=120 (50/50), p_null=1.

### Eval sets (never trained on; dedupe `(premise,hypothesis)` vs train)
- `snli_test` 5k hard; `snli_test_soft` same items with 5-vote targets (from zip test).
- `snli_test_paraphrase` — same as snli_test but candidates
  `["the hypothesis follows from the text","the hypothesis is undetermined by the text","the hypothesis contradicts the text"]`.
- `snli_null` — SNLI test, K=2, 50/50 gold present/absent.
- `mnli_val` 3k `validation_matched`; `chaos_mnli` all 1599 `metaeval/chaos-mnli-ambiguity` (`label_dist` = [e,n,c]).
- `unli_test` — UNLI test split.
- `clinc_test` — in-scope test, K=all 150 (includes held-out intents as candidates); `clinc_oos` — 1000 oos, K=150.
- `clinc_heldout` — test utterances of the 30 held-out intents, K=10 among held-out intents incl. gold.
- `clinc_k` — in-scope test, K ∈ {2,5,10,50}, gold present vs absent 50/50 (null-vs-K artifact check).

## Interfaces

```python
# encode.py
class Encoder:
    def __init__(self, name="Qwen/Qwen3-0.6B-Base", layers=20, device="mps"): ...
    @torch.inference_mode()
    def encode(self, texts: list[str], max_len: int, batch_size=64) -> list[torch.Tensor]  # each [len_i, 1024] bf16 on device
class FeatureCache:            # dict[str, (offset, length)] + flat bf16 feats
    def add(self, encoder, texts: list[str], max_len: int): ...
    def get(self, text: str) -> torch.Tensor      # [len, 1024]
    def save(self, path); @classmethod def load(cls, path, device)
def build_cache(data_dir="data", out="data/cache.pt")   # state 256, query 64, candidate 16

# model.py
class DecisionModel(nn.Module):   # d_in=1024, d=512, 2 decoder layers, 8 heads, ~9M params, fp32
    def forward(self, H, hmask, Q, qmask, C, cmask) -> logits [B, Kmax+1]   # last column = null
def decision_loss(logits, target, p_null, cmask) -> scalar   # soft CE over [(1-p_null)*target, p_null]; pads masked
def collate(cache, examples) -> (H,hmask,Q,qmask,C,cmask,target,p_null)     # C = mean-pooled candidate tokens
def decide(model, encoder, state: str, queries: list[tuple[str, list[str]]]) -> list[torch.Tensor]  # encodes state once, expands H to M

# metrics.py — all numpy float64 on CPU
def summarize(probs [N,K+1], target [N,K+1], label [N]) -> dict   # acc, nll, brier, ece, auroc_null, sel_acc@80, conf_wrong
```

Training: AdamW lr 3e-4, wd 0.01, B=128, 8 epochs, warmup 200 + cosine, clip 1.0, select by val NLL.
Flags: `--no_null` (ablation D), `--hard_only` (ablation E), default = F.
Temperature scaling: scalar T on val NLL, applied to all models incl. B; report raw and scaled.

Baseline B: full 28-layer backbone, 3-shot prompt, length-normalised candidate log-prob, null = literal `"none of the above"`.
Baseline C: layer-20 features of `"{premise} [SEP] {hypothesis}"` / utterance, mean-pooled, per-dataset linear head,
trained on the same (soft) targets. C-late: pooled state u, query v → MLP([u;v;|u−v|;u⊙v]) → per-dataset head.

Decision rules fixed up front: H1 holds if ours ≥ C − 2 pts on snli_test. H4 holds if AUROC_null > 0.85 on clinc_oos
and snli_null and clinc_heldout accuracy ≥ 0.8. H3 holds if F beats E and C on chaos_mnli NLL and snli_test ECE after temperature scaling.
