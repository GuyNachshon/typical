# PCDM v1 — full experiment on RunPod

Goal: test H1–H4 with an *adapted* backbone and label-diverse data, on one H100, ≤ ~$50 GPU (of $145 balance).
Rule: nothing is rented until the pipeline passes local tests + a 300-step mini run on the Mac; a bug on the pod costs cents.

## Decisions (from deep-reasoner design + Codex review; v0 code is extended, not rewritten)

**Backbone/adaptation.** `Qwen/Qwen3-1.7B-Base` (28 layers, d=2048). Hand-rolled LoRA (no peft): `LoRALinear(r=16, α=32, dropout=0.05)`
on q,k,v,o,gate,up,down of the top `--lora_layers 8` blocks; lower 20 frozen. `--lora_r 0` = frozen ablation. One forward with
`output_hidden_states=True` gives both `h_frozen = norm(hidden[L])` (L = n_layers − lora_layers) and `h_top` (post final norm).
States and queries go through the adapted backbone each step; **candidates stay in the frozen layer-L space** and are cached
(`FeatureCache`, max_len 16) → candidate cost is K-independent. bf16 weights, fp32 LoRA + tower, no autocast (same code on cuda/mps/cpu).

**Model (`model.py`).** `cand_null=True` default (G). Hybrid scorer: frozen-space features from masked-mean state `Uf`, query `Vf`
and candidate `C`: `sim = Linear(2·d_in, d)(cat[ln(Uf)⊙ln(C), ln(Vf)⊙ln(C)])`, `cos_u`, `cos_v`; scorer input
`cat[h, c, h⊙c, |h−c|, sim, cos_u, cos_v]` → `Linear(5d+2, d) → GELU → Linear(d,1)`. `--no_hybrid` zeros sim/cos. `d_in=2048`, ~13.5M tower.
MPS-safe decoder layer only on MPS; standard layer on CUDA. `decide()` chunks M instead of materialising M copies of H.

**Data (`data.py`, same JSONL schema + `meta` for baselines).** ~800k train / 4k val; sources and sizes in the table below.
Null/K synthesis: NLI 0.70/0.15/0.15 as v0; classification with N labels: 0.35 K=N, 0.25 K=min(10,N), 0.15 K=3, 0.25 gold-absent
(K∈{3,10,N−1}, p_null=1); SQuAD v2 / BoolQ natural nulls. Label paraphrase without LLM: snake/Camel→words; per-example style
(raw / title-case / synonym table ~120 entries p=0.5 per word / HWU descriptions); NLI `TRAIN_NAMES` (2 wordings held out).
10 query templates per family (nli, prop, intent, topic, emotion, qa, boolq): 0 = v0 wording (eval), 1–7 train, 8–9 → `*_qpara` eval.
Held out ENTIRELY: Banking77, TREC (SetFit/TREC-QC), 20 Newsgroups (+ v0's 30 CLINC intents). Leak audit: normalised text and
(premise,hypothesis) overlap between train and every eval set must be 0; group-split train/val by state text.
Build locally → `hf upload guychuk/pcdm-data --private` → `hf download` on the pod.

**Training (`train.py`).** Integerised examples + string table; `DataLoader(num_workers=4, pin_memory)` on CUDA; length-bucketed
batches of 64; 24k steps; AdamW tower lr 3e-4 wd 0.01, LoRA lr 1e-4 wd 0; warmup 500, cosine to 0.1×; clip 1.0.
val NLL every 1k steps; full eval table + T-fit every 4k and at end; `last.pt` (lora+tower+opt+sched+step+rng+args) every 2k steps,
`best.pt` by val NLL; auto-resume if `last.pt` exists; W&B project `pcdm` (`train/loss`, per-family loss, lrs, step_time, tokens/s,
gpu_mem, `val/nll`, `eval/<set>/<metric>` raw+scaled, T, final `wandb.Table`); `hf upload runs/<name> guychuk/pcdm-runs` at end.
`--device auto` (cuda > mps > cpu) everywhere.

**Baselines.** B: same log-prob protocol with a **KV-cached prefix** (score all candidates of an example from one prefix pass),
`--backbone` flag (1.7B and `Qwen/Qwen3-8B-Base`), caps 1000 / 300 (K≥50). C_lora: cross-encoder with the same LoRA config + per-dataset
heads. `bench.py`: ours vs prefix-cached B, K∈{4,32,150}, M∈{1,8,64,256}, latency + peak memory. Undefined cells reported as `N/A`, not sentinels.

## Run schedule (H100 SXM ≈ $3/h; each idempotent — `run_gpu.sh` skips runs with `results.json`)

| # | run | flags | h | $ | answers |
|---|---|---|---|---|---|
| 0 | setup + GPU smoke (300 steps, kill+resume, W&B, hf upload) | | 0.5 | 1.5 | pipeline |
| 1 | `main_s0` | 1.7B, LoRA 8, G, hybrid, full mix | 1.2 | 3.6 | H1/H3/H4 |
| 2 | `B_1.7B` | KV-cached log-prob | 0.3 | 0.9 | H1/H3/H4 |
| 3 | `B_8B` | Qwen3-8B-Base | 0.8 | 2.4 | H1 larger model |
| 4 | `C_lora` | fine-tuned cross-encoder ceiling | 0.8 | 2.4 | H1 |
| 5 | `bench` | | 0.2 | 0.6 | H2 |
| 6 | `abl_frozen` | `--lora_r 0` | 0.8 | 2.4 | is LoRA needed |
| 7 | `abl_nohybrid` | `--no_hybrid` | 1.2 | 3.6 | unseen labels |
| 8 | `abl_nlionly` | NLI+soft+BoolQ+SQuAD+CLINC only | 1.0 | 3.0 | label diversity |
| 9 | `abl_statenull` | `--cand_null false` | 1.2 | 3.6 | H3 leak at scale |
| 10 | `main_s1` | seed 1 | 1.2 | 3.6 | noise floor |
| 11 | `main_4B` | Qwen3-4B-Base, LoRA top 10 | 2.8 | 8.4 | scale |
| | total | | 12.7 | ~$38 (+20% ≈ $46) | |

Pre-registered rules: H1 main ≥ C_lora − 2 on snli/mnli and ≥ 0.9×B_8B on anli; H3 main beats C_lora and B_8B on chaos_mnli NLL
and snli ECE, and abl_statenull is worse by ≥ 0.05 NLL; H4 AUROC_null > 0.85 on clinc_oos, snli_null, squad_null and
banking77/trec acc ≥ 0.9×B_8B, clinc_heldout ≥ 0.8; hybrid / label-diversity must buy ≥ 5 / ≥ 10 pts on banking77+trec;
H2 per-query marginal ≥ 5× cheaper than KV-cached B at K=4, growing with K.

## Local gate (all must pass before `runpodctl pod create`)
1. `uv run pytest tests/ -q` (< 60 s, CPU): ragged collate masks/shapes; probs pads = 0; permutation equivariance; loss = hand soft-CE;
   p_null=1 → all mass on null; LoRA zero-init ⇒ adapted == frozen output and LoRA/tower get grads, frozen don't; checkpoint
   save→load→one step reproduces loss; W&B offline run; data self-test (schema, no overlap, held-out sets absent from train).
2. `uv run train.py --smoke` = real Qwen3-0.6B-Base + LoRA top 4, 64 examples, 20 steps, `--device auto`, W&B offline; kill at step 10, rerun → resumes.
3. `uv run data.py --small` (200/source) → `uv run train.py --name mini --steps 300 --bs 16 --backbone Qwen/Qwen3-0.6B-Base`: no NaN,
   snli acc > chance, every eval set present in the table, `results.json` + W&B offline run written.
4. `uv run baselines.py B --limit 10`, `uv run bench.py --quick` (KV-cache path works on transformers 5.17).
5. Full `uv run data.py` + leak audit + `hf upload`.

## RunPod mechanics
Secure Cloud DC with H100 SXM; 100 GB network volume at `/workspace` (HF_HOME, UV_CACHE_DIR, data, runs, `.env`);
template `runpod/pytorch` (pin at task time); `uv sync`; SSH-exec + `tmux`; `timeout 14h bash run_gpu.sh; runpodctl pod stop $POD`
as cost guard; terminate after final upload.

## Data sources (HF ids verified by scout; see data.py for the exact mapping)
(filled in by data.py; held-out: mteb/banking77, SetFit/TREC-QC, SetFit/20_newsgroups)

## Interfaces (contract between files; workers code against these)

```python
# encode.py
def pick_device(device="auto") -> str                       # cuda > mps > cpu
class Backbone(nn.Module):
    def __init__(self, name="Qwen/Qwen3-1.7B-Base", lora_layers=8, lora_r=16, lora_alpha=32, lora_dropout=0.05, device="auto")
    tokenizer; d (hidden size); split_layer L = n_layers - lora_layers; device
    def tokenize(self, texts: list[str], max_len: int) -> (input_ids [B,T] long, attention_mask [B,T] long)   # CPU; eos sink prepended; right pad
    def forward(self, input_ids, attention_mask) -> (h_top [B,T-1,d] fp32, h_frozen [B,T-1,d] fp32, mask [B,T-1] bool)  # sink position dropped; grads flow only through LoRA
    @torch.inference_mode() def frozen_features(self, texts, max_len, batch_size=64) -> list[Tensor [len_i, d] bf16]   # h_frozen per text (for FeatureCache)
    def lora_state_dict(self) -> dict; def load_lora_state_dict(self, sd)
    def trainable_parameters(self) -> list[nn.Parameter]   # LoRA A/B only (empty if lora_r == 0)
class FeatureCache  # unchanged API: add(backbone, texts, max_len) uses backbone.frozen_features; get/save/load; plus .pooled(text) -> mean vector (memoised)
def build_cache(data_dir, out, backbone) -> encodes ONLY candidates (all jsonl under data_dir), max_len 16

# model.py
class DecisionModel(d_in=2048, d=512, nhead=8, dim_feedforward=1024, num_layers=2, dropout=0.1,
                    no_null=False, cand_null=True, hybrid=True, mps_safe=False)
    forward(H, hmask, Q, qmask, C, cmask, Uf=None, Vf=None) -> logits [B, Kmax+1]   # Uf/Vf [B,d_in] frozen-space masked means; required if hybrid
def decision_loss(logits, target, p_null, cmask) -> scalar               # unchanged
def collate(cache, examples) -> dict(state: list[str], query: list[str], C [B,Kmax,d_in] fp32, cmask bool, target [B,Kmax], p_null [B], task: list[str])   # CPU tensors, no tokenisation
def run_batch(backbone, model, batch, max_state=256, max_query=64) -> logits   # tokenises state/query, backbone.forward twice, Uf/Vf from h_frozen, model(...)
@torch.inference_mode() def decide(backbone, model, cache_or_none, state, queries, chunk=64) -> list[Tensor]   # encodes state once; chunks queries

# train.py CLI
uv run train.py --name X [--backbone Qwen/Qwen3-1.7B-Base] [--lora_layers 8] [--lora_r 16] [--steps 24000] [--bs 64] [--lr 3e-4] [--lora_lr 1e-4]
                [--seed 0] [--no_hybrid] [--no_cand_null] [--no_null] [--hard_only] [--mix full|nlionly] [--eval_every 4000] [--val_every 1000]
                [--ckpt_every 2000] [--device auto] [--wandb] [--hf_repo guychuk/pcdm-runs] [--data data] [--smoke] [--eval_only]
writes runs/<name>/{last.pt,best.pt,results.json}; results.json format unchanged (+ "n/a" strings allowed for undefined cells)

# baselines.py
uv run baselines.py B [--backbone ...] [--limit N] [--kv_cache]      # KV-cached prefix scoring default
uv run baselines.py C [--backbone ...] [--lora_layers 8] [--lora_r 16] [--steps N]   # C_lora
# bench.py
uv run bench.py --model runs/main_s0 [--quick] [--backbone ...]    # K∈{4,32,150} × M∈{1,8,64,256}; ours vs KV-cached B; latency + peak mem
```

## Revision after main_s0 (2026-09-17, autonomous)

`main_s0` (last-layer tap, LoRA top 8) finished at SNLI 58.6 / MNLI 51.4 vs C_lora 85.5 / 76.8 — and below v0's frozen 0.6B (66.7).
Diagnosis (deep-reasoner + two pod diagnostics):
1. **Memory layer**: the tower's H/Q came from the LAST layer (next-token-shaped). SNLI 58.6 ≈ the hypothesis-only baseline — the
   tower never read the premise. `diag_tap20_frozen` (tap layer 20, no LoRA) → 65.3 / 57.4; `diag_tap20` (LoRA on 13–20) @4k
   → 66.3 / 59.6 and better null AUROC. Decision: `--tap_layer 20` + LoRA below the tap. Layers 21–28 are dropped (≈30% cheaper).
2. **Rogue dimensions**: 3 dims carry ~30% of token norm; per-dim z-score of backbone features (`--zscore`, stats from 512 train
   states) gave +3.5 SNLI / −0.08 NLL locally. Adopted for main_v2+.
3. **Null over-firing** on unfamiliar wordings: gold-absent synthesis used `K=N−1` near-miss sets (teaches "near-miss ⇒ null") and a
   25% rate. Data v3: gold-absent 0.15 with K∈{3,10}; CLINC 0.20. Train null fraction 0.203 → 0.171. New probe `snli_test_hyponly`.
4. LoRA dropout was active at eval (modules built after `from_pretrained`'s eval()) — fixed (`backbone.eval()` in eval loops).

Revised schedule (phase 3, `run_gpu3.sh`): main_v2 (tap20+zscore+LoRA, data v2 — isolates z-score vs diag_tap20) → main_v3
(data v3, 24k steps) → abl_nohybrid (v3) → main_v3_s1 → bench → B_1.7B → B_8B. Cut: abl_frozen (answered by the diag pair),
abl_statenull (v0, 3 seeds), abl_nlionly (matching, not label diversity, is the bottleneck), main_4B (0.6B→1.7B bought nothing;
the tap/features did). Spend so far ≈ $8; phase 3 ≈ $20.
