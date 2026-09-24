# Typical v0 launch — plan (2026-09-20)

Lead: this session. Ground truth for numbers: `REPORT.md`, `COMPARE.md`, `runs/*/results.json`, `runs/bench_a100/bench.json`.
Research summaries: `docs/launch/research-jev.md`, `docs/launch/research-releases.md`, `docs/launch/research-strategy.md`.

## Positioning
- Typical = open decision model. "Read the evidence once, answer hundreds of typed questions as calibrated probabilities with a built-in *none of the above*. No generation. Open weights, $4 to retrain."
- It is NOT a knowledge model (MMLU-Pro .09), not an LLM replacement, state ≤ 256 tokens, English, single seed, weaker on label vocabularies it never saw.
- Ship ONE model: `joint_emb_lw_v5` ("typical-v0", listwise, best null). Also publish `joint_emb` (v4, pointwise, strict add-irrelevant IIA = 0.000) as `typical-v0-pointwise`. `nc_n3` = research preview only, not demoed.

## Hero number
Per-decision cost is flat in K: 1.29 / 1.27 / 1.34 / 2.44 ms at K = 4 / 32 / 150 / 1000 (M=256, A100 PCIe, warm) vs 4.3 / 20 / 92 / 651 ms for a query-batched log-prob baseline on the same 1.7B → 3.3× / 16× / 69× / 267×. Always show the M=1 loss (131 vs 43 ms) too.

## Deliverables (this repo)
| path | what | owner |
|---|---|---|
| `typical/__init__.py` | `load()` → `Typical.decide(state, questions)`; Choice/Noul/Score derived from the one Choice primitive; temperature applied; token caps enforced | lead |
| `typical/serve.py` | FastAPI: `POST /v1/systemone` (Jev wire-compatible), `POST /v1/decide`, `GET /health`; serves `site/` | lead |
| `typical/precompute.py` | builds `site/data/*.json` from the real model on MPS: reliability bins, playground presets, choice-set lab presets, abstain presets; copies bench/cse/ksweep aggregates | lead |
| `site/` (index.html, app.js, style.css, data/) | landing page + 6 interactive demos, static-first, goes live when `/health` answers | frontend agent |
| `MODEL_CARD.md` | HF model card (YAML header, usage, evals with raw+scaled ECE, data sources + licenses, limitations) | docs agent |
| `LIMITATIONS.md` | "jaggedness"-style honest page | docs agent |
| `LAUNCH.md` | release checklist (blockers/nice-to-have), distribution kit (X thread, HN title, r/LocalLLaMA post, demo video shot list), evidence-bundle spec | docs agent |
| `BLOG.md` | launch post (ModernBERT skeleton: reframing title → TL;DR with 2 numbers → hero chart → how it works → benchmarks w/ caveats → 8-line usage → limitations → links) | blog agent |
| `LICENSE` | Apache-2.0 (code). Weights: flag ANLI CC BY-NC issue in LAUNCH.md | lead |
| `tests/test_typical.py` | schema conversions (score EV, confidence), wire adapter round-trip with a fake model | lead |

## Demos (site sections, in order)
1. **Playground** — doc (≤256 tok) + questions with option lists → probability bars incl. ∅, ms counter; presets canned (3 docs) + live when server up.
2. **Choice-set lab** — pick preset; buttons: shuffle / duplicate / add irrelevant / remove correct → Δlog-odds, Δp table. Toggle listwise (default, add-irr Δ .20) vs pointwise (0.000). Side panel: Jev third-party numbers (−0.28 log-odds, 0.84→0.93 reorder) cited to archerhume.com.
3. **Abstain lab** — CLINC-150 intents; type utterance → P(∅); K-sweep curve (P(∅|absent) .94→.41 K 2→150) shown honestly.
4. **Calibration** — reliability diagrams from real per-item dump; sets: SNLI, MNLI, CLINC, BoolQ, Banking77 (OOD, honest); raw vs T-scaled.
5. **Cost & latency calculator** — K, M, warm/cold, $/h → ms/decision, $/1k decisions, vs log-prob baseline; caveats block.
6. **How it works** — animated: prefix KV cache → M isolated suffixes → embedded candidates → energy + ∅ softmax.
7. **Benchmarks** — table (ours vs C_lora vs B_8B vs published refs), seed floor, links to results.json.
8. **Get started** — `pip install`, 8-line usage, `serve` command, Jev-wire drop-in curl.

## Data schemas (site/data/)
- `bench.json`: `{gpu, state_tokens, by_k: {K: {ours:[{m, total_ms, marginal_ms}], b_batched:[...], b_fair:[...]}}, cold:{K:{warm_ms, cold_ms}}}`
- `reliability.json`: `{set: {T, bins:[{lo,hi,conf,acc,n}], ece_raw, ece_scaled, n}}` (two variants: raw, scaled)
- `cse.json`: `{listwise:{clinc:{...}, banking77:{...}}, pointwise:{...}, jev:{add_irr_dlo:-0.28, reorder:"0.84→0.93", src:url}}`
- `ksweep.json`: `{clinc:[{K, p_null_absent, p_null_present, auroc}], banking77:[...]}`
- `presets.json`: `{playground:[{title, state, questions:[{q, options, probs:[..., null]}], ms}], cse:[{state, query, base:[...], variants:{shuffle:{options, probs}, dup:{...}, add_irr:{...}, remove_gold:{...}}, model:"listwise"|"pointwise"}], abstain:[{state, options(150), probs}]}`
- `benchmarks.json`: table rows from results.json.

## Order of work
1. lead: package + serve + precompute running (MPS, background) — schemas frozen above so the site can be built in parallel with sample data.
2. agents: site, docs, blog in parallel.
3. lead: integrate, screenshot-test site, run tests, commit.
