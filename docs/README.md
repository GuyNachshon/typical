# Docs map

Everything except `README.md` and `REPORT.md` moved off the repo root on 2026-09-23. Prose
written before that date cites these by bare filename ("PLAN2.md", "REVIEW.md §3"); the
section numbers are unchanged, only the path moved.

## Repo layout

| path | what |
|---|---|
| `pcdm/` | the training/research modules: `train.py`, `model.py`, `native.py`, `encode.py`, `data.py`, `mcq.py`, `metrics.py`, `baselines.py`, `bench.py`, `report*.py`. Run them from the repo root — `uv run pcdm/train.py …` — so `runs/` and `data_*/` resolve. |
| `pcdm_jev/` | the JevBench-facing decider/adapter over those modules. |
| `inference/` | the public, training-repo-independent `typical` package (+ parity test). |
| `scripts/` | one-off builders, evals, audits, and the shell drivers (`run_gpu*.sh`, `pod_setup.sh`, `watchdog.sh`, `costguard.sh`, `sync_research.sh`). |
| `site/` | the research + marketing site, its `server.py`/`precompute.py`, and its `package.json`. |
| `demo/`, `blog/`, `paper/`, `figures/`, `releases/`, `runs/`, `frozen/`, `vendor/` | the Gradio demo, posts, the paper source, generated figures, release cards, run artefacts, the frozen eval slice, and the vendored DOOM build. |
| `tests/` | `uv run pytest tests/ -q` (~2.5 min, CPU). `pyproject.toml`'s `pythonpath` puts `pcdm/` and the root on the path. |


## Docs

| where | what |
|---|---|
| `REPORT.md` (root) | the consolidated results log, §3a…§3ah. The number you are looking for is here. |
| `README.md` (root) | what Typical is, the model table, how to run inference and the demo. |
| `plan/PROJECT.md` | start here: document map, code map, run registry, findings ledger, ops rules. |
| `plan/PLAN.md` … `plan/PLAN7.md` | the roadmaps in order; `PLAN7.md` is the current one. |
| `plan/idea.md`, `plan/IDEA2.md` | the v0 and post-v1 statements of the thesis and the hypothesis stack. |
| `research/RESULTS.md` | one results table per model family/size, pulled from run artefacts. |
| `research/COMPARE.md` | the competitive picture vs the closed Jev/System One family and JevBench. |
| `research/NOVELTY.md` | what is and isn't novel in the contribution. |
| `research/REVIEW.md` | the stop-and-rethink review after `joint_v1`. |
| `research/RESEARCH_KNOWLEDGE*.md` | the field notes behind the above, incl. the adversarial pass. |
| `research/REPORT_*_draft.md` | per-file derivations behind REPORT §3x/§3y/§3ac. |
| `research/runs_bench_saved.txt` | a saved v0 bench table (MPS, state 256). |
| `site/CONTENT.md`, `site/DESIGN.md`, `site/PRODUCT.md` | the site's copy deck, visual lane and positioning. |
| `launch/` | the v0 launch plan, strategy notes, Jev wire schema, and v0 site screenshots. |
| `archive/v0/` | `BLOG.md`, `MODEL_CARD.md`, `LIMITATIONS.md` for typical-v0 (`joint_emb_lw_v5`). Superseded numbers — cite `releases/*.md` instead. |

Release cards live in `releases/`, the inference package's docs in `inference/README.md`, the
demo's in `demo/README.md`, and the site's in `site/README.md`.
