# Frozen evaluation artifacts (2026-09-19/20)
- `mmlu_pro.jsonl` (+ `.meta.json`): the 1,200-item stratified MMLU-Pro slice (seed 0) used for every knowledge number in REPORT.md §3f–§3l.
- `mmlu_cf.jsonl`: counterfactual option-set battery (6 variants × 1,200). Teacher-labelled version: HF `guychuk/pcdm-data` `v4/eval/mmlu_cf_teacher.jsonl`.
- `mmlu_choicesonly.jsonl`, `mmlu_shuffledq.jsonl`: the Δ_q probes (question blanked / swapped).
- `ksweep_clinc_nosib.jsonl`: sibling-excluded K-sweep (REPORT §3h).
- `leaked_qids.json`: 2 MMLU-Pro items with near-duplicates in data_kb (excluded); `manifest.json`: data_kb per-source counts.
Training data: HF `guychuk/pcdm-data` (`v4/`, `v5/`, `kb/`, `kbt/` teacher labels, `kbms/` multi-set labels). Checkpoints + results: HF `guychuk/pcdm-runs`.
Environment: `uv.lock`; torch cu126 on driver-570 hosts (see PROJECT.md ops).
