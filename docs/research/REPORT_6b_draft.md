# REPORT §3ac draft — Track C typed primitives (w-typed, 2026-09-21)
Five arms (init_from nc_v3_tap20_wf, 3k steps, qtype-filtered): score_A_kway, score_B_smooth (τ=0.7), score_C_cumlink, noul_A_2way, noul_B_bern.
Score: B wins pass rule (NLL/Brier/ECE improve broadly at matched accuracy; identical per-item decisions on JevBench ordinal items); C best JevBench std/ordinal Brier but calibration-worse internally (undertrained fresh params) — open follow-up.
Noul: B (Bernoulli) wins 6/7 sets, exact reversed-label invariance (0/0), PagerDuty .602→.886, typed-decisions NLL 1.72→1.21. JevBench excluded (few true yes/no items).
Numbers: see REPORT.md §3ac and runs/{score_*,noul_*}/results.json, runs/jev_jev_<arm>/ on guychuk/pcdm-runs. Code at 6ec3ca8.
Bugs fixed mid-run: bench.load_ours threads score_head/noul_head; bern head degrades gracefully on non yes/no rows.
