# Strategy report (deep-reasoner, 2026-09-20) — condensed

Ground-truth corrections:
- `joint_emb_lw_v5` does NOT have exact add-irrelevant IIA: cse add_irr/dlo_top2 = .204 (clinc) / .243 (banking77) — listwise SetMixer by design. Only pointwise `joint_emb` (v4) has 0.000. Exact for every energy model incl. lw_v5: reorder (max_dp 0.0) and duplicate slot gap (0.0).
- `decide()` truncates state to 256 tokens, queries to 64 tokens, candidates ≤32 embed tokens.
- `nc_n3` has no shared-state inference path and a broken null (abstains 61% on MMLU-Pro, OOS recall .20). Research preview only.
- At M=1 we are SLOWER than the prompted baseline (131 ms vs 43 ms at K=4). Win starts at M≥8.
- No LICENSE file. best.pt is hand-rolled LoRA+head (not PEFT).

Claim ledger (joint_emb_lw_v5, seed 0; seed floor NLI ±0.7, large-K ±3, unseen/OOS ±5):
- HERO: K-flat per-decision cost 1.29/1.27/1.34/2.44 ms at K=4/32/150/1000 (M=256, A100 PCIe, warm; cold +35–50%); 256 decisions over one state in 0.36 s. vs query-batched log-prob baseline same 1.7B: 3.3×/16×/69×/267×. Retire "73×".
- Calibration in-dist: SNLI ECE .009 raw/.020 scaled; MNLI .026/.011; CLINC .065/.034 (older joint_emb .007/.005).
- Evidence decisions: SNLI .909, MNLI .880, ANLI .555, BoolQ .834, CLINC-150 .784, HWU64 .818 (BoolQ/HWU ≤1–2 pts near-dup inflation).
- Null: AUROC .961 clinc-k / .980 snli / .978 squad; CLINC-OOS recall .735.
- Reorder + duplicate invariance exact (all energy runs). Jev: reorder 0.84→0.93, last-position 16/16 [AH].
- Add-irrelevant: 0.000 pointwise joint_emb only; .20 listwise; Jev −0.28 [AH].
- One state, isolated queries, bit-exact with concatenation (test_joint_causal_invariance).
- Open + reproducible ≈ $4/run.
MUST NOT claim: knowledge (MMLU-Pro .09; Jev 84.6), faster than Jev, 40–200× vs LLMs, OOD calibration (Banking77 ECE .21 raw), any label set (Banking77 .527), long docs, null robust to K (P(∅|absent) .94→.41), single-question latency, Score primitive (derived only), beats supervised classifiers on CLINC (.78 vs .97).

Ship: `joint_emb_lw_v5` as Typical v0; `joint_emb` (v4 pointwise) as strict-IIA checkpoint; nc_n3 research preview.

Release blockers: publish checkpoints; LICENSE (Apache-2.0 code; Qwen3 + Qwen3-Embedding are Apache-2.0); ANLI is CC BY-NC 4.0 (weights non-commercial OR drop ANLI and retrain ~$5); SNLI/MNLI/SQuAD/BoolQ CC BY-SA; imdb/yelp/bitext have restrictions; list all ~30 sources with licenses; loadable API + pip; limitations page.
Positioning: "Typical: read the evidence once, answer hundreds of typed questions with calibrated probabilities — no generation, open weights, $4 to retrain." What it is not: "not a knowledge model and not an LLM replacement: decides from the text you give it (≤250 tokens), 9% on MMLU-Pro, generates nothing, less accurate on label vocabularies it never saw."
