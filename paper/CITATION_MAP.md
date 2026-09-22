# Citation Map

Claim → citation keys, so the paper writer can cite without re-searching. All keys resolve in
`paper/references.bib` (142 verified entries). "S" = supports/grounds our claim, "C" = contextualises
or is the prior art a reviewer will raise, "A" = the attack citation (what a sceptic cites against us).

## Architecture and interface

| Claim / paper section | Source in repo | Keys |
|---|---|---|
| Runtime-supplied label sets are an established problem; label-embedding and schema-driven models | COMPARE §1–2 | C: pappas2019gile, yin2019zeroshot, halder2020tars, tunstall2022setfit, zaratiana2023gliner, zaratiana2025gliner2 |
| Encode state once, score many candidates: cross- vs dual-encoder, late interaction, decomposition | IDEA2, COMPARE §5 | C: nogueira2019bertrerank, reimers2019sbert, khattab2020colbert, cao2020deformer, macavaney2020prettr, yang2023mixencoder, dejong2023lumen, humeau2020polyencoders, lu2022erniesearch |
| Decoder LLM + classification head instead of generation | §1, §3l | C: li2023labelsupervised |
| The winner is present in hidden states before the answer symbol is bound | §3l, §3r, NOVELTY.md | C: wong2026models, bhatt2026representation, lieberum2023circuit, pal2023futurelens, wu2024planahead, burns2023latentknowledge |
| Probing / linear readout methodology and its controls | §3l Δ_q probes | S: alain2017probes, hewitt2019controltasks, belinkov2022probing, tenney2019pipeline, belrose2023tunedlens |
| The letter/MCQ interface is order- and symbol-fragile (motivates the direct readout) | §3l, §3q reversed-label control, §3v | S: zheng2024selectionbias, pezeshkpour2024optionorder, yang2025option, xue2024symbolbinding, wang2024firsttoken, robinson2023mcp |
| Choices-only / shuffled-question control (Δ_q) | §3j, §3l, §3r | S: balepur2024artifacts |
| Mid-depth tap (71%) rather than the last layer | §3r, §3s, §3g | S: gromov2024deeperlayers, men2024shortgpt, tenney2019pipeline, schwartz2020righttool, belrose2023tunedlens |
| LoRA on the top kept layers; backbone family | §1, §3ag | S: hu2022lora, yang2025qwen3; C: vaswani2017attention, brown2020gpt3, ouyang2022instructgpt |
| Structured output usually obtained by constrained generation (the alternative we avoid) | PROJECT.md framing | C: geng2023grammarconstrained, willard2023guided, beurerkellner2024domino; A: tam2024speakfreely |

## F1 — candidate-blind states carry priors and calibration, not question-conditioned knowledge

| Claim | Source | Keys |
|---|---|---|
| The factorisation being tested (compute the decision before candidates are known) | §3j, PLAN3 E3 | C: khattab2020colbert, dejong2023lumen, cao2020deformer, humeau2020polyencoders |
| What distillation transfers; why calibration transfers cheaply | §3j E3b/E3-ms | S: hinton2015distillation, kim2016seqkd, yuan2020teacherfree, zhang2020selfdistillation; C: snell2022contextdistillation, askell2021assistant, yang2024sdft |
| Above-chance score of a candidate-blind student is candidate priors | §3j control table | S: balepur2024artifacts, zhao2021calibrate |
| Mechanistic reason the negative is expected | §3s | C: wong2026models |
| Attack: under-training / small backbone / low-power measurement | §3j, §3s | A: biderman2024loralearnsless, mosbach2023fewshot |

## F2 — direct contextual readout preserves knowledge with less order fragility

| Claim | Source | Keys |
|---|---|---|
| N3 ≈ N1 on Δ_q, two seeds | §3l, §3v | C: wong2026models (mechanism), wang2024firsttoken |
| Slot identity is required (N2 fails, N3 works) | §3l | C: reimers2019sbert, khattab2020colbert (why slot-agnostic embeddings lose) |
| Order/IIA fragility quantified against the letter readout | §3l, §3t, §3v | S: zheng2024selectionbias, pezeshkpour2024optionorder, xue2024symbolbinding, yang2025option |
| No generation is needed for bounded decisions | §3l, §3m | C: sprague2025cot, li2024cotserial, merrill2024expressive, merrill2023parallelism, feng2023chain, goyal2024pause, pfau2024filler, deng2024internalize, hao2024coconut |

## F3 — the evidence/knowledge trade-off was depth plus a rendered abstain line

| Claim | Source | Keys |
|---|---|---|
| Depth, not the options-in-suffix formulation, cost evidence accuracy | §3r | S: gromov2024deeperlayers, men2024shortgpt, tenney2019pipeline |
| A rendered "none of the above" line was the null pathology | §3t, §3p | C: larson2019clinc, geifman2017selective; A: zhao2021calibrate, tam2024speakfreely (format effects) |
| Two-expert machinery unnecessary; learned gate recovers little of the envelope | §3n, §3o, §3u | C: hendrycks2017msp, liu2020energyood (confidence features) |

## F4 — rubric-conditioned training transfers inside its grammar only

| Claim | Source | Keys |
|---|---|---|
| In-grammar transfer (styles, families, rubric-flip) | §3w, §3x, §3ad | C: wei2022flan, sanh2022t0, chung2024flant5 |
| No transfer to unseen composition (level 7) or untouched external sets | §3x, §3ad | S: lake2018scan, keysers2020cfq, kim2020cogs, hupkes2020compositionality |
| Tuning mostly teaches format/shape | §3w verdict | A: zhou2023lima, lin2024urial |
| Serial/symbolic families are where a single forward pass should fail | §3ah (temporal/probability/tradeoff) | S: li2024cotserial, merrill2024expressive, merrill2023parallelism, feng2023chain, sprague2025cot |

## F5 — training helps standard, hurts hard; calibration needs soft targets and calibration-based selection

| Claim | Source | Keys |
|---|---|---|
| Fine-tuning can underperform a frozen backbone out of distribution | §3ab, §3ah | A: kumar2022finetunedistort; C: mosbach2023fewshot, biderman2024loralearnsless, luo2025forgetting |
| Fine-tuning degrades calibration; prior-knowledge interaction | §3ab, §3w | S: wang2025cogcalib, desai2020calibration, guo2017calibration |
| Soft / ordinal-smoothed targets fix probability quality without changing decisions | §3ac, §3ae | S: mueller2019labelsmoothing, szegedy2016labelsmoothing, diaz2019softordinal, gneiting2007scoringrules, brier1950verification, epstein1969rps, murphy1970rps |
| Brier term in the objective + checkpoint selection on held-out calibration NLL | §3ah, §3ag | S: gneiting2007scoringrules, brier1950verification, kumar2019verifiedcalibration, naeini2015calibration |
| We deliberately do not use a global post-hoc temperature | §3y, §6 | C: guo2017calibration, kumar2019verifiedcalibration |
| Frozen-teacher KD from the same backbone's few-shot distribution | §3ah | S: hinton2015distillation, askell2021assistant, snell2022contextdistillation, yang2024sdft, yuan2020teacherfree |
| Alternative uncertainty machinery we did not use (positioning) | §6 | C: yang2024laplacelora, angelopoulos2023conformal, romano2020adaptivecoverage, kumar2023conformalmcq, tian2023verbalized, kadavath2022know |

## F6 — long-state truncation bug

| Claim | Source | Keys |
|---|---|---|
| Position of evidence inside a long context changes answers | §3ag | S: liu2024lostmiddle, shi2023distracted |
| Length alone degrades reasoning; effective context < nominal context | §3ag, §3w state-length split | S: levy2024moretokens, hsieh2024ruler |
| Silent data-pipeline failures compound downstream | §3ag | S: sambasivan2021datacascades |

## F7 — serving cost is launch-overhead bound

| Claim | Source | Keys |
|---|---|---|
| Prefix/KV sharing across many suffixes is established systems work | §3m, §1 | C: juravsky2024hydragen, zheng2023sglang, kwon2023pagedattention, gim2023promptcache, cheng2023batch |
| What binds LLM inference at small batch | §3m, §3ab, §3ag | S: pope2023scaling, dao2022flashattention |
| Generation-side accelerators exist because decode steps are overhead-bound (contrast) | §3ab latency table | C: leviathan2023speculative, chen2023speculativesampling, cai2024medusa |

## Typed primitives (Choice / Score / Noul) and abstention

| Claim | Source | Keys |
|---|---|---|
| Ordinal targets / cumulative-link heads for Score | §3ac | S: mccullagh1980ordinal, frank2001ordinal, niu2016ordinalcnn, cao2020coral, shi2023corn, diaz2019softordinal |
| Ordinal-aware scoring metric (RPS) for the Score tier | §3ac, metrics.py | S: epstein1969rps, murphy1970rps, gneiting2007scoringrules |
| Bernoulli Noul head is exactly order-invariant; 2-way Choice is not | §3ac | S: zheng2024selectionbias, pezeshkpour2024optionorder, xue2024symbolbinding |
| Architectural abstain option; selective classification framing | §3h, §3t, §3y | S: chow1970reject, geifman2017selective, geifman2019selectivenet, hendrycks2017msp, liu2020energyood, wen2024abstention; C: chen2023selfevalselective, kumar2023conformalmcq |
| Out-of-scope intent detection as the abstention benchmark | §3c–§3e, COMPARE §2a | S: larson2019clinc, casanueva2020banking77 |

## Benchmarks, datasets and baselines

| Item | Source | Keys |
|---|---|---|
| MMLU-Pro (knowledge tier, Δ_q probes) and MMLU | §3f, §3j, §3l | S: wang2024mmlupro, hendrycks2021mmlu |
| TruthfulQA-MC1 | §3j, §3l | S: lin2022truthfulqa |
| NLI family and human-disagreement / graded sets | §3c–§3e, data_u | S: bowman2015snli, williams2018mnli, nie2020anli, nie2020chaosnli, chen2020unli |
| BoolQ | §3c | S: clark2019boolq |
| Encoder reference numbers in COMPARE §2a | COMPARE §2a | S: devlin2019bert, liu2019roberta, he2023debertav3 |
| Entailment-as-zero-shot baseline framing | COMPARE §5 | C: yin2019zeroshot |
| ModernBERT (backbone of the Laya leaderboard entry) | COMPARE §4a | C: warner2024modernbert |
| JevBench harness, Jev/TypeSafe claims, open replications | §3q, COMPARE §4a/§5 | URL-only; see the `% [VERIFY]` block at the end of references.bib. Do not invent BibTeX keys for these. |

## Keys the paper must NOT cite

`% [VERIFY]` in `references.bib`: Orca (OSDI 2022), El-Yaniv & Wiener (JMLR 2010), Menon et al.
"A Statistical Perspective on Distillation" (ICML 2021), the inverse-focal-loss paper (NeurIPS 2021),
GPT-2 tech report, and every TypeSafe/JevBench/replication URL. If any of these becomes load-bearing,
fetch and verify it first.
