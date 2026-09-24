# Related Work

Citation keys resolve against `paper/references.bib` (142 machine-verified entries). Non-archival
artifacts — the TypeSafe/Jev product pages, the third-party black-box analysis, the JevBench harness
and the open replications — have no verifiable bibliographic record and are cited as footnote URLs,
never as BibTeX keys; they are listed under `% [VERIFY]` at the end of the bib.

## 1. Runtime-defined label spaces: label embeddings, generalist extractors, and the cross-/dual-encoder trade-off

Deciding over a candidate set supplied at inference time is an old problem. GILE embeds inputs and
label descriptions in a shared space so unseen labels are scorable [pappas2019gile]; entailment
reformulation turns arbitrary label names into hypotheses [yin2019zeroshot]; TARS conditions a
sentence encoder on the label text itself [halder2020tars]; SetFit reaches the same regime with
contrastive few-shot tuning [tunstall2022setfit]; GLiNER and GLiNER2 encode text and a runtime schema
jointly in one bidirectional pass [zaratiana2023gliner, zaratiana2025gliner2] — GLiNER2 also appears
on the JevBench leaderboard we report against. The cost axis is the classical cross- versus
dual-encoder trade-off: a cross-encoder reranker attends candidate to input and pays one forward per
candidate [nogueira2019bertrerank]; a dual encoder is candidate-independent but interacts only at a
dot product [reimers2019sbert]. Late-interaction and decomposition methods sit between — ColBERT
[khattab2020colbert], DeFormer [cao2020deformer], PreTTR [macavaney2020prettr], MixEncoder
[yang2023mixencoder], LUMEN [dejong2023lumen] — and compiling a cross-encoder into a cheaper student
is standard practice [lu2022erniesearch]. Decoder LLMs with a classification head close the loop
[li2023labelsupervised]. *Positioning: our energy path is a member of this family (K-independent,
exactly order- and IIA-invariant); our contribution is not the family but the measured boundary —
candidate-independent scoring carries evidence-grounded decisions and loses question-conditioned
parametric knowledge, no matter how the decision state is parameterised.*

## 2. Reading the decision out of hidden states instead of generating it

This is our closest prior art. Linear probes and their controls established that hidden states carry
decodable structure [alain2017probes, hewitt2019controltasks, belinkov2022probing], layerwise analysis
that different depths carry different abstractions [tenney2019pipeline, belrose2023tunedlens], and
unsupervised readouts that a truth-like direction exists [burns2023latentknowledge,
azaria2023internalstate]. States also anticipate tokens that have not been emitted
[pal2023futurelens, wu2024planahead], and MCQ-specific circuit analysis localises the symbol-binding
step [lieberum2023circuit]. Most directly, [wong2026models] shows a two-stage computation: the winning
option is decodable in content space at the final option boundary, before the answer symbol is bound;
[bhatt2026representation] traces the same emergence across languages. Meanwhile the letter interface
itself is documented as unreliable: selection bias and option-order sensitivity
[zheng2024selectionbias, pezeshkpour2024optionorder], symbol-identity effects and binding fixes
[yang2025option, xue2024symbolbinding], first-token probabilities that disagree with the generated
answer [wang2024firsttoken], scoring-format effects [robinson2023mcp], and the choices-only control
showing how much of MCQ accuracy needs no question at all [balepur2024artifacts] — the control we
adopt as our primary metric. On depth, deeper layers are partly redundant [gromov2024deeperlayers,
men2024shortgpt] and early exit is an established efficiency lever [schwartz2020righttool].
*Positioning: prior work observes the latent decision and diagnoses the symbol interface; we turn the
readout into the product interface — a calibrated distribution over runtime-supplied labels plus
abstain — and measure what survives when generation is removed, including that the last layer is the
wrong tap and that the readout must be over contextual option spans, not slot-agnostic embeddings.*

## 3. Calibration of LLMs and of fine-tuned classifiers

Modern networks are miscalibrated and temperature scaling largely fixes in-distribution ECE
[guo2017calibration]; binned ECE is itself biased [naeini2015calibration, kumar2019verifiedcalibration],
so proper scoring rules are the honest target [gneiting2007scoringrules, brier1950verification], with
the ranked probability score as the ordinal-aware member [epstein1969rps, murphy1970rps]. Training-time
levers include label smoothing [szegedy2016labelsmoothing, mueller2019labelsmoothing] and focal loss
[lin2017focal, mukhoti2020focalcalibration]; Bayesian LoRA adds posterior uncertainty to adapters
[yang2024laplacelora]. For language models specifically: pretrained transformers calibrate well but
degrade out of distribution [desai2020calibration], QA calibration needs explicit work
[jiang2021calibration], models can verbalise or self-assess confidence [tian2023verbalized,
kadavath2022know], prompt-side priors distort label probabilities [zhao2021calibrate], and fine-tuning
interacts with prior knowledge to destroy calibration — the CogCalib line [wang2025cogcalib].
Distribution-free guarantees come from conformal and selective prediction
[angelopoulos2023conformal, romano2020adaptivecoverage], including for multiple-choice QA
[kumar2023conformalmcq]. *Positioning: our calibration claims are training-time, not post-hoc — soft
and ordinal-smoothed targets, and checkpoint selection on a held-out calibration metric (no run trains
a Brier term in the objective; `--brier_lambda` is 0.0 in every shipped/candidate recipe); we report no
global temperature fit.*

## 4. Distillation, context distillation, and calibration preservation

Soft-target distillation transfers more than argmax [hinton2015distillation, kim2016seqkd], and the
smoothing view explains why it improves probability quality even without a stronger teacher
[yuan2020teacherfree, zhang2020selfdistillation]. Context distillation compiles a prompt or system
context into weights [askell2021assistant, snell2022contextdistillation], and self-distillation from
the model's own outputs narrows the fine-tuning distribution gap [yang2024sdft]. *Positioning: our KD
teacher is the same frozen backbone read with a fixed few-shot rendering, so the relevant question is
componential — which parts of a teacher's behaviour survive a change of interface. Our answer (priors
and calibration yes, question-conditioned knowledge no) is the negative half of this literature.*

## 5. Prefix/KV sharing and inference efficiency

Sharing a long prefix across many suffixes is well-explored systems work: Hydragen
[juravsky2024hydragen], RadixAttention in SGLang [zheng2023sglang], PagedAttention [kwon2023pagedattention],
modular prompt-KV reuse [gim2023promptcache], plus the analysis of which regime inference is actually
bound by [pope2023scaling, dao2022flashattention]; batch prompting achieves a crude version at the API
level [cheng2023batch]. The generation-side accelerators are the informative contrast: speculative
decoding and Medusa exist because each sequential decode step is overhead- and bandwidth-bound rather
than compute-bound [leviathan2023speculative, chen2023speculativesampling, cai2024medusa]. Structured
output for software consumers is normally obtained by constraining generation
[geng2023grammarconstrained, willard2023guided, beurerkellner2024domino], which itself costs accuracy
[tam2024speakfreely]. *Positioning: we contribute no kernel. We contribute the measured consequence of
deleting the decode loop: with a cached state prefix, per-decision latency is nearly flat in backbone
size (45 to 60 ms from 1.7B to 14B), i.e. fixed launch overhead, not FLOPs, sets the price.*

## 6. Is generation necessary? CoT expressivity and latent reasoning

Chain-of-thought buys serial computation that a constant-depth transformer provably lacks in one pass
[wei2022cot, li2024cotserial, merrill2024expressive, merrill2023parallelism, feng2023chain], and
empirically it helps mainly on math and symbolic tasks [sprague2025cot]. Filler and pause tokens show
part of the gain is extra compute rather than verbalised reasoning [goyal2024pause, pfau2024filler],
and implicit/latent CoT internalises the steps [deng2023implicitcot, deng2024internalize, hao2024coconut].
*Positioning: this literature delimits our scope. A single non-generative forward pass is the right
instrument for bounded, evidence- or prior-driven decisions, and is predicted to fail on serial
composition — which is exactly where our hard tier (temporal, probability, trade-off families) still
sits near chance.*

## 7. Ordinal regression heads

Cumulative-link models [mccullagh1980ordinal], ordinal-to-binary decomposition [frank2001ordinal,
niu2016ordinalcnn], rank-consistent variants [cao2020coral, shi2023corn], and soft ordinal targets
[diaz2019softordinal]. *Positioning: our Score primitive is the minimal member of this family — an
ordinal-smoothed target over the rendered levels that changes no decision and fixes probability
quality; a cumulative-link head did not win at matched budget and stays an open follow-up.*

## 8. Abstention and selective classification

The reject option is classical [chow1970reject], with deep-network instantiations
[geifman2017selective, geifman2019selectivenet], confidence baselines [hendrycks2017msp] and
energy-based scores [liu2020energyood]; out-of-scope intent detection is the applied benchmark
[larson2019clinc, casanueva2020banking77]. Abstention in LLMs is now surveyed [wen2024abstention], with
self-evaluation [chen2023selfevalselective] and conformal coverage [kumar2023conformalmcq] as the two
main mechanisms. *Positioning: our abstain option is architectural — an extra logit inside the same
softmax over runtime candidates, trained on gold-absent rows — and our open phenomenon is that its
discrimination degrades with candidate-set cardinality even at controlled semantic difficulty.*

## 9. Long context, evidence position, and silent data failures

Retrieval position inside the context changes answers [liu2024lostmiddle], length alone degrades
reasoning [levy2024moretokens], irrelevant context distracts [shi2023distracted], and synthetic
long-context evaluation exposes effective-length limits [hsieh2024ruler]; data-pipeline failures
propagate silently through ML systems [sambasivan2021datacascades]. *Positioning: our truncation
finding is the training-side analogue — where the facts sit in the render interacts with the fixed
training window, so a length ablation cannot see it, and the resulting model is confidently wrong
rather than uncertain on long states.*

## 10. Fine-tuning versus frozen backbones with in-context examples

Fine-tuning distorts pretrained features and can underperform out of distribution
[kumar2022finetunedistort]; fair comparisons of few-shot fine-tuning against ICL are close
[mosbach2023fewshot]; LoRA both learns and forgets less [biderman2024loralearnsless]; continual
fine-tuning forgets [luo2025forgetting]; and the superficial-alignment line argues most of what tuning
adds is format, recoverable in-context [zhou2023lima, lin2024urial]. Instruction tuning generalises
across task formats [wei2022flan, sanh2022t0, chung2024flant5], while compositional-generalisation
benchmarks show where learned grammars stop [lake2018scan, keysers2020cfq, kim2020cogs,
hupkes2020compositionality]. *Positioning: two of our findings live here — rubric-conditioned training
transfers inside its rule grammar and not to unseen composition, and our trained 14B beats its frozen
self on the standard tier while losing to frozen-plus-three-exemplars on the hard tier.*

## 11. The System-One product line, its open replications, and JevBench

The commercial System-One line (TypeSafe's Jev) is documented only by vendor pages and a third-party
black-box probe; the vendor names an RL-for-calibrated-decisions objective without describing it, and
the probe infers shared-state encoding, isolated question branches and direct probability readouts
from token-accounting and latency evidence. JevBench is a community harness of typed decisions
(`choice` / `score` / `noul`) with a public subset and held-out judge items; its leaderboard carries
open replications (SemIf/OpenJev, djev, system-one and system-one-open, Laya, jeff,
open-jev-deberta-v3-large, open-alternative-jev) that publish weights and scores but no method
papers. The only entries with a documented method are GLiNER2 [zaratiana2025gliner2] and the
ModernBERT backbone used by Laya [warner2024modernbert]. *Positioning: we are the first open,
end-to-end account of this architecture class with matched ablations — and our frozen-backbone
rendering control (the same Qwen3.5-9B checkpoint moving .806 to .931 standard by rendering alone,
with no training) is evidence that a large share of published leaderboard standard-tier numbers is a
protocol effect rather than a model capability.*

## Benchmarks and backbones

Reported sets and models: MMLU and MMLU-Pro [hendrycks2021mmlu, wang2024mmlupro], TruthfulQA
[lin2022truthfulqa], SNLI/MNLI/ANLI/ChaosNLI/UNLI [bowman2015snli, williams2018mnli, nie2020anli,
nie2020chaosnli, chen2020unli], BoolQ [clark2019boolq], CLINC-150 with OOS and Banking77/HWU64
[larson2019clinc, casanueva2020banking77]; encoder references [devlin2019bert, liu2019roberta,
he2023debertav3]; decoder backbones and adapters [vaswani2017attention, brown2020gpt3,
ouyang2022instructgpt, hu2022lora, yang2025qwen3].
