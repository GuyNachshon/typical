# PCDM and MMLU-Pro-class tasks — adversarial memo (2026-09-19)

Scope: no code changed; reasoning from `REPORT.md` §1/§3c–§3f, `COMPARE.md`, `IDEA2.md`, `model.py`, `encode.py`,
`baselines.py`, `mcq.py`, `scripts/mmlu_pro_eval.py`. Budget: ~$40 H100, two weeks. Constraints kept: state encoded once,
per-query cost K-independent-ish, direct calibrated probabilities with a learned null, no generation.

## 0. Three facts that reframe the question

1. **Jev's 84.6 is not a reachable target at any size we can rent.** 84.6 on MMLU-Pro is frontier non-thinking territory
   (Qwen3-235B-A22B-Instruct-2507 ≈ 83, GPT-4o ≈ 73 with CoT). With ~10B active that requires a very large total MoE plus
   instruction tuning, or contamination. [AH]'s own probes (fresh 3-digit multiplication 86.7%, 2-step word problems 32%)
   are hard to square with 84.6 on a benchmark where ~40% of items (math/physics/chem/engineering) need multi-step
   computation. Cheap check, $0: per-category accuracy from the [AH] evidence bundle. Frontier non-thinking models show a
   15–25 pt gap between computation and knowledge categories; if Jev shows < 8 pts, or if accuracy on paraphrased stems
   drops > 10 pts, the number is memorised. We should not design a research programme around a number we haven't stress-tested.
2. **The honest ceiling for any PCDM is the backbone's own non-generative (direct-answer) accuracy.** A head cannot add
   knowledge; it can only fail to extract it. Qwen3-1.7B-Base is ~30–35 on MMLU-Pro *with 5-shot CoT* (Qwen3 tech report,
   approximate); direct-answer scores run well below CoT on MMLU-Pro (Wang et al. 2024 report CoT helps on MMLU-Pro, unlike
   MMLU). Expect the 1.7B direct-answer ceiling at ~18–25%. The metric that matters is **retention = PCDM / backbone-ceiling**,
   not absolute accuracy. Absolute accuracy is bought with parameters, which is exactly the axis that kills "16–73× cheaper".
3. **MMLU-Pro is one question per state.** PCDM's economics (encode once, M queries) do not apply to it at all. Even a
   perfect MMLU-Pro score demonstrates nothing the product sells. It is a diagnostic of knowledge extraction, not a demo.

Why we are at chance (9%, among-K 13%): `among-K` is also chance, so the null (reason 3 in §3f) is not the story. The
story is the candidate interface: `DecisionModel.forward` scores `h` (mean of projected query tokens, from the suffix
"Which option is correct?") against `Qwen3-Embedding-0.6B(option text)` through an MLP. That is a bi-encoder. MMLU-Pro
options are CoT-generated hard negatives: ten topically identical strings differing in one number or one clause. Bi-encoders
lose to cross-encoders precisely on hard negatives (BEIR, Thakur et al. 2021) — the LM never reads an option. Nothing about
backbone size, truncation, or training data changes that until the option text reaches the LM or the readout stops being
"does this option's topic match the question's topic".

## 1. The five obvious answers, adversarially

### (a) Bigger backbone (4B / 8B)

- **Mechanism of failure.** The head never reads options; a bigger backbone makes a better question embedding matched
  against the same context-free option vectors. The repo already has this lesson: 0.6B → 1.7B "bought nothing until the
  tap moved" (§2.4). 8B costs ~5× per training run (~$20) and ~5× per query, eats half the budget for one seed, and 8B-Base
  direct-answer is still ~40–45%, not 84. It also converts the pitch from "cheap decisions" into "a mid-size LLM with a
  calibration head".
- **Cheapest run ($0.5, no training).** The prompted `B_1.7B` log-prob baseline on the same 1,200 MMLU-Pro items (queued in
  §3f, never run) — that is the 1.7B ceiling. Do not train a 4B until the 1.7B retention ratio is known.
- **Rule.** Only after Step 1 (below) reaches retention ≥ 0.8 at 1.7B: one `joint` run on Qwen3-4B-Base ($10–12, tap at
  the same 71% depth = layer 26/36) plus `B_4B` ceiling ($1). Success: retention ≥ 0.75 at 4B and ECE ≤ .05 → "extraction
  scales with knowledge". Kill: retention < 0.6 → the interface loses more as there is more to lose; scaling the backbone is
  not the fix, and say so.

### (b) Distil teacher option-probabilities over MMLU auxiliary_train / ARC / OBQA

- **Mechanism of failure.** Two separate failures get conflated. (i) *Data*: nothing in v5 is knowledge QA, so the head
  never learned "answer-shaped" decisions — real, and fixable with hard labels for $4. (ii) *Teacher*: soft labels transfer
  the teacher's decision behaviour on the training distribution, not its facts; the student cannot represent what it
  never encoded. Worse for calibration: an 8B/32B teacher is ~near-one-hot on ARC/OBQA (little dark knowledge) and the
  student learns "be confident on items that look like these" without the knowledge — overconfidence on exactly the items
  it gets wrong. And the distribution is wrong: auxiliary_train/ARC/OBQA are 4-option grade-school science; MMLU-Pro is
  10-option with adversarial distractors. Distilling into a bi-encoder interface also changes nothing (see §0).
- **Cheapest run ($4).** Hard labels only: `data_v5` + ~25% knowledge-QA rows (MMLU auxiliary_train, ARC-E/C, OBQA, SciQ,
  CommonsenseQA; 4–10 options; 15% gold-dropped rows for ∅), `--init_from joint_emb_lw_v5`, 6k steps, eval on MMLU-Pro-1200 and
  ARC-C test plus the full 21-set regression suite. This is the *data* half of (b), and it is Step 1's K-independent arm.
- **Rule.** Success: MMLU-Pro ≥ 0.8 × `B_1.7B` ceiling and ARC-C ≥ 0.9 × the backbone's own ARC-C log-prob score, in-dist
  sets within seed floor (NLI ±0.7, large-K ±3). Kill for the data hypothesis: < 0.6 × ceiling on both → the interface, not
  data, is the ceiling. Teacher soft labels (open 8B log-probs over options on H100, ≈ $5 for 100k items — chat APIs do not
  return option probabilities) are run only if the hard-label arm has acc ≥ 0.8 × ceiling **and** ECE > .08; success =
  ECE halves with acc within 1 pt. Distillation is a calibration tool here, not a knowledge tool.

### (c) Stop truncating at layer 20

- **Mechanism of failure (of the *fix*).** Two pulls. For: factual attribute extraction happens late (logit-lens/tuned-lens
  answer tokens emerge in the last ~25% of layers; Geva et al. 2023, Meng et al. 2022) and layer 20/28 is 71% depth — so
  truncation plausibly removes the layers where the answer forms. Against: the "last layer = hypothesis-only" finding (§2.4)
  was measured with the *fresh tower*, on NLI; it was never re-tested under `--joint --tower_layers 0`, so the tap-20
  default is inherited, not chosen. Last-layer states are in next-token-logit geometry (anisotropic; `zscore` is off in the
  joint runs), and mean-pooling them over "Which option is correct?" gives a "next token after each query word" vector, not
  an answer vector. And un-truncating costs +40% FLOPs on both state and query. Also: with a bi-encoder readout, better
  answer formation still has to be matched against frozen option embeddings — (c) cannot rescue (a bad) interface either.
- **Cheapest run ($0.3, no training).** In Step 0, score `B_1.7B` option log-probs with the model truncated at 20 (two lines:
  `lm.model.layers = lm.model.layers[:20]`) vs full. Asymmetric read: if C_20 ≈ C_full, truncation is fine (layer-20 hidden
  through the untuned lm_head is a *lower* bound on what a trained head extracts). If C_20 ≪ C_full it is inconclusive and
  the trained arm decides.
- **Rule.** Run a tap-28 arm ($5) only if C_full − C_20 ≥ 5 pts. Success: tap-28 arm ≥ tap-20 arm + 3 on MMLU-Pro with NLI
  within floor; then also re-check `bench_fair` (per-query ms rises ~1.4×). Otherwise keep tap 20.

### (d) Feed the LM's own candidate log-probs as a head feature

- **Mechanism of failure.** Candidate log-probs require running every candidate as a continuation through the LM against
  the KV cache — that *is* the `B` baseline's cost (`score_example_kv`, K continuations), the thing we are 16–73× cheaper
  than. The result is "prompted LM + temperature + a learned null", with the economics thesis deleted. Not a head feature,
  a different model.
- **The K-independent version that is worth one run.** The next-token distribution at the suffix's last position is one
  softmax per query (`logp_first` in `baselines.py:265` already computes it; `lm_head` = `embed_tokens` for the tied 1.7B,
  see `mcq.py:34`). Feeding the *last-position* hidden state (the answer-start latent) to the scorer instead of / alongside
  the mean-pooled query tokens is a free change to `DecisionModel.forward` and fixes a real defect: the current `h` averages
  over the question words, which is fine for NLI/intents and wrong for "what is the answer". First-token log-probs of each
  option are a cheap, K-independent extra feature (gather K entries of one V-softmax) but weak on their own (options share
  first tokens: "The", digits).
- **Cheapest run.** Fold into the Step 1 K-independent arm (same $4): `h` ← concat(mean-pooled query, last-position state).
- **Rule.** Success/kill shared with (b). Do not run the full log-prob feature at all; if we want that model it already
  exists as `B_1.7B` + `fit_temperature`.

### (e) Latent multi-step suffix (pause tokens / looped depth)

- **Mechanism of failure.** Pause tokens help mainly when the model is pre-trained with them; fine-tuning-only gains are
  small and inconsistent (Goyal et al. 2024). Coconut (Hao et al. 2024) needs a CoT curriculum and was shown on GSM8K/ProsQA-
  scale synthetic tasks with GPT-2-size models; recurrent depth (Geiping et al. 2025) is a pretraining recipe — looping
  layers 13–20 of a model that was never trained to loop destroys the representation. A LoRA on 8 layers with $4 will not
  make a 1.7B model integrate a circuit or do a 2-step conversion latently. The economics survive (8–16 extra suffix
  tokens) — the capability does not appear. Jev's own 32% on 2-step word problems says the competitor did not solve this either.
- **Cheapest run ($4).** N = 8 learned pause tokens appended to the suffix, trained with the LoRA, on the Step 1 data;
  compare against the identical run without them, per MMLU-Pro category group.
- **Rule.** Success: ≥ +3 pts on the computation group (math/physics/chem/engineering) beyond the no-pause run, ≥ 2× the
  seed floor. Kill: < +2 → drop the idea for this budget class; the honest product boundary is "not a calculator". I would
  not spend the $4 before Steps 0–2 are done; it is last on the list.

## 2. The framing question: is "parametric knowledge" the right axis?

**No — but the alternative pitch has its own trap.**

Against the parametric axis: knowledge in weights is the one thing this model class is worst placed to sell — it scales with
parameters (kills "cheap"), it is unverifiable and stale, and it is exactly what a non-generative single-pass model cannot
compound with CoT. TypeSafe's own docs concede "quite literal" and "accuracy falls as the state grows with unrelated
content" — behaviour of an evidence reader, not a knowledge model. And §0.3: MMLU-Pro has no amortisation structure.

For "retrieval puts the knowledge in the state; we decide over it, calibrated and cheap": this is what PCDM has actually
demonstrated (NLI .91, BoolQ .83, 150-way intents .80, null AUROC .95–.98, ECE .005–.02, 16–73× cheaper). It also matches
the architecture's one genuine advantage — one long state, many questions — which is what a retrieved context with M
decisions looks like.

The trap: an open-book MMLU-Pro item is still a 10-way discrimination over hard-negative options. Putting the passage in
the state does **not** escape the bi-encoder problem in §0; the model must read the options *and* the passage. So the
retrieval framing is only testable after the interface is fixed (Step 1), and "we decide over evidence" is a claim about the
readout as much as about the state.

**Evidence that settles it (pre-registered).** Split the 1,200 MMLU-Pro items into a knowledge group (history, law,
philosophy, psychology, health, biology, business, economics, other) and a computation group (math, physics, chemistry,
engineering, CS). Three conditions on the same items with the Step 1 model: closed-book; BM25 top-3 Wikipedia passages in
the state (pyserini prebuilt `wikipedia-dpr-100w` index, CPU, $0); and an oracle passage (LLM-written ~150-word background
that states the governing fact/principle — deliberately leaky; this is the reading-comprehension ceiling).

| outcome | reading |
|---|---|
| knowledge group: Δ(oracle − closed) ≥ +25, Δ(BM25 − closed) ≥ +8, ECE ≤ .06; computation group Δ < 5 in both | framing holds: knowledge is a state property; publish the category split as the product boundary ("decides over evidence; not a calculator") |
| knowledge group: Δ(oracle) ≥ +25 but Δ(BM25) < +3 | model is fine, retrieval is the bottleneck; publish oracle as ceiling; retrieval is the customer's/integration's job |
| knowledge group: Δ(oracle) < +10 | PCDM cannot read evidence in this format even when it is handed to it → interface still broken; the framing is untested, go back to Step 1 |
| computation group: Δ(oracle) ≥ +15 | surprising; means "worked examples in state" substitute for latent computation — a real product story, replicate on a second seed before saying it |

The better public demonstrator of the actual pitch is not MMLU-Pro at all: ContractNLI (Koreeda & Manning 2021) — one
contract (1–2k tokens) as the state, 17 fixed hypotheses per contract, 3-way + calibrated — tests encode-once at a state
length we have never benchmarked (REPORT: nothing past 256 tokens) and is exactly "many decisions over one document".
Cost: one eval $0.5, one fine-tune $5. It belongs in the plan if the MMLU-Pro framing test comes out as predicted.

## 3. Ranked plan (~$40)

**Step 0 — decompose the 9% ($1, day 1–2, no training).** Same 1,200 items:
`B_1.7B` full (option-text log-prob, KV-cached, `baselines.py`) = C_full; `B_1.7B` truncated at layer 20 = C_20;
`mcq_lora` checkpoint (`--readout mcq --eval_only`, options in context, tap 20 + LoRA, v3 data) = M_20;
Jev per-category from the [AH] bundle. Decision tree:
- C_full < 15 → closed-book at 1.7B is dead; skip Step 1's closed-book targets, go to Step 2 (framing) directly, note the size axis is out of budget.
- C_full ≥ 20 and M_20 ≥ 0.8·C_20 → the loss is the candidate interface → Step 1 as written.
- C_full − C_20 ≥ 5 → add the tap-28 arm to Step 1.
- Jev's computation-vs-knowledge gap < 8 pts → treat 84.6 as suspect in every write-up.

**Step 1 — fix the readout, keep the null and the state cache ($13–18, week 1).** Data: `data_v5` + 25% knowledge-QA
hard-label rows (auxiliary_train/ARC/OBQA/SciQ/CSQA, 15% gold-dropped for ∅); all arms `--init_from joint_emb_lw_v5`, 6k steps.
- Arm K-indep ($4): existing head + last-position query state + first-token option log-prob feature. This is (b)+(d) and the
  measure of what K-independence costs on this task class.
- Arm span ($5): options rendered into the suffix after the query; candidate vector `c_j` = tap-layer mean of option j's
  span (extend `tokenize_joint`/`split_joint` with per-option offsets; `DecisionModel.forward` and `decide`'s KV path
  unchanged — the suffix is just longer). Listwise by construction, IIA broken as in Jev, cost O(K·L_opt) suffix tokens —
  Jev's cost model — still one pass, no generation, no option-letter symbol binding. Keep the cached-embedding interface
  for large-K label spaces; the scorer is the same module fed from either source.
- Control ($4): `--readout mcq` on the same data — the letter-readout "Qwen does everything" comparison with ∅ letter + T.
- Conditional ($5): span arm at tap 28.
Rules: success = MMLU-Pro ≥ 0.85·C_20 (0.85·C_full for tap 28) with post-T ECE ≤ .05 and *raw* ECE reported; ARC-C ≥ 0.9×
backbone log-prob; 21-set regression within seed floor. Kill = < 0.6·C_20 on both arms → the energy head cannot extract
what the log-probs show; ship MCQ-LoRA-with-null as the knowledge readout and stop.
Expected: span ≫ K-indep on MMLU-Pro (hard negatives), ≈ on intents; K-indep ≥ current 9% by ≥ 5 pts from the data alone.

**Step 2 — the framing test ($6–7, week 2).** Build `mmlu_pro_openbook_{bm25,oracle}.jsonl` (state = passages + stem),
evaluate the Step 1 winner as-is ($1), then one fine-tune with 20% open-book synthetic rows (SQuAD/BoolQ-style passages
with MCQ options, $5). Read against the §2 table. Then ContractNLI eval ($0.5) as the encode-once demonstrator if the
framing holds.

**Step 3 — replicate and scale, only if green ($15).** Second seed of the Step 1 winner ($5, every claim on unseen spaces
needs it) and the single 4B retention run from (a) ($10–12). Teacher soft-label distillation and pause tokens only from
leftover money, in that order, with the rules above.

Total: $1 + $13–18 + $7 + $15 ≈ $36–41.

## 4. Verdict

The 9% is not a knowledge problem first; it is a readout problem: a K-independent bi-encoder over frozen option embeddings
cannot discriminate MMLU-Pro's hard-negative options no matter how much the backbone knows, so (a), (c) and teacher
distillation all fail for the same reason and (d)-as-posed and (e) fail on cost and on the literature respectively. The
cheapest decisive experiment is $1 of evals that already exist (`B_1.7B` full/truncated, `mcq_lora` on the 1,200 items),
and the one design change worth $5 is letting option text into the suffix while keeping the state cache, the learned null
and the calibration — which is also Jev's cost model, so "K-independent" should be softened to "K-independent for label
spaces, O(option tokens) for decisions" rather than defended. Chasing 84.6 closed-book is a parameter-buying race that
erases the cost advantage and whose target may be contaminated; the honest scoreboard is retention of the backbone's own
direct-answer ceiling, calibrated, plus the open-book category split, which if it comes out as predicted *is* the pitch:
the knowledge is in the state, PCDM decides over it, calibrated and cheap, and it is not a calculator.

## References (all exist; numbers marked approximate are from memory of the cited reports)
Wang et al. 2024, MMLU-Pro (NeurIPS D&B) · Qwen3 Technical Report 2025 · Thakur et al. 2021, BEIR · Robinson & Wingate 2023,
multiple-choice symbol binding · Goyal et al. 2024, pause tokens (ICLR) · Hao et al. 2024, Coconut · Geiping et al. 2025,
recurrent-depth latent reasoning · Belrose et al. 2023, tuned lens · Geva et al. 2023, dissecting factual recall ·
Meng et al. 2022, ROME · Hinton et al. 2015, distillation · Kadavath et al. 2022, LMs (mostly) know what they know ·
Lewis et al. 2020, RAG · Lin et al. 2021, Pyserini · Koreeda & Manning 2021, ContractNLI · [AH] Archer Hume 2026, Jev evidence bundle.
