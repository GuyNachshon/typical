---
title: "We Removed Generation from an LLM. Here's What Broke."
date: 2026-09-22
---

# We Removed Generation from an LLM. Here's What Broke.

[Typical](./typical-launch.md) is a family of open decision models. You give one a state and a typed question, and it returns a probability distribution over options you defined at runtime. No tokens are generated and nothing is parsed.

That interface sounds like a simplification of a language model. In practice, removing generation removed a lot of scaffolding that was doing load-bearing work, and most of what we learned came from finding out which scaffolding mattered.

This is the archaeology: the architecture that didn't work, the bugs that looked like results, and the results that looked like bugs. Section references (§3j, §3t and so on) point at our internal experiment report, which publishes with the training code.

One measurement caveat applies throughout. Our JevBench runs are against the public subset, unranked: 72 standard items that come from only 36 independent states (each state appears as two paraphrases), and 111 hard items. Cluster-bootstrapped, that puts roughly ±9 points on any hard-tier number. Where a claim below rests on a small JevBench gap we say so.

## 1. The obvious architecture failed

The first design was the clean one. Compute a decision state `Z(state, question)` once, without looking at the candidates, then score each candidate against it with a cheap head. Candidates are then free: their cost is a dot product, K scales for nothing, and you can cache label vectors forever.

We built three versions of it and distilled a listwise teacher into them: a single decision vector, eight probes with token-level MaxSim, and probes plus O(K) cross-attention over the candidate tokens. On MMLU-Pro among-K they scored .171, .157 and .147, against .310 for the teacher. That looked like partial success. About half the teacher's score had transferred.

Then we ran the control that killed it (§3j). We re-scored every student with the real question replaced by a shuffled, unrelated one, holding the candidate set fixed. The gap between the two is Δ_sh, and it is the only number in the table that says whether the question is doing any work:

| among-K | normal | shuffled question | Δ_sh |
|---|---|---|---|
| teacher (options in context) | .310 | .208 | **+.102** |
| `z1` student | .171 | .147 | +.023 |
| `zr` student | .157 | .182 | −.026 |
| `zr_set` student | .147 | .151 | −.003 |
| multi-set + Δ-log-odds student | .130 | .130 | +.000 |
| `zr` at full depth (28/28) | .147 | .156 | −.008 |

The teacher gains a tenth of a point of accuracy from being told what the question is. The students gain between −.026 and +.023, in both directions, at 1,200 items — which is to say nothing we can distinguish from zero, and nothing like the teacher. Their whole above-chance score was candidate-set priors: which option strings look plausible together, learned from the corpus.

That result held against every lever we had. Head capacity is not the bottleneck — a single decision vector retains as much as eight probes. Set conditioning is not — O(K) cross-attention over the candidate tokens changes nothing on this axis. Supervision is not: multi-set training, where the same state appears with seven different option sets and the loss targets the teacher's Δ-log-odds between them, produced Δ_sh of exactly 0.000.

One note on how we first got this wrong, because it is the kind of mistake that is easy to repeat. Our original control replaced the question with a period rather than with a different question, and by that measure every student scored *better* with the question destroyed. The shuffled control is the right one — it holds the amount of question-shaped text fixed and varies only whether that text is the relevant question — and it moves the individual students around by a couple of points. It does not change the finding, and we report it because the weaker control is what we ran first.

## 2. Candidates have to participate in the computation

The suspicion at that point was depth: every candidate-blind student read the backbone at layer 20 of 28, and maybe the last eight layers were where question-conditioned knowledge lived. So we re-ran the candidate-blind student at full depth (§3s). Among-K *fell*, .157 to .147, and Δ_q(shuffled) stayed at −.008 while the evidence tasks regressed hard (BoolQ −11, ANLI −8, MNLI −7).

That closed it. What transfers into a state computed before the options are known is priors and calibration, not question-conditioned parametric knowledge, at any tap depth.

The architecture that works renders the options into the suffix, so the candidate text goes through the same pretrained layers as the question. The released models built on it carry Δ_q(shuffled) of .127 at 1.7B and .193 at 4B: real, measurable, question-dependent behaviour, from the same backbone that couldn't produce any when the options arrived after the fact.

There's a much dumber version of the same lesson from earlier in the project. Qwen3 has no BOS token, so when we embedded a single-token candidate on its own, that token landed at position 0 — the attention-sink position, where the residual stream carries activations orders of magnitude larger than anything token identity contributes ([Sun et al., 2024](https://arxiv.org/abs/2402.17762)). The cosine similarity between the position-0 states of `neutral`, `yes` and `transfer` was 0.9998. The state does depend on its token; it is just that the sink swamps the dependence. Every single-token candidate we embedded was, for scoring purposes, the same vector. Prepending `<|endoftext|>` and dropping it fixed it.

## 3. The final layer was the wrong layer

The first full-scale version read the backbone's last hidden layer, which is what you'd do without thinking about it. It scored 58.6 on SNLI, and the number alone told us nothing except that it was bad.

What told us what was wrong was a control we should have been running from the start: blank the premise and re-score. A model that has learned entailment should collapse. This family of models did not. At SNLI .701 it scored **.648 with the premise blanked** — the premise was worth five points. It was reading the hypothesis for the well-known SNLI annotation artefacts and ignoring the evidence, which is the classic dataset-artifact failure ([Gururangan et al., 2018](https://arxiv.org/abs/1803.02324)). Our decision head had been handed a state already specialised for next-token prediction, and whatever cross-sentence evidence structure existed in the middle of the network was gone by the top.

The premise-blanked score is not itself the diagnostic — a hypothesis-only classifier trained on SNLI reaches the high 60s, so .648 in isolation is unremarkable. The **gap** is the diagnostic, and ours was five points. Once the state and the query interacted inside the pretrained layers rather than above them, the same probe read .440 against .910 with the premise: a gap of forty-seven.

Tapping at layer 20 of 28 took SNLI to 65.3 frozen and 68.2 with LoRA, and the NLI family loss started descending for the first time. Backbone size had been a red herring the whole time: going from 0.6B to 1.7B bought nothing at all until the tap moved.

The same effect shows up later in the option-conditioned architecture (§3r). Same head, same data, same steps, only the tap depth changed:

| | native readout @ 28 layers | native readout @ tap 20 |
|---|---|---|
| SNLI / MNLI / ANLI / BoolQ | .863 / .752 / .431 / .737 | **.906 / .865 / .519 / .834** |
| CLINC-150 | .740 | **.862** |
| best val NLL | .430 | **.341** |

We later swept tap depth again against the current recipe (tap 15, 17 and 18 of 28) and found nothing that clears the seed-to-seed spread we'd measured, so the parameter stays where it is (§3aj). The finding is narrower than "shallower is better": the top of a generative stack is specialised for generating, and a decision head wants a representation from before that specialisation.

## 4. "None of the above" broke the model

Every decision model needs an escape hatch. The obvious implementation is to add "none of the above" to the candidate list and let the softmax handle it.

It produced one of the strangest failure modes in the project. The abstention rate became a pure function of K — of how many options you happened to pass, not of whether the right one was among them.

The number that shows it is P(∅ | the gold answer is absent): how often the model correctly abstains when it should. Sweeping K on CLINC with the gold label removed:

| P(∅ \| gold absent), CLINC | K=5 | K=50 | K=150 |
|---|---|---|---|
| rendered ∅ as an option | .02 | .61 | **1.00** |
| ∅ as a head decision | .89 | .67 | .64 |

The top row is the pathology in one line. At K=5 the model essentially never abstained; at K=150 it always did. Not "abstained more often" — *always*, on every item, which is a model that has stopped answering. On Banking77 at K=77 the same collapse drove null AUROC to .00.

We looked at AUROC first and nearly missed it. On the same sweep it reads .97 / .91 / 1.00 for the broken version against .98 / .92 / .95 for the fixed one — a rounding error at small K, and *better* for the broken one at K=150, because a model that abstains on everything ranks every positive above every negative. AUROC measures whether the score separates. It cannot see that the score's meaning has moved.

Downstream, the effects were not subtle. MMLU-Pro false-abstention was .692, dropping accuracy to .133. Banking77 with all 77 labels scored .063.

Moving abstention out of the candidate list and into a separate head, computed independently of the softmax over the real options, fixed all of it at once (§3t). MMLU-Pro false-abstention went .692 to .003 with accuracy .133 to .353, KL to the listwise teacher went 2.43 to .27, and Banking77-77 went .063 to .579. It cost 2 to 7 points on intent and topic label spaces (HWU64 .801 to .757, 20NG .589 to .515) and raised option-order sensitivity.

The practical version: if you have added a "none of these" option to any classifier, check whether it behaves the same way at K=5 and K=150. A null score that means something different at every candidate count cannot be thresholded, and a threshold is the entire point.

## 5. We trained on long documents after silently deleting the evidence

Our long-policy training corpus, 23k rows, renders the case facts at the end of the text: policy document first, then `Case: <facts> <request>`.

State length in that corpus runs p10/p50/p90 = 1,190 / 1,845 / 2,490 tokens. The training pipeline right-truncated states at `--max_state`. At the 1,024-token window Release 1 used, **98.8% of those rows lost their facts before the model ever saw them**. At the 256-token window the earlier scaling ladder used, all of them did.

So for a stretch of this project we were training models to answer confidently about text that had been cut off. The symptom was visible in the metrics, if you knew to read it: long-document policy accuracy of .05 for the 14B ladder run and .21 for `typical-medium`, with high confidence attached.

The fixes were to regenerate the corpus facts-first (case position inside the first 8% of the text), and to stop truncating. `--drop_truncated` discards any row that doesn't fit the window rather than quietly cutting it, which is the right default for a training loader handling anything with a payload at a known position.

**Then we tried to prove it mattered, and for two days we had it backwards.** The data defect is verified and not in dispute: we can count the rows, 98.8% is a measurement, and we confirmed empirically that the tokenizer path keeps the start of a state and drops the end. What needed establishing is that the defect *caused* the metric movement we saw afterwards. The run that fixed it changed five things at once (facts-first rendering, `--drop_truncated`, a wider window, a Brier term, calibration-based checkpoint selection), so the long-policy recovery had five candidate explanations and we had separated none of them.

The counterfactual corpus was still on disk, so the ablation was clean to set up: two 1.7B arms, byte-identical flags, a 1,024-token window with `--drop_truncated` deliberately off so truncation bites exactly as it did in Release 1, differing only in which render of the same rows is the training file. Same rows, same labels, same state lengths, `Case:` at the start or at the end.

On JevBench it came back a null. Long-policy accuracy was .316 for the facts-first arm against .105 for facts-last — the predicted direction, but 6 of 19 items against 2 of 19. Fisher's exact test gives two-sided p = 0.232, the bootstrap interval on the difference contains zero, and the hard-tier aggregate goes the other way (.378 against .396). We wrote that up as "consistent with the mechanism, underpowered to confirm it."

Then we scored both arms on a purpose-built held-out set: 605 long states, no overlap with either training corpus, a 4,096-token window so nothing is truncated at eval. Our first pass only had the facts-last arm, and it showed something striking — .830 when the case sits at the end of the state where its training put it, .640 when the same 605 items put the case at the start, and on the two-way `policy_permit` subset (n = 330) .842 against .615 against a majority-class floor of .612. Moving the facts to a position the model wasn't trained to look at cost 23 points and landed it exactly on the floor.

That looked like the answer, and we drafted it as the answer: the mechanism is train/test render mismatch, and truncation is a side issue. It was the wrong conclusion, for a reason that is easy to miss. Scoring each arm only in the render its own training used confounds the variable under test with render compatibility. The facts-first arm's cells landed a day later and completed the 2×2:

| trained on | scored on | overall | `policy_permit` (floor .612) | `action_select` (floor .233) |
|---|---|---|---|---|
| facts-first | facts-first *(matched)* | **.942** | **.933** | **.953** |
| facts-first | facts-last | .797 | .788 | .807 |
| facts-last | facts-first | .640 | .615 *(at floor)* | .669 |
| facts-last | facts-last *(matched)* | .830 | .842 | .815 |

Compare each model in its own matched condition — the diagonal, which removes the render confound entirely — and facts-first training is ahead by **11.2 points**: .942 against .830, 95% CI [+.077, +.147], p = 5e-10. The facts-first model is also the more robust of the two, losing 14.5 points when the render is switched against it against the facts-last model's 19.0.

So three effects were stacked on top of each other, and only the full design separates them. The truncation fix works, unambiguously at 605 items. Render mismatch is separately real and large, and it is what dragged the facts-last arm to its floor in the half-finished comparison. And JevBench's long-policy subfamily, at 19 items, could not see either one: it returned p = 0.232 on a mechanism that a better-powered measurement puts at p = 5e-10.

We are publishing the sequence and not just the endpoint, because the intermediate state is the part that generalises. We had a pre-registered primary metric, it came back null, and the null was an artefact of nineteen items and a single-render design — not of the mechanism. If you have run a matched ablation where each arm is evaluated in the condition its own training assumed, you have not run a matched ablation.

<p align="center"><img src="../figures/fig_truncation.png" alt="Left: state token length distribution against truncation cutoffs at 256/1024/2048/3072 tokens. Right: long-policy accuracy across checkpoints" width="640"></p>
<p align="center"><em>Left: how much of a long-policy row survives each token budget. Right: long-policy accuracy across checkpoints.</em></p>

## 6. Training distribution moved capability as much as architecture

It's tempting to write this project as an architecture story, because architecture is what's interesting. The mixture sweeps say otherwise (§3z, §3ad).

Raising the evidence share of the training mixture from .35 to .45–.50 brought NLI and BoolQ back to within a point of the untrained base model and intent and topic sets to within 3–4 points, with workflow performance unchanged. No architecture changed.

Augmenting 20% of workflow rows with the gold answer removed, so the correct answer is ∅, lowered false-abstention on CLINC from .10 to .08 and on TREC from .27 to .12, and *raised* accuracy on CLINC, TREC, HWU64 and 20NG by 2.6, 3.6, 4.3 and 6.0 points. Teaching the model when to abstain made it better at not abstaining.

The hard curriculum is the sharpest case. Adding a rule-engine corpus where the same state and options recur under different rubrics with different gold answers took held-out family, grammar and style accuracy from .481 / .507 / .491 to .827 / .881 / .888, and rubric-flip accuracy from .477 to .710. On JevBench's hard tier, the families that corpus covers moved in the same direction, though those are 6 and 7 items respectively and we'd treat them as directional: adversarial .33 to .83, ambiguous .57 to .71.

And level-7 composition, which mixes temporal, unit, expected-value and trade-off reasoning and which the generator never produces, went .477 to .498. What we generated was learned. What we didn't generate was not, and no amount of scale substituted for it.

There is a fourth result in this family that we found unwelcome, and that we read wrong the first time. Rerunning the Release-1 candidate at batch 16 instead of 64 — simply 4× fewer examples seen — improved the hard tier on both axes at once: accuracy .369 to .450, Brier .90 to .76. It lost ground on everything in-distribution (CLINC .733, MMLU among-K .323, held-out noul .609, validation NLL up).

We first wrote this up as evidence that the hard tier is a probability-quality problem — the same model, less confident, scoring better. That explanation doesn't survive contact with the accuracy column. Argmax accuracy is invariant to how sharp the distribution is; you cannot move it by being less confident. If fitting our mixture *less* makes the model pick the right answer more often on the hard tier, then fitting it is damaging something the hard tier needs, and calibration is a second, separate effect on top.

Which is the same shape as §8 below, where a backbone we didn't train at all does better still. Under-fit beats fit; untrained beats under-fit. That ordering is the project's central unresolved problem, and it points at our training distribution rather than at the model's confidence.

## 7. A loss change bought calibration; it was not free

Score is an ordinal primitive: severity 0 through 3, priority 1 through 5. A K-way softmax treats those levels as unrelated buckets, so predicting 0 when the answer is 1 costs exactly what predicting 0 when the answer is 3 costs.

Training the same head with ordinal-smoothed targets (τ = 0.7), which is a loss change and no new parameters, gave this (§3ac). The held-out urgency set is K = 4 on every item, so we've printed what a model that knows nothing would score:

| held-out urgency (K = 4) | uniform predictor | K-way | ordinal-smoothed |
|---|---|---|---|
| accuracy | .25 | .505 | .508 |
| NLL | 1.39 | 2.07 | **1.23** |
| Brier | .75 | .79 | **.68** |
| ECE | – | .35 | **.19** |
| ordinal MAE | – | .58 | **.55** |

That reference column is not decoration. The K-way model's probabilities were *worse than uniform* on both NLL and Brier — it picked the right level half the time and the numbers it attached were actively misleading. Ordinal smoothing is what moved them to the useful side of the line, and even then 1.23 against 1.39 is a modest margin. We print ln K next to every NLL in the report for this reason, and we'd suggest anyone reporting an NLL do the same; without it, 2.07 and 1.23 look like two points on the same scale rather than opposite sides of "knows nothing."

Accuracy barely moved, and on the 12 ordinal items in JevBench the two models produce *identical per-item predictions*. That is the claim we can make narrowly, and it is the one we originally over-generalised into a section heading. Across the rest of the same ablation the change did move decisions, and mostly for the better: an external severity set went .853 to .939 in accuracy, JevBench hard went .360 to .459, and JevBench standard went the other way, .764 to .708. "Improved probabilities, identical decisions" is true of twelve ordinal items and false of the run as a whole.

The same loss change scaled. At 14B, the recipe with the long-state fix, a Brier term and calibration-based checkpoint selection took held-out score NLL from 2.87 to 0.95 (against the same ln 4 = 1.39 reference: from far worse than uniform to meaningfully better) and typed-decisions NLL from 1.96 to 1.04, with JevBench hard Brier .85 to .66 (§3ah).

The Noul head is a structural version of the same idea. A two-way Choice over `["no", "yes"]` moves P(yes) by up to .55 when you reverse the label order. A Bernoulli head reads the decision state with no candidates rendered at all, so there is no order to reverse: in the matched ablation it moved by exactly zero on all nine test sets. It also cleared an external floor the two-way head couldn't: PagerDuty .602 to .886, against a constant-prediction floor of .792.

One caveat on "by construction," because it holds for the head and not automatically for a call you make. Noul routes per row: a question goes to the Bernoulli head only if its candidate set is exactly `{yes, no}`. Anything else two-way — `{true, false}`, `{approve, reject}` — is scored as ordinary Choice and keeps Choice's order sensitivity. On the eleven held-out sets we ran against the shipped `typical-small`, ten are exactly zero and one moves by .009, and that one is the set that isn't literally yes/no. The invariance is a property of the head, so use `noul()` and let it render the proposition rather than passing your own two labels.

<p align="center"><img src="../figures/fig_calibration.png" alt="Held-out score NLL and typed-decisions NLL across checkpoints" width="640"></p>
<p align="center"><em>Held-out score and typed-decisions NLL across checkpoints.</em></p>

One thing we added and then couldn't justify. The 14B run bundled knowledge distillation from a frozen 14B teacher, on the theory that it would buy hard-tier reasoning. We ran the matched control with `--distill_beta 0`, one flag different and nothing else (§3ai). The control came out slightly ahead: hard accuracy .477 against .450, long-document policy .211 against .158, standard Brier .127 against .175, validation NLL 0.410 against 0.438, at the cost of 1.4 points of standard accuracy.

The direction is consistent across those metrics, and it is still not a result. Cluster-bootstrapped, the hard-tier numbers are .450 [.360, .541] with KD and .477 [.387, .568] without, intervals that overlap along almost their whole length. We cannot show the teacher helped and we cannot show it hurt; 111 items is not enough to tell. What we can say is that the change we added specifically to buy hard-tier reasoning bought nothing we can detect, which is reason enough not to credit it for the calibration gains above. Those belong to the loss and the data.

## 8. Fine-tuning made the hardest cases worse

The uncomfortable number in this project: on JevBench's hard tier, a frozen 14B base model reading letter logits with three examples in its prompt scores .559. Our trained 14B decision checkpoint scores .450, and its no-KD control .477.

Our first draft of this section said the intervals overlap so we wouldn't call it a defeat. That was the wrong test, and we'd flag it in someone else's paper. The marginal intervals are [.468, .649] and [.360, .541]; differencing two marginal intervals by eye is not a comparison, and it throws away the fact that both models answered the *same 111 items*. The paired test is the one that applies. A paired cluster bootstrap on the per-item difference gives **frozen minus trained = +.108, 95% CI [+.027, +.189], p = .015**. Against the no-KD control it gives +.081 [−.009, +.171], p = .082 — consistent in direction, not separable.

So for the checkpoint we actually trained, the frozen backbone wins, and it is not a coin flip. On the standard tier the ordering reverses (−.111 [−.250, +.014], p = .10), which is what our training was for.

The per-family breakdown shows where it goes wrong, and it is not where we said it was. Here is the trained 14B against the frozen one on the same items:

| hard family | frozen 14B, 3-shot | `tl1b` |
|---|---|---|
| trap, routing | 1.00 | 1.00 |
| adversarial (n=6) | .833 | .667 |
| ambiguous (n=7) | .714 | .429 |
| judge-hard (n=17) | .588 | .529 |
| multi-hop (n=18) | .444 | **.611** |
| probability | .500 | .300 |
| long-policy (n=19) | .421 | .158 |
| temporal/numeric | .333 | .200 |
| trade-off | .500 | .167 |

We had been explaining the hard tier as missing data coverage: our generators never produce temporal arithmetic, unit conversion or expected-value trade-offs, so the model never learned them. That story predicts the frozen backbone fails these families too. It doesn't. It leads on every serial family in the table, and trails only on multi-hop. Individual cells are single digits and none of them separates alone, but the direction is uniform.

Missing coverage still explains why training didn't *add* the behaviour. It doesn't explain why training appears to have removed it. That distinction matters for what we build next, and we had it wrong until we looked at the columns side by side.

Two related findings sharpen it. First, the backbone generation appears to matter more than our training does here: a frozen Qwen3.5-9B scores .541 hard with three examples and .595 under a different rendering, above the .495 of the best checkpoint we've trained at any size, again with intervals that overlap. Second, that rendering effect is visible on its own. The same frozen Qwen3.5-9B goes from .806 to .931 on the standard tier purely by changing how the state and options are laid out, with zero training, and all three Qwen3.5 sizes we tested moved the same way under the same change. We'd read that as a sign that a meaningful share of the public leaderboard's standard-tier spread is a protocol effect, and as a reason to be careful reading ours.

The lesson we'd generalise: a frozen model with a few examples in context is a ceiling to verify you've cleared, per tier, and we publish the tiers where we haven't. One control we have *not* run and should have: the same frozen backbone with a scratchpad, on the same 111 items. Until that exists, nothing here distinguishes "this task needs intermediate computation" from "our fine-tuning damaged a backbone that already had some of it." We think it's the second. It's first on the list.

## 9. One deep copy cost 25–33% of serving latency

The serving path took a state, encoded it once into a KV cache, and then scored each question's suffix against that cache. Correct, and the whole point of the architecture.

It also deep-copied the entire prefix KV cache on every single decision, because mutating a shared cache is the kind of bug you only find in production and the copy made it impossible. Replacing the copy with a stride-0 view (bit-identical output, zero bytes allocated), plus caching the attention masks and position tensors and capping rendered option text:

| warm p50 per decision | before | after |
|---|---|---|
| `typical-small` (1.7B) | 21–25 ms | **15.5–17 ms** |
| `typical-medium` (4B) | 26–27 ms | **19–21 ms** |
| Qwen3.5-4B | 42–53 ms | 34–46 ms |

A quarter to a third of serving latency, for a change that removes code.

We also tried `torch.compile` and CUDA graphs and rejected both. They reached 8 ms on a single shape, but probabilities moved by up to .1 across shape buckets and the mutable HF cache produced stale-buffer crashes. Manual per-bucket graph capture is the remaining path to roughly 10 ms, and we haven't done it.

A second one in the same neighbourhood, with a more specific mechanism than we first wrote down. Our packing function built the 2D padding mask as `torch.long`. Hand `transformers` a non-boolean 2D padding mask and its SDPA mask preparation expands it into a 4D additive float bias — which is no longer eligible for the flash or memory-efficient kernels, so attention falls back to the math backend and materialises B×H×L² scores. Irrelevant at 256 tokens. At the 3,072-token states the 14B run used, it was the difference between OOM at 79 GB of 80 with micro-batch 4 and gradient checkpointing on, and running cleanly at 62 GB with micro-batch 2. The fix was `dtype=torch.bool` (commit `6d0a7e3`).

The transferable version isn't "use bool." It's that mask *dtype* silently selects your attention kernel two libraries away from where you wrote it, and the symptom is a memory number, not an error.

<p align="center"><img src="../figures/fig_serving.png" alt="Cold and warm p50 latency before and after removing the per-decision KV cache deep copy" width="640"></p>
<p align="center"><em>Removing the deep copy took roughly a quarter to a third off warm p50 latency across the family.</em></p>

## 10. Where direct decisions stop working

Pulling the failures together, they point at one boundary.

Decisions that are a readout of the state under a rule work well, and scale the way you'd hope: routing, extraction, adequacy, intent, severity, eligibility, rubric application including rubrics the model has never seen. That's the standard tier, and it's what the released models are for.

Decisions that require carrying an intermediate value through several steps do not work, at any size we've trained. Temporal arithmetic, unit conversion, expected-value comparison, multi-constraint trade-offs. Adding capacity moved them a little — on our own level-7 composition set, against a .441 uniform-guess rate, 1.7B scores .495, 4B .544 and 14B .620, while the same models score .84 to .90 on the held-out families, grammars and styles our generator does produce. The curriculum that moved every in-distribution family by thirty points moved this one by five.

<p align="center"><img src="../figures/fig_ladder.png" alt="JevBench standard and hard accuracy across backbone size, trained and frozen" width="640"></p>
<p align="center"><em>JevBench standard and hard accuracy across the size ladder, trained and frozen.</em></p>

Long-document policy looked like it belonged in between, and it turned out to be the clearest case of all once we ran the right experiment: a data problem, at p = 5e-10, that a 19-item benchmark subfamily reported as a null. Section 5 is the whole story.

The open question is narrower than we'd been framing it, and §8 is why. We had been asking when a decision needs intermediate computation and when a single direct read is enough. The data doesn't support that question yet, because the thing beating us on the serial families is another single direct read — a frozen backbone, one forward pass, letter logits, no scratchpad. Whatever those families need, the untrained backbone has more of it than our trained model does, and no amount of arguing about depth explains a result where the same architecture does better without our training.

So the question we can actually pose is: what does our fine-tuning remove? Two controls decide it, and neither is expensive. Frozen 14B with a scratchpad on the same 111 items tells us whether intermediate computation is the missing ingredient at all. A frozen backbone trained only on the families it already handles tells us whether the damage is coverage or interference. Both are queued.

---

The models are on Hugging Face: [`OzLabs/typical-small`](https://huggingface.co/OzLabs/typical-small) and [`OzLabs/typical-medium`](https://huggingface.co/OzLabs/typical-medium), with the inference package and the full evaluation artefacts in each repo. The launch post is [here](./typical-launch.md). Training code and the complete experiment report follow.
