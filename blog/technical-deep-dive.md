---
title: "We Removed Generation from an LLM. Here's What Broke."
date: 2026-09-22
---

# We Removed Generation from an LLM. Here's What Broke.

[Typical](./typical-launch.md) is a family of open decision models. You give one a state and a typed question, and it returns a probability distribution over options you defined at runtime. No tokens are generated and nothing is parsed.

That interface sounds like a simplification of a language model. In practice, removing generation removed a lot of scaffolding that was doing load-bearing work, and most of what we learned came from finding out which scaffolding mattered.

This is the archaeology: the architecture that didn't work, the bugs that looked like results, and the results that looked like bugs. Section references (§3j, §3t and so on) point at our internal experiment report, which publishes with the training code.

One measurement caveat applies throughout. Our JevBench runs are against the public subset, unranked: 72 standard items that come from only 36 independent states (each state appears as two paraphrases), and 111 hard items. Cluster-bootstrapped, that puts roughly ±9 points on any hard-tier number. Where a claim below rests on a small JevBench gap we say so, and we'd rather report a direction we can't resolve than dress it up as a result.

## 1. The obvious architecture failed

The first design was the clean one. Compute a decision state `Z(state, question)` once, without looking at the candidates, then score each candidate against it with a cheap head. Candidates are then free: their cost is a dot product, K scales for nothing, and you can cache label vectors forever.

We built three versions of it and distilled a listwise teacher into them: a single decision vector, eight probes with token-level MaxSim, and probes plus O(K) cross-attention over the candidate tokens. On MMLU-Pro among-K they scored .171, .157 and .147, against .310 for the teacher. That looked like partial success. About half the teacher's score had transferred.

Then we ran the control that killed it (§3j). We re-scored every student with the question replaced by a period, and again with the question shuffled:

| among-K | normal | choices only | shuffled question | question-dependent gain |
|---|---|---|---|---|
| teacher (options in context) | .310 | .222 | .208 | **+.088** |
| `z1` student | .171 | .181 | .147 | −.010 |
| `zr` student | .157 | .166 | .182 | −.009 |
| `zr_set` student | .147 | .162 | .151 | −.014 |

Every student did the same or better with the question destroyed. Their entire above-chance score was candidate-set priors: which option strings look plausible together, learned from the corpus. Nothing question-dependent had compiled into `Z` at all.

We tried the strongest version of the fix. Multi-set supervision, where the same state appears with seven different option sets and the loss targets the teacher's Δ-log-odds between them, gave a model with Δ_q(shuffled) of exactly 0.000. The set-conditioning cross-attention path stayed inert through 4k steps of direct supervision.

## 2. Candidates have to participate in the computation

The suspicion at that point was depth: every candidate-blind student read the backbone at layer 20 of 28, and maybe the last eight layers were where question-conditioned knowledge lived. So we re-ran the candidate-blind student at full depth (§3s). Among-K *fell*, .157 to .147, and Δ_q(shuffled) stayed at −.008 while the evidence tasks regressed hard (BoolQ −11, ANLI −8, MNLI −7).

That closed it. What transfers into a state computed before the options are known is priors and calibration, not question-conditioned parametric knowledge, at any tap depth.

The architecture that works renders the options into the suffix, so the candidate text goes through the same pretrained layers as the question. The released models built on it carry Δ_q(shuffled) of .127 at 1.7B and .193 at 4B: real, measurable, question-dependent behaviour, from the same backbone that couldn't produce any when the options arrived after the fact.

There's a much dumber version of the same lesson from earlier in the project. Qwen3 has no BOS token, so position 0's hidden state is content-independent. The cosine similarity between the position-0 states of `neutral`, `yes` and `transfer` was 0.9998. Every single-token candidate we embedded was, numerically, the same vector. Prepending `<|endoftext|>` and dropping it fixed it.

## 3. The final layer was the wrong layer

The first full-scale version read the backbone's last hidden layer, which is what you'd do without thinking about it. It scored 58.6 on SNLI.

58.6 is the hypothesis-only baseline for SNLI. The model was making NLI predictions without using the premise, which is the classic dataset-artifact failure. Our decision head had been handed a state that had already been specialised for next-token prediction, and whatever cross-sentence evidence structure existed in the middle of the network was gone by the top.

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

It produced one of the strangest failure modes in the project. Null behaviour became a function of K:

| null AUROC, CLINC-150 | K=5 | K=20 | K=50 | K=150 |
|---|---|---|---|---|
| rendered ∅ as an option | .97 | .94 | .91 | 1.00† |
| ∅ as a head decision | .98 | .95 | .92 | .95 |

† Degenerate. At K=150 the model abstained on everything, which makes AUROC perfect and the model useless. At the other end, at K=5 it never abstained: P(∅ | gold absent) was .02. On Banking77 at K=77 the same degeneracy gave a null AUROC of .00.

Downstream, the effects were not subtle. MMLU-Pro false-abstention was .692, dropping accuracy to .133. Banking77 with all 77 labels scored .063.

Moving abstention out of the candidate list and into a separate head, computed independently of the softmax over the real options, fixed all of it at once (§3t). The K-sweep became monotone and smooth, MMLU-Pro false-abstention went .692 to .003 with accuracy .133 to .353, KL to the teacher went 2.43 to .27, and Banking77-77 went .063 to .579. It cost 2 to 7 points on intent and topic label spaces (HWU64 .801 to .757, 20NG .589 to .515) and raised option-order sensitivity.

The practical version: if you have added a "none of these" option to any classifier, check whether it behaves the same way at K=5 and K=150. A null score that means something different at every candidate count cannot be thresholded, and a threshold is the entire point.

## 5. We trained on long documents after silently deleting the evidence

Our long-policy training corpus, 23k rows, renders the case facts at the end of the text: policy document first, then `Case: <facts> <request>`.

State length in that corpus runs p10/p50/p90 = 1,190 / 1,845 / 2,490 tokens. The training pipeline right-truncated states at `--max_state`. At the 1,024-token window Release 1 used, **98.8% of those rows lost their facts before the model ever saw them**. At the 256-token window the earlier scaling ladder used, all of them did.

So for a stretch of this project we were training models to answer confidently about text that had been cut off. The symptom was visible in the metrics, if you knew to read it: long-document policy accuracy of .05 for the 14B ladder run and .21 for `typical-medium`, with high confidence attached.

The fixes were to regenerate the corpus facts-first (case position inside the first 8% of the text), and to stop truncating. `--drop_truncated` discards any row that doesn't fit the window rather than quietly cutting it, which is the right default for a training loader handling anything with a payload at a known position.

**Then we tried to prove it mattered, and couldn't.** The data defect is verified and not in dispute: we can count the rows, 98.8% is a measurement, and we confirmed empirically that the tokenizer path keeps the start of a state and drops the end. What is *not* established is that the defect caused the metric movement we saw afterwards. The run that fixed it changed five things at once (facts-first rendering, `--drop_truncated`, a wider window, a Brier term, calibration-based checkpoint selection), so the long-policy recovery had five candidate explanations and we had separated none of them.

The counterfactual corpus was still on disk, so the ablation was clean to set up: two 1.7B arms, byte-identical flags, a 1,024-token window with `--drop_truncated` deliberately off so truncation bites exactly as it did in Release 1, differing only in which render of the same rows is the training file. Same rows, same labels, same state lengths, `Case:` at the start or at the end.

It came back underpowered. Long-policy accuracy was .316 for the facts-first arm against .105 for facts-last, which is the predicted direction, but that is 6 of 19 items against 2 of 19. Fisher's exact test gives two-sided p = 0.232, the bootstrap interval on the difference is [−0.053, +0.474] and contains zero, and the hard-tier aggregate goes the other way (.378 facts-first against .396 facts-last). Nineteen items cannot settle this.

**What the same experiment measured well is a different effect.** On a held-out long-state set of 605 items, with no truncation at eval in either condition, we rendered each item both ways and scored the facts-last-trained arm on both. It gets .842 when the case sits at the end of the state, where its training data put it, and .615 when the same 605 items put the case at the start. Its majority-class floor on that set is .612. Moving the facts to a position the model wasn't trained to look at costs 23 points and takes it to the floor.

So the honest reading is not the one we started with. The defect is real, the model's long-document failure is real, and the mechanism we can actually demonstrate at adequate sample size is train/test render mismatch: the model learned where in the state to look, and fails when that changes. Whether the truncation itself contributed on top of that is a question our ablation was too small to answer, and we're leaving it open rather than claiming the tidier version.

<p align="center"><img src="../figures/fig_truncation.png" alt="Left: state token length distribution against truncation cutoffs at 256/1024/2048/3072 tokens. Right: long-policy accuracy across checkpoints" width="640"></p>
<p align="center"><em>Left: how much of a long-policy row survives each token budget. Right: long-policy accuracy across checkpoints.</em></p>

## 6. Training distribution moved capability as much as architecture

It's tempting to write this project as an architecture story, because architecture is what's interesting. The mixture sweeps say otherwise (§3z, §3ad).

Raising the evidence share of the training mixture from .35 to .45–.50 brought NLI and BoolQ back to within a point of the untrained base model and intent and topic sets to within 3–4 points, with workflow performance unchanged. No architecture changed.

Augmenting 20% of workflow rows with the gold answer removed, so the correct answer is ∅, lowered false-abstention on CLINC from .10 to .08 and on TREC from .27 to .12, and *raised* accuracy on CLINC, TREC, HWU64 and 20NG by 2.6, 3.6, 4.3 and 6.0 points. Teaching the model when to abstain made it better at not abstaining.

The hard curriculum is the sharpest case. Adding a rule-engine corpus where the same state and options recur under different rubrics with different gold answers took held-out family, grammar and style accuracy from .481 / .507 / .491 to .827 / .881 / .888, and rubric-flip accuracy from .477 to .710. On JevBench's hard tier, the families that corpus covers moved in the same direction, though those are 6 and 7 items respectively and we'd treat them as directional: adversarial .33 to .83, ambiguous .57 to .71.

And level-7 composition, which mixes temporal, unit, expected-value and trade-off reasoning and which the generator never produces, went .477 to .498. What we generated was learned. What we didn't generate was not, and no amount of scale substituted for it.

There is a fourth result in this family that we found unwelcome. Rerunning the Release-1 candidate at batch 16 instead of 64, which is simply 4× fewer examples seen, produced the best hard-tier Brier of any run in the project (.76) and hard accuracy .450, while losing ground on everything in-distribution (CLINC .733, MMLU among-K .323). Small batch isn't a recipe, it's under-fitting: the same model, less confident, scores higher on the hard tier. That is fairly direct evidence that our hard-tier problem is substantially a probability-quality problem rather than a knowledge problem.

## 7. Calibration improved without changing decisions

Score is an ordinal primitive: severity 0 through 3, priority 1 through 5. A K-way softmax treats those levels as unrelated buckets, so predicting 0 when the answer is 1 costs exactly what predicting 0 when the answer is 3 costs.

Training the same head with ordinal-smoothed targets (τ = 0.7), which is a loss change and no new parameters, gave this (§3ac):

| held-out urgency | K-way | ordinal-smoothed |
|---|---|---|
| accuracy | .505 | .508 |
| NLL | 2.07 | **1.23** |
| Brier | .79 | **.68** |
| ECE | .35 | **.19** |
| ordinal MAE | .58 | **.55** |

Accuracy didn't move. On the 12 ordinal items in JevBench the two models produce *identical per-item predictions*. Every probability-quality metric improved anyway. On an external severity set the same change took accuracy .853 to .939 and MAE .15 to .06.

The same thing scaled. At 14B, the recipe with the long-state fix, a Brier term and calibration-based checkpoint selection took held-out score NLL from 2.87 to 0.95 and typed-decisions NLL from 1.96 to 1.04, with JevBench hard Brier .85 to .66 (§3ah).

The Noul head is a structural version of the same idea. A two-way Choice over `["no", "yes"]` moves P(yes) by up to .55 when you reverse the label order. A Bernoulli head reading the decision state with no candidates rendered moves it by exactly zero, on all nine test sets, by construction. It also cleared an external floor the two-way head couldn't: PagerDuty .602 to .886, against a constant-prediction floor of .792.

<p align="center"><img src="../figures/fig_calibration.png" alt="Held-out score NLL and typed-decisions NLL across checkpoints" width="640"></p>
<p align="center"><em>Held-out score and typed-decisions NLL across checkpoints.</em></p>

One thing we added and then couldn't justify. The 14B run bundled knowledge distillation from a frozen 14B teacher, on the theory that it would buy hard-tier reasoning. We ran the matched control with `--distill_beta 0`, one flag different and nothing else (§3ai). The control came out slightly ahead: hard accuracy .477 against .450, long-document policy .211 against .158, standard Brier .127 against .175, validation NLL 0.410 against 0.438, at the cost of 1.4 points of standard accuracy.

The direction is consistent across those metrics, and it is still not a result. Cluster-bootstrapped, the hard-tier numbers are .450 [.360, .541] with KD and .477 [.387, .568] without, intervals that overlap along almost their whole length. We cannot show the teacher helped and we cannot show it hurt; 111 items is not enough to tell. What we can say is that the change we added specifically to buy hard-tier reasoning bought nothing we can detect, which is reason enough not to credit it for the calibration gains above. Those belong to the loss and the data.

## 8. Fine-tuning made the hardest cases worse

The uncomfortable number in this project: on JevBench's hard tier, a frozen 14B base model reading letter logits with three examples in its prompt scores .559. Our trained 14B decision checkpoint scores .450, and its no-KD control .477.

At 111 items those intervals overlap, so we won't call that a defeat. What it rules out is the claim we'd have liked to make. We trained a decision model on top of that backbone and cannot demonstrate that it got any better at the tier we care most about.

The per-family breakdown shows where. On the no-KD control: adversarial .833, trap 1.00, routing 1.00, multi-hop .50, judge-hard .588, probability .50, ambiguous .429, long-policy .211, temporal and numeric .20, trade-off .167. Everything that resembles a workflow decision under a rubric is fine. Everything that requires carrying a value through several steps is near chance.

Two related findings sharpen it. First, the backbone generation appears to matter more than our training does here: a frozen Qwen3.5-9B scores .541 hard with three examples and .595 under a different rendering, above the .495 of the best checkpoint we've trained at any size, again with intervals that overlap. Second, that rendering effect is visible on its own. The same frozen Qwen3.5-9B goes from .806 to .931 on the standard tier purely by changing how the state and options are laid out, with zero training, and all three Qwen3.5 sizes we tested moved the same way under the same change. We'd read that as a sign that a meaningful share of the public leaderboard's standard-tier spread is a protocol effect, and as a reason to be careful reading ours.

The lesson we'd generalise: a frozen model with a few examples in context is not a strawman to beat before shipping. It's a ceiling to verify you've actually cleared, per tier, and we publish the tiers where we haven't.

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

A second one in the same neighbourhood: the SDPA padding mask was being built as a `long` tensor, which silently forces PyTorch into its O(L²) math kernel. That was the cause of our 14B out-of-memory failures at 3,072-token states, and of a lot of memory pain before that. The fix was `bool`.

<p align="center"><img src="../figures/fig_serving.png" alt="Cold and warm p50 latency before and after removing the per-decision KV cache deep copy" width="640"></p>
<p align="center"><em>Removing the deep copy took roughly a quarter to a third off warm p50 latency across the family.</em></p>

## 10. Where direct decisions stop working

Pulling the failures together, they point at one boundary.

Decisions that are a readout of the state under a rule work well, and scale the way you'd hope: routing, extraction, adequacy, intent, severity, eligibility, rubric application including rubrics the model has never seen. That's the standard tier, and it's what the released models are for.

Decisions that require carrying an intermediate value through several steps do not work, at any size we've trained. Temporal arithmetic, unit conversion, expected-value comparison, multi-constraint trade-offs. Adding capacity moved them very little, and the curriculum that fixed every other family didn't touch them because we never generated them.

<p align="center"><img src="../figures/fig_ladder.png" alt="JevBench standard and hard accuracy across backbone size, trained and frozen" width="640"></p>
<p align="center"><em>JevBench standard and hard accuracy across the size ladder, trained and frozen.</em></p>

Long-document policy sits in between and is the most interesting case, because it's the one where we currently can't tell a data problem from an architecture problem. Section 5's ablation is running precisely because the answer changes what we build next.

That leaves the open question in a shape we can state but not yet answer: when is a single direct decision enough, and when does a decision genuinely need intermediate computation? Our current guess is that the boundary has little to do with difficulty, and a lot to do with whether the answer is a function of the state that a fixed-depth read can express, or a sequence of operations that needs somewhere to put its intermediate results. If that's right, the fix is not a bigger decision model. It's a decision model that knows when to stop and ask for one.

---

The models are on Hugging Face: [`OzLabs/typical-small`](https://huggingface.co/OzLabs/typical-small) and [`OzLabs/typical-medium`](https://huggingface.co/OzLabs/typical-medium), with the inference package and the full evaluation artefacts in each repo. The launch post is [here](./typical-launch.md). Training code and the complete experiment report follow.
