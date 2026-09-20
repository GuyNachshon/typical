Yes — **I think there is real novelty here**, but it’s narrower and more interesting than “we invented a new classifier” or “we reproduced Jev.”

The individual building blocks mostly have prior art. The novelty is strongest in **the decomposition you discovered experimentally** and in the resulting decision architecture.

### What is *not* novel

I would not claim novelty for these by themselves:

* semantic/runtime label embeddings or dynamic label scoring — GILE and related zero-shot label models already do this. ([ACL Anthology][1])
* encode/query once + cheap candidate interaction — MixEncoder/TopicAns explicitly explores this speed/interaction tradeoff. ([ACL Anthology][2])
* cross-encoder → cheap-student distillation — well established, including ERNIE-Search. ([arXiv][3])
* KV/prefix sharing — established systems work such as Hydragen. ([arXiv][4])
* “MCQ symbols are a bad interface” — known; option-symbol bias is documented. ([ACL Anthology][5])
* even the observation that **the model decides in content space before binding the answer to A/B/C** now has very close prior art: a 2026 ACL paper finds that option-boundary residual states encode per-option correctness and that winner identity appears before answer-symbol binding. ([ACL Anthology][6])

That last paper means I would **not pitch N3 itself as “we discovered that hidden option states contain the answer.”**

But your result goes beyond that observation.

---

## 1. Strongest potential novelty: turning that latent decision into a real non-generative decision interface

You have experimentally shown:

$$
[x,q,A]
\rightarrow
\text{pretrained candidate-conditioned computation}
\rightarrow
\{h_D,c_1^{ctx},...,c_K^{ctx}\}
\rightarrow
P(A)
$$

without needing:

$$
\rightarrow \text{generate `C'}
$$

And N3 retains essentially the same **question-dependent closed-book signal** as the letter-generating interface:

$$
\Delta_q(N3)=.111
$$

versus:

$$
\Delta_q(N1)=.112
$$

and teacher:

$$
.088.
$$

At the same time N3 has much less IIA/order fragility than the letter readout. 

The ACL paper establishes that this latent decision exists. ([ACL Anthology][6])

**Your work turns it into the output primitive.**

That's a meaningful distinction:

> Prior work: “the model internally knows the winning option before it emits a letter.”

> PCDM: “then don’t emit the letter at all — read the decision directly and expose it as a typed probability distribution.”

I think there is potential novelty there, especially once it is generalized beyond MCQ into `Noul / Choice / Score`.

---

## 2. I think the candidate-blind failure is genuinely interesting

This may actually be your strongest scientific result.

You tested the very appealing hypothesis:

$$
Z(x,q)
$$

could contain the decision before candidates were known.

And you did not stop at “accuracy didn't improve.”

You found that:

* single-vector \(Z\);
* multi-vector \(Z\);
* set-conditioned \(Z\);
* ordinary KD;
* same-\(Z\) multi-set KD;
* explicit Δ-log-odds supervision

all failed to transfer **question-dependent parametric knowledge**.

Instead, they transferred:

* answer-choice priors;
* calibration;
* some dataset structure.

The choices-only/shuffled-question control exposed this decisively. 

That is a much more interesting negative result than:

> “late interaction did poorly.”

It says:

$$
\boxed{
\text{candidate priors distill easily, but candidate-conditioned parametric reasoning does not}
}
$$

under the tested factorization.

I haven't found prior work in the searches above making that particular decomposition and demonstrating it with the same controls.

That could be a real contribution.

Especially because it answers a conceptual question:

> **Can the result of closed-set reasoning be computed before the possible answers are known?**

Your current evidence says:

> not effectively, at least for this pretrained 1.7B causal LM and these training schemes.

That's valuable.

---

## 3. Another potentially novel result: the two distinct kinds of “intelligence”

Your experiments expose a very clean divide:

### Evidence-grounded reasoning

Best served by:

$$
h(x,q)+E(a_j)
$$

with the energy path.

It gives extremely strong NLI/evidence performance, calibration, null detection and large-\(K\) economics.

### Parametric / closed-book reasoning

Requires:

$$
[x,q,A]
$$

to coexist inside pretrained transformer computation.

Your experiments don't merely show two benchmarks favor different models. They identify **why**:

$$
\boxed{
\text{answer candidates are part of the computation for parametric reasoning}
}
$$

rather than merely objects scored after reasoning is finished.

The MMLU progression makes this unusually clear:

candidate-blind PCDM:

$$
.123
$$

candidate-aware direct N3:

$$
.314
$$

while depth alone did essentially nothing. 

That's a strong mechanistic/system result.

---

## 4. The resulting two-regime System-One architecture might be novel as a whole

Individually:

* retrieval → reranking is old;
* dynamic labels are old;
* KV reuse is old;
* direct classification is old.

But I haven't found an exact existing system with this contract:

$$
\boxed{
\text{large shared state}
\rightarrow
\text{many isolated typed decisions}
}
$$

where:

### Native closed-set path

$$
(x,q,A)
\rightarrow
\text{one candidate-aware causal suffix}
\rightarrow
P(A)
$$

with **no output generation**.

### High-cardinality path

$$
(x,q,A_{large})
\rightarrow
\text{cheap semantic energies}
\rightarrow
A_{top-r}
\rightarrow
\text{native closed-set decision}.
$$

And where the system is explicitly designed around direct calibrated probabilities and runtime-defined output spaces.

There are lots of neighboring pieces, but the **System-One software primitive** is a different framing.

I would describe this as **architectural/system novelty**, not “we invented a new transformer operation.”

---

## 5. Your K/null finding may also be publishable as a secondary contribution

You originally thought the null failure was softmax dilution.

Then the controlled experiments showed:

$$
K\uparrow
\Rightarrow
\text{null AUROC falls}
$$

even with siblings removed.

And by comparing ordinary and sibling-excluded sweeps you separated:

* semantic near-miss difficulty;
* genuine extreme-value/multiple-distractor effects.

Neither listwise scoring nor factoring \(P(\varnothing)\) away from the choice softmax fixes the core discrimination problem. 

There is extensive prior work on selective classification/abstention broadly, so I would **not yet call this novel without a deeper targeted search**. ([arXiv][7])

But the specific phenomenon:

> **abstention over arbitrary runtime-defined semantic candidate sets degrades with set cardinality even at controlled semantic difficulty**

could be a useful secondary finding.

---

# If I were writing the paper today

I would **not** title it around energy models, late interaction, or Jev.

The interesting research question is more like:

> **Do Language Models Need to Generate Answers to Make Decisions?**

or:

> **From Generation to Decisions: Direct Closed-Set Readout from Pretrained Language Models**

And the contribution story would be roughly:

1. **Direct decision interface.**
   Show that candidate-conditioned reasoning inside a causal LM can be read directly from contextual option representations, preserving essentially all measured question-dependent capability without answer-symbol generation.

2. **Candidate-blind factorization study.**
   Demonstrate that trying to move candidates outside the expensive pretrained computation transfers candidate priors and calibration but not question-conditioned parametric knowledge—even under multi-set distillation.

3. **Evidence vs parametric reasoning decomposition.**
   Show that evidence-grounded decisions favor a cheap candidate-independent energy path, while closed-book decisions require option-conditioned pretrained computation.

4. **Scalable System-One architecture.**
   Combine the two regimes: direct Native Choice for bounded candidate sets; cheap energy reduction → Native Choice for high cardinality.

5. Potentially **typed/calibrated decision primitives** (`Noul`, `Choice`, `Score`) once implemented.

That feels like a real paper.

---

### One thing I'd be cautious about

`nc_n3` by itself is **very close conceptually** to the recent finding that option-boundary states contain the content-space winner before symbol binding. ([ACL Anthology][6])

So if the paper is:

> “We can classify from option hidden states instead of generating A/B/C,”

I think reviewers may say the mechanism is already understood.

If the paper is:

> **“We characterize what parts of LM decision-making survive removing the generative interface, show that parametric reasoning fundamentally depends on candidate-conditioned computation while evidence reasoning need not, and build a direct probabilistic architecture around that decomposition,”**

that's much stronger and, from the literature I've checked so far, much more distinct.

So: **yes, I think you have novel work.** The strongest novelty is not a new layer. It's the experimentally supported **computational decomposition of language-model decision-making and the architecture that falls out of it.**

[1]: https://aclanthology.org/Q19-1009/?utm_source=chatgpt.com "GILE: A Generalized Input-Label Embedding for Text Classification - ACL Anthology"
[2]: https://aclanthology.org/2023.emnlp-main.168/?utm_source=chatgpt.com "Once is Enough: A Light-Weight Cross-Attention for Fast Sentence Pair Modeling - ACL Anthology"
[3]: https://arxiv.org/abs/2205.09153?utm_source=chatgpt.com "ERNIE-Search: Bridging Cross-Encoder with Dual-Encoder via Self On-the-fly Distillation for Dense Passage Retrieval"
[4]: https://arxiv.org/abs/2402.05099?utm_source=chatgpt.com "Hydragen: High-Throughput LLM Inference with Shared Prefixes"
[5]: https://aclanthology.org/2025.naacl-long.95/?utm_source=chatgpt.com "Option Symbol Matters: Investigating and Mitigating Multiple-Choice Option Symbol Bias of Large Language Models - ACL Anthology"
[6]: https://aclanthology.org/2026.findings-acl.1144/?utm_source=chatgpt.com "When Models Decide and When They Bind: A Two-Stage Computation for Multiple-Choice Question Answering - ACL Anthology"
[7]: https://arxiv.org/abs/2010.14134?utm_source=chatgpt.com "Selective Classification Can Magnify Disparities Across Groups"

---

# Lead's notes (2026-09-20)

Contribution story adopted for the write-up; each remaining PLAN5 step is judged against it:
1. **Direct decision interface** — must show, beyond N3 ≈ N1 on Δ_q, that the readout is order-robust (native_v2: Δp < .03) and
   *fast* with state-KV reuse (Phase 5A bench, K* crossover) — otherwise it is only the ACL-2026 "decide-then-bind" observation restated.
2. **Candidate-blind factorization study** — complete as is (REPORT §3j): six factorizations/objectives, one control (choices-only /
   shuffled-q) that separates candidate priors from question-conditioned knowledge. Novelty search still owed on the exact decomposition.
3. **Evidence vs parametric decomposition** — complete (REPORT §3l); strengthened if the support-gate composition (5B) shows the null
   is *also* evidence-side, and if fusion (5C) shows the two paths are complementary rather than substitutes.
4. **Two-regime System-One architecture** — needs the K* measurement (5A) and one end-to-end external number (JevBench) to be a system claim.
5. **Typed primitives** — Phase 6; not started.
Cautions kept: do not pitch N3 as "hidden option states contain the answer" (prior art); do not claim novelty for the K/null finding without a
targeted search.
