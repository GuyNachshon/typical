# Jev / TypeSafe launch — research (2026-09-20), condensed

Launch Sept 15 2026, $40M seed, CEO Diogo Almeida (ex-OpenAI). HN ~1,655 pts / 480 comments. Closed weights, US API, waitlist.

## Artifacts they shipped
- Blog https://typesafe.ai/blog/introducing-system-one-models-and-jev — "System One" (Kahneman), RLCD training, "70ms–500ms", "40x–200x faster", $0.042/MTok in, output FREE, "0% type errors", Doom + Wikiracing demos, admitted caveats.
- Homepage: "Decisions, not strings", "Calibrated confidence", "More like code", "Zero Hallucinations"; "$42 per billion input tokens"; split-screen video vs LLM workflow.
- Docs https://docs.typesafe.ai/llms.txt — ~110 pages: primitives Choice/Score/Noul, confidence, patterns (fan-out, confidence-routing, composite-scoring, intent-routing), 18 cookbooks, SDKs, HTTP `POST /v1/systemone`, **model-jaggedness/jev-1.13** (11 admitted failure modes — unusual candor). No calibration curves anywhere. No reproducibility artifacts.
- Playground console.typesafe.ai/playground — login-walled; shareable query links.
- Evals https://evals.typesafe.ai/ — 4 workflows (security_incidents, agent_trace_observability, invoice_processing, customer_service); baselines Haiku 4.5/Sonnet 5/Opus 5/DS v4/Luna/Sol/Terra; metrics accuracy, $/case (log), s/case (log); two scatter plots; Jev 67.8% / $0.0004 / 0.4s vs Opus 5 73.1% / $0.1761 / 37.8s. Labels = average of GPT-6 Astra + Claude Fable 5.1 (LLM consensus, not human). No raw download, no calibration plot, no classical baselines.
- Adapter https://github.com/typesafe-ai/system-one-adapter-python (MIT) — answer System One requests via OpenAI/Anthropic so users compute their own Nx numbers.
- Agent skill (Claude Code plugin) — distribution via coding agents.
- Demo videos: Doom (~10 q/s, per-action probability bars — best visual), Wikiracing (255-option cap, 2-stage), Smart-home SPA.
- Ecosystem: OpenRouter, Cloudflare AI Gateway, Vercel evaluator; 3 awesome-lists (~200 projects in 5 days).

## Their demos (UX)
- Quickstart: support ticket + 3 questions in one call (Noul urgency, Choice department, Score frustration) → `noul: 0.999`, `score: 1.035, confidence: 0.842`, probabilities.
- Parallel-questions cookbook: 53k-char GDPR article + 13 questions → "12.2x cheaper and 10.0x faster" ($0.000497 vs $0.006090; 0.27 s vs 2.71 s).
- Every.to: 37 docs × 21 questions = 777 judgments in 0.7 s; 1,709 judgments < 1¢ — most-cited independent demo.

## Reception
Resonated: speed/cost, output tokens free, batching many questions over one state ("the missing primitive"), confidence-routing (act/confirm/human), typed API no JSON parsing, Doom bars.
Criticized: "can't hallucinate" = shape guarantee only; "frontier" false (mid-tier on own evals); vendor-run evals with LLM labels; calibration asserted never shown (only third-party ECE .031 on MMLU by Archer Hume); closed weights / no self-host / waitlist; "really smart switch statement" (590K views); real speedups ~7× median vs 193× claimed; IIA violations (Hume: add irrelevant option shifts log-odds; last-position accuracy 16/16 vs ~50% earlier); no built-in null (docs: add "other" yourself); complementary questions don't sum to 1.
Open questions: fine-tuning, determinism, data resale, rate limits, model size, calibration under shift, vs constrained-decoding small LLMs, local/privacy path.
Open clones: vinnylarouge/jevlike (972★), wfzyx/von (sub-15 ms drop-in for POST /v1/systemone), ikermoel/open-alternative-jev (T-scaled), decider (Qwen3.5-2B), NanoJev (0.6B). None address null or choice-set invariance.

## Gaps Typical can exploit
Open weights + local (loudest ask); architectural null; choice-set invariance (reproduce Hume's experiments live); calibration proven (reliability diagrams, ECE per task, under shift); reproducible evals with public ground truth + raw JSON + harness + classical baselines; honest ratios (median, p95); explain it in 3 sentences.

## Their phrasing (avoid copying)
"System One model", "Decisions, not strings", "frontier-intelligence function call: unstructured state in, typed probabilistic decisions out", "speculative fan-out", "confidence routing", "model jaggedness", "RLCD". Primitives: Choice (`criteria` map ≤255, returns `choice`, `probabilities`, `confidence`), Score (2–10 ordered levels → expected-value `score` + `legend`), Noul (yes/no → `noul` ∈ [0,1]). Input `state`; endpoint `POST /v1/systemone`; model ids `jev-1.13.0` / `jev-latest`.

Sources: HN 49717558, 49731282, 49745752; https://archerhume.com/posts/jevs-architecture-unmasked (+ evidence.json); https://openchamber.dev/blog/jev-typesafe-ai/; https://www.latent.space/p/ainews-jev-a-system-one-model-that; https://every.to/also-true-for-humans/mini-vibe-check-typesafe-s-jev-judged-everything-i-ve-written-in-0-7-seconds; https://kingy.ai/blog/typesafe-jev-review-the-ai-model-that-doesnt-generate-text/; https://www.modemguides.com/blogs/ai-news/jev-typesafe-reality-check-run-locally; https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711
