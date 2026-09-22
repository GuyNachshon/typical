"""End-to-end decision latency: Typical against a hosted LLM, measured the same way.

    uv run --no-sync python scripts/bench_llm.py --model gpt-5.1-mini --trials 30

The claim ledger forbids "faster than any hosted system" precisely because no apples-to-apples
run existed (.context/content-spec.md, MUST NOT #1). This is that run, so the rule it exists to
enforce -- do not compare numbers measured under different conditions -- is what shapes it:

  * Both arms are timed from a client process, wall clock, request to parsed answer. The 45 ms in
    releases/typical-small.md is in-process on an H100 with model load excluded, and it is NOT
    comparable to anything that crosses a network. It is not used here.
  * Both arms do the same task: read one ticket, pick one of four teams, and say how confident.
  * Both arms are given their best shape. Typical batches M questions into one call because its
    cache makes that natural; the LLM gets the same courtesy -- one call returning M fields --
    and is also measured one-question-at-a-time, because questions in a real workflow arrive as
    the workflow reaches them, and that is the case the cached prefix is for.
  * Warmups are discarded. Medians are reported, with p10/p90, because a mean over a network is a
    story about the worst request.

Typical runs on whatever the local server has (Apple silicon here, not an H100) -- the output
records the device, and any chart built from this file must print it. A local MPS number is
slower than the H100 ladder; that is the honest direction for the comparison to be wrong in.

Writes site/data/llm_latency.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site/data/llm_latency.json"
TYPICAL_URL = "http://localhost:8787/api/decide"
HEALTH_URL = "http://localhost:8787/api/health"

# One call per provider family. All of them speak the OpenAI chat-completions shape except
# Anthropic, which is close enough that one branch covers it.
PROVIDERS = {
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1/chat/completions", "TOGETHER_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY"),
}


def load_case() -> tuple[str, dict, list[dict]]:
    """The ticket and questions the site already uses, so the benchmark and the page agree."""
    pg = json.loads((ROOT / "site/data/presets.json").read_text())["playground"]
    queries = pg["queries"]
    choice = next(q for q in queries if q["type"] == "choice")
    return pg["state"], choice, queries


def schema_for(questions: list[dict]) -> dict:
    """A JSON schema pinning the LLM to the same answer space Typical is given.

    Without this the arms are not the same task: a free-text answer needs parsing and can miss the
    label set, which is the failure Typical's typed head makes impossible. Constraining the LLM is
    the fair thing to do and also the slower thing for it, so it is stated plainly here.
    """
    props = {}
    for i, q in enumerate(questions):
        labels = q.get("labels") or ["no", "yes"]
        props[f"q{i}"] = {"type": "string", "enum": labels}
        props[f"q{i}_confidence"] = {"type": "number"}
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def prompt_for(state: str, questions: list[dict]) -> str:
    lines = [f"Ticket:\n{state}\n", "Answer each question. Use only the options given."]
    for i, q in enumerate(questions):
        labels = q.get("labels") or ["no", "yes"]
        lines.append(f'q{i}: {q["question"]} Options: {", ".join(labels)}')
    lines.append('Reply as JSON with a "qN" and "qN_confidence" (0-1) for each.')
    return "\n".join(lines)


def call_llm(client: httpx.Client, provider: str, model: str, state: str, questions: list[dict]) -> float:
    """One request, timed from just before the send to just after the answer parses."""
    url, key_env = PROVIDERS[provider]
    key = os.environ[key_env]
    schema = schema_for(questions)
    text = prompt_for(state, questions)
    if provider == "anthropic":
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": text}],
            "tools": [{"name": "answer", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": "answer"},
        }
    else:
        headers = {"Authorization": f"Bearer {key}"}
        body = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
            },
        }
    t0 = time.perf_counter()
    res = client.post(url, json=body, headers=headers, timeout=180)
    res.raise_for_status()
    data = res.json()
    if provider == "anthropic":
        answer = next(b["input"] for b in data["content"] if b["type"] == "tool_use")
    else:
        answer = json.loads(data["choices"][0]["message"]["content"])
    ms = (time.perf_counter() - t0) * 1000
    # a timing that did not produce every answer is not a timing of the task
    missing = [k for k in schema["required"] if k not in answer]
    if missing:
        raise RuntimeError(f"{provider}/{model} left {len(missing)} field(s) unanswered: {missing[:3]}")
    return ms


def call_typical(client: httpx.Client, state: str, questions: list[dict]) -> float:
    t0 = time.perf_counter()
    res = client.post(
        TYPICAL_URL,
        json={"state": state, "queries": [{k: q[k] for k in ("type", "question", "labels") if k in q} for q in questions]},
        timeout=180,
    )
    res.raise_for_status()
    data = res.json()
    ms = (time.perf_counter() - t0) * 1000
    if len(data.get("results", [])) != len(questions):
        raise RuntimeError("typical returned the wrong number of results")
    return ms


def summarise(samples: list[float]) -> dict:
    s = sorted(samples)
    return {
        "n": len(s),
        "median_ms": round(statistics.median(s), 1),
        "p10_ms": round(s[int(0.1 * (len(s) - 1))], 1),
        "p90_ms": round(s[int(0.9 * (len(s) - 1))], 1),
    }


def run(label: str, fn, trials: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for i in range(trials):
        samples.append(fn())
        print(f"  {label}: {i + 1}/{trials}  {samples[-1]:.0f} ms", end="\r", flush=True)
    print(f"  {label}: {summarise(samples)}          ")
    return summarise(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai", choices=sorted(PROVIDERS))
    ap.add_argument("--model", required=True, help="the hosted model id, e.g. gpt-5.1-mini")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--many", type=int, default=8, help="questions in the batched arm")
    args = ap.parse_args()

    key_env = PROVIDERS[args.provider][1]
    if not os.environ.get(key_env):
        raise SystemExit(f"{key_env} is not set. Put it in the environment (not in a tracked file) and re-run.")

    state, choice, queries = load_case()
    # The batched arm repeats the four real questions up to --many, so every question is one a
    # reader can see on the page rather than filler invented to pad a benchmark.
    many = [queries[i % len(queries)] for i in range(args.many)]
    one = [choice]

    device = "unknown"
    with httpx.Client() as client:
        try:
            device = client.get(HEALTH_URL, timeout=10).json().get("device") or "unknown"
        except Exception:
            raise SystemExit("the local Typical server is not answering on :8787 -- start it first")

        print(f"typical ({device}) vs {args.provider}/{args.model}, {args.trials} trials")
        out = {
            "method": (
                "Wall clock from a client process: request sent to answer parsed, both arms, same "
                "ticket and same four options, the hosted model constrained to the same label set "
                "by JSON schema. Medians over the stated trial count, warmups discarded. Typical "
                f"runs locally on {device}, not the H100 the 45 ms ladder was measured on."
            ),
            "typical_device": device,
            "llm": f"{args.provider}/{args.model}",
            "trials": args.trials,
            "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "arms": {},
        }
        out["arms"]["typical_1"] = run("typical  ×1 ", lambda: call_typical(client, state, one), args.trials, args.warmup)
        out["arms"][f"typical_{args.many}"] = run(f"typical  ×{args.many} ", lambda: call_typical(client, state, many), args.trials, args.warmup)
        out["arms"]["llm_1"] = run("llm      ×1 ", lambda: call_llm(client, args.provider, args.model, state, one), args.trials, args.warmup)
        # best case for the hosted model: every question in one request
        out["arms"][f"llm_{args.many}_batched"] = run(f"llm      ×{args.many} batched", lambda: call_llm(client, args.provider, args.model, state, many), args.trials, args.warmup)
        # and the case the cached prefix is actually for: questions arriving one at a time
        out["arms"][f"llm_{args.many}_serial"] = run(
            f"llm      ×{args.many} serial ",
            lambda: sum(call_llm(client, args.provider, args.model, state, [q]) for q in many),
            max(4, args.trials // 4),
            1,
        )

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


def demo() -> None:
    """Offline check of the parts that do not need a network or a key."""
    state, choice, queries = load_case()
    assert state and len(queries) >= 3
    s = schema_for(queries[:2])
    assert s["properties"]["q0"]["enum"] == queries[0]["labels"], "the LLM is pinned to Typical's label set"
    assert set(s["required"]) == set(s["properties"]), "every field is required, so a partial answer fails"
    assert "q1:" in prompt_for(state, queries[:2]) and state[:20] in prompt_for(state, queries[:2])
    d = summarise([10.0, 20.0, 30.0, 40.0, 50.0])
    assert d["median_ms"] == 30.0 and d["p10_ms"] == 10.0 and d["p90_ms"] == 40.0
    assert summarise([5.0])["median_ms"] == 5.0, "a single sample does not index out of range"
    print("bench_llm.py self-test OK")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo()
    else:
        main()
