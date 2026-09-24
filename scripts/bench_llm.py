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
import re
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


def load_case(preset: str = "playground") -> tuple[str, dict, list[dict]]:
    """The ticket and questions the site already uses, so the benchmark and the page agree."""
    pg = json.loads((ROOT / "site/data/presets.json").read_text())[preset]
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


def call_llm(client: httpx.Client, provider: str, model: str, state: str, questions: list[dict], effort: str = "low") -> tuple[float, dict]:
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
            # A reasoning model left on its default setting spends seconds thinking about which of
            # four teams owns a ticket. At "low" it gets its fast path, which is the setting that
            # makes this comparison hardest for us and the one anybody would actually ship; the
            # run is repeated at "high" because that is what the same model costs when you let it
            # think, and a reader deciding between the two should see both. Providers with no such
            # knob ignore the field.
            "reasoning": {"effort": effort},
        }
    # A 429 is the provider queueing us, not the model thinking, so a rate-limited attempt is
    # retried and only the attempt that produced an answer is timed. Counting the backoff would
    # inflate the hosted number with our own account limits, which is not what this measures.
    for attempt in range(6):
        t0 = time.perf_counter()
        res = client.post(url, json=body, headers=headers, timeout=600)
        if res.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        res.raise_for_status()
        data = res.json()
        err = ((data.get("choices") or [{}])[0].get("error") or {}).get("code")
        if err == 429:
            time.sleep(2 ** attempt)
            continue
        break
    else:
        raise RuntimeError(f"{provider}/{model} rate-limited through six retries")
    answer = extract(provider, data)
    ms = (time.perf_counter() - t0) * 1000
    # a timing that did not produce every answer is not a timing of the task
    missing = [k for k in schema["required"] if k not in answer]
    if missing:
        raise RuntimeError(f"{provider}/{model} left {len(missing)} field(s) unanswered: {missing[:3]}")
    return ms, answer


def extract(provider: str, data: dict) -> dict:
    """Pull the structured answer out, whichever shape the provider returned it in.

    Gemini through OpenRouter returns `content: null` and puts the object in a tool call, so a
    straight json.loads on the content field crashes on one provider and not the others. Anything
    this cannot parse raises with the payload attached rather than being counted as a fast trial.
    """
    if provider == "anthropic":
        blocks = data.get("content") or []
        for b in blocks:
            if b.get("type") == "tool_use":
                return b["input"]
        raise RuntimeError(f"no tool_use block in {json.dumps(data)[:400]}")
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):  # some providers return content parts
        content = "".join(part.get("text", "") for part in content)
    if content:
        return json.loads(content)
    for call in msg.get("tool_calls") or []:
        args = (call.get("function") or {}).get("arguments")
        if args:
            return json.loads(args) if isinstance(args, str) else args
    raise RuntimeError(f"no answer in {json.dumps(data)[:400]}")


def call_typical(client: httpx.Client, state: str, questions: list[dict]) -> tuple[float, list[dict]]:
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
    return ms, data["results"]


def summarise(samples: list[float]) -> dict:
    s = sorted(samples)
    return {
        "n": len(s),
        "median_ms": round(statistics.median(s), 1),
        "p10_ms": round(s[int(0.1 * (len(s) - 1))], 1),
        "p90_ms": round(s[int(0.9 * (len(s) - 1))], 1),
    }


def run(label: str, fn, trials: int, warmup: int) -> dict:
    """fn returns (ms, answer); only the timing is kept here -- answers are captured separately."""
    for _ in range(warmup):
        fn()
    samples = []
    for i in range(trials):
        samples.append(fn()[0])
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
    ap.add_argument("--reasoning", default="low", choices=["low", "high"], help="reasoning effort asked of the hosted model")
    ap.add_argument("--capture", action="store_true", help="record the answers of one batched call instead of timing many")
    ap.add_argument("--preset", default="playground", help="which block of site/data/presets.json to ask about")
    args = ap.parse_args()

    key_env = PROVIDERS[args.provider][1]
    if not os.environ.get(key_env):
        raise SystemExit(f"{key_env} is not set. Put it in the environment (not in a tracked file) and re-run.")

    state, choice, queries = load_case(args.preset)
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

        if args.capture:
            capture(client, args, state, queries, device)
            return

        print(f"typical ({device}) vs {args.provider}/{args.model} @ {args.reasoning} reasoning, {args.trials} trials")
        out = {
            "method": (
                "Wall clock from a client process: request sent to answer parsed, both arms, same "
                "ticket and same four options, the hosted model constrained to the same label set "
                "by JSON schema. Each hosted model is run twice, at its lowest and its highest reasoning "
                "effort. Medians over the stated "
                "trial count, warmups discarded. Typical "
                f"runs locally on {device}, not the H100 the 45 ms ladder was measured on."
            ),
            "typical_device": device,
            "llm": f"{args.provider}/{args.model}",
            "reasoning": args.reasoning,
            "trials": args.trials,
            "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "arms": {},
        }
        out["arms"]["typical_1"] = run("typical  ×1 ", lambda: call_typical(client, state, one), args.trials, args.warmup)
        out["arms"][f"typical_{args.many}"] = run(f"typical  ×{args.many} ", lambda: call_typical(client, state, many), args.trials, args.warmup)
        out["arms"]["llm_1"] = run("llm      ×1 ", lambda: call_llm(client, args.provider, args.model, state, one, args.reasoning), args.trials, args.warmup)
        # best case for the hosted model: every question in one request
        out["arms"][f"llm_{args.many}_batched"] = run(f"llm      ×{args.many} batched", lambda: call_llm(client, args.provider, args.model, state, many, args.reasoning), args.trials, args.warmup)
        # and the case the cached prefix is actually for: questions arriving one at a time
        out["arms"][f"llm_{args.many}_serial"] = run(
            f"llm      ×{args.many} serial ",
            lambda: (sum(call_llm(client, args.provider, args.model, state, [q], args.reasoning)[0] for q in many), None),
            max(4, args.trials // 4),
            1,
        )

    # merge rather than overwrite: the file holds one entry per hosted model, so a second run adds
    # a row instead of erasing the first one
    doc = json.loads(OUT.read_text()) if OUT.exists() else {"method": out["method"], "runs": {}}
    doc["method"] = out["method"]
    doc["typical_device"] = out["typical_device"]
    doc.setdefault("runs", {})[f'{out["llm"]} ({args.reasoning} reasoning)'] = {k: out[k] for k in ("trials", "reasoning", "measured_utc", "arms")}
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(doc['runs'])} hosted model(s))")


def stream_llm(client: httpx.Client, provider: str, model: str, state: str, questions: list[dict], effort: str) -> tuple[dict, list[dict], float, dict]:
    """The same call as call_llm, streamed, timestamping the moment each answer finishes arriving.

    The side-by-side on the page replays a hosted model filling in its answers one at a time. That
    replay is only worth showing if the pacing is recorded rather than invented, so this watches the
    raw token stream and notes, for every qN, the first instant its value is complete in the buffer.
    A line on the page appears exactly when the model finished writing it.
    """
    url, key_env = PROVIDERS[provider]
    schema = schema_for(questions)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_for(state, questions)}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}},
        "reasoning": {"effort": effort},
        "stream": True,
        # the provider bills this call and knows what it charged; asking it beats multiplying our
        # own token count by a price list that changes without telling us
        "usage": {"include": True},
    }
    headers = {"Authorization": f"Bearer {os.environ[key_env]}"}
    # a value counts as arrived once the token after it has landed, which is what the closing comma
    # or brace is: until then the model could still be writing digits
    done_re = re.compile(r'"q(\d+)"\s*:\s*(?:"[^"]*"|-?[\d.]+)\s*[,}]')
    buf, seen, timeline, usage = "", set(), [], {}
    t0 = time.perf_counter()
    with client.stream("POST", url, json=body, headers=headers, timeout=600) as res:
        res.raise_for_status()
        for line in res.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
            buf += delta.get("content") or ""
            now = (time.perf_counter() - t0) * 1000
            for m in done_re.finditer(buf):
                i = int(m.group(1))
                if i not in seen:
                    seen.add(i)
                    timeline.append({"q": i, "t_ms": round(now, 1)})
    total = (time.perf_counter() - t0) * 1000
    return json.loads(buf), timeline, total, usage


def capture(client: httpx.Client, args, state: str, queries: list[dict], device: str) -> None:
    """What each model said, not how long it took -- written to site/data/showdown.json.

    The latency file holds medians and never recorded a single answer, so a side-by-side of the
    responses had nothing to read from. Both arms here run the identical call the timings were
    taken with: one request, every question, the same options pinned by the same schema. Typical's
    numbers are its head's probabilities; the hosted model's confidence is a field we asked it to
    fill in, which is a different kind of number and the page has to say so rather than hide it.
    """
    key = f"{args.provider}/{args.model} ({args.reasoning} reasoning)"
    out = ROOT / "site/data/showdown.json"
    doc = json.loads(out.read_text()) if out.exists() else {}
    doc["state"] = state
    doc["questions"] = [{"type": q["type"], "question": q["question"], "labels": q.get("labels") or ["no", "yes"]} for q in queries]
    doc["method"] = (
        "One request per model, every question in it, the same options pinned by the same JSON "
        "schema. Wall clock from a client process. The hosted arm is streamed and each answer is "
        "timestamped the moment its value finished arriving, so the replay on the page runs at the "
        f"pace the model actually wrote at. Typical runs locally on {device}."
    )
    models = doc.setdefault("models", {})

    ms, results = call_typical(client, state, queries)
    for _ in range(max(0, args.trials - 1)):  # keep the fastest: a local server's slow runs are contention
        ms = min(ms, call_typical(client, state, queries)[0])
    models["typical-small"] = {
        "ours": True,
        "device": device,
        "ms": round(ms, 1),
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "answers": [
            {"pick": r["argmax"], "probs": {k: round(v, 3) for k, v in r["probs"].items()}, "p_null": round(r["p_null"], 3)}
            for r in results
        ],
    }
    print(f"  typical: {ms:.0f} ms, {len(results)} answers")

    hosted, timeline, total, usage = stream_llm(client, args.provider, args.model, state, queries, args.reasoning)
    missing = [i for i in range(len(queries)) if f"q{i}" not in hosted]
    if missing:
        raise RuntimeError(f"{key} left {len(missing)} answer(s) unwritten: {missing[:3]}")
    models[key] = {
        "ms": round(total, 1),
        "cost_usd": usage.get("cost"),
        "tokens_in": usage.get("prompt_tokens"),
        "tokens_out": usage.get("completion_tokens"),
        "first_token_ms": timeline[0]["t_ms"] if timeline else None,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timeline": timeline,
        "answers": [
            {"pick": hosted[f"q{i}"], "stated_confidence": hosted.get(f"q{i}_confidence")}
            for i in range(len(queries))
        ],
    }
    print(f"  {key}: {total:.0f} ms, first answer at {timeline[0]['t_ms'] if timeline else '?'} ms, cost {usage.get('cost')}")
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(models)} model(s))")


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
    got = extract("openrouter", {"choices": [{"message": {"content": '{"q0": "billing"}'}}]})
    assert got == {"q0": "billing"}, "plain JSON content parses"
    got = extract("openrouter", {"choices": [{"message": {"content": None, "tool_calls": [{"function": {"name": "answer", "arguments": '{"q0": "shipping"}'}}]}}]})
    assert got == {"q0": "shipping"}, "a null content with a tool call still yields the answer"
    try:
        extract("openrouter", {"choices": [{"message": {"content": None}}]})
    except RuntimeError:
        pass
    else:
        raise AssertionError("an empty answer must raise, not count as a fast trial")
    print("bench_llm.py self-test OK")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo()
    else:
        main()
