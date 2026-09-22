"""Profile a single Typical decision on the serving path, and produce the before/after
latency table (p50/p95 ms) across models x state length x K x {cold, warm}.

"cold" = the state text is different on every call (state-cache miss every time, i.e. the
pre-optimisation baseline path: full re-encode of the state prefix). "warm" = the same
state text is reused across calls with a new question each time (state-cache hit after the
first call) -- the path Typical.choice/score/noul/decide take automatically via the LRU in
core.py.

Usage:
  uv run --no-sync python inference/bench_serve.py --models typical-small --out_dir runs/serve_bench
  uv run --no-sync python inference/bench_serve.py --models typical-small --profile   # chrome trace + key_averages table
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from typical import Typical

MODELS = {
    "typical-small": ("guychuk/pcdm-runs", "typical-small/best.pt"),   # 1.7B, Qwen3
    "typical-medium": ("guychuk/pcdm-runs", "typical-medium/best.pt"),  # 4B, Qwen3
    "tm2": ("guychuk/pcdm-runs", "tm2/best.pt"),                       # 4B, Qwen3.5
}
STATE_LENS = (256, 1024)
KS = (2, 4, 10, 32)
FILLER = ("Ticket log entry: the customer reported an issue with their recent order and "
          "followed up twice asking for a status update on the resolution. ")


def make_state(tok, n_tokens: int) -> str:
    """Filler text truncated to exactly n_tokens tokens (no special tokens)."""
    reps = n_tokens // len(tok(FILLER, add_special_tokens=False)["input_ids"]) + 3
    ids = tok(FILLER * reps, add_special_tokens=False)["input_ids"][:n_tokens]
    return tok.decode(ids)


def make_labels(k: int) -> list[str]:
    return [f"candidate option {j} short description text" for j in range(k)]


def make_items(tok, n: int = 20) -> list[dict]:
    """20 JevBench-style items spanning the (state_len, K) grid."""
    grid = [(sl, k) for sl in STATE_LENS for k in KS]
    items = []
    for i in range(n):
        sl, k = grid[i % len(grid)]
        items.append({"state_len": sl, "K": k, "state": make_state(tok, sl),
                      "question": f"Question {i}: which option best applies?",
                      "labels": make_labels(k)})
    return items


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit_ms(fn, n: int = 12) -> dict:
    times = []
    for _ in range(n):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"p50_ms": statistics.median(times), "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
            "n": n}


def bench_model(name: str, repo_id: str, filename: str) -> dict:
    m = Typical.from_pretrained(repo_id, filename=filename)
    tok = m.tok
    configs = [(sl, k) for sl in STATE_LENS for k in KS]
    results = {"model": name, "backbone": m.head.backbone.model.config.name_or_path
               if hasattr(m.head.backbone.model.config, "name_or_path") else None, "configs": []}
    for sl, k in configs:
        state = make_state(tok, sl)
        question, labels = "Which option best applies?", make_labels(k)
        nonce = [0]

        def cold_call():
            nonce[0] += 1
            m.choice(f"[{nonce[0]}] " + state, question, labels)

        def warm_call():
            m.choice(state, question, labels)

        warm_call()  # warm up CUDA kernels/allocator, populate state cache for warm_call
        cold_call()  # warm up the cold path's own kernels too (same shapes, different state text)
        cold = timeit_ms(cold_call)
        warm = timeit_ms(warm_call)
        print(f"  [{name}] state={sl:5d} K={k:3d}  cold p50={cold['p50_ms']:7.2f}ms p95={cold['p95_ms']:7.2f}ms"
              f"   warm p50={warm['p50_ms']:7.2f}ms p95={warm['p95_ms']:7.2f}ms"
              f"   speedup={cold['p50_ms']/max(warm['p50_ms'], 1e-6):.2f}x")
        results["configs"].append({"state_tokens": sl, "K": k, "cold": cold, "warm": warm})
    if torch.cuda.is_available():
        results["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return results


def profile_one(name: str, repo_id: str, filename: str, out_dir: str, state_len: int = 256, k: int = 10):
    """PyTorch profiler over a single warm choice() call -- reports the split between
    state-prefix forward, suffix forward, tokenisation, mask building, and cpu syncs (see
    the record_function blocks in typical/native.py)."""
    from torch.profiler import ProfilerActivity, profile

    m = Typical.from_pretrained(repo_id, filename=filename)
    tok = m.tok
    state, question, labels = make_state(tok, state_len), "Which option best applies?", make_labels(k)
    for _ in range(3):
        m.choice(state, question, labels)  # warmup (state cache hit after call 1)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    with profile(activities=activities) as prof:
        m.choice(state, question, labels)
    sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_by, row_limit=20)
    print(f"\n=== profile: {name} warm choice() state={state_len} K={k} ===\n{table}")
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, f"{name}_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"trace written to {trace_path}")

    # cold, for contrast
    m._state_kv.clear()
    with profile(activities=activities) as prof_cold:
        m.choice(state + " [cold]", question, labels)
    table_cold = prof_cold.key_averages().table(sort_by=sort_by, row_limit=20)
    print(f"\n=== profile: {name} cold choice() state={state_len} K={k} ===\n{table_cold}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    ap.add_argument("--out_dir", default="runs/serve_bench")
    ap.add_argument("--profile", action="store_true", help="also run torch.profiler over one warm+cold call")
    ap.add_argument("--tag", default="results", help="output json filename stem")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    for name in args.models:
        repo_id, filename = MODELS[name]
        print(f"=== {name} ({repo_id}/{filename}) ===")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        res = bench_model(name, repo_id, filename)
        all_results.append(res)
        if args.profile:
            profile_one(name, repo_id, filename, args.out_dir)

    out_path = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {out_path}")
