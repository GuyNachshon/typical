"""Run PCDM through the JevBench harness (github.com/fstandhartinger/jevbench, MIT) on its
public files and print a compact table.

uv run scripts/jevbench_run.py --model runs/joint_emb_lw_v5 --mode energy --name e_v5
uv run scripts/jevbench_run.py --model runs/nc_n3 --mode native --name n3
uv run scripts/jevbench_run.py --model runs/nc_n3 --mode compose --energy_run runs/joint_emb_lw_v5 --name comp
uv run scripts/jevbench_run.py --mode mcq_zero_shot --backbone Qwen/Qwen3-4B-Base --name zs_4b

-> runs/jev_<name>/<tier>/{results.jsonl, summary.json, manifest.json, raw/} per public file
   (original | easy | hard; the harness keeps tiers apart) + runs/jev_<name>/summary.json.
The run loop is jevbench's own Runner/Ledger (the CLI's adapter table is a local dict, so a
new adapter can't be passed through `cli run` without patching it); `summarize` is the CLI.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TIER_DISPLAY = {"original": "standard (public 72/96)"}  # jevbench's own name for the 72-item public standard tier
LEADERBOARD_TIER = {"original": "standard", "easy": "easy", "hard": "hard"}  # our tier name -> jevbench-v1.2-per-task.json's tier name


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="run dir (best.pt); unused for --mode mcq_zero_shot")
    ap.add_argument("--mode", default="energy", choices=["energy", "native", "compose", "mcq_zero_shot"])
    ap.add_argument("--energy_run", default=None, help="energy run dir (support gate) for --mode compose")
    ap.add_argument("--backbone", default=None, help="--mode mcq_zero_shot: frozen HF backbone, no checkpoint")
    ap.add_argument("--tap_layer", type=int, default=0, help="--mode mcq_zero_shot: layer to truncate to (0 = full depth)")
    ap.add_argument("--shots", type=int, default=0,
                     help="--mode mcq_zero_shot: N worked examples from data_v5/val.jsonl prepended to the "
                          "state (fixed seed, same exemplars at every size; never from JevBench or W eval sets)")
    ap.add_argument("--prompt_style", choices=["ours", "semif"], default="ours",
                     help="--mode mcq_zero_shot: 'semif' renders SemIf's (github.com/TheoLeeCJ/SemIf) `direct` "
                          "chat-template JSON prompt + bare-letter answer slots instead of our default "
                          "raw-text 'A. opt\\nAnswer:' rendering; incompatible with --shots (0-shot only)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", default="original,easy,hard", help="public files, comma-separated")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_state", type=int, default=4096)
    ap.add_argument("--jevbench_dir", default="/tmp/jevbench")
    ap.add_argument("--reverse_labels", action="store_true",
                     help="reverse each task's label order (order-sensitivity probe); stored under runs/jev_<name>_rev/")
    args = ap.parse_args()

    if args.mode == "compose":
        ap.error("compose ≡ native under the harness (r cancels out of the renormalised label "
                  "distribution -- see to_labels); use scripts/compose_support.py for the abstention-gate analysis instead")
    if args.mode == "mcq_zero_shot" and not args.backbone:
        ap.error("--mode mcq_zero_shot needs --backbone")
    if args.mode != "mcq_zero_shot" and not args.model:
        ap.error("--model is required for --mode energy|native")

    sys.path.insert(0, args.jevbench_dir)
    import jevbench.adapters
    from jevbench import cli
    from jevbench.budget import Ledger
    from jevbench.runner import Runner
    from jevbench.tasks import dataset_hash, load_jsonl
    from pcdm_jev.adapter import LocalPCDMAdapter
    from pcdm_jev.decider import MAX_QUERY, MAX_STATE
    jevbench.adapters.LocalPCDMAdapter = LocalPCDMAdapter  # the registry is the module namespace

    out = ROOT / "runs" / f"jev_{args.name}{'_rev' if args.reverse_labels else ''}"
    out.mkdir(parents=True)  # exclusive, like every jevbench run dir
    adapter = LocalPCDMAdapter(endpoint=args.energy_run, model=args.model, mode=args.mode,
                               device=args.device, max_state=args.max_state,
                               backbone=args.backbone, tap_layer=args.tap_layer, shots=args.shots,
                               prompt_style=args.prompt_style)
    adapter.load()
    print(f"[jev] loaded {args.mode} in {adapter.load_s:.1f}s", flush=True)
    if args.reverse_labels:  # ponytail: swap label order at the request boundary, no Task/adapter internals touched
        _build = adapter.build_request

        def _reversed_build(task, _build=_build):
            req = _build(task)
            req["labels"] = list(reversed(req["labels"]))
            return req
        adapter.build_request = _reversed_build

    try:
        harness_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.jevbench_dir,
                                        capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        harness_commit = None
    ckpt = Path(args.model, "best.pt") if args.model else None

    ledger = Ledger(out / "ledger.jsonl")
    summaries = {}
    manifests = {}
    for tier in args.tasks.split(","):
        tfile = Path(args.jevbench_dir, "datasets/public", f"{tier}.jsonl")
        tasks = load_jsonl(str(tfile))[: args.limit]
        d = out / tier
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        records = Runner(adapter, ledger, raw_dir=d / "raw", default_reserve_usd=0.0).run_all(
            tasks, results_path=d / "results.jsonl")
        manifest = {
            "run_label": args.name, "adapter": adapter.name, "requested_model": args.model or args.backbone,
            "mode": args.mode, "backbone": args.backbone, "tap_layer": args.tap_layer, "shots": args.shots,
            "prompt_style": args.prompt_style,
            "energy_run": args.energy_run, "device": adapter.load().device, "max_state": args.max_state,
            "reverse_labels": args.reverse_labels,
            "cost_basis": adapter.cost_basis, "dataset_hash": dataset_hash(tasks), "n_planned": len(tasks),
            "n_attempted": len(records), "started_utc": started,
            "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and "cuda" in adapter.load().device else None,
            "torch_version": torch.__version__,
            "checkpoint_sha256": _sha256(ckpt) if ckpt and ckpt.exists() else None,
            "harness_commit": harness_commit,
            "trained_max_state": MAX_STATE, "trained_max_query": MAX_QUERY,
            "load_s": adapter.load_s,
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        manifests[tier] = manifest
        with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):  # the CLI prints the full JSON; we keep the export
            cli.main(["summarize", "--tasks", str(tfile), "--results", str(d / "results.jsonl"),
                      "--public-export", str(d / "summary.json"), "--ledger", str(ledger.path)])
        summaries[tier] = json.loads((d / "summary.json").read_text())
        summaries[tier]["records"] = records

    incomplete = [t for t, m in manifests.items() if m["n_attempted"] < m["n_planned"]]
    if incomplete and args.limit is None:
        print(f"[jev] refusing to write top-level summary.json: incomplete tiers {incomplete} "
              f"(pass --limit if partial runs are expected)", flush=True)
    else:
        (out / "summary.json").write_text(json.dumps(
            {t: {k: v for k, v in s.items() if k != "records"} for t, s in summaries.items()}, indent=2, sort_keys=True))
    print_table(summaries, manifests)
    print_leaderboard_comparison(summaries, args.jevbench_dir)


def _f(x, w=6, nd=3):
    return f"{x:{w}.{nd}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"


def print_table(summaries: dict, manifests: dict = None):
    manifests = manifests or {}
    print(f"\n{'tier/family':<28}{'n':>4}{'acc':>7}{'brier':>7}{'ece':>7}{'p50s':>7}{'p95s':>7}")
    for tier, s in summaries.items():
        for fam, m in sorted(s["per_family"].items()):
            if not m["n_scorable"]:
                continue
            print(f"{tier + '/' + fam:<28}{m['n_scorable']:>4}{_f(m['accuracy'])}{_f(m['brier_mean'])}"
                  f"{_f((m['ece'] or {}).get('ece'))}{_f(m['latency']['p50_s'])}{_f(m['latency']['p95_s'])}")
        label = TIER_DISPLAY.get(tier, tier)
        print(f"{label + ' (all)':<28}{s['n_scorable']:>4}{_f(s['accuracy'])}{_f(s['brier_mean'])}"
              f"{_f((s['ece'] or {}).get('ece'))}{_f(s['latency']['p50_s'])}{_f(s['latency']['p95_s'])}")
        m = manifests.get(tier)
        if m:
            complete = m["n_attempted"] >= m["n_planned"]
            rt = [r.get("runtime") or {} for r in s["records"]]
            n = max(len(rt), 1)
            beyond_state = sum(bool(x.get("state_beyond_train_len")) for x in rt)
            beyond_query = sum(bool(x.get("query_beyond_train_len")) for x in rt)
            p_nulls = [x.get("p_null", 0) for x in rt]
            failures = m["n_planned"] - m["n_attempted"]
            print(f"  {label}: complete={complete} attempted/planned={m['n_attempted']}/{m['n_planned']} "
                  f"failures={failures} state_beyond_train={100 * beyond_state / n:.0f}% "
                  f"query_beyond_train={100 * beyond_query / n:.0f}% mean_p_null={sum(p_nulls) / n:.3f} "
                  f"p_null>0.5={100 * sum(p > 0.5 for p in p_nulls) / n:.0f}%")
    recs = [r for s in summaries.values() for r in s["records"]]
    rt = [r.get("runtime") or {} for r in recs]
    lat = sorted(r["latency_s"] for r in recs)
    trunc = sum(bool(x.get("state_truncated")) for x in rt)
    qtrunc = sum(bool(x.get("query_truncated")) for x in rt)
    print(f"\noverall: n={len(recs)} valid={sum(bool(r['valid']) for r in recs)} "
          f"p50={lat[len(lat) // 2]:.3f}s p95={lat[int(0.95 * (len(lat) - 1))]:.3f}s "
          f"states truncated={100 * trunc / max(len(rt), 1):.0f}% queries truncated={100 * qtrunc / max(len(rt), 1):.0f}% "
          f"mean p_null={sum(x.get('p_null', 0) for x in rt) / max(len(rt), 1):.3f}")


def print_leaderboard_comparison(summaries: dict, jevbench_dir: str):
    """Per-tier accuracy on exactly our public task ids, ours vs. every leaderboard system that
    publishes per-item public outcomes in jevbench-v1.2-per-task.json (systems[*].public_tasks:
    task_id -> [c|w|f|n, latency_s]). This is the only apples-to-apples comparison for a
    public-subset run: tier-level leaderboard accuracy (results.json/per-task.json by_tier) also
    counts held-out and imported items we never see, so it isn't comparable to a public-only run."""
    pt_path = Path(jevbench_dir, "results/v1.2/jevbench-v1.2-per-task.json")
    if not pt_path.exists():
        return
    lb = json.loads(pt_path.read_text())
    print(f"\n{'tier':<28}{'system':<40}{'n':>5}{'acc':>7}")
    for tier, s in summaries.items():
        lb_tier = LEADERBOARD_TIER.get(tier)
        if lb_tier is None:
            continue
        ids = [r["task_id"] for r in s["records"]]
        ours_acc = sum(bool(r["correct"]) for r in s["records"]) / max(len(ids), 1)
        print(f"{TIER_DISPLAY.get(tier, tier):<28}{'ours':<40}{len(ids):>5}{_f(ours_acc)}")
        for sys_id, sysd in sorted(lb.get("systems", {}).items()):
            pub = sysd.get("public_tasks", {})
            outcomes = [pub[i][0] for i in ids if i in pub]
            if not outcomes:
                continue
            acc = sum(o == "c" for o in outcomes) / len(outcomes)
            print(f"{'':<28}{sysd.get('display', sys_id)[:39]:<40}{len(outcomes):>5}{_f(acc)}")


if __name__ == "__main__":
    main()
