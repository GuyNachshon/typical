"""PLAN6 Phase 6: post-hoc evaluation of a checkpoint on data_wf / data_wf_hf eval files at full state
length (train-time evals truncate states to 256 tokens and are head-capped by --eval_cap).

    uv run python scripts/eval_wf.py --run runs/nc_v3_tap20_wf --mode native \
        --files data_wf/eval/*.jsonl data_wf_hf/eval/*.jsonl --limit 2000 --out runs/nc_v3_tap20_wf/eval_wf.json

Rows are the training schema (state/query/candidates/target/p_null/label), so the query is used verbatim
(no query_text re-rendering). Per file: metrics.summarize (raw, T=1) + metrics.rubric_flip for wf_rubric_flip*.
--limit is head-N so adjacent flip pairs survive. Soft-gold sets (typed_decisions) report NLL/Brier; acc there is
argmax-vs-argmax agreement, not truth."""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import rubric_flip, summarize  # noqa: E402
from pcdm_jev.decider import PCDMDecider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", default="native", choices=["native", "energy"])
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None, help="head-N rows per file")
    ap.add_argument("--max_state", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dec = PCDMDecider(args.run, mode=args.mode, max_state=args.max_state)
    out = {"run": args.run, "mode": args.mode, "max_state": args.max_state, "limit": args.limit, "eval": {}}
    for f in sorted(p for pat in args.files for p in glob.glob(pat)):
        rows = [json.loads(l) for l in open(f)][: args.limit]
        kmax = max(len(r["candidates"]) for r in rows)
        probs, target = np.zeros((len(rows), kmax + 1)), np.zeros((len(rows), kmax + 1))
        label, overflow = np.full(len(rows), np.nan), 0
        with torch.inference_mode():
            for i, r in enumerate(tqdm(rows, desc=Path(f).stem, leave=False)):
                k = len(r["candidates"])
                try:
                    p = dec._probs(args.mode, r["state"], r["query"], r["candidates"]).float().cpu().numpy()
                except AssertionError:  # native suffix overflow (e.g. tree_choice flat K=320): count it, score as uniform
                    overflow += 1
                    p = np.full(k + 1, 1.0 / (k + 1))
                probs[i, :k], probs[i, -1] = p[:k], p[-1]
                target[i, :k] = (1 - r["p_null"]) * np.asarray(r["target"], dtype=float)
                target[i, -1] = r["p_null"]
                if r.get("label") is not None:
                    label[i] = r["label"]
        name = Path(f).stem
        out["eval"][name] = {"n": len(rows), "suffix_overflow": overflow, "raw": summarize(probs, target, label)}
        if name.startswith("wf_rubric_flip"):
            out["eval"][name]["flip"] = rubric_flip(probs, rows)
        print(name, f"overflow={overflow}", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["eval"][name]["raw"].items()}),
              out["eval"][name].get("flip", ""), flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1))  # checkpoint after every file


if __name__ == "__main__":
    main()
