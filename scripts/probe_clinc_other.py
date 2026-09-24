"""PLAN7/S3w calib #3a ($0 probe): append a catch-all "other" candidate to every CLINC-150 test
row and see whether mass that would otherwise go to null instead lands on "other" -- the wf
checkpoint's main E-regression story (REPORT S3w) is that W trains routing/extraction/cua rows
against a rendered catch-all (other/not_stated/skip) that competes with the null column on
exactly the intent/topic label spaces where CLINC-150 false-abstain regressed.

Reports, for the plain set and the +"other" set: acc, mean P(null), mean P(other) (only
meaningful for the +"other" set). Uses the checkpoint's own fitted T (results.json) and the
exact eval file/config the checkpoint was scored on (data_v5/eval/clinc_test.jsonl, readout
native) so numbers are directly comparable to REPORT's cached clinc_test row.

uv run scripts/probe_clinc_other.py [--model runs/nc_v3_tap20_wf] [--data data_v5/eval/clinc_test.jsonl]
                                     [--limit N] [--bs 32] [--out runs/probe_clinc_other_wf/results.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pcdm"))


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def add_other(ex):
    ex2 = dict(ex)
    ex2["candidates"] = list(ex["candidates"]) + ["other"]
    ex2["target"] = list(ex["target"]) + [0.0]  # "other" is never gold in CLINC-150
    return ex2


def run(bb, model, examples, bs, T, null_mode, tag=""):
    import train as Tm
    n_correct, n = 0, 0
    p_null_sum, p_other_sum, nll_sum = 0.0, 0.0, 0.0
    for bi, (logits, cmask, target, p_null, ex_batch) in enumerate(
            Tm.forward_batches(bb, model, None, examples, bs, readout="native")):
        if bi % 10 == 0:
            print(f"  [{tag}] batch {bi} ({n}/{len(examples)} rows)", flush=True)
        probs = Tm.probs_from_logits(logits, cmask, T, null=null_mode)
        tgt = Tm.full_target_from(target, p_null)
        nll = -(tgt * torch.log(probs.clamp_min(1e-12))).sum(-1)
        nll_sum += nll.sum().item()
        pred = probs.argmax(-1)
        Kmax = cmask.shape[1]
        for i, ex in enumerate(ex_batch):
            lbl = ex.get("label")
            gold_idx = Kmax if (lbl is None or lbl == -1) else lbl
            n_correct += int(pred[i].item() == gold_idx)
            p_null_sum += probs[i, -1].item()
            p_other_sum += probs[i, len(ex["candidates"]) - 1].item()  # last real candidate slot
            n += 1
    return {"n": n, "acc": n_correct / n, "p_null_mean": p_null_sum / n,
            "p_other_mean": p_other_sum / n, "nll": nll_sum / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/nc_v3_tap20_wf")
    ap.add_argument("--data", default="data_v5/eval/clinc_test.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="shuffle before --limit (local-compute mode); unset = head-N")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--T", type=float, default=None, help="default: this run's fitted T from results.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import bench
    from encode import pick_device

    device = pick_device("auto")
    bb, model = bench.load_ours(args.model, None, device)
    model.eval()

    T = args.T
    if T is None:
        res_path = Path(args.model, "results.json")
        T = json.loads(res_path.read_text())["T"] if res_path.exists() else 1.0
    null_mode = getattr(model, "null", "softmax")

    examples = load_jsonl(args.data)
    if args.seed is not None:
        import random
        random.Random(args.seed).shuffle(examples)
    if args.limit:
        examples = examples[: args.limit]
    aug = [add_other(ex) for ex in examples]

    print(f"[probe] model={args.model} T={T:.4f} n={len(examples)}", flush=True)
    plain = run(bb, model, examples, args.bs, T, null_mode, tag="plain")
    other = run(bb, model, aug, args.bs, T, null_mode, tag="other")

    out = {
        "model": args.model, "data": args.data, "T": T, "n": len(examples),
        "plain": plain, "with_other": other,
        "delta_p_null": other["p_null_mean"] - plain["p_null_mean"],
        "delta_acc": other["acc"] - plain["acc"],
    }
    print(json.dumps(out, indent=2))

    out_path = Path(args.out) if args.out else ROOT / "runs" / f"probe_clinc_other_{Path(args.model).name}" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
