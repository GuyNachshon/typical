"""Soft labels from an options-in-context (mcq-readout) teacher, for PLAN3 E3-T.

Loads runs/<name>/best.pt (must be a --readout mcq checkpoint), scores every row of a
jsonl file with mcq.py's own batching (collate_mcq/run_batch_mcq -- the K>51 two-stage
chunked scorer just works, unchanged), and writes <file>.teacher.jsonl: each input row's
fields plus:
  teacher:   softmax(logits/T) over [candidates..., null], K+1 floats (K = len(candidates))
  teacher_T: the temperature used (--T, default 1.0)
  teacher_perm_std: (--perms P > 1 only) mean per-candidate std of the P per-permutation
             probability vectors -- how order-sensitive the teacher was on this row

--perms P (PI memo, "symmetrize the teacher first"): score each row under P orderings of its
candidates (identity + P-1 seeded random permutations, all P copies in the same forward batch),
map every probability vector back to candidate identity and average -> `teacher`. P=1 (default)
is today's behaviour, byte-identical. Reorder then becomes an invariance constraint for the
student, not something it imitates.

Resumable: if the output file already exists, its line count is treated as "rows already
done" and labeling continues from there (assumes --file hasn't changed between runs).

uv run scripts/teacher_label.py --run runs/mcq_lora --file data_kb/train.jsonl [--bs 32] [--T 1.0]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcq import MCQHead, collate_mcq, run_batch_mcq
from encode import pick_device

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score_batch(head, examples, T):
    """-> list of (K+1)-float probability vectors, one per example, in order."""
    batch = collate_mcq(examples)
    logits = run_batch_mcq(head, batch, examples)
    probs = torch.softmax(logits.float() / T, dim=-1).cpu()
    out = []
    for ex, p in zip(examples, probs):
        K = len(ex["candidates"])
        out.append(torch.cat([p[:K], p[-1:]]).tolist())  # null is always the last column
    return out


def label_batch(head, examples, T, perms=1, seed=0, start_idx=0):
    """-> (probs, perm_std): probs as score_batch; perm_std None for perms=1 (score_batch verbatim),
    else the per-row mean per-candidate std across the P orderings. Permutation p>0 of global row
    r is seeded by (seed, r, p), so labels are reproducible across --bs and resumes."""
    if perms == 1:
        return score_batch(head, examples, T), None
    copies, orders = [], []
    for i, ex in enumerate(examples):
        for p in range(perms):
            order = list(range(len(ex["candidates"])))
            if p:
                random.Random(f"{seed}:{start_idx + i}:{p}").shuffle(order)
            copies.append(dict(ex, candidates=[ex["candidates"][k] for k in order]))
            orders.append(order)
    probs = score_batch(head, copies, T)  # all P*B copies in one batch
    out, std = [], []
    for i, ex in enumerate(examples):
        K = len(ex["candidates"])
        stack = torch.zeros(perms, K + 1)
        for p in range(perms):
            pv = probs[i * perms + p]
            stack[p, orders[i * perms + p]] = torch.tensor(pv[:K])  # permuted slot k -> orig index order[k]
            stack[p, K] = pv[K]
        out.append(stack.mean(0).tolist())
        std.append(stack.std(0).mean().item())
    return out, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<name> dir containing best.pt (readout=mcq)")
    ap.add_argument("--file", required=True, help="jsonl to label")
    ap.add_argument("--out", default=None, help="default: <file with .jsonl replaced by .teacher.jsonl>")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--T", type=float, default=1.0, help="teacher softmax temperature")
    ap.add_argument("--perms", type=int, default=1, help="P orderings per row, averaged (1 = plain, byte-identical)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the --perms permutations")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16", help="backbone compute dtype (cuda: fp16/bf16)")
    args = ap.parse_args()

    device = pick_device(args.device)
    ckpt_path = Path(args.run) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)  # trusted, self-produced
    rargs = ckpt.get("args", {})
    assert rargs.get("readout") == "mcq", f"{ckpt_path} is not a --readout mcq checkpoint"

    head = MCQHead(rargs["backbone"], lora_layers=rargs["lora_layers"], lora_r=rargs["lora_r"],
                   device=device, tap_layer=rargs.get("tap_layer", 0))
    head.load_lora_state_dict(ckpt["lora"])
    head.eval()
    dtype = DTYPES[args.dtype]
    if dtype != torch.bfloat16:  # Backbone always loads bf16; only cast away from it on request
        head.backbone.model.to(dtype)
        head.lm_head_weight = head.lm_head_weight.to(dtype)

    rows = load_jsonl(args.file)
    out_path = Path(args.out) if args.out else Path(args.file).with_suffix("").with_suffix(".teacher.jsonl")
    n_done = 0
    if out_path.exists():
        with open(out_path) as f:
            n_done = sum(1 for _ in f)
        print(f"resuming {out_path}: {n_done}/{len(rows)} rows already labeled")
    remaining = rows[n_done:]
    if not remaining:
        print("nothing to do")
        return

    t0 = time.time()
    with open(out_path, "a" if n_done else "w") as f, torch.inference_mode():
        n_written = 0
        for bi, i in enumerate(range(0, len(remaining), args.bs)):
            ex_batch = remaining[i:i + args.bs]
            teacher_probs, perm_std = label_batch(head, ex_batch, args.T, args.perms, args.seed, n_done + i)
            for r, (ex, tp) in enumerate(zip(ex_batch, teacher_probs)):
                out_row = dict(ex, teacher=tp, teacher_T=args.T)
                if perm_std is not None:
                    out_row["teacher_perm_std"] = perm_std[r]
                f.write(json.dumps(out_row) + "\n")
            n_written += len(ex_batch)
            if bi % 100 == 0:
                elapsed = time.time() - t0
                rate = n_written / max(elapsed, 1e-9)
                print(f"batch {bi}: {n_done + n_written}/{len(rows)} rows "
                      f"({rate:.1f} rows/s, {elapsed:.0f}s elapsed)")
    print(f"done: {out_path} ({len(rows)} rows total)")


if __name__ == "__main__":
    main()
