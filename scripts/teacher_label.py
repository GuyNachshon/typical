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
from mcq import MCQHead, collate_mcq, run_batch_mcq, MAXK_DIRECT, _ids, _pack
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


def score_batch_zero_shot(head, examples, T, shots_prefix="", shots_prefix_tokens=0, max_state=256, max_suffix=512):
    """PLAN7 tl1b item 6: frozen-backbone teacher, reusing pcdm_jev.decider's mcq_zero_shot
    rendering (_render_zero_shot: no null option is ever rendered, so p_null is hardcoded 0)
    and --shots machinery (_data_shots: worked examples from data_v5/val, not mcq.build_shots'
    generic trivia) -- so labels match the exact conditioning of the frozen-14B JevBench runs
    (REPORT S3ab: jev_zs3_14b, hard .559). Batched version of decider._probs_zero_shot; every
    example must have 2 <= K <= MAXK_DIRECT candidates (the caller filters -- see main()).
    -> list of (K+1)-float probability vectors (null always last, always 0.0), one per example."""
    from pcdm_jev.decider import _render_zero_shot
    tok = head.backbone.tokenizer
    states = [shots_prefix + ex["state"] for ex in examples]
    suffixes = [_render_zero_shot(ex["query"], ex["candidates"]) for ex in examples]
    s_ids = _ids(tok, states, max_state + shots_prefix_tokens)
    x_ids = _ids(tok, suffixes, max_suffix)
    input_ids, attention_mask, lengths = _pack(tok, s_ids, x_ids)
    dev = head.device
    input_ids, attention_mask = input_ids.to(dev), attention_mask.to(dev)
    out = head.backbone.model(input_ids=input_ids, attention_mask=attention_mask)
    last = out.last_hidden_state[torch.arange(len(examples), device=dev), (lengths - 1).to(dev)].float()
    letter_ids = head.letter_ids.to(dev)
    w = head.lm_head_weight.to(dev)[letter_ids].float()
    letter_logits = (last @ w.T) / T  # [B, 52]
    return [torch.softmax(letter_logits[i, :len(ex["candidates"])], dim=-1).cpu().tolist() + [0.0]
            for i, ex in enumerate(examples)]


def label_batch(head, examples, T, perms=1, seed=0, start_idx=0, scorer=None):
    """-> (probs, perm_std): probs as score_batch; perm_std None for perms=1 (score_batch verbatim),
    else the per-row mean per-candidate std across the P orderings. Permutation p>0 of global row
    r is seeded by (seed, r, p), so labels are reproducible across --bs and resumes.
    scorer: (head, examples, T) -> probs, defaulting to the module-level score_batch (trained
    mcq checkpoint); main() passes score_batch_zero_shot for --zero_shot."""
    score = scorer if scorer is not None else score_batch
    if perms == 1:
        return score(head, examples, T), None
    copies, orders = [], []
    for i, ex in enumerate(examples):
        for p in range(perms):
            order = list(range(len(ex["candidates"])))
            if p:
                random.Random(f"{seed}:{start_idx + i}:{p}").shuffle(order)
            copies.append(dict(ex, candidates=[ex["candidates"][k] for k in order]))
            orders.append(order)
    probs = score(head, copies, T)  # all P*B copies in one batch
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
    ap.add_argument("--run", default=None, help="runs/<name> dir containing best.pt (readout=mcq); unused with --zero_shot")
    ap.add_argument("--file", required=True, help="jsonl to label")
    ap.add_argument("--out", default=None, help="default: <file with .jsonl replaced by .teacher.jsonl>")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--T", type=float, default=1.0, help="teacher softmax temperature")
    ap.add_argument("--perms", type=int, default=1, help="P orderings per row, averaged (1 = plain, byte-identical)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the --perms permutations (and --shots exemplar draw)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16", help="backbone compute dtype (cuda: fp16/bf16)")
    ap.add_argument("--zero_shot", action="store_true",
                     help="PLAN7 tl1b item 6: frozen --backbone, no checkpoint -- reuses pcdm_jev.decider's "
                          "mcq_zero_shot rendering (null forced to 0, never rendered) and --shots worked-example "
                          "machinery. Only rows with an integer label >= 0 and 2 <= K <= mcq.MAXK_DIRECT candidates "
                          "get a teacher (others pass through unlabeled -- decision_loss ignores them via has_teacher)")
    ap.add_argument("--backbone", default=None, help="--zero_shot only, e.g. Qwen/Qwen3-14B-Base")
    ap.add_argument("--lora_r", type=int, default=0, help="--zero_shot only (0 = fully frozen)")
    ap.add_argument("--lora_layers", type=int, default=0, help="--zero_shot only")
    ap.add_argument("--tap_layer", type=int, default=0, help="--zero_shot only")
    ap.add_argument("--shots", type=int, default=0, help="--zero_shot only: N worked examples from data_v5/val (pcdm_jev.decider._data_shots)")
    ap.add_argument("--max_state", type=int, default=256, help="--zero_shot only: decision-state truncation length in tokens")
    args = ap.parse_args()

    device = pick_device(args.device)
    scorer = None
    if args.zero_shot:
        assert args.backbone, "--zero_shot needs --backbone"
        head = MCQHead(args.backbone, lora_layers=args.lora_layers, lora_r=args.lora_r,
                       device=device, tap_layer=args.tap_layer)
    else:
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

    if args.zero_shot:
        from pcdm_jev.decider import _data_shots
        shots_prefix = _data_shots(args.shots, seed=args.seed)
        shots_prefix_tokens = len(head.backbone.tokenizer(shots_prefix, add_special_tokens=False)["input_ids"]) \
            if shots_prefix else 0
        scorer = lambda h, exs, T: score_batch_zero_shot(h, exs, T, shots_prefix=shots_prefix,
                                                           shots_prefix_tokens=shots_prefix_tokens,
                                                           max_state=args.max_state)

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
    n_skip_label, n_skip_k1, n_skip_bigk = 0, 0, 0
    with open(out_path, "a" if n_done else "w") as f, torch.inference_mode():
        n_written = 0
        for bi, i in enumerate(range(0, len(remaining), args.bs)):
            ex_batch = remaining[i:i + args.bs]
            if args.zero_shot:
                # item 6: label>=0 rows only, skip K=1 (trivial) and K > MAXK_DIRECT (not chunked --
                # ponytail: JevBench-style zero-shot label sets are always small; add chunking if a
                # corpus ever needs it). Order preserved: unlabeled rows pass through untouched.
                eligible_idx = []
                for j, ex in enumerate(ex_batch):
                    lbl, K = ex.get("label"), len(ex["candidates"])
                    if not (isinstance(lbl, int) and lbl >= 0):
                        n_skip_label += 1
                    elif K == 1:
                        n_skip_k1 += 1
                    elif K > MAXK_DIRECT:
                        n_skip_bigk += 1
                    else:
                        eligible_idx.append(j)
                eligible = [ex_batch[j] for j in eligible_idx]
                teacher_probs, perm_std = label_batch(head, eligible, args.T, args.perms, args.seed, n_done + i,
                                                       scorer=scorer) if eligible else ([], None)
                tp_by_idx = dict(zip(eligible_idx, teacher_probs))
                std_by_idx = dict(zip(eligible_idx, perm_std)) if perm_std is not None else {}
                for j, ex in enumerate(ex_batch):
                    if j in tp_by_idx:
                        out_row = dict(ex, teacher=tp_by_idx[j], teacher_T=args.T)
                        if j in std_by_idx:
                            out_row["teacher_perm_std"] = std_by_idx[j]
                    else:
                        out_row = ex
                    f.write(json.dumps(out_row) + "\n")
            else:
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
    if args.zero_shot:
        print(f"zero-shot skips: {n_skip_label} label<0, {n_skip_k1} K=1, {n_skip_bigk} K>{MAXK_DIRECT}")
    print(f"done: {out_path} ({len(rows)} rows total)")


if __name__ == "__main__":
    main()
