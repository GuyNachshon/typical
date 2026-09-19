"""PLAN3 E3-ms, PI memo "train E3-ms on log-odds effects": after scripts/multiset.py expanded
a corpus and scripts/teacher_label.py labeled the expansion, write onto every variant row
  meta.orig_idx  index (in this file) of its group's orig row
  meta.delta_t   [[i, j, D_ij], ...] over pairs of candidates shared with the orig row, aligned by
                 candidate string; i/j index the VARIANT's candidates and
                 D_ij = [log P_T(i|A') - log P_T(j|A')] - [log P_T(i|A) - log P_T(j|A)]
                 (teacher log-odds shift), capped to the top-4 shared candidates by orig teacher prob.
Rows whose group has no labeled orig row get delta_t=[] (they still get CE/KL). The null column
cancels in every difference, so the K+1 `teacher` vectors are used as stored. In-place unless --out.
model.collate_delta turns delta_t into batch index tensors; decision_loss(..., gamma) applies it.

uv run scripts/ms_targets.py --file data_kbms/train.teacher.jsonl
"""
import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import write_jsonl
from scripts.multiset import load_jsonl

TOP_SHARED = 4


def delta_targets(orig, var, top=TOP_SHARED):
    """-> [[i_var, j_var, D_ij], ...] for the top-`top` shared candidates (by orig teacher prob)."""
    if not orig.get("teacher") or not var.get("teacher"):
        return []
    lo = {c: math.log(max(p, 1e-12)) for c, p in zip(orig["candidates"], orig["teacher"])}  # first dup wins
    lv = {c: (k, math.log(max(var["teacher"][k], 1e-12))) for k, c in reversed(list(enumerate(var["candidates"])))}
    shared = sorted((c for c in lv if c in lo), key=lambda c: -lo[c])[:top]
    return [[lv[a][0], lv[b][0], (lv[a][1] - lv[b][1]) - (lo[a] - lo[b])] for a, b in combinations(shared, 2)]


def add_targets(rows):
    orig_idx = {r["meta"]["ms_group"]: i for i, r in enumerate(rows) if r["meta"].get("ms_variant") == "orig"}
    n = 0
    for r in rows:
        m = r["meta"]
        if m.get("ms_variant") in (None, "orig"):
            continue
        oi = orig_idx.get(m["ms_group"])
        m["orig_idx"] = oi
        m["delta_t"] = delta_targets(rows[oi], r) if oi is not None else []
        n += bool(m["delta_t"])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="multiset-expanded, teacher-labeled jsonl")
    ap.add_argument("--out", default=None, help="default: overwrite --file")
    args = ap.parse_args()
    rows = load_jsonl(args.file)
    n = add_targets(rows)
    write_jsonl(Path(args.out or args.file), rows)
    print(f"[ms_targets] {n} variant rows got delta_t -> {args.out or args.file}")


if __name__ == "__main__":
    main()
