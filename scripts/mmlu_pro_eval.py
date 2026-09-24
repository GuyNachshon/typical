"""Build <data>/eval/mmlu_pro.jsonl (+ .meta.json) from TIGER-Lab/MMLU-Pro's test
split: a fixed-seed, category-stratified sample. Row schema matches the other eval
sets (see pcdm/data.py / data_v4/eval/boolq_val.jsonl): state/query/candidates/target/
p_null/task/label/meta. Gold is always present (p_null=0.0), candidates = the option
texts with letters dropped, state = the question stem.

uv run scripts/mmlu_pro_eval.py [--data data_v4] [--n 1200] [--seed 0]
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcdm"))
from data import trunc, write_jsonl  # reuse the same state-truncation/writer convention as pcdm/data.py

QUERY = "Which option is correct?"
MAX_CAND = 300  # sane cap; MMLU-Pro options are occasionally long multi-clause strings


def stratified_sample(rows, n, rng):
    """Proportional-by-category sample of `rows` totalling exactly n. Largest-remainder
    (Hamilton) apportionment: floor each category's exact share, then hand out the
    leftover seats to the categories with the biggest fractional remainder -- avoids the
    "+diff on one category" shortcut going negative when n is small relative to the
    number of categories."""
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    cats = sorted(by_cat)
    total = len(rows)
    raw = {c: n * len(by_cat[c]) / total for c in cats}
    quota = {c: int(raw[c]) for c in cats}
    remainder = n - sum(quota.values())
    order = sorted(cats, key=lambda c: raw[c] - quota[c], reverse=True)
    for c in order[:remainder]:
        quota[c] += 1

    picked = []
    for c in cats:
        v = list(by_cat[c])
        rng.shuffle(v)
        picked.extend(v[:quota[c]])
    rng.shuffle(picked)
    return picked


def build_row(r):
    options = [trunc(o, MAX_CAND) for o in r["options"]]
    label = r["answer_index"]
    target = [1.0 if i == label else 0.0 for i in range(len(options))]
    return {
        "state": trunc(r["question"]), "query": QUERY, "candidates": options,
        "target": target, "p_null": 0.0, "task": "mmlu_pro", "label": label,
        "meta": {"family": "qa", "question": QUERY, "category": r["category"], "question_id": r["question_id"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v4", help="data_v4 or data_v5; writes <data>/eval/mmlu_pro.jsonl")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = list(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
    rng = random.Random(args.seed)
    sample = stratified_sample(ds, min(args.n, len(ds)), rng)

    rows = [build_row(r) for r in sample]
    out_dir = Path(args.data) / "eval"
    write_jsonl(out_dir / "mmlu_pro.jsonl", rows)

    meta = [{"question_id": r["question_id"], "category": r["category"]} for r in sample]
    with open(out_dir / "mmlu_pro.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    by_cat = defaultdict(int)
    for r in sample:
        by_cat[r["category"]] += 1
    print(f"[mmlu_pro_eval] wrote {len(rows)} rows -> {out_dir / 'mmlu_pro.jsonl'}")
    for c in sorted(by_cat):
        print(f"  {c:<20}{by_cat[c]}")


if __name__ == "__main__":
    main()
