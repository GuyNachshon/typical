"""PLAN3 E3-ms: multi-set expansion. Given a jsonl of MCQ rows (data_kb/train.jsonl
schema: state/query/candidates/target/p_null/task/label/meta), write <out> where each
row becomes the original plus --variants extra rows sharing the same state/query but a
different candidate set -- same-Z, multiple-set supervision so a downstream student is
forced to factor candidate-conditioned reasoning through a candidate-blind decision
state (PLAN3.md E3 row's E3-ms follow-up). Variant types, cycled in this order (--variants N
takes the first N; the PI memo's invariance / set_context split is written to meta.ms_kind):
  remove        set_context  drop up to 2 random non-gold options (gold-absent rows: any), min K=2
  add_unrel     invariance   add 2 options drawn from other rows of the same `task` source
  reorder       invariance   random permutation of the existing options
  replace_hard  set_context  swap the teacher's top non-gold option for an unrelated one
  add_neardup   set_context  append a near-duplicate of gold, target split 0.5/0.5 (uniform if gold-absent)
  remove_strong set_context  drop the teacher's top-2 non-gold options, min K=2
The teacher-guided ones read the orig row's teacher probs from --teacher_probs FILE (a
teacher_label.py output of the SAME --file, aligned by row order) or, failing that, from the
row's own `teacher` field; with neither they fall back to random picks and say so.
Gold stays present on gold-present rows (p_null=0); on gold-absent rows (p_null=1) the
target stays uniform over whatever candidates remain -- p_null itself is never touched.
meta.ms_group = source row index, meta.ms_variant = "orig"/<variant name>, meta.ms_kind.
Deterministic given --seed.

head -n 30000 data_kb/train.teacher.jsonl > data_kbms/orig.jsonl   # rows carry `teacher` already
uv run scripts/multiset.py --file data_kbms/orig.jsonl --out data_kbms/train.jsonl --variants 6
cp data_kb/val.jsonl data_kbms/val.jsonl   # val stays single-set, becomes the data_kbms_val eval set
uv run scripts/teacher_label.py --run runs/mcq_lora --file data_kbms/train.jsonl --perms 3
uv run scripts/ms_targets.py --file data_kbms/train.teacher.jsonl   # -> meta.orig_idx / meta.delta_t
uv run pcdm/train.py --extra_data data_kbms --delta_gamma 1 ...   # (same flags as the plain data_kb E3 run)
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcdm"))
from data import write_jsonl  # same writer convention as the rest of the repo
from scripts.mmlu_counterfactual import near_dup_text

VARIANT_TYPES = ["remove", "add_unrel", "reorder", "replace_hard", "add_neardup", "remove_strong"]
VARIANT_KIND = {"reorder": "invariance", "add_unrel": "invariance"}  # everything else: set_context


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_pools(rows):
    """task -> flat list of candidate strings, for add_unrel sampling."""
    pools = {}
    for r in rows:
        pools.setdefault(r["task"], []).extend(r["candidates"])
    return pools


def one_hot(idx, n):
    return [1.0 if i == idx else 0.0 for i in range(n)]


def uniform(n):
    return [1.0 / n] * n


def non_gold_by_teacher(cands, gold_idx, teacher, rng):
    """Non-gold indices, strongest first: by teacher prob when given, else a random order."""
    idx = [i for i in range(len(cands)) if i != gold_idx]
    if teacher is not None:
        return sorted(idx, key=lambda i: -teacher[i])
    rng.shuffle(idx)
    return idx


def variant_remove(cands, gold_idx, rng, teacher=None, strong=False):
    """Kept indices after dropping non-gold options (min K=2): a random 1-2 of them, or with
    strong=True the teacher's top-2 (random 2 without a teacher)."""
    n = len(cands)
    droppable = non_gold_by_teacher(cands, gold_idx, teacher, rng)
    max_drop = min(2, n - 2, len(droppable))
    if max_drop <= 0:
        return list(range(n))
    drop = set(droppable[:max_drop] if strong else rng.sample(droppable, rng.randint(1, max_drop)))
    return [i for i in range(n) if i not in drop]


def variant_add_unrel(cands, task, pools, rng):
    pool = [c for c in pools.get(task, []) if c not in cands]
    rng.shuffle(pool)
    return pool[:2]


def make_variant(row, group_idx, name, pools, rng, teacher=None):
    cands = row["candidates"]
    gold_idx = row["label"] if row["p_null"] < 1.0 else None
    target = None

    if name in ("remove", "remove_strong"):
        kept = variant_remove(cands, gold_idx, rng, teacher, strong=name == "remove_strong")
        new_cands = [cands[i] for i in kept]
        new_label = kept.index(gold_idx) if gold_idx is not None else -1
    elif name == "add_unrel":
        new_cands = cands + variant_add_unrel(cands, row["task"], pools, rng)
        new_label = gold_idx if gold_idx is not None else -1
    elif name == "reorder":
        order = list(range(len(cands)))
        rng.shuffle(order)
        new_cands = [cands[i] for i in order]
        new_label = order.index(gold_idx) if gold_idx is not None else -1
    elif name == "replace_hard":
        new_cands = list(cands)
        hard = non_gold_by_teacher(cands, gold_idx, teacher, rng)
        if hard:
            new_cands[hard[0]] = (variant_add_unrel(cands, row["task"], pools, rng) or ["an unrelated option"])[0]
        new_label = gold_idx if gold_idx is not None else -1
    elif name == "add_neardup":
        # ponytail: gold-absent rows near-dup the teacher's top / a random option, target stays uniform
        src = gold_idx if gold_idx is not None else non_gold_by_teacher(cands, None, teacher, rng)[0]
        new_cands = cands + [near_dup_text(cands[src], rng)]
        new_label = gold_idx if gold_idx is not None else -1
        if gold_idx is not None:
            target = one_hot(gold_idx, len(new_cands))
            target[gold_idx] = target[-1] = 0.5
    else:
        raise ValueError(name)

    if target is None:
        target = one_hot(new_label, len(new_cands)) if gold_idx is not None else uniform(len(new_cands))
    meta = dict(row.get("meta", {}), ms_group=group_idx, ms_variant=name,
                ms_kind=VARIANT_KIND.get(name, "set_context"))
    return {"state": row["state"], "query": row["query"], "candidates": new_cands, "target": target,
            "p_null": row["p_null"], "task": row["task"], "label": new_label, "meta": meta}


def expand(rows, n_variants, seed, teachers=None):
    """teachers: per-row candidate probs (len(rows) lists / None entries) for the teacher-guided variants."""
    rng = random.Random(seed)
    pools = build_pools(rows)
    out = []
    for i, row in enumerate(rows):
        out.append({**row, "meta": dict(row.get("meta", {}), ms_group=i, ms_variant="orig")})
        for j in range(n_variants):
            name = VARIANT_TYPES[j % len(VARIANT_TYPES)]
            out.append(make_variant(row, i, name, pools, rng, teachers[i] if teachers else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="input jsonl (data_kb/train.jsonl schema)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher_probs", default=None,
                    help="teacher_label.py output of --file (same row order); default: the rows' own `teacher` field")
    args = ap.parse_args()

    rows = load_jsonl(args.file)
    trows = load_jsonl(args.teacher_probs) if args.teacher_probs else rows
    assert len(trows) == len(rows), f"--teacher_probs has {len(trows)} rows, --file {len(rows)}"
    teachers = [t.get("teacher") for t in trows]
    n_t = sum(t is not None for t in teachers)
    if n_t < len(rows):
        print(f"[multiset] {len(rows) - n_t}/{len(rows)} rows have no teacher probs: "
              "replace_hard/add_neardup/remove_strong fall back to random picks there")
    out_rows = expand(rows, args.variants, args.seed, teachers)
    write_jsonl(Path(args.out), out_rows)
    print(f"[multiset] {len(rows)} rows x (1 + {args.variants}) -> {len(out_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
