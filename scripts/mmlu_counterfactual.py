"""PLAN3 Task A (E3-cf): from <data>/eval/mmlu_pro.jsonl (scripts/mmlu_pro_eval.py) build
<data>/eval/mmlu_cf.jsonl -- per question, the option-set counterfactual battery: orig,
remove3, add3_unrel, replace_hard, add_neardup, reorder. All variants of one question share
meta.qid (choice_set_effects/cse_variants' meta.pair convention, renamed since a question
has 6 variants, not a base+one-variant pair) and carry meta.variant. Gold is present in
every variant (p_null=0.0), fixed seed. See metrics.counterfactual for the matching scorer.

--teacher_probs FILE (jsonl): either {"qid": ..., "probs": {candidate_str: prob, ...}} rows
or {"qid": ..., "candidate": ..., "prob": ...} rows (one per candidate) -- used by
replace_hard to pick the non-gold option the teacher ranks highest. Without it, replace_hard
swaps a random non-gold option and records that in meta.

uv run scripts/mmlu_counterfactual.py [--data data_v4] [--n 0] [--seed 0] [--teacher_probs FILE]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcdm"))
from data import write_jsonl  # same state-length filter / writer convention as pcdm/data.py/mmlu_pro_eval.py

TASK = "mmlu_cf"


def near_dup_text(s, rng):
    """Trivial surface change: trailing period, a leading "The ", or a case flip -- whichever
    of the three the rng picks actually changes the string (falls through if not)."""
    s = s.rstrip()
    for choice in rng.sample(["period", "article", "case"], 3):
        if choice == "period":
            cand = s[:-1] if s.endswith(".") else s + "."
        elif choice == "article":
            cand = s if s[:4].lower() == "the " else "The " + s[:1].lower() + s[1:]
        else:
            cand = s.swapcase()
        if cand != s:
            return cand
    return s + " "  # ponytail: only reachable for a string all three transforms are no-ops on


def build_variants(row, pool, teacher_by_qid, rng):
    qid = row["meta"]["question_id"]
    state, query = row["state"], row["query"]
    cands = list(row["candidates"])
    K = len(cands)
    gold_i = row["label"]
    gold = cands[gold_i]
    fam, cat = row["meta"].get("family", "qa"), row["meta"].get("category")

    def mk(variant, cs, gold_idx, target=None, **extra_meta):
        if target is None:
            target = [1.0 if i == gold_idx else 0.0 for i in range(len(cs))]
        return {"state": state, "query": query, "candidates": cs, "target": target,
                "p_null": 0.0, "task": TASK, "label": gold_idx,
                "meta": {"family": fam, "category": cat, "qid": qid, "variant": variant, **extra_meta}}

    rows = [mk("orig", cands, gold_i)]

    # remove3: drop up to 3 random non-gold options, keeping >=1 distractor when possible.
    non_gold = [i for i in range(K) if i != gold_i]
    rng.shuffle(non_gold)
    drop = set(non_gold[:min(3, max(len(non_gold) - 1, 0))])
    r3_cands = [c for i, c in enumerate(cands) if i not in drop]
    rows.append(mk("remove3", r3_cands, r3_cands.index(gold), n_removed=len(drop)))

    # add3_unrel: 3 options drawn from other questions' option pools (never this question's own).
    others = [c for c in pool if c not in cands]
    rng.shuffle(others)
    added = others[:3]
    rows.append(mk("add3_unrel", cands + added, gold_i, added=added))

    # replace_hard: swap the non-gold option the teacher ranks highest (else a random one)
    # for an unrelated option from another question; record which one was replaced + how it
    # was picked so metrics/analysis can condition on it.
    teacher = teacher_by_qid.get(qid)
    if teacher:
        rep_i, rep_text = max(((i, cands[i]) for i in non_gold), key=lambda ic: teacher.get(ic[1], -1.0))
        picked_by = "teacher"
    else:
        rep_i = rng.choice(non_gold) if non_gold else gold_i
        rep_text, picked_by = cands[rep_i], "random"
    unrelated = next((c for c in others if c not in added), added[0] if added else "an unrelated option")
    rh_cands = list(cands)
    rh_cands[rep_i] = unrelated
    rows.append(mk("replace_hard", rh_cands, gold_i, replaced_idx=rep_i, replaced_text=rep_text, picked_by=picked_by))

    # add_neardup: append a trivial surface variant of the gold; label stays the original
    # gold index, target mass splits 0.5/0.5 across gold and the near-dup.
    nd_cands = cands + [near_dup_text(gold, rng)]
    nd_target = [1.0 if i == gold_i else 0.0 for i in range(len(nd_cands))]
    nd_target[gold_i], nd_target[-1] = 0.5, 0.5
    rows.append(mk("add_neardup", nd_cands, gold_i, target=nd_target))

    # reorder: random permutation, label remapped.
    order = list(range(K))
    rng.shuffle(order)
    rows.append(mk("reorder", [cands[i] for i in order], order.index(gold_i)))

    return rows


def load_teacher_probs(path):
    """-> {qid: {candidate_str: prob}} from either row shape described in the module docstring."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "probs" in d:
                out[d["qid"]] = d["probs"]
            else:
                out.setdefault(d["qid"], {})[d["candidate"]] = d["prob"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v4", help="reads <data>/eval/mmlu_pro.jsonl, writes <data>/eval/mmlu_cf.jsonl")
    ap.add_argument("--n", type=int, default=0, help="cap the number of source questions (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher_probs", default=None, help="jsonl teacher distribution per question, see module docstring")
    args = ap.parse_args()

    src = Path(args.data) / "eval" / "mmlu_pro.jsonl"
    with open(src) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.n:
        rows = rows[:args.n]

    rng = random.Random(args.seed)
    pool = sorted({c for r in rows for c in r["candidates"]})
    teacher_by_qid = load_teacher_probs(args.teacher_probs) if args.teacher_probs else {}

    out_rows = []
    for row in rows:
        out_rows += build_variants(row, pool, teacher_by_qid, rng)

    out_path = Path(args.data) / "eval" / "mmlu_cf.jsonl"
    write_jsonl(out_path, out_rows)
    print(f"[mmlu_counterfactual] {len(rows)} questions x 6 variants -> {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
