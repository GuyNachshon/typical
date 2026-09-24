"""PLAN3 E3 leak audit ($0): before reading any MMLU-Pro gain as generalization, check
data_kb/{train,val}.jsonl against the frozen data_v4/eval/mmlu_pro.jsonl (1,200 items)
for (1) exact stem match, (2) normalized stem match, (3) near-dup stem via the same
MinHash char-5-gram @0.8 leak_audit.py already uses for data/, (4) normalized-stem
match with >=50% option-text overlap (a stricter sub-check: two different questions can
coincidentally share a normalized stem, an option-overlap floor confirms it's the same
question). Also dedupes data_kb/train.jsonl against itself across `task` sources
(MMLU-aux bundles ARC/OBQA/etc, so the same question can enter train twice under
different task names) and within a single source.

Writes runs/kb_audit.json (per-check counts + offending MMLU-Pro qids + duplicate-group
sizes + the source-pair overlap matrix). If any MMLU-Pro item leaks by (1)-(4), also
writes data_kb/leaked_qids.json (question_ids to exclude when reporting eval numbers --
data_kb itself is never modified).

uv run scripts/kb_audit.py
"""
import json
import re
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcdm"))
from scripts.leak_audit import build_index, best_match  # reuse the MinHash machinery (DRY)

EVAL_PATH = Path("data_v4/eval/mmlu_pro.jsonl")
KB_DIR = Path("data_kb")
RUNS = Path("runs")
THRESH = 0.8
PUNCT = re.compile(r"[^\w\s]")
DIGITS = re.compile(r"\d+")


def strict_norm(s, collapse_numbers=False):
    """lowercase, strip punctuation, collapse whitespace; optionally collapse digit runs
    (catches stems/options that are identical except for reworded numbers)."""
    s = PUNCT.sub(" ", s.lower())
    if collapse_numbers:
        s = DIGITS.sub("#", s)
    return " ".join(s.split())


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def option_overlap(a, b):
    """Fraction of the smaller normalized option set also present in the other."""
    A, B = {strict_norm(x) for x in a}, {strict_norm(x) for x in b}
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


def main():
    t0 = time.time()
    mmlu = load_jsonl(EVAL_PATH)
    kb_train = load_jsonl(KB_DIR / "train.jsonl")
    kb_val = load_jsonl(KB_DIR / "val.jsonl")
    kb_rows = [r for r in kb_train] + [r for r in kb_val]
    print(f"[{time.time()-t0:.0f}s] mmlu_pro={len(mmlu)} kb_train={len(kb_train)} kb_val={len(kb_val)}")

    # ---- indices over data_kb (train+val) ----
    exact_idx, norm_idx, numcol_idx = {}, {}, {}
    for r in kb_rows:
        exact_idx.setdefault(r["state"], []).append(r)
        norm_idx.setdefault(strict_norm(r["state"]), []).append(r)
        numcol_idx.setdefault(strict_norm(r["state"], True), []).append(r)

    uniq_norm_states = list(norm_idx)
    lsh, mh = build_index(uniq_norm_states)
    print(f"[{time.time()-t0:.0f}s] indices built ({len(uniq_norm_states)} unique normalized kb stems)")

    leaked_qids = set()
    checks = {"exact": [], "normalized": [], "normalized_numcollapse": [], "near_dup": [], "stem_and_options": []}
    for item in mmlu:
        qid = item.get("meta", {}).get("question_id", item["state"][:40])
        stem, cands = item["state"], item["candidates"]

        if stem in exact_idx:
            checks["exact"].append(qid); leaked_qids.add(qid)

        ns = strict_norm(stem)
        norm_hits = norm_idx.get(ns, [])
        if norm_hits:
            checks["normalized"].append(qid); leaked_qids.add(qid)
            if any(option_overlap(cands, r["candidates"]) >= 0.5 for r in norm_hits):
                checks["stem_and_options"].append(qid); leaked_qids.add(qid)

        if strict_norm(stem, True) in numcol_idx:
            checks["normalized_numcollapse"].append(qid); leaked_qids.add(qid)

        j, _ = best_match(lsh, mh, ns)
        if j >= THRESH:
            checks["near_dup"].append(qid); leaked_qids.add(qid)

    print(f"[{time.time()-t0:.0f}s] mmlu_pro audit done ({len(leaked_qids)}/{len(mmlu)} leaked by any check)")

    # ---- cross-source dedup inside data_kb/train.jsonl ----
    groups = {}  # strict_norm(state) -> [(task, idx)]
    for i, r in enumerate(kb_train):
        groups.setdefault(strict_norm(r["state"]), []).append((r["task"], i))
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    tasks = sorted({r["task"] for r in kb_train})
    matrix = {t: {u: 0 for u in tasks} for t in tasks}  # matrix[a][b] = # dup groups touching both a,b
    for members in dup_groups.values():
        present = sorted({t for t, _ in members})
        if len(present) == 1:
            matrix[present[0]][present[0]] += 1  # within-source dup group
        else:
            for a, b in combinations(present, 2):
                matrix[a][b] += 1; matrix[b][a] += 1

    unique_train = len(groups)
    dup_sizes = sorted((len(v) for v in dup_groups.values()), reverse=True)

    report = {
        "n_mmlu_pro": len(mmlu),
        "n_kb_train": len(kb_train), "n_kb_val": len(kb_val),
        "checks": {k: {"n": len(v), "pct": round(100 * len(v) / len(mmlu), 2), "qids": v} for k, v in checks.items()},
        "leaked_qids_total": sorted(leaked_qids, key=str),
        "cross_source_dedup": {
            "n_train_rows": len(kb_train),
            "n_unique_normalized_stems": unique_train,
            "n_duplicate_groups": len(dup_groups),
            "duplicate_group_sizes_top20": dup_sizes[:20],
            "source_pair_matrix": matrix,
        },
    }
    RUNS.mkdir(exist_ok=True)
    (RUNS / "kb_audit.json").write_text(json.dumps(report, indent=2))
    if leaked_qids:
        (KB_DIR / "leaked_qids.json").write_text(json.dumps(sorted(leaked_qids, key=str), indent=2))

    print(f"\n{'check':<26}{'n':>6}{'pct':>8}")
    for k, v in checks.items():
        print(f"{k:<26}{len(v):>6}{100*len(v)/len(mmlu):>7.2f}%")
    print(f"\nleaked (any check): {len(leaked_qids)}/{len(mmlu)}")
    print(f"data_kb/train.jsonl: {len(kb_train)} rows -> {unique_train} unique normalized stems "
          f"({len(dup_groups)} duplicate groups, largest={dup_sizes[0] if dup_sizes else 0})")
    print(f"[{time.time()-t0:.0f}s] wrote runs/kb_audit.json"
          + (" and data_kb/leaked_qids.json" if leaked_qids else ""))


if __name__ == "__main__":
    main()
