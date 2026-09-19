"""Near-duplicate leakage audit: train vs each eval set (and val), via MinHash LSH
over char 5-gram shingles. data.py's dedupe() removes EXACT (premise, hyp)
matches; this catches near-dups it can't (paraphrased premises/hyps, near-
identical passages). NLI sets also get hyp-alone and (premise, hyp)-pair audits,
since SNLI/MNLI reuse premises across splits by design.
Run: uv run scripts/leak_audit.py
"""
import json
import sys
import time
from pathlib import Path

from datasketch import MinHash, MinHashLSH

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import norm_text  # reuse the repo's normalizer (DRY)

DATA = Path("data")
EVAL_DIR = DATA / "eval"
RUNS = Path("runs")
NUM_PERM = 64
K = 5
THRESH_LO, THRESH_HI = 0.8, 0.9
NLI_SETS = {"snli_test", "mnli_val", "anli_test", "chaos_mnli", "snli_test_soft",
            "snli_null", "snli_test_paraphrase", "snli_test_qpara", "unli_test"}
# Which field defines "near-dup row" for the accuracy-impact subset below.
# Reusing a premise alone doesn't leak a label; reusing a near-identical
# (premise, hyp) pair does, so pair-level near-dup is the risk that matters.
NEARDUP_SETS = {"snli_test", "mnli_val"}

def shingles(s, k=K):
    if len(s) < k:
        return [s.encode()]
    return [s[i:i + k].encode() for i in range(len(s) - k + 1)]

def minhash(s):
    m = MinHash(num_perm=NUM_PERM)
    m.update_batch(shingles(s))
    return m

def load_rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def hyp_of(r):
    h = r.get("meta", {}).get("hyp")
    if h:
        return h
    q = r["query"]
    return q.split(": ", 1)[-1] if ": " in q else q

def build_index(norm_texts):
    """norm_texts: iterable of unique normalized strings. -> (lsh, {text: minhash})."""
    lsh = MinHashLSH(threshold=THRESH_LO, num_perm=NUM_PERM)
    mh = {}
    for t in norm_texts:
        m = minhash(t)
        mh[t] = m
        lsh.insert(t, m)
    return lsh, mh

def best_match(lsh, mh, norm_q):
    qm = minhash(norm_q)
    best_j, best_t = 0.0, None
    for cand in lsh.query(qm):
        j = qm.jaccard(mh[cand])
        if j > best_j:
            best_j, best_t = j, cand
    return best_j, best_t

def field_audit(lsh, mh, exact_set, eval_texts):
    """Returns stats dict + list of (jaccard, eval_text, train_text, row_idx) hits >= THRESH_LO."""
    n = len(eval_texts)
    exact = near_lo = near_hi = 0
    hits = []
    for i, raw in enumerate(eval_texts):
        norm = norm_text(raw)
        if not norm:
            continue
        if norm in exact_set:
            exact += 1
        j, t = best_match(lsh, mh, norm)
        if j >= THRESH_LO:
            near_lo += 1
            hits.append((j, raw, t, i))
        if j >= THRESH_HI:
            near_hi += 1
    hits.sort(key=lambda x: -x[0])
    pct = lambda c: round(100 * c / n, 2) if n else 0.0
    return dict(n=n, exact=exact, near_lo=near_lo, near_hi=near_hi,
                near_lo_pct=pct(near_lo), near_hi_pct=pct(near_hi)), hits

def top5(hits):
    return [{"jaccard": round(j, 3), "eval": e[:200], "train": t[:200]} for j, e, t, _ in hits[:5]]

def main():
    t0 = time.time()
    train_rows = load_rows(DATA / "train.jsonl")
    val_rows = load_rows(DATA / "val.jsonl")

    train_states = {norm_text(r["state"]) for r in train_rows}
    train_hyps = {norm_text(hyp_of(r)) for r in train_rows if r.get("meta", {}).get("hyp")}
    train_pairs = {norm_text(r["state"]) + " ||| " + norm_text(hyp_of(r))
                   for r in train_rows if r.get("meta", {}).get("hyp")}
    val_states = {norm_text(r["state"]) for r in val_rows}

    print(f"[{time.time()-t0:.0f}s] building train indices: "
          f"{len(train_states)} states, {len(train_hyps)} hyps, {len(train_pairs)} pairs")
    state_lsh, state_mh = build_index(train_states)
    hyp_lsh, hyp_mh = build_index(train_hyps)
    pair_lsh, pair_mh = build_index(train_pairs)
    val_lsh, val_mh = build_index(val_states)
    print(f"[{time.time()-t0:.0f}s] train indices built")

    report = {}
    header = f"{'set':<24}{'n':>7}{'exact':>7}{'nd@.8':>8}{'nd@.9':>8}{'val@.8':>8}"
    print(header)
    for path in sorted(EVAL_DIR.glob("*.jsonl")):
        name = path.stem
        rows = load_rows(path)
        state_stats, state_hits = field_audit(state_lsh, state_mh, train_states, [r["state"] for r in rows])
        val_stats, _ = field_audit(val_lsh, val_mh, val_states, [r["state"] for r in rows])
        entry = {"n": state_stats["n"], "state": state_stats, "state_top5": top5(state_hits),
                 "val_state_near_lo_pct": val_stats["near_lo_pct"]}
        row_fmt = f"{name:<24}{state_stats['n']:>7}{state_stats['exact']:>7}" \
                  f"{state_stats['near_lo_pct']:>7}%{state_stats['near_hi_pct']:>7}%{val_stats['near_lo_pct']:>7}%"
        if name in NLI_SETS:
            hyps = [hyp_of(r) for r in rows]
            pairs = [norm_text(r["state"]) + " ||| " + norm_text(h) for r, h in zip(rows, hyps)]
            hyp_stats, _ = field_audit(hyp_lsh, hyp_mh, train_hyps, hyps)
            pair_stats, pair_hits = field_audit(pair_lsh, pair_mh, train_pairs, pairs)
            entry["hyp"] = hyp_stats
            entry["pair"] = pair_stats
            entry["pair_top5"] = top5(pair_hits)
            row_fmt += f"  hyp@.8={hyp_stats['near_lo_pct']}% pair@.8={pair_stats['near_lo_pct']}%"
            if name in NEARDUP_SETS:
                idx = sorted(i for _, _, _, i in pair_hits)
                out = RUNS / f"leak_audit_nearDup_{name}.json"
                out.write_text(json.dumps({"set": name, "field": "pair@0.8", "row_indices": idx}))
                entry["near_dup_indices_file"] = str(out)
        report[name] = entry
        print(row_fmt)

    RUNS.mkdir(exist_ok=True)
    (RUNS / "leak_audit.json").write_text(json.dumps(report, indent=2))
    print(f"[{time.time()-t0:.0f}s] done, wrote runs/leak_audit.json")

if __name__ == "__main__":
    main()
