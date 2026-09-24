"""Score a model on the long-composition eval and run the paired test the set was built for.

    uv run --no-sync python scripts/eval_longcomp.py evals/longcomp_v1.jsonl
    uv run --no-sync python scripts/eval_longcomp.py --demo

The set renders every item at several lengths with identical content, so the interesting number is
not the accuracy at each length -- it is the paired comparison of one item against itself. McNemar
over those pairs is the test JevBench's hard tier cannot support: there, long and hard are the same
19 items, so a length split compares different questions.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import urllib.request
from pathlib import Path

SERVER = "http://localhost:8787/api/decide"


def decide(state: str, question: str, labels: list[str], timeout: int = 180) -> dict:
    body = json.dumps({"state": state, "queries": [{"type": "choice", "question": question, "labels": labels}]}).encode()
    req = urllib.request.Request(SERVER, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["results"][0]


def mcnemar(pairs) -> tuple[int, int, float]:
    """pairs: (a_correct, b_correct). Exact two-sided binomial over the discordant ones."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)
    return b, c, p


def summarise(rows):
    out = []
    by = collections.defaultdict(list)
    for r in rows:
        by[("length", r["target_tokens"])].append(r["correct"])
        by[("position", r["evidence_position"])].append(r["correct"])
        by[("kind", r["kind"])].append(r["correct"])
    for (dim, k), v in sorted(by.items()):
        out.append((dim, k, len(v), sum(v) / len(v)))
    return out


def paired(rows, dim: str):
    """Group renderings by item, then compare one value of `dim` against every other."""
    idx = collections.defaultdict(dict)
    for r in rows:
        idx[r["item_id"]][(r[dim], r["evidence_position"] if dim == "target_tokens" else r["target_tokens"])] = r["correct"]
    keys = sorted({k[0] for v in idx.values() for k in v})
    res = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pairs = [(v[k1], v[k2]) for v in idx.values()
                     for k1 in v for k2 in v if k1[0] == a and k2[0] == b and k1[1] == k2[1]]
            if pairs:
                res.append((a, b, len(pairs), *mcnemar(pairs)))
    return res


def demo() -> None:
    b, c, p = mcnemar([(True, False)] * 10 + [(False, True)] * 0)
    assert b == 10 and c == 0 and p < 0.01, "ten wins and no losses is significant"
    b, c, p = mcnemar([(True, False)] * 5 + [(False, True)] * 5)
    assert p == 1.0, "an even split cannot separate"
    assert mcnemar([(True, True)] * 20)[2] == 1.0, "no discordant pairs, nothing to test"
    rows = [{"item_id": f"i{i}", "target_tokens": L, "evidence_position": "first", "kind": "k",
             "correct": (L == 700)} for i in range(8) for L in (700, 3300)]
    r = paired(rows, "target_tokens")
    assert r and r[0][0] == 700 and r[0][1] == 3300, "pairs the two lengths"
    assert r[0][3] == 8 and r[0][4] == 0, "short wins every pair here"
    print("eval_longcomp.py self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_file", nargs="?", type=Path)
    ap.add_argument("--out", type=Path, default=Path("evals/longcomp_scored.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo or not a.eval_file:
        demo()
        return

    rows = [json.loads(l) for l in a.eval_file.open()]
    if a.limit:
        rows = rows[: a.limit]
    scored = []
    for i, r in enumerate(rows):
        try:
            res = decide(r["state"], r["question"], r["labels"])
        except Exception as e:  # a failed call is not a wrong answer and must not be scored as one
            print(f"  [{i}] request failed: {type(e).__name__}", file=sys.stderr)
            continue
        pick = res["argmax"]
        scored.append({**{k: v for k, v in r.items() if k != "state"},
                       "predicted": pick, "correct": pick == r["gold"],
                       "p_gold": res["probs"].get(r["gold"]), "p_null": res.get("p_null")})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}", file=sys.stderr)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for s in scored:
            f.write(json.dumps(s) + "\n")

    acc = sum(s["correct"] for s in scored) / len(scored)
    print(f"\nscored {len(scored)} renderings of {len({s['item_id'] for s in scored})} items")
    print(f"overall {acc:.4f}   (chance {scored[0]['chance']:.2f})\n")
    for dim, k, n, v in summarise(scored):
        print(f"  {dim:9} {str(k):8} n={n:4}  acc {v:.4f}")
    print("\npaired over the same item, one length against another:")
    for a_, b_, n, w, l, p in paired(scored, "target_tokens"):
        print(f"  {a_:5} vs {b_:5}: {n:4} pairs, {a_} wins {w}, {b_} wins {l}, exact p = {p:.4f}")
    print("\npaired over the same item, evidence first against last:")
    for a_, b_, n, w, l, p in paired(scored, "evidence_position"):
        print(f"  {a_:5} vs {b_:5}: {n:4} pairs, {a_} wins {w}, {b_} wins {l}, exact p = {p:.4f}")


if __name__ == "__main__":
    main()
