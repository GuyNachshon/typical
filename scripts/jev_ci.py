"""Bootstrap CIs for JevBench public-subset accuracy (paper review §24).

The standard tier is 72 items but only 36 independent states: each state appears as two
paraphrases sharing a `group`. Resampling items would treat those as independent and
report an interval that is too narrow, so standard is CLUSTER-bootstrapped over `group`.
Hard has one item per group, so it reduces to an ordinary item bootstrap.

uv run scripts/jev_ci.py [run ...]
"""
import json, random, sys, collections
from huggingface_hub import hf_hub_download

def ci(runs, split, B=20000, seed=0):
    p = hf_hub_download("guychuk/pcdm-runs", f"jev_native_{runs}/{split}/results.jsonl")
    rows = [json.loads(l) for l in open(p)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("group") or id(r)].append(bool(r.get("correct")))
    cl = list(by.values())
    n_items = sum(len(c) for c in cl)
    point = sum(sum(c) for c in cl) / n_items
    rng = random.Random(seed); k = len(cl); out = []
    for _ in range(B):
        s = [cl[rng.randrange(k)] for _ in range(k)]
        tot = sum(len(c) for c in s)
        out.append(sum(sum(c) for c in s) / tot)
    out.sort()
    return point, out[int(.025*B)], out[int(.975*B)], n_items, k

if __name__ == "__main__":
    names = sys.argv[1:] or ["ladder_14b","tl1b","tl1b_nokd","trunc_first","trunc_last"]
    print(f"{'run':22s} {'split':9s} {'acc':>6s}  95% CI (cluster bootstrap)   items/clusters")
    for n in names:
        for sp in ("original","hard"):
            try:
                pt, lo, hi, ni, nc = ci(n, sp)
                print(f"{n:22s} {sp:9s} {pt:6.3f}  [{lo:.3f}, {hi:.3f}]  +/-{(hi-lo)/2*100:4.1f}pt   {ni}/{nc}")
            except Exception as e:
                print(f"{n:22s} {sp:9s} unavailable ({type(e).__name__})")
