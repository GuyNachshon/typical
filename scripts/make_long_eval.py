"""Held-out long-state eval set for the render-order ablation (REPORT §3ai).

JevBench's hard tier measures long-document policy questions with n=19 -- two items of
difference swings it 10 points, which cannot settle whether facts-first rendering caused
the long_policy recovery. This builds the same kind of rows at n~800 from the corpus
generator's own long-capable families, with the held-out rubric styles (the styles that
never appear in train/val) and a distinct seed.

Rows render facts-first, i.e. the Case IS inside a 1024-token window. That is the point:
a model trained on facts-last data learned to answer from filler it could still see, so it
should do WORSE here than one trained to read the Case, and the gap is the effect size.

uv run scripts/make_long_eval.py [--n 800] [--seed 9174] [--out data_wf_long/eval/wf_long_policy.jsonl]
"""
import argparse, json, random, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from workflow_corpus import gen_policy_permit, gen_action_select, LONG_EXTRA_SPLIT

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--seed", type=int, default=9174)  # not 0: train used seed 0
ap.add_argument("--out", default="data_wf_long/eval/wf_long_policy.jsonl")
a = ap.parse_args()

rng = random.Random(a.seed)
n_p = round(a.n * LONG_EXTRA_SPLIT["policy_permit"] / sum(LONG_EXTRA_SPLIT.values()))
rows = (gen_policy_permit(n_p, rng, heldout_style=True, force_long=True)
        + gen_action_select(a.n - n_p, rng, heldout_style=True, force_long=True))
rows = [r for r in rows if r["meta"].get("long")]
rng.shuffle(rows)

out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for r in rows:
        r["meta"]["eval_set"] = "wf_long_policy"
        f.write(json.dumps(r) + "\n")

pos = []
for r in rows:
    c = r["state"].find("Case:")
    if c >= 0 and r["state"]:
        pos.append(c / len(r["state"]))
pos.sort()
print(f"wrote {len(rows)} rows -> {out}")
print(f"  all long: {all(r['meta'].get('long') for r in rows)}")
print(f"  case_pos p50={pos[len(pos)//2]:.3f} (facts-first)")
print(f"  state chars p50={sorted(len(r['state']) for r in rows)[len(rows)//2]}")
print(f"  families: {ic(rows)}" if False else f"  families: {dict((f, sum(1 for r in rows if r['meta']['family']==f)) for f in {r['meta']['family'] for r in rows})}")
