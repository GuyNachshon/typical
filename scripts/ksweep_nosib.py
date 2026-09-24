"""Derive ksweep_clinc_nosib.jsonl from an existing ksweep_clinc.jsonl: same 400 utterances and
label space, but the gold's sibling intents are excluded from the distractors (REPORT §3h).
uv run scripts/ksweep_nosib.py data_v4/eval/ksweep_clinc.jsonl [--out data_v4/eval/ksweep_clinc_nosib.jsonl]"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcdm"))
from data import build_ksweep, build_sibling_map, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out", default=None)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.src)]
present = [r for r in rows if r["meta"]["present"]]
names = max((r["candidates"] for r in present), key=len)  # the K=N row lists the whole label space
id_of = {n: i for i, n in enumerate(names)}
utter = {}
for r in present:
    utter.setdefault(r["meta"]["u"], (r["state"], r["query"], id_of[r["candidates"][r["label"]]]))
utter = [utter[u] for u in sorted(utter)]
out = build_ksweep("ksweep_clinc_nosib", utter, names, list(range(len(names))), random.Random(0),
                   sibling_map=build_sibling_map(names, "ksweep_clinc_nosib"))
dst = a.out or str(Path(a.src).with_name("ksweep_clinc_nosib.jsonl"))
write_jsonl(Path(dst), out)
print(f"{len(utter)} utterances, {len(names)} labels -> {len(out)} rows -> {dst}")
