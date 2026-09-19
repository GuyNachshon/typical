"""Choices-only / shuffled-question probes (PI memo 2026-09-19; Balepur et al. ACL 2024): same options as
mmlu_pro.jsonl but the question blanked ("."), or replaced by another item's question. If a model keeps its
MMLU-Pro score without the question, it is exploiting candidate-set priors, not knowledge in Z(x, q).
uv run scripts/mmlu_probes.py --data data_v4  -> eval/mmlu_choicesonly.jsonl, eval/mmlu_shuffledq.jsonl"""
import argparse, json, random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import write_jsonl

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data_v4"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rows = [json.loads(l) for l in open(Path(a.data) / "eval" / "mmlu_pro.jsonl")]
rng = random.Random(a.seed)
states = [r["state"] for r in rows]
perm = list(range(len(rows))); rng.shuffle(perm)
perm = [(p + 1) % len(rows) if p == i else p for i, p in enumerate(perm)]  # never map an item to itself
blank = [dict(r, state=".", task="mmlu_choicesonly", meta=dict(r["meta"], probe="choices_only")) for r in rows]
shuf = [dict(r, state=states[perm[i]], task="mmlu_shuffledq", meta=dict(r["meta"], probe="shuffled_q")) for i, r in enumerate(rows)]
write_jsonl(Path(a.data) / "eval" / "mmlu_choicesonly.jsonl", blank)
write_jsonl(Path(a.data) / "eval" / "mmlu_shuffledq.jsonl", shuf)
print(len(blank), len(shuf))
