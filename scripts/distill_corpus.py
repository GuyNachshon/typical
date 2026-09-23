"""Closed-book MCQ distillation corpus for PLAN3 E3-0.

Builds data_kb/train.jsonl + data_kb/val.jsonl (eval-row schema: state/query/candidates/
target/p_null/task/label/meta, family "qa" -- same shape as scripts/mmlu_pro_eval.py's
rows, see pcdm/data.py) from ~10 knowledge-MCQ HF sources. Never touches MMLU test/validation
or MMLU-Pro. truthfulqa_mc1 and gpqa are held out entirely to data_kb/eval/*.jsonl.

uv run scripts/distill_corpus.py [--out data_kb] [--limit N] [--seed 0]
  --limit N: cap every source's raw rows at N (fast pipeline check, e.g. --limit 50)
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcdm"))
from data import trunc, norm_text, group_split, write_jsonl  # reuse pcdm/data.py's conventions

QUERY = "Which option is correct?"  # same fixed query mmlu_pro_eval.py uses for this family
MAX_CAND = 300
CAP_DEFAULT = 40000
CAP_MEDMCQA = 20000
NULL_FRAC = 0.05  # fraction of rows per source that get the gold option removed (p_null=1)
VAL_TOTAL = 3000

# "A." / "(a)" / "A)" -> stripped; a bare word starting with a letter+space is untouched
# since the required '.'/')' must immediately follow the single letter.
LETTER_PREFIX = re.compile(r"^\s*\(?[A-Za-z][.)]\s*")


def strip_prefix(text):
    return LETTER_PREFIX.sub("", text.strip()).strip()


def _rows(ds):
    return list(ds)


# ---- per-source loaders: -> list of (state, query_or_None, candidates, gold_idx) ----
# query_or_None: None means "use the fixed QUERY constant"; sources with a genuine
# separate question (LogiQA) fold {context}\n{question} into state and still use QUERY,
# so every row in the corpus shares one query string (matches mmlu_pro_eval.py's convention).

def load_mmlu_aux(limit, dedupe_against):
    ds = _rows(load_dataset("cais/mmlu", "auxiliary_train", split="train"))
    out = []
    for r in ds:
        r = r["train"]  # ponytail: HF wraps auxiliary_train rows under a "train" key
        if norm_text(r["question"]) in dedupe_against:
            continue
        out.append((r["question"], None, list(r["choices"]), int(r["answer"])))
    random.Random(0).shuffle(out)
    return out[:limit] if limit else out


def load_arc(config, limit):
    ds = _rows(load_dataset("allenai/ai2_arc", config, split="train"))
    out = []
    for r in ds:
        labels = r["choices"]["label"]
        if r["answerKey"] not in labels:
            continue
        out.append((r["question"], None, list(r["choices"]["text"]), labels.index(r["answerKey"])))
    return out[:limit] if limit else out


def load_obqa(limit):
    ds = _rows(load_dataset("allenai/openbookqa", "main", split="train"))
    out = []
    for r in ds:
        labels = r["choices"]["label"]
        if r["answerKey"] not in labels:
            continue
        out.append((r["question_stem"], None, list(r["choices"]["text"]), labels.index(r["answerKey"])))
    return out[:limit] if limit else out


def load_csqa(limit):
    ds = _rows(load_dataset("tau/commonsense_qa", split="train"))
    out = []
    for r in ds:
        labels = r["choices"]["label"]
        if r["answerKey"] not in labels:
            continue
        out.append((r["question"], None, list(r["choices"]["text"]), labels.index(r["answerKey"])))
    return out[:limit] if limit else out


def load_sciq(limit):
    ds = _rows(load_dataset("allenai/sciq", split="train"))
    out = [(r["question"], None,
            [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]], 3) for r in ds]
    return out[:limit] if limit else out


def load_qasc(limit):
    ds = _rows(load_dataset("allenai/qasc", split="train"))
    out = []
    for r in ds:
        labels = r["choices"]["label"]
        if r["answerKey"] not in labels:
            continue
        out.append((r["question"], None, list(r["choices"]["text"]), labels.index(r["answerKey"])))
    return out[:limit] if limit else out


def load_logiqa(limit):
    # jeggers/logiqa2_formatted: lucasmccabe-lmi/logiqa and jeggers/logiqa (PLAN3's named
    # ids) no longer resolve on the Hub (script-based / gone); this is the closest live
    # equivalent -- same LogiQA-2.0 content, "text" = passage, "question" separate,
    # "answer" = 0-indexed gold, "options" = 4-way MCQ.
    ds = _rows(load_dataset("jeggers/logiqa2_formatted", split="train"))
    out = [(f'{r["text"]}\n{r["question"]}', None, list(r["options"]), int(r["answer"])) for r in ds]
    return out[:limit] if limit else out


def load_aqua(limit):
    # math_qa (PLAN3's first choice) is a script-based dataset the `datasets` lib can no
    # longer load; falls back to aqua_rat (raw) as PLAN3 allows ("or deepmind/aqua_rat").
    ds = _rows(load_dataset("deepmind/aqua_rat", "raw", split="train"))
    out = []
    for r in ds:
        opts = [strip_prefix(o) for o in r["options"]]
        letter = r["correct"].strip()
        gold = ord(letter) - ord("A") if letter.isalpha() else None
        if gold is None or not (0 <= gold < len(opts)):
            continue
        out.append((r["question"], None, opts, gold))
    return out[:limit] if limit else out


def load_medmcqa(limit):
    ds = _rows(load_dataset("openlifescienceai/medmcqa", split="train"))
    out = []
    for r in ds:
        cop = r["cop"]
        if not (0 <= cop <= 3):
            continue
        out.append((r["question"], None, [r["opa"], r["opb"], r["opc"], r["opd"]], cop))
    return out[:limit] if limit else out


def load_truthfulqa_mc1(limit):
    ds = _rows(load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation"))
    out = []
    for r in ds:
        labels = r["mc1_targets"]["labels"]
        if 1 not in labels:
            continue
        out.append((r["question"], None, list(r["mc1_targets"]["choices"]), labels.index(1)))
    return out[:limit] if limit else out


def load_gpqa(limit):
    ds = _rows(load_dataset("Idavidrein/gpqa", "gpqa_main", split="train"))
    out = []
    for r in ds:
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        out.append((r["Question"], None, opts, 0))
    return out[:limit] if limit else out


TRAIN_SOURCES = [
    ("mmlu_aux", load_mmlu_aux, CAP_DEFAULT),  # special-cased below (needs the dedupe set)
    ("arc_easy", lambda limit: load_arc("ARC-Easy", limit), CAP_DEFAULT),
    ("arc_challenge", lambda limit: load_arc("ARC-Challenge", limit), CAP_DEFAULT),
    ("openbookqa", load_obqa, CAP_DEFAULT),
    ("commonsense_qa", load_csqa, CAP_DEFAULT),
    ("sciq", load_sciq, CAP_DEFAULT),
    ("qasc", load_qasc, CAP_DEFAULT),
    ("logiqa", load_logiqa, CAP_DEFAULT),
    ("aqua_rat", load_aqua, CAP_DEFAULT),
    ("medmcqa", load_medmcqa, CAP_MEDMCQA),
]
HELD_OUT_SOURCES = [
    ("truthfulqa_mc1", load_truthfulqa_mc1),
    ("gpqa", load_gpqa),
]


def make_rows(task, raw_items, rng, null_frac=NULL_FRAC):
    """raw_items: (state, query_or_None, candidates, gold_idx). Cleans + truncates, drops
    <2-option/bad-gold rows, and turns null_frac of the remainder into gold-absent rows
    (gold option removed, p_null=1, uniform target) so the null keeps a training signal."""
    rows = []
    for state, query, cands, gold in raw_items:
        cands = [trunc(strip_prefix(c), MAX_CAND) for c in cands]
        state = trunc(state)
        if len(cands) < 2 or gold is None or not (0 <= gold < len(cands)):
            continue
        if rng.random() < null_frac:
            kept = cands[:gold] + cands[gold + 1:]
            target = [1.0 / len(kept)] * len(kept)
            rows.append({"state": state, "query": query or QUERY, "candidates": kept,
                         "target": target, "p_null": 1.0, "task": task, "label": -1,
                         "meta": {"family": "qa"}})
        else:
            target = [1.0 if i == gold else 0.0 for i in range(len(cands))]
            rows.append({"state": state, "query": query or QUERY, "candidates": cands,
                         "target": target, "p_null": 0.0, "task": task, "label": gold,
                         "meta": {"family": "qa"}})
    return rows


def apportion(sizes, total):
    """Largest-remainder apportionment of `total` seats across `sizes` (same recipe as
    scripts/mmlu_pro_eval.py's stratified_sample), so val quotas sum to exactly `total`."""
    grand = sum(sizes.values()) or 1
    raw = {k: total * v / grand for k, v in sizes.items()}
    quota = {k: int(raw[k]) for k in sizes}
    remainder = total - sum(quota.values())
    order = sorted(sizes, key=lambda k: raw[k] - quota[k], reverse=True)
    for k in order[:remainder]:
        quota[k] += 1
    return quota


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_kb")
    ap.add_argument("--limit", type=int, default=0, help="cap every source's raw rows (fast check)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print("=== Held out (never in train/val) ===")
    manifest = {"sources": {}, "held_out": {}}
    for task, loader in HELD_OUT_SOURCES:
        try:
            raw = loader(args.limit)
            rows = make_rows(task, raw, rng, null_frac=0.0)  # eval sets: gold always present
            write_jsonl(out / "eval" / f"{task}.jsonl", rows)
            manifest["held_out"][task] = len(rows)
            print(f"  {task}: {len(rows)} rows -> {out}/eval/{task}.jsonl")
        except Exception as e:
            print(f"WARNING: skipping held-out {task}: {e}")
            manifest["held_out"][task] = f"skipped ({e})"

    print("\n=== Train sources ===")
    per_source_rows = {}
    dedupe_against = set()  # normalised ARC/OBQA question text -- mmlu_aux bundles these
    for task, loader, cap in TRAIN_SOURCES:
        if task == "mmlu_aux":
            continue  # loaded last, once the dedupe set below is populated
        try:
            raw = loader(args.limit or cap)
            rows = make_rows(task, raw, rng)
            per_source_rows[task] = rows
            if task in ("arc_easy", "arc_challenge", "openbookqa"):
                dedupe_against.update(norm_text(state) for state, *_ in raw)
            print(f"  {task}: {len(rows)} rows")
        except Exception as e:
            print(f"WARNING: skipping {task}: {e}")

    try:
        raw = load_mmlu_aux(args.limit or CAP_DEFAULT, dedupe_against)
        rows = make_rows("mmlu_aux", raw, rng)
        per_source_rows["mmlu_aux"] = rows
        print(f"  mmlu_aux: {len(rows)} rows (deduped against ARC/OBQA)")
    except Exception as e:
        print(f"WARNING: skipping mmlu_aux: {e}")

    print("\n=== Train/val split (per source, no shared normalised state) ===")
    val_quota = apportion({k: len(v) for k, v in per_source_rows.items()}, VAL_TOTAL)
    train_rows, val_rows = [], []
    for task, rows in per_source_rows.items():
        # ponytail: cap at len(rows)//2 so a source smaller than its proportional share
        # (only bites in --limit test runs; real sources dwarf their VAL_TOTAL slice)
        # still keeps train rows instead of donating everything to val.
        quota = min(val_quota.get(task, 0), len(rows) // 2)
        t, v = group_split(rows, quota, rng)
        train_rows += t; val_rows += v
        manifest["sources"][task] = {"n": len(rows), "train": len(t), "val": len(v)}
        print(f"  {task:<16}{len(rows):>8} rows  ({len(t)} train / {len(v)} val)")

    rng.shuffle(train_rows); rng.shuffle(val_rows)
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)
    manifest["total_train"] = len(train_rows)
    manifest["total_val"] = len(val_rows)
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nTOTAL: {len(train_rows)} train / {len(val_rows)} val -> {out}/")
    if len(train_rows) < 150000 and not args.limit:
        print(f"WARNING: train rows {len(train_rows)} < 150k target")


if __name__ == "__main__":
    main()
