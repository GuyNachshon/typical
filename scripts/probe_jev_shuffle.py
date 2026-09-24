"""PLAN7/S3w calib #3b ($0 probe): does the wf checkpoint execute the rubric, or pattern-match
state -> label some other way, at the length it was trained on?

Target set: the 36 JevBench hard-tier public items with state <= 256 tokens (the trained
native max_state -- no truncation confound), identified from the existing
runs/jev_native_v3t20_wf/hard/results.jsonl. For each, replace question["criteria"] (the
rubric text pcdm_jev.decider.query_text renders) with another hard-tier item's criteria of the
*same question type* (choice/noul/score -- criteria's data shape differs by type, so a
same-type donor is also the only structurally valid swap) -- labels, state, gold and
instructions are untouched, only the rubric-to-label mapping becomes nonsense. Donor choice is
seeded (deterministic, excludes self). Re-decides both the original and shuffled versions with
the wf model (mode=native, matching the original hard-tier run's config) so the "own rubric"
number here is a fresh, comparable control rather than a copy of the cached results.jsonl.

If shuffled accuracy stays near the .64 own-rubric accuracy, the model isn't using rubric
content at trained length (it's reading state/labels some other way); if it falls toward
chance, it is.

uv run scripts/probe_jev_shuffle.py [--jevbench_dir /tmp/jevbench] [--model runs/nc_v3_tap20_wf]
                                     [--seed 0] [--out runs/probe_jev_shuffle_wf]
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pcdm"))


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def short_hard_ids(cached_results_path, max_state=256):
    """The 36 ids: hard-tier rows from a prior run whose *actual* state token count (as measured
    by that run, native tokenizer) was <= max_state -- i.e. never truncated at the trained length."""
    rows = load_jsonl(cached_results_path)
    return [r["task_id"] for r in rows if r.get("runtime", {}).get("state_tokens", 10 ** 9) <= max_state]


def build_shuffled_question(task, donor_by_type, seed):
    """Deterministic donor (excluding self) of the same question["type"], criteria swapped in."""
    qtype = task["question"]["type"]
    pool = [t for t in donor_by_type[qtype] if t["id"] != task["id"]]
    donor = random.Random(f"{seed}:{task['id']}").choice(pool)
    q = dict(task["question"])
    q["criteria"] = donor["question"]["criteria"]
    return q, donor["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jevbench_dir", default="/tmp/jevbench")
    ap.add_argument("--model", default="runs/nc_v3_tap20_wf")
    ap.add_argument("--cached_results", default="runs/jev_native_v3t20_wf/hard/results.jsonl",
                     help="prior hard-tier run used only to pick the <=256-state-token id set")
    ap.add_argument("--max_state_tokens", type=int, default=256)
    ap.add_argument("--decider_max_state", type=int, default=4096, help="matches the original hard-tier run's --max_state")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/probe_jev_shuffle_wf")
    args = ap.parse_args()

    from pcdm_jev.decider import PCDMDecider

    tasks = load_jsonl(Path(args.jevbench_dir, "datasets/public/hard.jsonl"))
    by_id = {t["id"]: t for t in tasks}
    target_ids = short_hard_ids(args.cached_results, args.max_state_tokens)
    assert len(target_ids) == 36, f"expected 36 short hard-tier items, found {len(target_ids)}"
    targets = [by_id[i] for i in target_ids]

    donor_by_type = defaultdict(list)
    for t in tasks:
        donor_by_type[t["question"]["type"]].append(t)

    dec = PCDMDecider(str(ROOT / args.model), mode="native", max_state=args.decider_max_state)
    print(f"[probe] loaded wf native decider from {args.model}", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for task in targets:
        shuf_q, donor_id = build_shuffled_question(task, donor_by_type, args.seed)
        own_probs, own_rt = dec.decide(task["state"], task["question"], task["labels"])
        shuf_probs, shuf_rt = dec.decide(task["state"], shuf_q, task["labels"])
        own_pred = max(own_probs, key=own_probs.get)
        shuf_pred = max(shuf_probs, key=shuf_probs.get)
        records.append({
            "task_id": task["id"], "family": task["family"], "type": task["question"]["type"],
            "donor_id": donor_id, "expected": task["expected"], "n_labels": len(task["labels"]),
            "own_correct": own_pred == task["expected"], "own_pred": own_pred, "own_p_null": own_rt["p_null"],
            "shuffled_correct": shuf_pred == task["expected"], "shuffled_pred": shuf_pred,
            "shuffled_p_null": shuf_rt["p_null"],
        })
        print(f"  {task['id']:<32} own={'OK ' if records[-1]['own_correct'] else 'no '} "
              f"shuf={'OK ' if records[-1]['shuffled_correct'] else 'no '} (donor {donor_id})", flush=True)

    n = len(records)
    own_acc = sum(r["own_correct"] for r in records) / n
    shuf_acc = sum(r["shuffled_correct"] for r in records) / n
    chance_acc = sum(1.0 / r["n_labels"] for r in records) / n  # mean random-guess rate (K varies per item)
    by_type = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)
    per_type = {t: {"n": len(rs), "own_acc": sum(x["own_correct"] for x in rs) / len(rs),
                     "shuffled_acc": sum(x["shuffled_correct"] for x in rs) / len(rs)}
                for t, rs in by_type.items()}

    summary = {
        "model": args.model, "n": n, "seed": args.seed, "target_ids": target_ids,
        "own_rubric_acc": own_acc, "shuffled_rubric_acc": shuf_acc, "chance_acc": chance_acc,
        "cached_hard_acc_for_these_ids": None,  # filled below if available
        "per_type": per_type,
        "type_counts": dict(Counter(r["type"] for r in records)),
        "verdict": ("executes the rubric (falls toward chance)" if shuf_acc <= chance_acc + 0.10
                    else "does not clearly depend on rubric content at trained length"),
    }
    try:
        cached = load_jsonl(args.cached_results)
        cached_by_id = {r["task_id"]: r for r in cached}
        summary["cached_hard_acc_for_these_ids"] = sum(
            bool(cached_by_id[i]["correct"]) for i in target_ids) / n
    except Exception:
        pass

    with open(out_dir / "results.json", "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    print(f"\n[probe] n={n} own_rubric_acc={own_acc:.3f} shuffled_rubric_acc={shuf_acc:.3f} "
          f"chance_acc={chance_acc:.3f} cached_hard_acc={summary['cached_hard_acc_for_these_ids']}")
    print(f"[probe] per_type: {json.dumps(per_type, indent=2)}")
    print(f"[probe] wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
