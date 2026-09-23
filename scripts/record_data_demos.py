#!/usr/bin/env -S uv run python
"""Records every decide() call the three data-demo modules (sql.js, ads.js, compaction.js)
make into site/data/replays.json (merged, not clobbered) and writes per-demo summaries
(precision/recall, tallies, timings) into site/data/demos/<name>.summary.json, which the
modules read for their static-mode tallies.

State/query TEMPLATES below are byte-for-byte mirrors of the ones documented in the top
comment of each site/js/demos/{sql,ads,compaction}.js - the hash key api.js computes
client-side only matches a replay if the (state, queries) strings are identical, so any
drift between here and the JS breaks static mode silently.

Requires server.py running on :8787 (uv run uvicorn server:app --port 8787).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from record_replays import hash_key, post_decide, q, self_test_hash_parity  # noqa: E402

DEMO_DIR = ROOT / "site" / "data" / "demos"


# ---- sql: mirrors site/js/demos/sql.js's sqlQuestion() ----------------------------------
def sql_question(condition: str, criterion: str) -> str:
    return f"Does this row satisfy the condition: {condition}? Answer yes if {criterion}; otherwise answer no."


# ---- ads: mirrors site/js/demos/ads.js's AD_Q / brandQuestion() / categoryQuestion() ----
AD_Q = (
    "Is this text an advertisement or not? Each option's criterion is listed; pick the single best match.\n"
    "advertisement: promotes a product, service, or brand so the reader will buy or use it  "
    "not_an_advertisement: news, a personal message, a recipe, a notice, a job post, a support request, or other text that does not sell anything"
)


def brand_question(brands: dict) -> str:
    pairs = [f"{name}: {b['cue']}" for name, b in brands.items()]
    return "Which brand is this observation advertising? Each option's criterion is listed; pick the single best match.\n" + "  ".join(pairs)


def category_question(categories: list) -> str:
    pairs = [f"{c['id']}: {c['desc']}" for c in categories]
    return "Which product category does this observation belong to? Each option's criterion is listed; pick the single best match.\n" + "  ".join(pairs)


# ---- compaction: mirrors site/js/demos/compaction.js's KEEP_Q / VERB_Q / callState() ----
KEEP_Q = "Is this tool call's result relevant to the task? Answer yes if the result is about the file, function, or test named in the task; otherwise answer no."
VERB_Q = "Does the result contain source code, a diff, or test output? Answer yes if it does; otherwise answer no."


def call_state(task: str, call: dict) -> str:
    return f"Task: {task}\nTool call: {call['tool']} {call['args']}\nResult:\n{call['result']}"


# ---- auroc: mirrors site/js/demos/compaction.js's auroc() (rank-sum / n_pos*n_neg) ------
def auroc(scores: list, golds: list):
    pos = [s for s, g in zip(scores, golds) if g]
    neg = [s for s, g in zip(scores, golds) if not g]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def prf(pred, gold):
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(g and not p for p, g in zip(pred, gold))
    P = tp / (tp + fp) if tp + fp else 1.0
    R = tp / (tp + fn) if tp + fn else 1.0
    acc = sum(p == g for p, g in zip(pred, gold)) / len(gold)
    return P, R, acc


# ponytail: other agents in this session write site/data/replays.json too (see the
# "replays" worker in the team list) - a plain read-once/write-once here already lost one
# full recording run to a race (another writer's save landed between our read and our
# write and dropped all our new keys). flush_replays() re-reads and unions right before
# every write, and main() calls it after each demo instead of once at the end, so a
# collision only costs the last few calls, not the whole run.
REPLAYS_PATH = None


def flush_replays(new_entries: dict) -> dict:
    on_disk = json.loads(REPLAYS_PATH.read_text()) if REPLAYS_PATH.exists() else {}
    on_disk.update(new_entries)  # ours win for keys we just computed; everyone else's keys pass through
    REPLAYS_PATH.write_text(json.dumps(on_disk, indent=2, ensure_ascii=False) + "\n")
    return on_disk


def record(state, queries, cache):
    key = hash_key(state, queries)
    if key in cache:  # already recorded (re-runs of this script shouldn't re-hit the server)
        res = cache[key]
    else:
        res = post_decide(state, queries)
        cache[key] = res
    return res


def run_sql(cache):
    pool = json.loads((DEMO_DIR / "sql.json").read_text())
    t0 = time.time()
    per_cond = {}
    for cond in pool["conditions"]:
        question = sql_question(cond["condition"], cond["criterion"])
        pred, gold, mss = [], [], []
        for i, row in enumerate(pool["rows"]):
            res = record(row["text"], [q("noul", question, ["no", "yes"])], cache)
            p_yes = res["results"][0]["probs"]["yes"]
            pred.append(p_yes > 0.5)
            gold.append(row["gold"] == cond["id"])
            mss.append(res["ms"])
        P, R, acc = prf(pred, gold)
        n_pos = sum(gold)
        per_cond[cond["id"]] = {
            "condition": cond["condition"],
            "n": len(pool["rows"]),
            "positives": n_pos,
            "matches": sum(pred),
            "precision": round(P, 3),
            "recall": round(R, 3),
            "accuracy": round(acc, 3),
            "mean_ms": round(sum(mss) / len(mss), 1),
        }
        print(f"  sql[{cond['id']}] {cond['condition']!r}: P {P:.2f} R {R:.2f} acc {acc:.3f} matches {sum(pred)}/{len(pool['rows'])}")
    wall_s = time.time() - t0
    summary = {"conditions": per_cond, "n_rows": len(pool["rows"]), "wall_s_all_conditions": round(wall_s, 1)}
    (DEMO_DIR / "sql.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_ads(cache):
    pool = json.loads((DEMO_DIR / "ads.json").read_text())
    brand_names = list(pool["brands"])
    cat_ids = [c["id"] for c in pool["categories"]]
    brand_q = brand_question(pool["brands"])
    cat_q = category_question(pool["categories"])

    ad_pred, ad_gold, mss = [], [], []
    brand_correct = brand_total = cat_correct = cat_total = 0
    for i, obs in enumerate(pool["observations"]):
        queries = [
            q("choice", AD_Q, ["advertisement", "not_an_advertisement"]),
            q("choice", brand_q, brand_names),
            q("choice", cat_q, cat_ids),
        ]
        res = record(obs["text"], queries, cache)
        ad_r, brand_r, cat_r = res["results"]
        is_ad_pred = ad_r["argmax"] == "advertisement"
        ad_pred.append(is_ad_pred)
        ad_gold.append(obs["isAd"])
        mss.append(res["ms"])
        if obs["isAd"]:
            brand_total += 1
            if brand_r["argmax"] == obs["brand"]:
                brand_correct += 1
            cat_total += 1
            if cat_r["argmax"] == pool["brands"][obs["brand"]]["category"]:
                cat_correct += 1

    P, R, ad_acc = prf(ad_pred, ad_gold)
    always_yes_acc = sum(ad_gold) / len(ad_gold)
    summary = {
        "n_observations": len(pool["observations"]),
        "is_ad": {"accuracy": round(ad_acc, 3), "precision": round(P, 3), "recall": round(R, 3), "baseline_always_yes": round(always_yes_acc, 3)},
        "brand": {"accuracy": round(brand_correct / brand_total, 3), "n": brand_total, "chance": round(1 / len(brand_names), 3)},
        "category": {"accuracy": round(cat_correct / cat_total, 3), "n": cat_total, "chance": round(1 / len(cat_ids), 3)},
        "mean_ms": round(sum(mss) / len(mss), 1),
    }
    print(f"  ads is-ad: acc {ad_acc:.3f} (baseline {always_yes_acc:.3f}) P {P:.2f} R {R:.2f}")
    print(f"  ads brand: acc {brand_correct}/{brand_total} = {brand_correct/brand_total:.3f} (chance {1/len(brand_names):.3f})")
    print(f"  ads category: acc {cat_correct}/{cat_total} = {cat_correct/cat_total:.3f} (chance {1/len(cat_ids):.3f})")
    (DEMO_DIR / "ads.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_compaction(cache):
    pool = json.loads((DEMO_DIR / "compaction.json").read_text())
    p_keep, p_verb, gold_keep, gold_verb, mss = [], [], [], [], []
    for i, call in enumerate(pool["calls"]):
        state = call_state(pool["task"], call)
        queries = [q("noul", KEEP_Q, ["no", "yes"]), q("noul", VERB_Q, ["no", "yes"])]
        res = record(state, queries, cache)
        p_keep.append(res["results"][0]["probs"]["yes"])
        p_verb.append(res["results"][1]["probs"]["yes"])
        gold_keep.append(call["keep"])
        gold_verb.append(call["verbatim"])
        mss.append(res["ms"])

    keep_pred = [p > 0.5 for p in p_keep]
    verb_pred = [p > 0.5 for p in p_verb]
    _, _, keep_acc = prf(keep_pred, gold_keep)
    _, _, verb_acc = prf(verb_pred, gold_verb)
    keep_majority = max(sum(gold_keep), len(gold_keep) - sum(gold_keep)) / len(gold_keep)
    verb_majority = max(sum(gold_verb), len(gold_verb) - sum(gold_verb)) / len(gold_verb)
    auc = auroc(p_keep, gold_keep)
    max_p_keep = max(p_keep)
    summary = {
        "n_calls": len(pool["calls"]),
        "keep": {"accuracy": round(keep_acc, 3), "majority": round(keep_majority, 3), "auroc": round(auc, 3) if auc is not None else None, "max_p_keep": round(max_p_keep, 3)},
        "verbatim": {"accuracy": round(verb_acc, 3), "majority": round(verb_majority, 3)},
        "mean_ms": round(sum(mss) / len(mss), 1),
    }
    print(f"  compaction keep: acc {keep_acc:.3f} (majority {keep_majority:.3f}) AUROC {auc:.3f} max P(keep) {max_p_keep:.3f}")
    print(f"  compaction verbatim: acc {verb_acc:.3f} (majority {verb_majority:.3f})")
    (DEMO_DIR / "compaction.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    global REPLAYS_PATH
    self_test_hash_parity()
    REPLAYS_PATH = ROOT / "site" / "data" / "replays.json"
    before = len(json.loads(REPLAYS_PATH.read_text())) if REPLAYS_PATH.exists() else 0

    # each cache starts as whatever's on disk right now (so record() can skip calls other
    # agents already recorded under the same key) and is flushed - merged, not clobbered -
    # right after its demo finishes, so a concurrent writer elsewhere only ever races the
    # last few seconds of work, not the whole ~5 minute run.
    for name, run in (("sql", run_sql), ("ads", run_ads), ("compaction", run_compaction)):
        print(f"recording {name}...")
        cache = json.loads(REPLAYS_PATH.read_text()) if REPLAYS_PATH.exists() else {}
        run(cache)
        on_disk = flush_replays(cache)
        print(f"  flushed ({len(on_disk)} keys on disk)")

    on_disk = json.loads(REPLAYS_PATH.read_text())
    print(f"\n{REPLAYS_PATH} now has {len(on_disk)} keys ({len(on_disk) - before} new this run)")


if __name__ == "__main__":
    main()
