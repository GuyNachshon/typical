#!/usr/bin/env -S uv run python
"""Record site/data/replays.json: precomputed /api/decide responses for every demo in
presets.json, keyed by the same hash js/api.js::hashKey computes client-side from
(state, queries) - so static deployments (no local server) can replay a live result
instead of showing nothing. Requires site/server.py running on :8787 (uv run uvicorn
server:app --port 8787).

Query dicts are built with keys in the exact order {type, question, labels} everywhere,
matching how the frontend constructs them (js/demo.js's row.get(), js/snake-ui.js's
typicalPolicy) - JSON.stringify/json.dumps both serialize dict/object keys in insertion
order, so the hash only matches if key order matches too.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
from typical.core import query_text  # noqa: E402  (verbatim instructions+criteria renderer)

SERVER = "http://127.0.0.1:8787"
PRESETS = json.loads((ROOT / "site" / "data" / "presets.json").read_text())

DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def to_base36(n: int) -> str:
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, 36)
        out.append(DIGITS[r])
    return "".join(reversed(out))


def hash_key(state: str, queries: list) -> str:
    """Port of js/api.js::hashKey (djb2-ish, 32-bit wraparound, base36)."""
    s = state + json.dumps(queries, separators=(",", ":"), ensure_ascii=False)
    h = 5381
    for ch in s:
        h = (((h << 5) & 0xFFFFFFFF) + h + ord(ch)) & 0xFFFFFFFF
    return to_base36(h)


def q(type_, question, labels):
    return {"type": type_, "question": question, "labels": labels}  # key order matters, see module docstring


def post_decide(state: str, queries: list, model: str = "typical-small") -> dict:
    body = json.dumps({"model": model, "state": state, "queries": queries}).encode()
    req = urllib.request.Request(f"{SERVER}/api/decide", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def build_pairs() -> list[tuple[str, str, list]]:
    """-> [(demo_label, state, queries), ...] - the exact calls the page will make."""
    pairs = []

    pg = PRESETS["playground"]
    pairs.append(("playground", pg["state"], pg["queries"]))
    # single-question call the page makes first, to show M=1 vs M=4-in-one-pass (H2)
    pairs.append(("playground[single]", pg["state"], pg["queries"][:1]))
    # the hero instrument holds this state fixed and cycles the question, so each one is also
    # recorded on its own - otherwise only the first cycles offline
    for i, query in enumerate(pg["queries"]):
        pairs.append((f"playground[q{i}]", pg["state"], [query]))
    # The board lets a visitor take a candidate out of the Choice set (and put one back). Each
    # variant is a different call, so each is recorded: dropping the winner is how abstention
    # becomes something you watch happen rather than something the page claims.
    base = pg["queries"][0]
    extra = "send a replacement"
    variants = [[l for l in base["labels"] if l != drop] for drop in base["labels"]]
    variants.append([*base["labels"], extra])
    for labels in variants:
        pairs.append((f"playground[cand:{'|'.join(labels)}]", pg["state"], [q("choice", base["question"], labels)]))

    flip = PRESETS["flip"]
    for i, order in enumerate(flip["orders"]):
        criteria = {label: flip["rules"][label] for label in order}
        question = query_text({"type": "choice", "instructions": flip["question_prefix"], "criteria": criteria})
        pairs.append((f"flip[{i}]", flip["state"], [q("choice", question, order)]))

    pol = PRESETS["policy"]
    pairs.append(("policy", pol["state"], pol["queries"]))

    ab = PRESETS["abstain"]
    for variant in ab["variants"]:
        pairs.append((f"abstain[{variant['label']}]", ab["state"], variant["queries"]))

    inbox = PRESETS["inbox"]
    for utt in inbox["examples"]:
        pairs.append((f"inbox[{utt[:24]!r}]", utt, [q("choice", inbox["question"], inbox["labels"])]))

    triage = PRESETS["triage"]
    rendered_q = query_text(triage["question"])
    for row in triage["examples"]:
        pairs.append((f"triage[{row['label']}]", row["state"], [q("choice", rendered_q, triage["labels"])]))

    return pairs


def self_test_hash_parity():
    """hashKey must match js/api.js's version bit-for-bit - check one key against node."""
    state, queries = "a", [{"q": 1}]
    py_key = hash_key(state, queries)
    js = f"""
    function hashKey(state, queries) {{
      const s = state + JSON.stringify(queries);
      let h = 5381;
      for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
      return (h >>> 0).toString(36);
    }}
    console.log(hashKey({json.dumps(state)}, {json.dumps(queries)}));
    """
    import subprocess
    node_key = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout.strip()
    assert py_key == node_key, f"hash mismatch: python={py_key!r} node={node_key!r}"
    print(f"hashKey parity OK ({py_key})")


def main():
    self_test_hash_parity()
    pairs = build_pairs()
    replays = {}
    rows = []
    for label, state, queries in pairs:
        t0 = time.time()
        try:
            res = post_decide(state, queries)
        except urllib.error.URLError as e:
            print(f"FAILED {label}: {e} - is site/server.py running on :8787?", file=sys.stderr)
            raise SystemExit(1)
        key = hash_key(state, queries)
        replays[key] = res
        wall_ms = (time.time() - t0) * 1000
        for i, r in enumerate(res["results"]):
            top_label = r["argmax"]
            top_p = r["probs"].get(top_label, 0.0)
            qtext = queries[i]["question"][:40]
            rows.append((f"{label}#{i}", qtext, top_label, top_p, r["p_null"], res["ms"]))
        print(f"recorded {label} ({len(res['results'])} result(s), {wall_ms:.0f} ms wall)")

    out_path = ROOT / "site" / "data" / "replays.json"
    out_path.write_text(json.dumps(replays, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path} ({len(replays)} keys)\n")

    w = [24, 40, 20, 8, 8, 8]
    header = ("demo", "question", "argmax", "top p", "p_null", "ms")
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(header)))
    for demo, qtext, top_label, top_p, p_null, ms in rows:
        cells = [demo, qtext, top_label, f"{top_p:.3f}", f"{p_null:.3f}", f"{ms:.0f}"]
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(cells)))


if __name__ == "__main__":
    main()
