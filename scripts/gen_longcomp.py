"""Generate a long-state composition eval whose items exist at several lengths.

    uv run --no-sync python scripts/gen_longcomp.py --items 140 --out evals/longcomp_v1.jsonl
    uv run --no-sync python scripts/gen_longcomp.py --demo

Why this exists. JevBench's hard tier cannot tell us whether long states hurt, because its long
items are also its hardest: every one of the 19 long_policy items is over 2,048 tokens and they are
long *and* compositional at once. Splitting that tier by length compares different questions.

So here the same decision is rendered at several lengths. The evidence, the question, the candidate
set and the gold answer are byte-identical across renderings; only the amount of irrelevant policy
around them changes, and where in it the evidence sits. A model that answers the short rendering and
misses the long one failed on length, and nothing else -- the pairing is what makes that statement
available, and it is the statement JevBench cannot make.

Gold is computed, never labelled. Each item states facts, applies a rule to them arithmetically, and
the correct option follows from the arithmetic. Distractors are the answers you get from the same
facts under a plausibly wrong reading -- an off-by-one on a deadline, a sum that misses a line, the
higher number when the rule asks for the better ratio -- so a model cannot score by eliminating
nonsense. K is fixed at 4, so chance is .25 on every item.

Nothing here shares a generator with any training corpus: the clause grammar, the filler and the
arithmetic are written in this file. That matters because the project's own rubric-generalisation
results already share one rule engine between train and test, and an eval built on the training
generator would measure in-distribution fit rather than the thing we want to know.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# Filler. Real-sounding policy prose with no bearing on any question, used only to push the state
# past a token budget. Every sentence is inert: no dates, no amounts, no thresholds, so it can
# never accidentally become evidence.
FILLER = [
    "The administrator maintains a register of all correspondence relating to this policy.",
    "Notices under this section may be delivered by post or by electronic means.",
    "Headings in this document are for convenience and do not affect interpretation.",
    "Where the context permits, the singular includes the plural.",
    "The schedule forms part of this agreement and is to be read with it.",
    "Nothing in this clause limits any right the parties have at law.",
    "The parties agree to act reasonably in giving any consent required here.",
    "A waiver of one breach is not a waiver of any later breach.",
    "This section survives the termination of the agreement.",
    "Any reference to a statute includes a reference to that statute as amended.",
    "Records are retained in accordance with the administrator's retention schedule.",
    "The administrator may appoint an agent to perform its functions under this part.",
    "Correspondence is deemed received on the business day after it is sent.",
    "The parties may agree in writing to vary the procedure set out in this part.",
    "Defined terms carry the meaning given to them in the definitions section.",
    "This part applies subject to any mandatory provision of applicable law.",
    "An election under this clause must be made in the form the administrator requires.",
    "The administrator publishes guidance on the operation of this part from time to time.",
]

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
DEPTS = ["Claims", "Underwriting", "Recoveries", "Compliance"]


def _money(n: int) -> str:
    return f"${n:,}"


# ---------------------------------------------------------------------------------------------
# Three composition types. Each returns (facts, rule, question, options, gold_index, depth).
# `depth` is how many stated numbers you have to combine before the rule can be applied.

def build_threshold(rng: random.Random):
    """Sum several line items, compare the total against a stated excess."""
    n = rng.choice([3, 4])
    lines = [rng.randrange(400, 3200) for _ in range(n)]
    excess = rng.randrange(2000, 7000)
    total = sum(lines)
    parts = [f"The claim file records {len(lines)} itemised losses."]
    for i, v in enumerate(lines, 1):
        parts.append(f"Line {i} is assessed at {_money(v)}.")
    rule = (f"A claim is payable in full when the sum of its itemised losses exceeds the policy "
            f"excess of {_money(excess)}. Otherwise the claim is declined.")
    gold = "pay in full" if total > excess else "decline"
    # the distractors are the answers you reach by dropping a line, or by comparing the largest
    # single line against the excess instead of the sum
    opts = ["pay in full", "decline", "pay the excess only", "refer to underwriting"]
    return parts, rule, "How should the administrator dispose of this claim?", opts, opts.index(gold), len(lines)


def build_temporal(rng: random.Random):
    """Add a stated notice period to a stated date and compare against a deadline."""
    m = rng.randrange(0, 11)
    day = rng.randrange(1, 25)
    notice = rng.choice([14, 21, 30, 45])
    elapsed = rng.choice([10, 20, 35, 60])
    parts = [
        f"The loss was notified to the administrator on {day} {MONTHS[m]}.",
        f"The administrator acknowledged the notification {elapsed} days after it was received.",
    ]
    rule = (f"An acknowledgement is timely when it is issued within {notice} days of notification. "
            f"A timely acknowledgement preserves the claim; an untimely one refers it to Compliance.")
    gold = "preserve the claim" if elapsed <= notice else "refer to Compliance"
    opts = ["preserve the claim", "refer to Compliance", "reissue the acknowledgement", "close the file"]
    return parts, rule, "What follows from the acknowledgement date?", opts, opts.index(gold), 2


def build_tradeoff(rng: random.Random):
    """Two options, each with a cost and a recovery; the rule asks for the better ratio, not the
    larger recovery -- so the higher absolute number is always a distractor."""
    a_cost, b_cost = rng.randrange(200, 900), rng.randrange(200, 900)
    a_rec = a_cost * rng.choice([2, 3, 4]) + rng.randrange(0, 120)
    b_rec = b_cost * rng.choice([2, 3, 4]) + rng.randrange(0, 120)
    parts = [
        f"Route A costs {_money(a_cost)} to pursue and is expected to recover {_money(a_rec)}.",
        f"Route B costs {_money(b_cost)} to pursue and is expected to recover {_money(b_rec)}.",
    ]
    rule = ("Recoveries selects the route with the higher ratio of expected recovery to cost, "
            "not the route with the larger expected recovery.")
    gold = "Route A" if (a_rec / a_cost) > (b_rec / b_cost) else "Route B"
    opts = ["Route A", "Route B", "pursue both routes", "abandon recovery"]
    return parts, rule, "Which route should Recoveries pursue?", opts, opts.index(gold), 4


BUILDERS = {"threshold": build_threshold, "temporal": build_temporal, "tradeoff": build_tradeoff}


def render(facts, rule, rng: random.Random, target_tokens: int, position: str, tok) -> str:
    """Pad with inert filler to about `target_tokens`, with the evidence first or last."""
    head = f"POLICY EXTRACT — {rng.choice(DEPTS)} department\n\n{rule}\n"
    ev = "\n".join(facts)
    body, used = [], len(tok(head + ev)["input_ids"])
    i = 0
    while used < target_tokens:
        s = FILLER[(i + rng.randrange(0, len(FILLER))) % len(FILLER)]
        body.append(f"{len(body) + 1}. {s}")
        used += len(tok(s)["input_ids"]) + 4
        i += 1
    pad = "\n".join(body)
    return f"{head}\n{ev}\n\n{pad}\n" if position == "first" else f"{head}\n{pad}\n\n{ev}\n"


def generate(n_items: int, lengths, positions, seed: int, tok):
    rng = random.Random(seed)
    out = []
    for i in range(n_items):
        kind = list(BUILDERS)[i % len(BUILDERS)]
        facts, rule, q, opts, gold, depth = BUILDERS[kind](rng)
        for L in lengths:
            for pos in positions:
                state = render(facts, rule, random.Random(seed * 7919 + i), L, pos, tok)
                out.append({
                    "item_id": f"{kind}-{i:04d}",
                    "render_id": f"{kind}-{i:04d}-L{L}-{pos}",
                    "kind": kind, "depth": depth,
                    "target_tokens": L, "state_tokens": len(tok(state)["input_ids"]),
                    "evidence_position": pos,
                    "state": state, "question": q, "labels": opts,
                    "gold": opts[gold], "chance": 1.0 / len(opts),
                })
    return out


def demo() -> None:
    class T:  # a stand-in tokenizer: word count is enough to exercise the shape offline
        def __call__(self, s): return {"input_ids": s.split()}
    rng = random.Random(0)
    for name, fn in BUILDERS.items():
        facts, rule, q, opts, gold, depth = fn(rng)
        assert len(opts) == 4, f"{name} must offer four options so chance is .25"
        assert len(set(opts)) == 4, f"{name} has a duplicated option"
        assert 0 <= gold < 4 and facts and rule and q
    rows = generate(6, [40, 200], ["first", "last"], 1, T())
    assert len(rows) == 6 * 2 * 2, "every item is rendered at every length and position"
    by_item = {}
    for r in rows:
        by_item.setdefault(r["item_id"], set()).add(r["gold"])
    assert all(len(v) == 1 for v in by_item.values()), "an item's gold cannot change with its rendering"
    short = [r for r in rows if r["target_tokens"] == 40]
    long_ = [r for r in rows if r["target_tokens"] == 200]
    assert min(r["state_tokens"] for r in long_) > max(r["state_tokens"] for r in short), "long renderings are longer"
    first = {r["item_id"] for r in rows if r["evidence_position"] == "first"}
    assert first == set(by_item), "every item exists in both positions"
    print("gen_longcomp.py self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=140)
    ap.add_argument("--lengths", default="700,1900,3300")
    ap.add_argument("--positions", default="first,last")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out", type=Path, default=Path("evals/longcomp_v1.jsonl"))
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    rows = generate(a.items, [int(x) for x in a.lengths.split(",")], a.positions.split(","), a.seed, tok)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"wrote {len(rows)} renderings of {a.items} items -> {a.out}")
    print("  per kind:", kinds)
    for L in sorted({r["target_tokens"] for r in rows}):
        t = [r["state_tokens"] for r in rows if r["target_tokens"] == L]
        print(f"  target {L:5}: actual {min(t)}-{max(t)} tokens")


if __name__ == "__main__":
    main()
