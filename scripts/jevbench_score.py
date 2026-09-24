"""Score a JevBench run on the four published axes, and say plainly what it cannot score.

    uv run --no-sync python scripts/jevbench_score.py runs/jev_native_ts1c runs/jev_native_tm2
    uv run --no-sync python scripts/jevbench_score.py --demo

The benchmark scores four axes rather than the tier accuracies we currently publish:

  Intelligence  (accuracy - chance) / (1 - chance) per tier, clipped at 0.
                Weights: hard .30, easy .14, standard .28, judge .28.
  Calibration   hard tier: ECE plus fidelity to the exact gold distributions.
  Speed         mean of score(p50), score(p95); score(s) = 100 - 20*log10(s / 0.1)
  Cost          100 - 30*log10(usd per 1,000 decisions / 0.001)

Two of those we cannot compute from outside, and this prints that rather than a number:

  * JUDGE is 28% of Intelligence and its items are not public. A run over the 231 public ids
    covers 72% of the weight. Reporting the remaining 72% as if it were the axis would overstate
    a model that is bad at the hard tier, because the missing 28% is the tier most like it.
  * COST needs a price per 1,000 decisions. A self-hosted checkpoint has no provider tariff, so
    the harness writes null and our adapter declares cost_basis local_gpu_no_provider_tariff.
    Any figure here is an operator's GPU hourly rate divided by a throughput they chose; it is a
    declaration, not a measurement, so this refuses to invent one.

Speed and the hard-tier ECE that Calibration is built on both come straight out of the harness.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# app:leaderboard; the majority baselines the benchmark publishes for its own tiers
CHANCE = {"original": 0.311, "easy": 0.284, "hard": 0.336}
WEIGHT = {"hard": 0.30, "easy": 0.14, "original": 0.28, "judge": 0.28}
PUBLIC = ("original", "easy", "hard")
TIER_NAME = {"original": "standard", "easy": "easy", "hard": "hard"}


def normalised(acc: float, chance: float) -> float:
    """Headroom above chance, clipped at zero: a model at chance scores 0, not 'a bit of credit'."""
    return max(0.0, (acc - chance) / (1.0 - chance))


def speed_score(seconds: float) -> float:
    """0.1 s = 100, and every 10x slower costs 20. Unbounded above, which is why a 77 ms decision
    scores over 100 -- the scale was drawn for systems that answer in seconds."""
    return 100.0 - 20.0 * math.log10(seconds / 0.1)


def cost_score(usd_per_1000: float) -> float:
    return 100.0 - 30.0 * math.log10(usd_per_1000 / 0.001)


def score_run(run: Path) -> dict:
    d = json.loads((run / "summary.json").read_text())
    tiers = {}
    covered = 0.0
    intelligence = 0.0
    for t in PUBLIC:
        if t not in d:
            continue
        acc = d[t]["accuracy"]
        n = normalised(acc, CHANCE[t])
        tiers[TIER_NAME[t]] = {"accuracy": acc, "chance": CHANCE[t], "normalised": round(n * 100, 2), "weight": WEIGHT[t]}
        intelligence += WEIGHT[t] * n * 100
        covered += WEIGHT[t]
    hard = d.get("hard", {})
    std = d.get("original", {})
    lat = std.get("latency") or {}
    out = {
        "run": run.name,
        "version": std.get("version"),
        "tiers": tiers,
        "intelligence_partial": round(intelligence, 2),
        "intelligence_weight_covered": round(covered, 2),
        "intelligence_complete": False,
        "missing": ["judge tier is not public (28% of Intelligence)"],
        "calibration_inputs": {
            "hard_ece": (hard.get("ece") or {}).get("ece"),
            "hard_ece_n": (hard.get("ece") or {}).get("n"),
            "hard_brier": hard.get("brier_mean"),
            "gold_distribution_fidelity": None,
        },
        "speed": None,
        "cost": None,
        "cost_basis": std.get("cost_basis"),
        "price_per_1000_decisions_usd": std.get("price_per_1000_decisions_usd"),
    }
    if lat.get("p50_s") and lat.get("p95_s"):
        p50, p95 = speed_score(lat["p50_s"]), speed_score(lat["p95_s"])
        out["speed"] = {
            "p50_s": lat["p50_s"], "p95_s": lat["p95_s"],
            "score_p50": round(p50, 1), "score_p95": round(p95, 1), "score": round((p50 + p95) / 2, 1),
        }
    return out


def demo() -> None:
    assert abs(speed_score(0.1) - 100.0) < 1e-9, "0.1 s is the anchor"
    assert abs(speed_score(1.0) - 80.0) < 1e-9, "ten times slower costs twenty"
    assert speed_score(0.0774) > 100, "a decision faster than the anchor scores above it"
    assert abs(cost_score(0.001) - 100.0) < 1e-9, "a tenth of a cent per thousand is the anchor"
    assert abs(cost_score(0.01) - 70.0) < 1e-9, "ten times dearer costs thirty"
    assert normalised(0.336, 0.336) == 0.0, "a model at chance scores zero, not a fraction"
    assert normalised(0.2, 0.336) == 0.0, "and below chance is clipped, not negative"
    assert abs(normalised(1.0, 0.284) - 1.0) < 1e-9, "a perfect tier is the whole headroom"
    assert abs(sum(WEIGHT.values()) - 1.0) < 1e-9, "the four Intelligence weights are a partition"
    assert abs(sum(WEIGHT[t] for t in PUBLIC) - 0.72) < 1e-9, "the public tiers are 72% of it"
    print("jevbench_score.py self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo or not a.runs:
        demo()
        return
    for r in a.runs:
        s = score_run(r)
        print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
