"""Reads runs/*.json (+ hand-transcribed release-card tables) and writes site/data/*.json --
small, flat, chart-ready. Run: uv run python site/precompute.py

ponytail: JevBench acc/Brier/ECE, ECE bins, chance baselines and the K/M latency sweep have one
unambiguous JSON source each, so those are read programmatically. NLU/topic-intent/MMLU/held-out/
external/latency-ladder numbers live only in the release cards' markdown tables (releases/*.md,
REPORT.md) as already-curated raw-vs-scaled/acc-vs-acc_k selections -- PLAN.md names those tables
the project's source of truth, so they're hand-transcribed here (verified once against the
underlying results.json/eval_wf*.json in the process that built this file) rather than re-derived
with a table parser. Every model dict carries a "sources" map, one path per metric group.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # this file lives in site/
OUT = ROOT / "site" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def load(rel):
    return json.loads((ROOT / rel).read_text())


def jevbench(rel):
    """runs/jev_native_*/summary.json -> {std, easy, hard: {acc, brier, ece, n}}."""
    d = load(rel)
    tiers = {"std": "original", "easy": "easy", "hard": "hard"}
    out = {}
    for tier, key in tiers.items():
        t = d[key]
        out[tier] = {"acc": t["accuracy"], "brier": t["brier_mean"], "ece": t["ece"]["ece"], "n": t["n_attempted"]}
    return out


def ece_bins(rel):
    d = load(rel)
    return {
        tier: [
            {"lo": b["lo"], "hi": b["hi"], "n": b["n"], "mean_confidence": b["mean_confidence"], "accuracy": b["accuracy"]}
            for b in d[key]["ece"]["bins"]
        ]
        for tier, key in [("std", "original"), ("easy", "easy"), ("hard", "hard")]
    }


# ---------------------------------------------------------------------------
# models.json
# ---------------------------------------------------------------------------
MODELS = [
    {
        "id": "typical-small-preview",
        "backbone": "Qwen3-1.7B-Base", "params": "1.7B", "tap": "20/28",
        "released": True, "hf_url": "https://huggingface.co/OzLabs/typical-small-preview",
        "jevbench": jevbench("runs/jev_native_v3t20_wf/summary.json"),
        "latency": {"single_ms": {"k2": 45, "k32": 46, "k256": 106},
                    "marginal_ms_m32": {"k2": 2.7, "k32": 3.7, "k256": 28.0},
                    "peak_gb": {"k2": 6.4, "k256": 9.3},
                    "note": "borrows typical-small's ladder -- same 1.7B recipe shape"},
        "mmlu_pro_among_k": 0.330,
        "nlu": {"snli": 0.894, "mnli": 0.849, "boolq": 0.831, "anli": 0.481},
        "topic_intent": {"clinc": 0.734, "trec": 0.372, "hwu": 0.735, "ng20": 0.522},
        "external": {"pagerduty": 0.779, "jevlogs": 0.522, "mind2web": 0.427},
        "held_out": {"noul": 0.699, "score": 0.498, "flip_both_correct": 0.565},
        "sources": {
            "jevbench": "runs/jev_native_v3t20_wf/summary.json",
            "latency": "releases/typical-small.md#latency",
            "mmlu_pro_among_k": "releases/typical-small.md#§3ae-table",
            "nlu": "releases/typical-small.md#§3ae-table",
            "topic_intent": "releases/typical-small.md#§3ae-table",
            "external": "releases/typical-small.md#§3ae-table",
            "held_out": "releases/typical-small.md#§3ae-table",
        },
    },
    {
        "id": "typical-small",
        "backbone": "Qwen3-1.7B-Base", "params": "1.7B", "tap": "20/28",
        "released": True, "hf_url": "https://huggingface.co/OzLabs/typical-small",
        "jevbench": jevbench("runs/jev_native_ts1b/summary.json"),
        "latency": {"single_ms": {"k2": 45, "k32": 46, "k256": 106},
                    "marginal_ms_m32": {"k2": 2.7, "k32": 3.7, "k256": 28.0},
                    "peak_gb": {"k2": 6.4, "k256": 9.3}},
        "mmlu_pro_among_k": 0.343,
        "nlu": {"snli": 0.894, "mnli": 0.859, "boolq": 0.824, "anli": 0.497},
        "topic_intent": {"clinc": 0.804, "trec": 0.508, "hwu": 0.761, "ng20": 0.540},
        "external": {"pagerduty": 0.817, "jevlogs": 0.710, "mind2web": 0.357},
        "held_out": {"noul": 0.715, "score": 0.520, "flip_both_correct": 0.581},
        "sources": {
            "jevbench": "runs/jev_native_ts1b/summary.json",
            "latency": "releases/typical-small.md#latency",
            "mmlu_pro_among_k": "releases/typical-small.md#§3ae-table",
            "nlu": "releases/typical-small.md#§3ae-table",
            "topic_intent": "releases/typical-small.md#§3ae-table",
            "external": "releases/typical-small.md#§3ae-table",
            "held_out": "releases/typical-small.md#§3ae-table",
        },
    },
    {
        "id": "typical-medium",
        "backbone": "Qwen3-4B-Base", "params": "4B", "tap": "26/36",
        "released": True, "hf_url": "https://huggingface.co/OzLabs/typical-medium",
        "jevbench": jevbench("runs/jev_native_tm1b/summary.json"),
        "latency": {"single_ms": {"k2": 56, "k32": 56, "k256": 118},
                    "marginal_ms_m32": {"k2": 3.3, "k32": 6.1, "k256": 50.6},
                    "peak_gb": {"k2": 14.6, "k256": 18.6},
                    "bench": "runs/bench_tm1b/bench.json"},  # measured on the released checkpoint; card numbers match
        "mmlu_pro_among_k": 0.458,
        "nlu": {"snli": 0.909, "mnli": 0.861, "boolq": 0.843, "anli": 0.544},
        "topic_intent": {"clinc": 0.847, "trec": 0.414, "hwu": 0.769, "ng20": 0.588},
        "external": {"pagerduty": 0.838, "jevlogs": 0.673, "mind2web": 0.544},
        "held_out": {"noul": 0.811, "score": 0.528, "flip_both_correct": 0.786},
        "sources": {
            "jevbench": "runs/jev_native_tm1b/summary.json",
            "latency": "releases/typical-medium.md#latency",
            "mmlu_pro_among_k": "releases/typical-medium.md#§3af-table",
            "nlu": "releases/typical-medium.md#§3af-table",
            "topic_intent": "releases/typical-medium.md#§3af-table",
            "external": "releases/typical-medium.md#§3af-table",
            "held_out": "releases/typical-medium.md#§3af-table",
        },
    },
    {
        "id": "typical-14b-ladder",
        "backbone": "Qwen3-14B-Base", "params": "14B", "tap": "28/40",
        "released": False, "hf_url": None,
        "jevbench": jevbench("runs/jev_native_ladder_14b/summary.json"),
        "latency": {"single_ms": {"k2": 60, "k32": 62, "k256": 157},
                    "marginal_ms_m32": {"k2": 3.9, "k32": 11.7, "k256": 109.7},
                    "peak_gb": {"k2": 51.6, "k256": 57.5}},
        "mmlu_pro_among_k": 0.514,
        "nlu": {"snli": 0.910, "mnli": 0.864, "boolq": 0.894, "anli": 0.588},
        "topic_intent": {"clinc": 0.834, "trec": 0.472, "hwu": 0.792, "ng20": 0.668},
        "external": None,  # not reported in the §3ab scaling-ladder table -- unsourced, not run
        "held_out": {"noul": 0.887, "score": 0.558, "flip_both_correct": None},  # flip: not in ladder table
        "sources": {
            "jevbench": "runs/jev_native_ladder_14b/summary.json",
            "latency": "gpu-runpod-full-experiment:REPORT.md#§3ab-apples-to-apples-table",
            "mmlu_pro_among_k": "gpu-runpod-full-experiment:REPORT.md#§3ab-table",
            "nlu": "gpu-runpod-full-experiment:REPORT.md#§3ab-table",
            "topic_intent": "gpu-runpod-full-experiment:REPORT.md#§3ab-table",
            "held_out": "gpu-runpod-full-experiment:REPORT.md#§3ab-table",
        },
        "note": "scaling-ladder point (PLAN7 Track A), not a public release -- no HF repo, no PagerDuty/jevlogs/Mind2Web/flip numbers in the source table",
    },
]
(OUT / "models.json").write_text(json.dumps(MODELS, indent=2))

# ---------------------------------------------------------------------------
# reliability.json -- ECE bins per model per tier
# ---------------------------------------------------------------------------
RELIABILITY = {
    "typical-small-preview": ece_bins("runs/jev_native_v3t20_wf/summary.json"),
    "typical-small": ece_bins("runs/jev_native_ts1b/summary.json"),
    "typical-medium": ece_bins("runs/jev_native_tm1b/summary.json"),
    "typical-14b-ladder": ece_bins("runs/jev_native_ladder_14b/summary.json"),
    "source": "runs/jev_native_*/summary.json (.<tier>.ece.bins)",
}
(OUT / "reliability.json").write_text(json.dumps(RELIABILITY, indent=2))

# ---------------------------------------------------------------------------
# latency.json -- ladder (per release card / REPORT) + two raw sweeps:
#   energy_sweep_legacy: runs/bench_a100 -- OLD energy checkpoint (joint_emb_lw_v5), kept for
#     reference only, NOT the native releases.
#   native_sweep: runs/bench_native_{L256c8,L1000,L2000} -- model nc_n3, 1.7B native readout,
#     same readout shape as the typical-small/-medium releases.
# ---------------------------------------------------------------------------
_a100 = load("runs/bench_a100/bench.json")
energy_sweep_legacy = {}
for k, by_m in _a100["by_k"].items():
    energy_sweep_legacy[k] = {
        "ours": [{"m": e["m"], "total_ms": e["total_ms"], "marginal_ms": e["marginal_ms"]} for e in by_m["ours"]],
        "prompted_baseline": [{"m": e["m"], "total_ms": e["time_ms"], "marginal_ms": e["marginal_ms"]} for e in by_m["b"]],
    }

_FIELDS = ("m", "t_state_ms", "t_queries_ms", "total_ms", "marginal_ms")
native_sweep = {}
for path in ["runs/bench_native_L256c8/bench.json", "runs/bench_native_L1000/bench.json", "runs/bench_native_L2000/bench.json"]:
    d = load(path)
    crossover_by_l = d.get("crossover") or {}
    for L, by_k in d["native"].items():
        Ld = native_sweep.setdefault(L, {})
        for K, entry in by_k.items():
            Ld[K] = {
                "native": [{f: e[f] for f in _FIELDS} for e in entry["native"]],
                "baseline": [{f: e[f] for f in _FIELDS} for e in entry["b_batched"]],
                "peak_mb": entry["native_peak_mem_mb"],
                "crossover": crossover_by_l.get(L),
            }

LATENCY = {
    "ladder": {m["id"]: m["latency"] | {"source": m["sources"].get("latency")} for m in MODELS},
    "energy_sweep_legacy": energy_sweep_legacy,
    "energy_sweep_legacy_note": "OLD energy checkpoint (joint_emb_lw_v5), not the native releases -- kept for reference only. See native_sweep for the shape the releases actually use.",
    "energy_sweep_legacy_source": "runs/bench_a100/bench.json (by_k[K].ours / .b, model=runs/joint_emb_lw)",
    "native_sweep": native_sweep,
    "native_sweep_source": "runs/bench_native_{L256c8,L1000,L2000}/bench.json (native[L][K]: .native/.b_batched/.native_peak_mem_mb, model=runs/nc_n3, Qwen3-1.7B-Base)",
}
(OUT / "latency.json").write_text(json.dumps(LATENCY, indent=2))

# ---------------------------------------------------------------------------
# chance.json -- JevBench majority/chance baselines
# ---------------------------------------------------------------------------
CHANCE = {
    "std": {"acc": 0.311, "n": 72, "n_eff": 36},
    "easy": {"acc": 0.284, "n": 48},
    "hard": {"acc": 0.336, "n": 111},
    "source": "releases/typical-small.md#jevbench-disclosure (identical text in typical-medium.md)",
}
(OUT / "chance.json").write_text(json.dumps(CHANCE, indent=2))

# ---------------------------------------------------------------------------
# frozen.json -- frozen-backbone controls on the same 231 public JevBench ids
# (letter logits over rendered options, 3 exemplars). Transcribed from REPORT §3ag (2026-09-22).
# ---------------------------------------------------------------------------
FROZEN = {
    "protocol": "frozen backbone, letter logits over the rendered options, 3 exemplars, same 231 public ids",
    "source": "gpu-runpod-full-experiment:REPORT.md#§3ag-frozen-controls",
    "rows": [
        {"backbone": "Qwen3-1.7B-Base", "std": 0.528, "easy": 1.00, "hard": 0.369, "brier_hard": 0.71, "trained": "typical-small"},
        {"backbone": "Qwen3-4B-Base", "std": 0.778, "easy": 1.00, "hard": 0.441, "brier_hard": 0.67, "trained": "typical-medium"},
        {"backbone": "Qwen3-4B (instruct)", "std": 0.778, "easy": 1.00, "hard": 0.423, "brier_hard": 0.95, "trained": None},
        {"backbone": "Qwen3-8B-Base", "std": 0.556, "easy": 0.958, "hard": 0.369, "brier_hard": 0.74, "trained": None},
        {"backbone": "Qwen3-14B-Base", "std": 0.819, "easy": 1.00, "hard": 0.559, "brier_hard": 0.60, "trained": "typical-14b-ladder"},
        {"backbone": "Qwen3.5-4B-Base", "std": 0.764, "easy": 1.00, "hard": 0.495, "brier_hard": 0.60, "trained": None},
    ],
    "in_flight": "a 14B release candidate, not yet a release (Qwen3-14B, facts-first long rows, 3,072-token states, frozen-14B distillation) and tm2 (Qwen3.5-4B); pass rule: hard >= .559 or hard Brier <= .65, long_policy >= .35 (REPORT §3ag)",
    "long_state_bug": "data_wf_long rendered the case facts last and training right-truncated at max_state, so 98.8% of long-policy rows lost their facts at 1,024 tokens; fixed by facts-first regeneration + --drop_truncated (REPORT §3ag, commit 2fad326)",
}
(OUT / "frozen.json").write_text(json.dumps(FROZEN, indent=2))

# ---------------------------------------------------------------------------
print(f"{'file':<20}{'top-level keys'}")
for f in sorted(OUT.glob("*.json")):
    d = json.loads(f.read_text())
    keys = list(d.keys()) if isinstance(d, dict) else f"[{len(d)} items]"
    print(f"{f.name:<20}{keys}")
