"""Reads runs/**/*.json (+ hand-transcribed REPORT.md/release-card numbers) and writes
site/data/research-*.json for research.html. Run: uv run python scripts/precompute_research.py

ponytail: same pattern as precompute.py -- G6/G7 (held-out + external sets) have one
unambiguous JSON source (eval_wf*.json) so they're read programmatically; G8 (the six-day
lineage) and G5's family-weight donut live only as prose/args in REPORT.md and the release
cards, so they're hand-transcribed here with a "source" cite per point, same convention
precompute.py already uses for models.json.
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def load(rel):
    return json.loads((ROOT / rel).read_text())


_SKIP_RUN_DIR = re.compile(r"^(probe_|jev_|bench_|zs_|smoke|dump_|leak_|kb_)")


def run_count():
    """Training/baseline/ablation run directories on the source branch's runs/
    tree -- this worktree only checks out a subset, so `ls runs/` locally
    undercounts. Mirrors: git ls-tree --name-only gpu-runpod-full-experiment:runs
    | grep -v -E '^(probe_|jev_|bench_|zs_|smoke|dump_|leak_|kb_)' | grep -v '\\.json$'"""
    out = subprocess.run(
        ["git", "ls-tree", "--name-only", "gpu-runpod-full-experiment:runs"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return sum(1 for name in out if not _SKIP_RUN_DIR.match(name) and not name.endswith(".json"))


# ---------------------------------------------------------------------------
# research-heldout.json -- G6 (curriculum, trained-on-adjacent) + G7 (external,
# never trained on) bars, each with its majority/constant-prediction floor.
# ---------------------------------------------------------------------------
HELDOUT_SETS = [
    ("wf_heldout_noul", "held-out noul (W)"),
    ("wf_heldout_score", "held-out score (W)"),
    ("wf_heldout_style", "held-out style (W)"),
    ("wh_heldout_family", "held-out family (DecisionMix v2)"),
    ("wh_heldout_grammar", "held-out grammar (DecisionMix v2)"),
    ("wh_heldout_style", "held-out style (DecisionMix v2)"),
    ("wh_rubric_flip", "rubric-flip, both correct"),
    ("wh_level7", "level 7 -- untrained composition (chance)"),
]
EXTERNAL_SETS = [
    ("pagerduty_trigger", "PagerDuty (never trained)"),
    ("mind2web_choice", "Mind2Web (never trained)"),
    ("jevlogs_triage", "jevlogs (research-licensed, block-level labels)"),
    ("tree_choice_cap", "tree-choice K=320 (never trained)"),
    ("typed_decisions_test", "typed-decisions (held-out)"),
]

_ts1b = load("runs/ts1b/eval_wf_full.json")["eval"]
_tm1b = load("runs/tm1b/eval_wf.json")["eval"]


def _rows(spec, ts_src, tm_src):
    out = []
    for key, label in spec:
        s = _ts1b[key]
        m = _tm1b[key]
        assert s["baselines"]["majority_acc"] == m["baselines"]["majority_acc"], key
        out.append({
            "key": key,
            "label": label,
            "small": round(s["raw"]["acc"], 4),
            "medium": round(m["raw"]["acc"], 4),
            "floor": round(s["baselines"]["majority_acc"], 4),
            "n_small": s["n"],
            "n_medium": m["n"],
        })
    return out


HELDOUT = {
    "curriculum": _rows(HELDOUT_SETS, "runs/ts1b/eval_wf_full.json", "runs/tm1b/eval_wf.json"),
    "external": _rows(EXTERNAL_SETS, "runs/ts1b/eval_wf_full.json", "runs/tm1b/eval_wf.json"),
    "source": "runs/ts1b/eval_wf_full.json + runs/tm1b/eval_wf.json (eval.<set>.raw.acc, .baselines.majority_acc)",
    "note": "curriculum = REPORT.md G6 (in-grammar held-out, level 7 kept in as an untrained-composition control); external = G7 (sets never trained on)",
}
(OUT / "research-heldout.json").write_text(json.dumps(HELDOUT, indent=2))

# ---------------------------------------------------------------------------
# research-jevbench-family.json -- optional G9, JevBench standard-tier per family
# ---------------------------------------------------------------------------
_ts1b_jev = load("runs/jev_native_ts1b/summary.json")["original"]["per_family"]
JEV_FAMILY = {
    "typical-small": {
        fam: {"acc": round(v["accuracy"], 4), "n": v["calibration_n"]}
        for fam, v in sorted(_ts1b_jev.items())
    },
    "source": "runs/jev_native_ts1b/summary.json (original.per_family.<family>.accuracy)",
}
(OUT / "research-jevbench-family.json").write_text(json.dumps(JEV_FAMILY, indent=2))

# ---------------------------------------------------------------------------
# research-mix.json -- G5: training-batch family weights (donut) + corpus row
# counts (bars). Numbers are the release card's Training args / Data section,
# not derived from a JSON file (there is no single machine-readable source).
# ---------------------------------------------------------------------------
MIX = {
    "family_weights": [
        {"key": "E", "label": "E -- evidence (NLI/BoolQ/SQuAD-style)", "weight": 0.40},
        {"key": "K", "label": "K -- knowledge (MCQ distillation)", "weight": 0.15},
        {"key": "W", "label": "W -- workflow (rubric-conditioned)", "weight": 0.35},
        {"key": "U", "label": "U -- uncertainty (soft-target)", "weight": 0.10},
    ],
    "corpus_rows": [
        {"key": "kb", "label": "data_kb (K)", "rows": 147446},
        {"key": "wf", "label": "data_wf (W)", "rows": 72000},
        {"key": "wf_hf", "label": "data_wf_hf (W)", "rows": 86600},
        {"key": "wh", "label": "data_wh (W, DecisionMix v2)", "rows": 60605},
        {"key": "u", "label": "data_u (U)", "rows": 31029},
    ],
    "source": "releases/typical-small.md Training args (family_weights) + Data section",
}
(OUT / "research-mix.json").write_text(json.dumps(MIX, indent=2))

# ---------------------------------------------------------------------------
# research-timeline.json -- G8: the lineage's JevBench-standard trajectory plus
# the bug/verdict annotations. Every point cites the REPORT.md section it comes
# from; dates are section dates (or the v0/v1 phase date where a subsection has
# none) -- REPORT.md itself only date-stamps at the section level.
# ---------------------------------------------------------------------------
TIMELINE = {
    "lineage": [
        {"date": "2026-09-18", "label": "energy (joint_emb_lw_v5)", "std": 0.403, "source": "REPORT.md §3q table"},
        {"date": "2026-09-19", "label": "native n3, 28L", "std": 0.417, "source": "REPORT.md §3q table"},
        {"date": "2026-09-20", "label": "native v2, letter-free", "std": 0.472, "source": "REPORT.md §3q table"},
        {"date": "2026-09-20", "label": "v3_tap20, no ∅ line", "std": 0.694, "source": "REPORT.md §3t"},
        {"date": "2026-09-20", "label": "+ workflow (typical-small-preview)", "std": 0.750, "source": "REPORT.md §3w; releases/typical-small-preview.md"},
        {"date": "2026-09-21", "label": "typical-small", "std": 0.694, "source": "REPORT.md §3ae (ts1b, +DecisionMix v2 +typed heads)"},
        {"date": "2026-09-21", "label": "typical-medium", "std": 0.806, "source": "REPORT.md §3af (tm1b, 4B)"},
        {"date": "2026-09-21", "label": "14B ladder (unreleased)", "std": 0.875, "source": "REPORT.md §3ab"},
    ],
    "bugs": [
        {"date": "2026-09-16", "label": "attention-sink token (position 0 content-independent)", "source": "REPORT.md §2.1"},
        {"date": "2026-09-16", "label": "fixed classifier in disguise (3 label strings memorised)", "source": "REPORT.md §2.2"},
        {"date": "2026-09-16", "label": "state-only null leaks its base rate", "source": "REPORT.md §2.3"},
        {"date": "2026-09-17", "label": "last-layer tap = hypothesis-only model (the big one)", "source": "REPORT.md §2.4"},
        {"date": "2026-09-17", "label": "rogue dimensions dominate token norm", "source": "REPORT.md §2.5"},
        {"date": "2026-09-17", "label": "null over-fires on unfamiliar wordings (data bug)", "source": "REPORT.md §2.6"},
        {"date": "2026-09-17", "label": "LoRA dropout active at eval", "source": "REPORT.md §2.7"},
        {"date": "2026-09-17", "label": "the cross-encoder is not the \"expensive joint\" alternative", "source": "REPORT.md §2.8"},
        {"date": "2026-09-21", "label": "global (not per-row) Bernoulli routing collapsed ts1 to chance", "source": "REPORT.md §3ae"},
    ],
    "verdicts": [
        {"date": "2026-09-19", "label": "candidate-blind states carry priors, not question-dependent knowledge", "source": "REPORT.md §3j"},
        {"date": "2026-09-20", "label": "depth confound resolved: tap 20, not 28, keeps evidence", "source": "REPORT.md §3r"},
        {"date": "2026-09-20", "label": "candidate-blind negative closed at full depth too", "source": "REPORT.md §3s"},
        {"date": "2026-09-20", "label": "rendered ∅ line was the null pathology, not the options-in-suffix formulation", "source": "REPORT.md §3t"},
        {"date": "2026-09-20", "label": "rubric-conditioned workflow data teaches the decision shapes it contains", "source": "REPORT.md §3w"},
        {"date": "2026-09-21", "label": "ordinal-smoothed Score + Bernoulli Noul adopted", "source": "REPORT.md §3ac"},
        {"date": "2026-09-21", "label": "DecisionMix v2: curriculum transfers within its grammar, not beyond it", "source": "REPORT.md §3ad"},
        {"date": "2026-09-21", "label": "typical-small / typical-medium frozen", "source": "REPORT.md §3ae, §3af"},
    ],
    "run_count": run_count(),
    "run_count_source": "git ls-tree --name-only gpu-runpod-full-experiment:runs | grep -v -E '^(probe_|jev_|bench_|zs_|smoke|dump_|leak_|kb_)' | grep -v '\\.json$' | wc -l -- this worktree only checks out a subset of runs/, so the count is read from the source branch's tree, not the local directory",
}
(OUT / "research-timeline.json").write_text(json.dumps(TIMELINE, indent=2))

# ---------------------------------------------------------------------------
print(f"{'file':<28}{'top-level keys'}")
for f in sorted(OUT.glob("research-*.json")):
    d = json.loads(f.read_text())
    print(f"{f.name:<28}{list(d.keys())}")
