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
# runs-index.json -- P4 (run explorer). One row per runs/<d>/results.json (skip
# smoke*/bench_*/jev_*, which don't carry the 75-set eval or aren't results.json at all).
# eval numbers come from results.json only (not the eval_wf_full/eval_wf re-runs -- those
# use a different max_state and already feed research-heldout.json; mixing them into this
# index would silently change the comparison protocol mid-run, so they're left out here and
# `group`/`released_as`/floors are the only cross-run context added).
# ---------------------------------------------------------------------------
_SKIP_INDEX_DIR = re.compile(r"^(smoke|bench_|jev_)")

_SET_FAMILIES = [
    ("evidence", ["snli_test", "snli_null", "snli_test_hyponly", "snli_test_paraphrase",
                  "snli_test_qpara", "snli_test_soft", "cse_snli", "mnli_val", "anli_test", "boolq_val"]),
    ("topic_intent", ["clinc_test", "clinc_heldout", "clinc_oos", "clinc_k", "ksweep_clinc",
                      "ksweep_clinc_nosib", "cse_clinc", "null_nearmiss_clinc",
                      "banking77_test", "banking77_k", "cse_banking77", "ksweep_banking77",
                      "null_nearmiss_banking77", "hwu64_test", "cse_hwu64", "null_nearmiss_hwu64",
                      "ng20_test", "trec_coarse", "trec_fine"]),
    ("knowledge", ["mmlu_pro", "mmlu_choicesonly", "mmlu_shuffledq", "mmlu_cf", "mmlu_cf_teacher",
                   "truthfulqa_mc1"]),
    ("workflow_heldout", ["wf_heldout_choice", "wf_heldout_noul", "wf_heldout_score", "wf_heldout_style",
                          "wf_rubric_flip", "wf_rubric_shuffled", "wh_heldout_family", "wh_heldout_grammar",
                          "wh_heldout_style", "wh_level7", "wh_rubric_flip", "wh_rubric_shuffled"]),
    ("uncertainty", ["u_chaosnli", "u_real_heldout", "u_synthetic_heldout", "unli_test", "chaos_mnli"]),
    ("external", ["pagerduty_trigger", "jevlogs_triage", "mind2web_choice", "tree_choice_cap",
                  "typed_decisions_test", "typed_decisions_train"]),
]
_FAMILY_SOURCE = "site/research.md #eval-suite (family order) + Architecture §b (CLINC-150/Banking77 K)"

# 1/K chance floors for fixed-label-set benchmarks that carry no majority-class baseline of
# their own -- K per research.md (CLINC-150, Banking77 in Architecture §b) or the corpus's
# published class count (SNLI/MNLI/ANLI 3-way, BoolQ 2-way, TREC coarse/fine 6/50-way, MMLU-Pro
# 10-way). Not introspected from any run; a fixed table, same convention as chance.json.
_FIXED_K = {
    **{k: 150 for k in ("clinc_test", "clinc_heldout", "clinc_oos", "clinc_k", "ksweep_clinc",
                         "ksweep_clinc_nosib", "cse_clinc", "null_nearmiss_clinc")},
    **{k: 77 for k in ("banking77_test", "banking77_k", "cse_banking77", "ksweep_banking77",
                        "null_nearmiss_banking77")},
    **{k: 64 for k in ("hwu64_test", "cse_hwu64", "null_nearmiss_hwu64")},
    "ng20_test": 20, "trec_coarse": 6, "trec_fine": 50,
    **{k: 3 for k in ("snli_test", "snli_null", "snli_test_hyponly", "snli_test_paraphrase",
                       "snli_test_qpara", "snli_test_soft", "cse_snli", "mnli_val", "anli_test", "chaos_mnli")},
    "boolq_val": 2,
    **{k: 10 for k in ("mmlu_pro", "mmlu_choicesonly", "mmlu_shuffledq", "mmlu_cf", "mmlu_cf_teacher")},
}
_FIXED_K_SOURCE = "fixed label-set size (not from a run) -- CLINC-150/Banking77 per research.md Architecture §b, others per the corpus's published class count"


def _run_group(name):
    if name.startswith("probe_"):
        return "probe"
    if name.startswith("zs_"):
        return "zero_shot"
    if name.startswith("abl_"):
        return "ablation"
    if "mcq" in name:
        return "baseline"
    return "lineage"


_RELEASED_AS = {"ts1b": "typical-small", "tm1b": "typical-medium"}  # only the two directly
# eval_wf-sourced release runs; no other results.json directory names its release checkpoint.


def _run_date(rel_path):
    out = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%ad", "--date=short",
         "gpu-runpod-full-experiment", "--", rel_path],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    return out[-1] if out else None  # last line = earliest (oldest) add, git log is newest-first


def build_runs_index():
    run_dirs = sorted(p.parent.name for p in (ROOT / "runs").glob("*/results.json"))
    seen_sets = set()
    runs = []
    for name in run_dirs:
        if _SKIP_INDEX_DIR.match(name):
            continue
        d = load(f"runs/{name}/results.json")
        eval_out = {}
        for set_key, v in d.get("eval", {}).items():
            raw = v.get("raw", {})
            if not raw:
                continue
            seen_sets.add(set_key)
            row = {"acc": raw.get("acc"), "nll": raw.get("nll"), "ece": raw.get("ece"),
                   "null_recall": raw.get("null_recall"), "n": v.get("n")}
            eval_out[set_key] = {k: (round(v2, 4) if isinstance(v2, float) else v2) for k, v2 in row.items()}
        runs.append({
            "name": name,
            "date": _run_date(f"runs/{name}/results.json"),
            "group": _run_group(name),
            "released_as": _RELEASED_AS.get(name),
            "eval": eval_out,
        })

    family_of = {s: fam for fam, sets in _SET_FAMILIES for s in sets}
    ordered = [s for _, sets in _SET_FAMILIES for s in sets if s in seen_sets]
    ordered += sorted(seen_sets - set(ordered))
    sets = [{"key": s, "family": family_of.get(s, "other")} for s in ordered]

    floors = {}
    for src_name, src_file in (("ts1b", "runs/ts1b/eval_wf_full.json"), ("tm1b", "runs/tm1b/eval_wf.json")):
        ev = load(src_file)["eval"]
        for set_key, v in ev.items():
            b = v.get("baselines", {})
            if "majority_acc" in b and set_key not in floors:
                floors[set_key] = {"value": round(b["majority_acc"], 4),
                                    "source": f"{src_file} (.eval.{set_key}.baselines.majority_acc)"}
    for set_key, k in _FIXED_K.items():
        if set_key in seen_sets and set_key not in floors:
            floors[set_key] = {"value": round(1 / k, 4), "source": _FIXED_K_SOURCE}

    index = {
        "sets": sets,
        "floors": floors,
        "runs": runs,
        "source": "runs/<name>/results.json (.eval.<set>.raw); dates: git log --diff-filter=A "
                   "gpu-runpod-full-experiment -- runs/<name>/results.json; group by directory-name "
                   "prefix (probe_/zs_/abl_/*mcq*); families: " + _FAMILY_SOURCE,
    }
    payload = json.dumps(index, indent=2)
    size = len(payload.encode())
    assert size <= 1_000_000, f"runs-index.json is {size} bytes, over the 1 MB budget"
    (OUT / "runs-index.json").write_text(payload)
    print(f"runs-index.json: {len(runs)} runs, {len(sets)} sets, {len(floors)} floors, {size} bytes")


build_runs_index()

# ---------------------------------------------------------------------------
# trace.json -- P3 (architecture trace). Tokenizes presets.playground with the real Qwen3-1.7B
# tokenizer (no weights needed) and renders each query exactly as native_kv_decide would, then
# copies the replayed probabilities from replays.json under record_replays.py's hash. Run this
# script with `uv run --with transformers --with torch python scripts/precompute_research.py`
# (or plain `uv run python ...` if the project env already has them) -- import is local to this
# block so the rest of the script still runs without transformers/torch installed.
# ---------------------------------------------------------------------------
def build_trace():
    import sys
    sys.path.insert(0, str(ROOT / "inference"))
    from transformers import AutoTokenizer
    from typical.native import RENDERS, _is_bern_row, _ids, _yes_idx

    presets = load("site/data/presets.json")
    replays = load("site/data/replays.json")
    pg = presets["playground"]
    state, queries = pg["state"], pg["queries"]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    state_ids = _ids(tok, [state], 4096)[0]
    Ls = 1 + len(state_ids)  # +1 eos sink

    def render_query(q):
        cand_texts = q["labels"]
        is_bern = q["type"] == "noul" and _is_bern_row(cand_texts)
        render_fn = RENDERS["query_only" if is_bern else "letters_nonull"]
        text, spans = render_fn(q["question"], cand_texts)
        x_ids = tok(text, add_special_tokens=False)["input_ids"]
        return {
            "type": q["type"], "question": q["question"], "labels": cand_texts,
            "render": "query_only" if is_bern else "letters_nonull",
            "is_bern": is_bern, "suffix_text": text, "suffix_tokens": len(x_ids),
            "T": len(x_ids) + 1,  # + trailing eos tail (native.py's _pack `tail=(eos,)`)
            "spans": spans, "K": len(cand_texts),
        }

    rendered = [render_query(q) for q in queries]

    hash_key = _replay_hash(state, queries)
    replay = replays.get(hash_key)
    assert replay is not None, f"replays.json has no entry for playground hash {hash_key}"
    for r, res in zip(rendered, replay["results"]):
        r["probs"] = {k: round(p, 4) for k, p in res["probs"].items()}
        r["p_null"] = round(res["p_null"], 4)
        r["argmax"] = res["argmax"]
        if "expected" in res:
            r["expected"] = round(res["expected"], 4)

    # Noul label-swap: render the escalate question's candidates in both orders. query_only
    # ignores candidate text entirely, so both renders are byte-identical -- that's the proof,
    # not an assertion about the head, of exact order-invariance (native.py::_render_query_only).
    noul_q = next(q for q in queries if q["type"] == "noul")
    order_a = noul_q["labels"]
    order_b = list(reversed(order_a))
    text_a, _ = RENDERS["query_only"](noul_q["question"], order_a)
    text_b, _ = RENDERS["query_only"](noul_q["question"], order_b)
    yes_idx_a = _yes_idx([order_a], "cpu").item()
    yes_idx_b = _yes_idx([order_b], "cpu").item()
    noul_result = next(r for r, q in zip(replay["results"], queries) if q["type"] == "noul")
    noul_swap = {
        "order_a": order_a, "order_b": order_b,
        "suffix_text_a": text_a, "suffix_text_b": text_b,
        "suffix_identical": text_a == text_b,
        "p_yes": round(noul_result["probs"][order_a[yes_idx_a]], 4),
        "note": "query_only renders the question alone (no candidate text), so both label orders "
                "produce the identical suffix and therefore the identical P(yes) -- see "
                "inference/typical/native.py::_render_query_only and ::_yes_idx",
    }

    h100 = load("site/data/latency.json")["native_sweep"]["256"]["4"]["native"][0]
    trace = {
        "state": {"text": state, "n_tokens": len(state_ids)},
        "Ls": Ls,
        "trunk": {
            "backbone": "Qwen3-1.7B-Base", "tap_layer": "20 of 28",
            "lora": "LoRA r16 on the top 8 kept layers (13-20)",
            "h100_state_ms": round(h100["t_state_ms"], 1),
            "h100_source": "site/data/latency.json .native_sweep.256.4.native[0] (m=1, one-off state encode)",
        },
        "questions": rendered,
        "noul_swap": noul_swap,
        "head": {
            "choice_formula": "s_k = u·v_k ; r = σ(gate(top2, margin, mean, lse, h, mean_c, var_c)) ; "
                               "P(k) = (1-r)p_k, P(∅) = r  (inference/typical/native.py::factored_null_logits)",
            "noul_formula": "P(yes) = σ(w_n · h_D)  (inference/typical/native.py::NativeHead._bern_probs)",
        },
        "replay": {"hash": hash_key, "ms": round(replay["ms"], 1), "device": replay["device"], "model": replay["model"]},
        "source": "site/data/presets.json#playground + site/data/replays.json['" + hash_key + "'] "
                   "+ inference/typical/native.py (RENDERS, factored_null_logits, _bern_probs) "
                   "+ Qwen/Qwen3-1.7B-Base tokenizer + site/data/latency.json.native_sweep.256.4",
    }
    payload = json.dumps(trace, indent=2)
    (OUT / "trace.json").write_text(payload)
    print(f"trace.json: {len(rendered)} questions, {len(payload.encode())} bytes")


def _replay_hash(state, queries):
    """Port of scripts/record_replays.py::hash_key (djb2-ish, 32-bit wraparound, base36) --
    duplicated rather than imported since record_replays.py isn't a package module."""
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    s = state + json.dumps(queries, separators=(",", ":"), ensure_ascii=False)
    h = 5381
    for ch in s:
        h = (((h << 5) & 0xFFFFFFFF) + h + ord(ch)) & 0xFFFFFFFF
    if h == 0:
        return "0"
    out = []
    n = h
    while n:
        n, r = divmod(n, 36)
        out.append(digits[r])
    return "".join(reversed(out))


build_trace()

# ---------------------------------------------------------------------------
print(f"{'file':<28}{'top-level keys'}")
for f in sorted(OUT.glob("research-*.json")):
    d = json.loads(f.read_text())
    print(f"{f.name:<28}{list(d.keys())}")
