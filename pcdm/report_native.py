"""PLAN4 §15 key table: evidence retention / parametric knowledge (Δ_q) / decision quality / systems, one row per run.
Reads runs/<name>/results.json (+ runs/probe_<name>/results.json for Δ_q, runs/bench_*<name>*/bench.json for latency).
uv run pcdm/report_native.py [names...]   (default: every run with an mmlu_pro eval)"""
import glob
import json
import sys
from pathlib import Path

EVIDENCE = [("snli_test", "acc"), ("mnli_val", "acc"), ("anli_test", "acc"), ("boolq_val", "acc"), ("clinc_test", "acc"), ("hwu64_test", "acc")]
QUALITY = [("snli_test", "nll"), ("snli_test", "brier"), ("snli_test", "ece"), ("chaos_mnli", "nll"), ("clinc_k", "auroc_null"),
           ("mmlu_pro", "false_abstain"), ("clinc_oos", "null_recall")]
TEACHER_DQ = 0.088  # teacher_kb: normal .310 - choices-only .222 (REPORT §3j)
TEACHER_DQS = 0.102  # teacher_kb: normal .310 - shuffled-question .208 (the stricter, primary variant)


def load(p):
    return json.load(open(p)) if Path(p).exists() else None


def val(r, s, m):
    v = (r or {}).get("eval", {}).get(s, {}).get("scaled", {}).get(m)
    return v if isinstance(v, (int, float)) else None


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def row(name):
    r = load(f"runs/{name}/results.json")
    p = load(f"runs/probe_{name}/results.json")
    mm, co, sh = val(r, "mmlu_pro", "acc_k"), val(p, "mmlu_choicesonly", "acc_k"), val(p, "mmlu_shuffledq", "acc_k")
    dq = None if mm is None or co is None else mm - co
    dqs = None if mm is None or sh is None else mm - sh  # shuffled-question variant (REPORT §3j: stricter, primary)
    ev = [val(r, s, m) for s, m in EVIDENCE]
    q = [val(r, s, m) for s, m in QUALITY]
    cf = (r or {}).get("eval", {}).get("mmlu_cf", {}).get("cf", {})
    lat = None
    for b in glob.glob(f"runs/bench*{name}*/bench.json"):
        j = json.load(open(b)); k4 = j.get("by_k", {}).get("4") or j.get("by_k", {}).get(4) or {}
        lat = (k4.get("marginal_at_m_max", {}) or {}).get("ours_ms") or lat
    return {"run": name, "mmlu_among_k": mm, "choices_only": co, "shuffled_q": sh, "delta_q": dq,
            "dq_ratio": None if dq is None else dq / TEACHER_DQ, "delta_q_shuf": dqs, "dqs_ratio": None if dqs is None else dqs / TEACHER_DQS,
            "evidence": ev, "quality": q, "reorder_dp": cf.get("reorder/max_dp"), "iia": cf.get("add3_unrel/dlo_top2"),
            "warm_ms_k4": lat}


def main():
    names = sys.argv[1:] or sorted(Path(x).parent.name for x in glob.glob("runs/*/results.json")
                                   if "mmlu_pro" in json.load(open(x)).get("eval", {}) and not Path(x).parent.name.startswith(("probe_", "mmlu_", "ks_", "B_")))
    hdr = f"{'run':<16}{'MMLU amK':>9}{'choices':>9}{'shuf':>7}{'Δq':>8}{'Δq/T':>6}{'Δq_sh':>7}{'/T':>5} | " + "".join(f"{s[:5]:>7}" for s, _ in EVIDENCE) + " | " + "".join(f"{(s[:4]+'/'+m[:4]):>11}" for s, m in QUALITY) + f"{'reord':>7}{'IIA':>7}{'ms@K4':>7}"
    print(hdr)
    for n in names:
        x = row(n)
        print(f"{x['run']:<16}{fmt(x['mmlu_among_k']):>9}{fmt(x['choices_only']):>9}{fmt(x['shuffled_q']):>7}{fmt(x['delta_q']):>8}{fmt(x['dq_ratio'],2):>6}{fmt(x['delta_q_shuf']):>7}{fmt(x['dqs_ratio'],2):>5} | "
              + "".join(f"{fmt(v):>7}" for v in x["evidence"]) + " | " + "".join(f"{fmt(v):>11}" for v in x["quality"])
              + f"{fmt(x['reorder_dp']):>7}{fmt(x['iia']):>7}{fmt(x['warm_ms_k4'],2):>7}")
    print(f"\n(Δq = normal − choices-only, Δq_sh = normal − shuffled-question (primary, REPORT §3j); /T vs the teacher's {TEACHER_DQ} / {TEACHER_DQS}; '-' = not run)")


if __name__ == "__main__":
    main()
