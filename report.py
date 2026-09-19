"""Compare runs/*/results.json side by side. `uv run report.py [--raw]`."""
import glob
import json
import sys
from pathlib import Path

# (eval set, metric) pairs that map onto the hypotheses in PLAN.md
COLS = [
    ("snli_test", "acc"), ("mnli_val", "acc"), ("clinc_test", "acc"),                 # H1
    ("snli_test_paraphrase", "acc"), ("clinc_heldout", "acc"), ("clinc_heldout", "acc_k"),  # dynamic candidates
    ("snli_test", "ece"), ("chaos_mnli", "nll"), ("snli_test_soft", "nll"), ("unli_test", "nll"),  # H3
    ("clinc_k", "auroc_null"), ("clinc_k", "auroc_conf"), ("snli_null", "auroc_null"), ("clinc_oos", "acc"),  # H4
    ("anli_test", "acc"), ("banking77_test", "acc"), ("trec_fine", "acc"), ("ng20_test", "acc"),  # held-out
    ("squad_null", "auroc_null"), ("boolq_val", "nll"),
    ("ksweep_clinc", "p_null_absent_range"), ("cse_banking77", "dup/mass_err"),       # data v4
    ("null_nearmiss_hwu64", "auroc_null"),
    ("mmlu_pro", "false_abstain"), ("clinc_oos", "null_recall"),                         # E3 abstention, disaggregated
]


def get_metric(run, s, m, key):
    """Normal metrics live at eval[s][key][m]. data v4's cse_*/ksweep_* sets are expected to be
    stored by train.py as a separate nested dict (eval[s]["cse"] / eval[s]["ksweep"], per-variant
    keys like "dup/mass_err"), possibly itself split by scaled/raw or flat -- try both shapes and
    return None (rendered as "-") if train.py hasn't written them yet."""
    ev = run.get("eval", {}).get(s, {})
    v = ev.get(key, {}).get(m)
    if v is not None:
        return v
    for nest in ("cse", "ksweep"):
        d = ev.get(nest)
        if isinstance(d, dict):
            v = d.get(key, {}).get(m) if isinstance(d.get(key), dict) else d.get(m)
            if v is not None:
                return v
    return None


def main():
    key = "raw" if "--raw" in sys.argv else "scaled"
    runs = {Path(p).parent.name: json.load(open(p)) for p in sorted(glob.glob("runs/*/results.json"))}
    runs = {n: r for n, r in runs.items() if "eval" in r and not n.startswith(("smoke", "bench", "diagloc", "intloc"))}
    head = f"{'run':<10}" + "".join(f"{s[:11] + '/' + m[:5]:>18}" for s, m in COLS)
    print(head)
    for name, r in runs.items():
        row = f"{name:<10}"
        for s, m in COLS:
            v = get_metric(r, s, m, key)
            if v is None:
                row += f"{'-':>18}"
            elif v == "n/a":
                row += f"{'n/a':>18}"
            else:
                row += f"{v:>18.3f}"
        print(row)
    print(f"\n({key} metrics; T fitted on val: " + ", ".join(f"{n}={r['T']:.2f}" for n, r in runs.items()) + ")")


if __name__ == "__main__":
    main()
