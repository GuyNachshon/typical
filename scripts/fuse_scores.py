"""PLAN5 sec 4: "try to unify the two experts" -- eval-time score-level fusion of the energy
and native experts, over the shared candidates aligned by compose_support.py.

    log   (default, geometric mixture in log-prob space):
        combined_logit_j = logit_energy_j + g * logit_native_j     (per-row normalising
        constants of each softmax cancel out of softmax(combined_logit), so this is exactly
        P(a_j|answerable) ~ softmax_j(log P_energy(a_j|answerable) + g*log P_native(a_j|answerable)))
    linear  (--fusion_mode linear):
        P(a_j|answerable) = (1-g) * P_energy(a_j|answerable) + g * P_native(a_j|answerable)

    P(null) = 1 - r,  r = 1 - P_energy(null | x, q, A)   -- same support gate as
    compose_support.py's --r_from energy: the null always comes from the energy path.
    P(a_j) = r * P(a_j | answerable)

g=0 reduces to the energy model exactly (both modes); large g is dominated by the native
candidate ranking (log mode's argmax converges to native's argmax).

No training: reuses compose_support.py's dump loaders, candidate alignment (by string), and
per-set eval (metrics.summarize + cse/ksweep/cf branches). Sweeps --g_grid, writes
runs/<name>_g<g>/results.json per g (train.py results.json format, so report.py /
report_native.py read them directly), and prints one table: rows = g, plus an "oracle" row
(best g per column -- the upper bound a learned per-input gate could reach) and a
"val-selected" row (single g chosen by min NLL on the dumps' val split, if present).

uv run scripts/fuse_scores.py DUMP_ENERGY DUMP_NATIVE --name compose_n3_fuse
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compose_support import align, _cand_softmax, eval_set  # noqa: E402
from null_bias import apply_bias, load_npz  # noqa: E402

G_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
VAL_NAMES = ("val", "data_v5_val")

# column, metric, direction ("max"/"min" -- which way is better, for the oracle row)
COLUMNS = [
    ("mmlu_pro", "acc_k", "max", "MMLU-Pro amK"),
    ("snli_test", "acc", "max", "SNLI acc"),
    ("mnli_val", "acc", "max", "MNLI acc"),
    ("anli_test", "acc", "max", "ANLI acc"),
    ("boolq_val", "acc", "max", "BoolQ acc"),
    ("clinc_test", "acc", "max", "CLINC-150"),
    ("clinc_heldout", "acc", "max", "CLINC-heldout"),
    ("banking77_test", "acc", "max", "Banking77-77"),
    ("clinc_oos", "null_recall", "max", "CLINC-OOS null_recall"),
    ("snli_test", "ece", "min", "SNLI ECE"),
    ("chaos_mnli", "nll", "min", "ChaosNLI NLL"),
]


def fuse(logits_e, K, logits_n_aligned, g, fusion_mode="log"):
    """-> probs [N, Kmax+1], null last (see module docstring for the two modes)."""
    r = 1.0 - apply_bias(logits_e, K, 0.0, 0.0, 1.0)[:, -1]
    if fusion_mode == "log":
        Kmax = logits_e.shape[1] - 1
        valid = np.arange(Kmax)[None, :] < K[:, None]
        logit_n_safe = np.where(valid, logits_n_aligned[:, :-1], 0.0)  # avoid 0 * -inf = nan at g=0
        z = np.where(valid, logits_e[:, :-1] + g * logit_n_safe, -np.inf)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p_answer = e / e.sum(axis=-1, keepdims=True)
    elif fusion_mode == "linear":
        p_answer = (1.0 - g) * _cand_softmax(logits_e, K) + g * _cand_softmax(logits_n_aligned, K)
    else:
        raise ValueError(f"unknown fusion_mode: {fusion_mode}")
    return np.concatenate([r[:, None] * p_answer, (1.0 - r)[:, None]], axis=1)


def _cell(entry, metric):
    if entry is None:
        return None
    v = entry["scaled"].get(metric)
    return v if isinstance(v, (int, float)) else None


def _fmt(v):
    return "-" if v is None else f"{v:.3f}"


def load_dumps(dump_e, dump_n):
    """-> {set: (logits_e, target_e, label_e, K, aligned_logits_n, examples_e)} for sets common
    to both dumps (skipped sets are printed as a note, mirroring compose_support.run)."""
    dump_e, dump_n = Path(dump_e), Path(dump_n)
    sets_e, sets_n = {p.stem for p in dump_e.glob("*.npz")}, {p.stem for p in dump_n.glob("*.npz")}
    for s in sorted(sets_e ^ sets_n):
        print(f"note: {s} only in {'energy' if s in sets_e else 'native'} dump -- skipped")
    loaded = {}
    for name in sorted(sets_e & sets_n):
        le, te, ye, Ke = load_npz(dump_e / f"{name}.npz")
        ln, tn, yn, Kn = load_npz(dump_n / f"{name}.npz")
        me, mn = json.load(open(dump_e / f"{name}.meta.json")), json.load(open(dump_n / f"{name}.meta.json"))
        try:
            assert (Ke == Kn).all(), f"{name}: K differs between dumps"
            aligned = align(me, mn, ln)
        except AssertionError as err:  # e.g. each dump's own "val" over different data
            print(f"note: {name} not over the same examples in both dumps -- skipped ({err})")
            continue
        loaded[name] = (le, te, ye, Ke, aligned, me)
    return loaded


def run(dump_e, dump_n, g_grid=G_GRID, fusion_mode="log", quiet=False):
    """-> {g: results_dict} (train.py results.json format, one per g)."""
    loaded = load_dumps(dump_e, dump_n)
    by_g = {}
    for g in g_grid:
        results = {"fusion_mode": fusion_mode, "g": g, "eval": {}}
        for name, (le, te, ye, Ke, aligned, me) in loaded.items():
            p = fuse(le, Ke, aligned, g, fusion_mode)
            results["eval"][name] = eval_set(name, p, te, ye, me)
        by_g[g] = results

    val_name = next((n for n in VAL_NAMES if n in loaded), None)
    val_g = None
    if val_name is not None:
        val_g = min(g_grid, key=lambda g: by_g[g]["eval"][val_name]["scaled"].get("nll", float("inf")))

    if not quiet:
        print_table(by_g, g_grid, val_name, val_g)
    return by_g, val_g


def print_table(by_g, g_grid, val_name, val_g):
    cols = [c for c in COLUMNS if any(c[0] in by_g[g]["eval"] for g in g_grid)]
    header = f"{'g':<14}" + "".join(f"{label:>15}" for _, _, _, label in cols)
    print(f"\nfusion sweep (mode={next(iter(by_g.values()))['fusion_mode']})")
    print(header)
    for g in g_grid:
        row = [_cell(by_g[g]["eval"].get(s), m) for s, m, _, _ in cols]
        print(f"{g:<14g}" + "".join(f"{_fmt(v):>15}" for v in row))

    oracle = []
    for s, m, direction, _ in cols:
        vals = [(g, _cell(by_g[g]["eval"].get(s), m)) for g in g_grid]
        vals = [(g, v) for g, v in vals if v is not None]
        best = (max if direction == "max" else min)(vals, key=lambda gv: gv[1]) if vals else (None, None)
        oracle.append(best[1])
    print(f"{'oracle (per-col g)':<14}" + "".join(f"{_fmt(v):>15}" for v in oracle))

    if val_name is None:
        print(f"note: no val split ({'/'.join(VAL_NAMES)}) in the dumps -- no val-selected row")
    else:
        row = [_cell(by_g[val_g]["eval"].get(s), m) for s, m, _, _ in cols]
        print(f"{'val-selected (g=' + str(val_g) + ')':<14}" + "".join(f"{_fmt(v):>15}" for v in row))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_energy")
    ap.add_argument("dump_native")
    ap.add_argument("--name", default="fuse", help="output dir prefix: runs/<name>_g<g>/results.json")
    ap.add_argument("--fusion_mode", choices=["log", "linear"], default="log")
    ap.add_argument("--g_grid", default=",".join(str(g) for g in G_GRID),
                     help="comma-separated g values")
    a = ap.parse_args()
    g_grid = [float(g) for g in a.g_grid.split(",")]
    by_g, val_g = run(a.dump_energy, a.dump_native, g_grid, a.fusion_mode)
    for g, results in by_g.items():
        out = Path("runs") / f"{a.name}_g{g:g}" / "results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {out}")
    if val_g is not None:
        print(f"\nval-selected g = {val_g:g}")


if __name__ == "__main__":
    main()
