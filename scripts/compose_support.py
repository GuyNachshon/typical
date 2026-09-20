"""PLAN5 sec 3: support gate (energy) x conditional choice (native), composed at eval time.

    r      = 1 - P_energy(null | x, q, A)                       (answerability / support)
    P_N(j) = softmax_j(native candidate logits)                (choice given answerable; null excluded)
    P(null) = 1 - r,   P(a_j) = r * P_N(j)

No training: both inputs are `train.py --eval_only --dump_logits DIR` dumps over the same eval
sets (DIR/<set>.npz logits/target/label/K + <set>.meta.json). The two runs may have rendered
the candidates in different orders (native shuffles), so every row is aligned by candidate
string onto the energy dump's order after asserting the same query + candidate multiset.

--r_from energy    r from the energy dump at T=1 (exact for a factored null; raw for softmax)
--r_from energy_T  softmax-null energy dump tempered by --energy_T (its results.json "T")
--r_from native    r from the native dump's own null (the ablation: no external gate)

Writes a train.py-format results.json (eval[set][raw|scaled] via metrics.summarize; cse/ksweep/cf
branches like train.run_full_eval) so report.py / report_native.py read it directly, and prints
a before/after table (energy alone, native alone, composed).

uv run scripts/compose_support.py DUMP_ENERGY DUMP_NATIVE --out runs/compose_n3/results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import summarize, choice_set_effects, ksweep, counterfactual  # noqa: E402
from null_bias import apply_bias, load_npz  # noqa: E402

TABLE = [  # (set, metric) -- "abst_err" = acc_k - acc; "ksweep" expands to K*/p_null_absent rows
    ("mmlu_pro", "acc"), ("mmlu_pro", "acc_k"), ("mmlu_pro", "abst_err"), ("mmlu_pro", "ece"),
    ("clinc_oos", "null_recall"),
    ("clinc_k", "auroc_null"), ("snli_null", "auroc_null"), ("squad_null", "auroc_null"),
    ("ksweep_clinc", "ksweep"),
    ("banking77_test", "acc"), ("clinc_heldout", "acc"),
    ("snli_test", "acc"), ("mnli_val", "acc"), ("boolq_val", "acc"),
]


def align(meta_e, meta_n, logits_n):
    """Permute native candidate logits into the energy dump's candidate order, row by row, by
    candidate string (duplicates matched in order). -> [N, Kmax_e + 1], null last, pads -inf."""
    N, Kmax = len(meta_e), max(len(m["candidates"]) for m in meta_e)
    out = np.full((N, Kmax + 1), -np.inf)
    for i, (e, n) in enumerate(zip(meta_e, meta_n)):
        assert e["query"] == n["query"] and sorted(e["candidates"]) == sorted(n["candidates"]), \
            f"row {i}: dumps are not over the same examples"
        pos = {}
        for j, c in enumerate(n["candidates"]):
            pos.setdefault(c, []).append(j)
        out[i, :len(e["candidates"])] = [logits_n[i, pos[c].pop(0)] for c in e["candidates"]]
        out[i, -1] = logits_n[i, -1]
    return out


def _cand_softmax(logits, K):
    """softmax over the K valid candidate columns only (null excluded) -> [N, Kmax], pads 0."""
    valid = np.arange(logits.shape[1] - 1)[None, :] < K[:, None]
    z = np.where(valid, logits[:, :-1], -np.inf)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def compose(logits_e, K_e, logits_n_aligned, r_from="energy", energy_T=1.0):
    """-> (probs_composed, probs_energy, probs_native_aligned), all [N, Kmax+1], null last."""
    p_e = apply_bias(logits_e, K_e, 0.0, 0.0, energy_T if r_from == "energy_T" else 1.0)
    p_n = apply_bias(logits_n_aligned, K_e, 0.0, 0.0, 1.0)
    r = 1.0 - (p_n if r_from == "native" else p_e)[:, -1]
    cand = _cand_softmax(logits_n_aligned, K_e)  # P_N(j | answerable): the native null cancels
    p = np.concatenate([r[:, None] * cand, (1.0 - r)[:, None]], axis=1)
    return p, p_e, p_n


def eval_set(name, probs, target, label, examples):
    m = summarize(probs, target, label)
    entry = {"raw": m, "scaled": m}
    if name.startswith("cse_"):
        entry["cse"] = choice_set_effects(probs, examples)
    elif name.startswith("ksweep_"):
        entry["ksweep"] = ksweep(probs, examples)
    elif name.startswith("mmlu_cf"):
        entry["cf"] = counterfactual(probs, examples)
    return entry


def cell(entry, metric):
    if entry is None:
        return "-"
    if metric == "abst_err":
        a, k = entry["scaled"].get("acc"), entry["scaled"].get("acc_k")
        return "-" if a is None or k is None else f"{k - a:.3f}"
    v = entry["ksweep"].get(metric) if "/" in metric else entry["scaled"].get(metric)
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def run(dump_e, dump_n, r_from="energy", energy_T=1.0, quiet=False):
    dump_e, dump_n = Path(dump_e), Path(dump_n)
    sets_e, sets_n = {p.stem for p in dump_e.glob("*.npz")}, {p.stem for p in dump_n.glob("*.npz")}
    for s in sorted(sets_e ^ sets_n):
        print(f"note: {s} only in {'energy' if s in sets_e else 'native'} dump -- skipped")
    results = {"T": energy_T if r_from == "energy_T" else 1.0, "r_from": r_from,
               "args": {"energy_dump": str(dump_e), "native_dump": str(dump_n), "r_from": r_from, "energy_T": energy_T},
               "eval": {}}
    before = {"energy": {}, "native": {}}
    for name in sorted(sets_e & sets_n):
        le, te, ye, Ke = load_npz(dump_e / f"{name}.npz")
        ln, tn, yn, Kn = load_npz(dump_n / f"{name}.npz")
        me, mn = json.load(open(dump_e / f"{name}.meta.json")), json.load(open(dump_n / f"{name}.meta.json"))
        assert (Ke == Kn).all(), name
        p, p_e, p_n = compose(le, Ke, align(me, mn, ln), r_from, energy_T)
        results["eval"][name] = eval_set(name, p, te, ye, me)
        before["energy"][name] = eval_set(name, p_e, te, ye, me)
        before["native"][name] = eval_set(name, p_n, te, ye, me)  # aligned -> same target/label as energy
    if not quiet:
        print(f"\n{'set':<16}{'metric':<20}{'energy':>10}{'native':>10}{'composed':>10}   (r_from={r_from})")
        for s, m in TABLE:
            if s not in results["eval"]:
                continue
            ms = [m] if m != "ksweep" else sorted(  # one row per K: P(null | absent)@K
                (k for k in results["eval"][s]["ksweep"] if k.endswith("/p_null_absent")), key=lambda k: int(k[1:].split("/")[0]))
            for m in ms:
                print(f"{s:<16}{m:<20}{cell(before['energy'][s], m):>10}{cell(before['native'][s], m):>10}"
                      f"{cell(results['eval'][s], m):>10}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_energy")
    ap.add_argument("dump_native")
    ap.add_argument("--out", required=True, help="results.json path (train.py format)")
    ap.add_argument("--r_from", choices=["energy", "energy_T", "native"], default="energy")
    ap.add_argument("--energy_T", type=float, default=1.0, help="with --r_from energy_T: the energy run's fitted T")
    a = ap.parse_args()
    results = run(a.dump_energy, a.dump_native, a.r_from, a.energy_T)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
