"""Post-hoc, val-fitted K-aware null bias: s_null' = s_null + alpha*log(K) + beta, fit jointly
with a shared temperature T by minimising val soft-CE, then applied (no retraining) to every
eval set dumped by `train.py --eval_only --dump_logits DIR`.

See REPORT.md sec 3c: P(null | gold absent) falls 0.98 -> 0.35 from K=2 to K=150 even with
nulls present at every K in training -- softmax dilutes the null logit as K grows, structurally,
not as a data-prior artifact. This is the cheapest fix that doesn't touch the checkpoint.

uv run scripts/null_bias.py DIR [--out runs/<name>_nullbias/results.json]
DIR must hold val.npz/val.meta.json + <set>.npz/<set>.meta.json for every eval set (as written
by train.py's --dump_logits). Also fits a beta-only (alpha=0) variant for comparison.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import summarize, choice_set_effects, ksweep

ALPHAS = np.arange(-3.0, 3.0 + 1e-9, 0.1)
BETAS = np.arange(-6.0, 6.0 + 1e-9, 0.25)
TS = np.geomspace(0.1, 10, 60)  # same grid as train.py's fit_temperature


def _valid_mask(K, Kmax):
    idx = np.arange(Kmax)
    cand_valid = idx[None, :] < K[:, None]
    null_valid = np.ones((K.shape[0], 1), dtype=bool)
    return np.concatenate([cand_valid, null_valid], axis=1)


def apply_bias(logits, K, alpha, beta, T):
    """probs = softmax((logits + bias)/T); bias adds alpha*log(K)+beta to the null column only.
    Validity (which candidate slots are real vs. padding) is reconstructed from K, not from the
    magnitude of the padded logits -- so it doesn't matter what pad value train.py wrote."""
    Kmax = logits.shape[1] - 1
    valid = _valid_mask(K, Kmax)
    biased = logits.copy()
    biased[:, -1] = biased[:, -1] + alpha * np.log(K) + beta
    z = np.where(valid, biased, -np.inf) / T
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def soft_ce(probs, target):
    return float(-(target * np.log(np.clip(probs, 1e-12, 1.0))).sum(-1).mean())


def _fit(logits, target, K, alphas):
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    nll_raw = soft_ce(apply_bias(logits, K, 0.0, 0.0, 1.0), target)
    best = {"alpha": 0.0, "beta": 0.0, "T": 1.0, "nll_raw": nll_raw, "nll_fit": float("inf")}
    for T in TS:
        for a in alphas:
            for b in BETAS:
                nll = soft_ce(apply_bias(logits, K, a, b, T), target)
                if nll < best["nll_fit"]:
                    best.update(alpha=float(a), beta=float(b), T=float(T), nll_fit=nll)
    return best


def fit_null_bias(logits, target, K):
    """Grid-search (alpha, beta, T) minimising val soft-CE: alpha in [-3,3] step 0.1,
    beta in [-6,6] step 0.25, T over the same geomspace as train.py's fit_temperature."""
    return _fit(logits, target, K, ALPHAS)


def fit_beta_only(logits, target, K):
    """Same fit with alpha pinned to 0 (bias = beta only), for comparison."""
    return _fit(logits, target, K, np.array([0.0]))


def load_npz(path):
    d = np.load(path)
    return d["logits"], d["target"], d["label"], d["K"]


def eval_set(name, logits, target, label, K, examples, alpha, beta, T):
    raw_probs = apply_bias(logits, K, 0.0, 0.0, 1.0)
    scaled_probs = apply_bias(logits, K, alpha, beta, T)
    entry = {"raw": summarize(raw_probs, target, label), "scaled": summarize(scaled_probs, target, label)}
    if name.startswith("cse_"):
        entry["cse_raw"] = choice_set_effects(raw_probs, examples)
        entry["cse"] = choice_set_effects(scaled_probs, examples)
    elif name.startswith("ksweep_"):
        entry["ksweep_raw"] = ksweep(raw_probs, examples)
        entry["ksweep"] = ksweep(scaled_probs, examples)
    return entry


def build_results(data_dir, fit):
    results = {"T": fit["T"], "alpha": fit["alpha"], "beta": fit["beta"],
               "val_nll_raw": fit["nll_raw"], "val_nll": fit["nll_fit"], "eval": {}}
    for npz_path in sorted(data_dir.glob("*.npz")):
        name = npz_path.stem
        if name == "val":
            continue
        logits, target, label, K = load_npz(npz_path)
        with open(data_dir / f"{name}.meta.json") as f:
            examples = json.load(f)
        results["eval"][name] = eval_set(name, logits, target, label, K, examples,
                                          fit["alpha"], fit["beta"], fit["T"])
    return results


# (set, metric, kind): kind picks the summarize dict ("summary") or the flat ksweep/cse dict
# ("ksweep"/"cse") we stash under `<kind>` (scaled) / `<kind>_raw` (raw, alpha=beta=0, T=1).
TABLE_COLS = [
    ("clinc_heldout", "acc", "summary"), ("clinc_heldout", "acc_k", "summary"),
    ("banking77_test", "acc", "summary"),
    ("banking77_k", "acc", "summary"), ("banking77_k", "acc_k", "summary"),
    ("trec_coarse", "acc", "summary"),
    ("ng20_test", "acc", "summary"),
    ("snli_test_paraphrase", "acc", "summary"),
    ("clinc_oos", "acc", "summary"),
    ("clinc_k", "auroc_null", "summary"),
    ("snli_null", "auroc_null", "summary"),
    ("squad_null", "auroc_null", "summary"),
    ("ksweep_clinc", "p_null_absent_range", "ksweep"),
    ("snli_test", "acc", "summary"), ("snli_test", "ece", "summary"),
    ("chaos_mnli", "nll", "summary"),
]


def get_val(entry, metric, kind, which):
    if entry is None:
        return None
    key = which if kind == "summary" else (kind if which == "scaled" else f"{kind}_raw")
    return entry.get(key, {}).get(metric)


def fmt(v):
    return f"{'-':>10}" if v is None or v == "n/a" else f"{v:>10.3f}"


def print_table(results):
    beta_only = results["beta_only"]
    print(f"\n{'set/metric':<34}{'before':>10}{'after(a+b)':>12}{'after(b0)':>12}")
    for name, metric, kind in TABLE_COLS:
        entry = results["eval"].get(name)
        bo_entry = beta_only["eval"].get(name)
        before = get_val(entry, metric, kind, "raw")
        after = get_val(entry, metric, kind, "scaled")
        after_b0 = get_val(bo_entry, metric, kind, "scaled")
        print(f"{name + '/' + metric:<34}{fmt(before):>10}{fmt(after):>12}{fmt(after_b0):>12}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data_dir = Path(args.dir)

    vlogits, vtarget, vlabel, vK = load_npz(data_dir / "val.npz")
    full = fit_null_bias(vlogits, vtarget, vK)
    beta_only = fit_beta_only(vlogits, vtarget, vK)
    print(f"val soft-CE: raw={full['nll_raw']:.4f}  "
          f"alpha+beta+T={full['nll_fit']:.4f} (alpha={full['alpha']:.2f}, beta={full['beta']:.2f}, T={full['T']:.2f})  "
          f"beta+T={beta_only['nll_fit']:.4f} (beta={beta_only['beta']:.2f}, T={beta_only['T']:.2f})")

    results = build_results(data_dir, full)
    results["beta_only"] = build_results(data_dir, beta_only)

    out_path = Path(args.out) if args.out else data_dir / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")

    print_table(results)


if __name__ == "__main__":
    main()
