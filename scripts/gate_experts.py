"""PLAN6 item 3: a learned per-input expert gate g(x) in [0, g_max] over the energy and native
experts, replacing the global g of fuse_scores.py (REPORT sec 3o: the oracle row is the envelope
a per-input gate could reach; a global g captures little of it).

Mixture (fuse_scores' `log` mode, energy null as the support gate as in sec 3n), tempered by ONE
val-fitted T so ECE/NLL are reportable:
    P(a_j | answerable) = softmax_j((logit_e_j + g(x) * logit_n_j) / T)      over the K shared candidates
    P(null) = P_energy(null | T),   P(a_j) = (1 - P(null)) * P(a_j | answerable)
g = 0 for every row reproduces the energy model at temperature T exactly.

g(x) = g_max * sigmoid(MLP(features)), features = inference-time signals only (no task ids):
    energy: max prob / entropy / top-2 margin of P_E(. | answerable), P_E(null)
    native: the same four from the aligned native logits
    shared: log K, log(1 + query chars)   (token counts are not in the dumps' meta)
    optional --use_query_emb: DUMP/<set>.query_emb.npy [N, D] concatenated (not in the dumps today)
2-layer MLP (F -> --hidden tanh -> 1), ~200 params by default; inputs are z-scored on val.

Trained by full-batch Adam on the soft-CE (NLL) of the mixed distribution over the dumps' val
split(s) only: "val", "data_v5_val", and every "data_*_val" set (the --extra_data convention,
e.g. data_kb_val) present in BOTH dumps -- never on mnli_val/boolq_val or any other eval set.
Then T is fitted on the same rows, and every remaining common set is evaluated. Writes
runs/<name>/results.json (pcdm/train.py format: eval[set][raw|scaled] + cse/ksweep/cf branches, so
pcdm/report.py / pcdm/report_native.py read it) with the gate's weights, features, val NLLs and mean g
per set under results["gate"], and prints one table: energy (g=0), g=1, native alone, oracle
(best fixed g per column over {0, .5, 1, 2}, each with its own val-fitted T), and the gate --
each row T-fitted on the same val rows -- plus mean g per eval set.

uv run scripts/gate_experts.py DUMP_ENERGY DUMP_NATIVE --name gate_n3 [--features all|energy_only|native_only]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcdm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compose_support import eval_set  # noqa: E402
from fuse_scores import load_dumps  # noqa: E402
from null_bias import apply_bias, soft_ce, TS  # noqa: E402

ORACLE_G = [0.0, 0.5, 1.0, 2.0]
FEATURE_SETS = {
    "energy_only": ["e_max", "e_ent", "e_margin", "e_null", "log_k", "log_qchars"],
    "native_only": ["n_max", "n_ent", "n_margin", "n_null", "log_k", "log_qchars"],
}
FEATURE_SETS["all"] = FEATURE_SETS["energy_only"] + FEATURE_SETS["native_only"][:4]

# (set, metric, direction, label) -- direction picks max/min for the oracle row
COLUMNS = [
    ("mmlu_pro", "acc_k", "max", "MMLU amK"),
    ("snli_test", "acc", "max", "SNLI"), ("mnli_val", "acc", "max", "MNLI"),
    ("anli_test", "acc", "max", "ANLI"), ("boolq_val", "acc", "max", "BoolQ"),
    ("clinc_test", "acc", "max", "CLINC-150"), ("clinc_heldout", "acc", "max", "CLINC-held"),
    ("banking77_test", "acc", "max", "Bank77"), ("clinc_oos", "null_recall", "max", "OOS rec"),
    ("snli_test", "nll", "min", "SNLI NLL"), ("snli_test", "ece", "min", "SNLI ECE"),
    ("banking77_test", "nll", "min", "Bank77 NLL"), ("banking77_test", "ece", "min", "Bank77 ECE"),
]


def is_val_set(name):
    return name in ("val", "data_v5_val") or (name.startswith("data_") and name.endswith("_val"))


def _cand_stats(logits, K, prefix):
    """max prob / entropy / top-2 margin of the candidate-conditional softmax + P(null) of the full one."""
    Kmax = logits.shape[1] - 1
    valid = np.arange(Kmax)[None, :] < K[:, None]
    z = np.where(valid, logits[:, :-1], -np.inf)
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    top2 = -np.sort(-np.pad(p, ((0, 0), (0, 1))), axis=-1)[:, :2]  # pad: K = 1 sets (single candidate) have no runner-up
    ent = -(np.where(p > 0, p * np.log(np.clip(p, 1e-12, 1)), 0.0)).sum(-1)
    return {f"{prefix}_max": top2[:, 0], f"{prefix}_ent": ent, f"{prefix}_margin": top2[:, 0] - top2[:, 1],
            f"{prefix}_null": apply_bias(logits, K, 0.0, 0.0, 1.0)[:, -1]}


def features(le, K, ln, examples, names, query_emb=None):
    """-> [N, F] float32 in the order of `names` (+ query_emb columns appended if given)."""
    f = {**_cand_stats(le, K, "e"), **_cand_stats(ln, K, "n"), "log_k": np.log(K),
         "log_qchars": np.log1p([len(ex.get("query", "")) for ex in examples])}
    X = np.stack([f[n] for n in names], axis=1)
    if query_emb is not None:
        X = np.concatenate([X, query_emb], axis=1)
    return X.astype(np.float32)


class Gate(torch.nn.Module):
    """g(x) = g_max * sigmoid(w2 . tanh(W1 x + b1) + b2), x z-scored with val stats."""

    def __init__(self, n_in, hidden=16, g_max=1.0):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sd", torch.ones(n_in))
        self.g_max = g_max

    def forward(self, X):
        return self.g_max * torch.sigmoid(self.net((X - self.mu) / self.sd)).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())


def mix(le, K, ln, g, T=1.0):
    """Tempered per-row mixture -> probs [N, Kmax+1], null last (torch; g is a tensor [N])."""
    le, ln = torch.as_tensor(le, dtype=torch.float64), torch.as_tensor(ln, dtype=torch.float64)
    K, g = torch.as_tensor(K), torch.as_tensor(g, dtype=torch.float64)
    Kmax = le.shape[1] - 1
    valid = torch.arange(Kmax)[None, :] < K[:, None]
    ln_safe = torch.where(valid, ln[:, :-1], torch.zeros(()))  # 0 * -inf = nan otherwise
    z = torch.where(valid, (le[:, :-1] + g[:, None] * ln_safe) / T, torch.full((), -torch.inf))
    p_answer = torch.softmax(z, dim=-1)
    valid_full = torch.cat([valid, torch.ones(len(K), 1, dtype=torch.bool)], dim=1)
    r = 1.0 - torch.softmax(torch.where(valid_full, le / T, torch.full((), -torch.inf)), dim=-1)[:, -1]
    return torch.cat([r[:, None] * p_answer, (1.0 - r)[:, None]], dim=1)


def const(g, n):
    return torch.full((n,), float(g), dtype=torch.float64)


def nll(probs, target):
    target = torch.as_tensor(target, dtype=torch.float64)
    return -(target * torch.log(probs.clamp_min(1e-12))).sum(-1).mean()


def fit_T(le, K, ln, g, target):
    """One temperature for the mixture at fixed g, min val NLL over pcdm/train.py's T grid."""
    return float(min(TS, key=lambda T: nll(mix(le, K, ln, g, T), target).item()))


def train_gate(gate, X, le, K, ln, target, steps=300, lr=1e-2, weight_decay=1e-4):
    """Full-batch Adam on the val NLL of the mixture. -> (nll_before, nll_after)."""
    X = torch.as_tensor(X)
    gate.mu.copy_(X.mean(0))
    gate.sd.copy_(X.std(0).clamp_min(1e-6))
    before = nll(mix(le, K, ln, gate(X), 1.0), target).item()
    opt = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        loss = nll(mix(le, K, ln, gate(X), 1.0), target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = nll(mix(le, K, ln, gate(X), 1.0), target).item()
    return before, after


def load_query_emb(dump_e, dump_n, name):
    for d in (Path(dump_e), Path(dump_n)):
        p = d / f"{name}.query_emb.npy"
        if p.exists():
            return np.load(p).astype(np.float32)
    raise FileNotFoundError(f"--use_query_emb: no {name}.query_emb.npy in either dump")


def _cell(entry, metric):
    v = None if entry is None else entry["scaled"].get(metric)
    return v if isinstance(v, (int, float)) else None


def _fmt(v):
    return "-" if v is None else f"{v:.3f}"


def run(dump_e, dump_n, name="gate_n3", feature_set="all", use_query_emb=False, hidden=16, g_max=1.0,
        steps=300, lr=1e-2, seed=0, quiet=False):
    loaded = load_dumps(dump_e, dump_n)
    val_names = sorted(n for n in loaded if is_val_set(n))
    eval_names = sorted(n for n in loaded if not is_val_set(n))
    if not val_names:
        raise SystemExit("no val split (val / data_v5_val / data_*_val) common to both dumps -- nothing to train on")
    if "data_kb_val" not in val_names:
        print("note: data_kb_val (the kb val) is not in both dumps -- the gate is trained on evidence val only")
    print(f"train on: {val_names}   eval on: {len(eval_names)} sets")

    names = FEATURE_SETS[feature_set]
    qe = {n: load_query_emb(dump_e, dump_n, n) for n in loaded} if use_query_emb else {}
    feats = {n: features(le, K, ln, me, names, qe.get(n)) for n, (le, te, ye, K, ln, me) in loaded.items()}

    Xv = np.concatenate([feats[n] for n in val_names])
    lev, tev, _, Kv, lnv, _ = (np.concatenate([loaded[n][i] for n in val_names]) for i in range(6))

    torch.manual_seed(seed)
    gate = Gate(Xv.shape[1], hidden, g_max)
    print(f"gate: {feature_set} features ({Xv.shape[1]} inputs), hidden={hidden}, {gate.n_params()} params, g in [0, {g_max:g}]")
    nll_before, nll_after = train_gate(gate, Xv, lev, Kv, lnv, tev, steps, lr)
    with torch.no_grad():
        gv = gate(torch.as_tensor(Xv))
        T = fit_T(lev, Kv, lnv, gv, tev)
        nll_T = nll(mix(lev, Kv, lnv, gv, T), tev).item()
    print(f"val NLL: init {nll_before:.4f} -> trained {nll_after:.4f} -> T={T:.3f} {nll_T:.4f}")

    # baselines, each with its own val-fitted T on the same rows
    rows = {f"g={g:g}": ("fixed", g, fit_T(lev, Kv, lnv, const(g, len(Kv)), tev)) for g in sorted(set(ORACLE_G) | {0.0, 1.0})}
    T_nat = float(min(TS, key=lambda T: soft_ce(apply_bias(lnv, Kv, 0.0, 0.0, T), tev)))
    rows["native"] = ("native", None, T_nat)
    rows["gate"] = ("gate", None, T)

    results = {"T": T, "val_nll_init": nll_before, "val_nll": nll_after, "val_nll_T": nll_T,
               "args": {"energy_dump": str(dump_e), "native_dump": str(dump_n), "features": feature_set,
                        "use_query_emb": use_query_emb, "hidden": hidden, "g_max": g_max, "steps": steps, "lr": lr, "seed": seed},
               "gate": {"features": names + (["query_emb"] if use_query_emb else []), "n_params": gate.n_params(),
                        "val_sets": val_names, "baseline_T": {k: v[2] for k, v in rows.items()}, "mean_g": {},
                        "state": {k: v.tolist() for k, v in gate.state_dict().items()}},
               "eval": {}}
    per_row = {k: {} for k in rows}
    with torch.no_grad():
        for n in eval_names:
            le, te, ye, K, ln, me = loaded[n]
            g_n = gate(torch.as_tensor(feats[n]))
            results["gate"]["mean_g"][n] = float(g_n.mean())
            for k, (kind, g, T_k) in rows.items():
                if kind == "native":
                    p = apply_bias(ln, K, 0.0, 0.0, T_k)
                else:
                    p = mix(le, K, ln, g_n if kind == "gate" else const(g, len(K)), T_k).numpy()
                per_row[k][n] = eval_set(n, p, te, ye, me)
            results["eval"][n] = per_row["gate"][n]
            # oracle envelope (PLAN6): P(either expert's argmax is correct) on labelled rows -- the ceiling any router can reach
            pe = apply_bias(le, K, 0.0, 0.0, 1.0); pn = apply_bias(ln, K, 0.0, 0.0, 1.0)
            lab = np.asarray(ye); ok = lab >= 0
            if ok.any():
                either = ((pe[ok].argmax(-1) == lab[ok]) | (pn[ok].argmax(-1) == lab[ok])).mean()
                results["gate"].setdefault("oracle_either_correct", {})[n] = float(either)
    if not quiet:
        print_table(per_row, results["gate"]["mean_g"], results["gate"].get("oracle_either_correct"))
    return results


def print_table(per_row, mean_g, oracle_either=None):
    cols = [c for c in COLUMNS if any(c[0] in per_row[k] for k in per_row)]
    print(f"\n{'row':<20}" + "".join(f"{lab:>11}" for _, _, _, lab in cols) + "   (each row: own val-fitted T)")
    for k in ("g=0", "g=1", "native"):
        print(f"{k + (' (energy)' if k == 'g=0' else ''):<20}" + "".join(f"{_fmt(_cell(per_row[k].get(s), m)):>11}" for s, m, _, _ in cols))
    oracle = []
    for s, m, d, _ in cols:
        vals = [v for g in ORACLE_G if (v := _cell(per_row[f'g={g:g}'].get(s), m)) is not None]
        oracle.append((max if d == "max" else min)(vals) if vals else None)
    print(f"{'oracle (per-col g)':<20}" + "".join(f"{_fmt(v):>11}" for v in oracle))
    print(f"{'gate':<20}" + "".join(f"{_fmt(_cell(per_row['gate'].get(s), m)):>11}" for s, m, _, _ in cols))
    if oracle_either:
        print("\noracle envelope P(either expert correct) vs gate acc:")
        for n, v in sorted(oracle_either.items()):
            ga = _cell(per_row["gate"].get(n), "acc"); print(f"  {n:<28}either={v:.3f}  gate={_fmt(ga)}")
    print("\nmean g per eval set:")
    for n, g in sorted(mean_g.items(), key=lambda kv: -kv[1]):
        print(f"  {n:<28}{g:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_energy")
    ap.add_argument("dump_native")
    ap.add_argument("--name", default="gate_n3", help="writes runs/<name>/results.json")
    ap.add_argument("--features", choices=sorted(FEATURE_SETS), default="all")
    ap.add_argument("--use_query_emb", action="store_true", help="append DUMP/<set>.query_emb.npy to the features")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--g_max", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    results = run(a.dump_energy, a.dump_native, a.name, a.features, a.use_query_emb, a.hidden, a.g_max, a.steps, a.lr, a.seed)
    out = Path("runs") / a.name / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
