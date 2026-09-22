#!/usr/bin/env python3
"""Publication figures for the Typical/PCDM paper + blog post.

Re-runnable. Writes PDF+PNG (200 dpi) into figures/, plus figures/latex_includes.tex
and figures/figures_manifest.json (which artefact fields backed each figure).

Data provenance rule: every plotted number comes from a local runs/*/*.json(l)
artefact, or is pulled once from the HF hub model repo guychuk/pcdm-runs into the
matching runs/<name>/... path (tracked by .gitignore's existing allow-list for
results.json / bench.json / eval_wf.json / summary.json / results.jsonl) so the
repo stays the source of truth on every subsequent run. Numbers that exist only
as REPORT.md prose (no backing JSON) are left out and reported as omitted.

Run: uv run --no-sync python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
FIGDIR = REPO / "figures"
FIGDIR.mkdir(exist_ok=True)
HF_REPO = "guychuk/pcdm-runs"

# Okabe-Ito colorblind-safe palette (muted).
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#767676"
YELLOW = "#E69F00"
SKY = "#56B4E9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
})

manifest: dict[str, dict] = {}
omitted: list[str] = []


def note_omitted(msg: str) -> None:
    omitted.append(msg)
    print(f"OMITTED: {msg}")


def save(fig, name: str, sources: list[str], desc: str) -> None:
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    manifest[name] = {"description": desc, "sources": sources}
    print(f"wrote {name}.pdf / {name}.png")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def ensure_artefact(rel_path: str) -> Path | None:
    """Return runs/<rel_path>, pulling it from the HF hub once if not present locally."""
    local = RUNS / rel_path
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(HF_REPO, rel_path, repo_type="model", local_dir=str(RUNS))
        return local if local.exists() else None
    except Exception as e:  # pragma: no cover - network/availability dependent
        print(f"could not fetch {rel_path} from {HF_REPO}: {e}")
        return None


def jev_summary(rel_dir: str) -> dict | None:
    p = ensure_artefact(f"{rel_dir}/summary.json")
    return load_json(p) if p else None


def eval_field(rel_dir: str, *keys):
    p = ensure_artefact(f"{rel_dir}/results.json")
    if p is None:
        return None
    d = load_json(p)
    ev = d.get("eval", d)
    for k in keys:
        if ev is None:
            return None
        ev = ev.get(k)
    return ev


def bench_single_decision_ms(rel_dir: str, k: str, state_tokens: str = "256") -> float | None:
    p = ensure_artefact(f"{rel_dir}/bench.json")
    if p is None:
        return None
    d = load_json(p)
    try:
        return d["native"][state_tokens][k]["native"][0]["total_ms"]
    except (KeyError, IndexError):
        return None


# ---------------------------------------------------------------------------
# fig_architecture — schematic of the decision pass (no run data, vector only)
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333333", fontsize=8.6, weight="normal"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                            linewidth=1.1, edgecolor=ec, facecolor=fc)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=fontsize, weight=weight, wrap=True)
        return b

    def arrow(p0, p1, color="#333333", style="-|>", lw=1.3, connectionstyle="arc3,rad=0.0"):
        a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                             linewidth=lw, color=color, connectionstyle=connectionstyle)
        ax.add_patch(a)

    # Inputs
    box(0.15, 4.55, 2.5, 0.95, "state text $x$\n(policy / case / evidence)", "#EAF2F8")
    box(0.15, 2.9, 2.5, 0.95, "query $q_i$ + rendered\ncandidates $A_i$", "#FDF2E3")

    # Prefix / suffix
    box(3.15, 4.55, 2.55, 0.95, "KV-cached prefix\nencoded ONCE per state", SKY + "33")
    box(3.15, 2.9, 2.55, 0.95, "suffix (per query, cheap)\noption spans + terminal token", YELLOW + "33")

    arrow((2.65, 5.02), (3.15, 5.02))
    arrow((2.65, 3.37), (3.15, 3.37))
    # prefix feeds every suffix pass (reused across i)
    arrow((4.4, 4.55), (4.4, 3.85), connectionstyle="arc3,rad=0.0")

    # Head
    box(6.2, 3.6, 2.6, 1.55,
        "native head\nreads: pooled option spans\n(N3) + terminal token $h_D$",
        "#F1EAF7")
    arrow((5.7, 5.02), (6.2, 4.55))
    arrow((5.7, 3.37), (6.2, 4.05))

    # Output
    box(6.55, 1.55, 1.9, 0.95, "$P(y\\,|\\,x, q, A)$\nover $A \\cup \\{\\varnothing\\}$", "#EAF7EE")
    arrow((7.5, 3.6), (7.5, 2.5))

    # Typed primitives row
    ax.text(5.0, 1.15, "three typed primitives, one readout", fontsize=9, style="italic",
             ha="center", color="#333333")
    prims = [
        ("Choice", "$K$-way, factored null", BLUE, 0.3, -0.35),
        ("Score", "ordinal levels, $\\tau{=}0.7$", GREEN, 3.55, -0.12),
        ("Noul", "Bernoulli, per-row routed", ORANGE, 6.8, 0.12),
    ]
    for name, sub, color, x0, rad in prims:
        box(x0, 0.05, 2.85, 0.85, f"{name}\n{sub}", color + "22", ec=color, fontsize=8.4, weight="bold")
        arrow((x0 + 1.4, 0.92), (7.05, 1.57), color=color, lw=1.1,
              connectionstyle=f"arc3,rad={rad}")

    save(
        fig, "fig_architecture",
        sources=["native.py:1-40 (docstring, class NativeHead)", "PROJECT.md:8-15"],
        desc="Schematic of the decision pass: KV-cached state prefix, per-query suffix, "
             "native head reading pooled option spans + terminal token, three typed "
             "primitives (Choice / Score / Noul) on one readout.",
    )


# ---------------------------------------------------------------------------
# fig_ladder — JevBench standard/hard vs model size
# ---------------------------------------------------------------------------
def fig_ladder():
    # (size_B, run_dir) tuples per series.
    q3_frozen = [(1.7, "jev_zs3_1p7b"), (4, "jev_zs3_4b"), (8, "jev_zs3_8b"), (14, "jev_zs3_14b")]
    q3_trained = [(1.7, "jev_native_ts1b"), (4, "jev_native_tm1b"),
                  (8, "jev_native_ladder_8b"), (14, "jev_native_tl1b")]
    q35_frozen = [(2, "jev_zs3_q35_2b"), (4, "jev_zs3_q35_4b"), (9, "jev_zs3_q35_9b")]
    q35_trained = [(4, "jev_native_tm2")]

    def series(pairs, field):
        xs, ys = [], []
        for size, d in pairs:
            s = jev_summary(d)
            if s is None:
                note_omitted(f"fig_ladder: {d} summary.json unavailable (locally or on HF hub)")
                continue
            xs.append(size)
            ys.append(s[field]["accuracy"])
        return xs, ys

    refs = [  # (label, std, hard) -- REPORT.md 3q, per-item on the public 231 ids
        ("Laya", 0.694, 0.351),
        ("jeff", 0.750, 0.387),
        ("open-alt-jev", 0.833, 0.568),
        ("system-one-open", 0.931, 0.486),
        ("SemIf", 0.986, 0.613),
        ("Jev 1.13", 0.986, 0.730),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True)
    for ax, field, title, ref_idx in ((axes[0], "original", "JevBench standard (72 items)", 1),
                                       (axes[1], "hard", "JevBench hard (111 items)", 2)):
        x, y = series(q3_frozen, field)
        ax.plot(x, y, "o--", color=BLUE, label="Qwen3 frozen (3-shot)", markerfacecolor="white")
        x, y = series(q3_trained, field)
        ax.plot(x, y, "o-", color=BLUE, label="Qwen3 trained")
        x, y = series(q35_frozen, field)
        ax.plot(x, y, "s--", color=ORANGE, label="Qwen3.5 frozen (3-shot)", markerfacecolor="white")
        x, y = series(q35_trained, field)
        ax.plot(x, y, "*", color=ORANGE, markersize=13, label="Qwen3.5 trained (tm2)")

        ref_vals = [(label, std if ref_idx == 1 else hard) for label, std, hard in refs]
        ref_vals.sort(key=lambda t: t[1])
        placed = []
        for label, val in ref_vals:
            y = val
            for p in placed:
                if abs(y - p) < 0.028:
                    y = p + 0.028
            placed.append(y)
            ax.axhline(val, color=GRAY, linewidth=0.8, linestyle=":")
            ax.text(14.3, y, label, fontsize=6.8, va="center", color=GRAY)

        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks([1.7, 2, 4, 8, 9, 14])
        ax.set_xticklabels(["1.7", "2", "4", "8", "9", "14"])
        ax.set_xlabel("backbone size (B parameters)")
        ax.set_ylabel("accuracy")
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0.15, 1.05)
        ax.set_xlim(1.4, 20)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.09), frameon=False)
    fig.text(0.5, -0.16,
              "8B trained point = ladder_8b, the matched untyped-head baseline (no typed-head release exists at 8B).",
              ha="center", fontsize=7.5, color="#555555")

    save(
        fig, "fig_ladder",
        sources=[
            "runs/jev_zs3_{1p7b,4b,8b,14b}/summary.json .original/.hard.accuracy (Qwen3 frozen 3-shot)",
            "runs/jev_native_{ts1b,tm1b,ladder_8b,tl1b}/summary.json .original/.hard.accuracy (Qwen3 trained)",
            "runs/jev_zs3_q35_{2b,4b,9b}/summary.json (Qwen3.5 frozen 3-shot, pulled from HF hub guychuk/pcdm-runs)",
            "runs/jev_native_tm2/summary.json (Qwen3.5-4B trained, pulled from HF hub)",
            "REPORT.md 3q reference table (Laya/jeff/open-alt-jev/system-one-open/SemIf/Jev-1.13, per-item on public 231 ids)",
        ],
        desc="JevBench standard and hard accuracy vs backbone size for Qwen3 (1.7/4/8/14B) and "
             "Qwen3.5 (2/4/9B); frozen-with-3-shots dashed, trained solid; open-field references "
             "as horizontal markers.",
    )


# ---------------------------------------------------------------------------
# fig_latency_quality — JevBench standard vs single-decision latency
# ---------------------------------------------------------------------------
def fig_latency_quality():
    points = [
        ("typical-small (1.7B)", "jev_native_ts1b", "bench_ts1c", "ts1b has no bench.json; "
         "latency proxy = bench_ts1c, same backbone/tap-20 re-run"),
        ("typical-medium (4B)", "jev_native_tm1b", "bench_tm1b", None),
        ("8B (ladder_8b, untyped baseline)", "jev_native_ladder_8b", "bench_ladder_8b", None),
        ("tl1b (14B, in-flight)", "jev_native_tl1b", "bench_tl1b", None),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    bands = [
        (25, 40, "#2ca02c", "one-letter decode"),
        (60, 200, "#ff7f0e", "JSON/label decode"),
        (3000, 60000, "#9467bd", "chain-of-thought"),
    ]
    for lo, hi, color, label in bands:
        ax.axvspan(lo, hi, color=color, alpha=0.12)
        ax.text((lo * hi) ** 0.5, 1.0, label, rotation=90, fontsize=7.5, color=color,
                 ha="center", va="top", transform=ax.get_xaxis_transform())

    used_sources = []
    for label, jev_dir, bench_dir, note in points:
        s = jev_summary(jev_dir)
        ms = bench_single_decision_ms(bench_dir, "32")
        if s is None or ms is None:
            note_omitted(f"fig_latency_quality: {label} missing summary or bench data")
            continue
        acc = s["original"]["accuracy"]
        ax.scatter([ms], [acc], color=BLUE, zorder=5, s=45)
        ax.annotate(label, (ms, acc), textcoords="offset points", xytext=(6, 5), fontsize=7.8)
        used_sources.append(f"{jev_dir}/summary.json + {bench_dir}/bench.json (K=32, state=256, m=1 total_ms)"
                             + (f" [{note}]" if note else ""))

    ax.set_xscale("log")
    ax.set_xlabel("single-decision latency (ms, log scale)")
    ax.set_ylabel("JevBench standard accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(15, 100000)

    save(
        fig, "fig_latency_quality",
        sources=used_sources + [
            "generative-LLM comparison bands are the ranges specified in the figure task "
            "(one-letter decode ~25-40ms, JSON/label decode ~60-200ms, CoT ~3-60s), not a "
            "measured artefact",
        ],
        desc="JevBench standard accuracy vs measured single-decision latency (ms, log scale) "
             "for our trained models, with generative-LLM decode-latency comparison bands.",
    )


# ---------------------------------------------------------------------------
# fig_calibration — held-out score NLL / typed-decisions NLL, before/after fix
# ---------------------------------------------------------------------------
def fig_calibration():
    runs = ["ladder_14b", "tl1b", "tl1b_nokd", "ts1b", "tm1b", "tm2"]
    score_nll, typed_nll, labels = [], [], []
    for r in runs:
        sn = eval_field(r, "wf_heldout_score", "scaled", "nll")
        tn = eval_field(r, "typed_decisions_test", "scaled", "nll")
        if sn is None or tn is None:
            note_omitted(f"fig_calibration: {r} results.json missing NLL fields")
            continue
        labels.append(r)
        score_nll.append(sn)
        typed_nll.append(tn)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(labels))
    w = 0.36
    ax.bar([i - w / 2 for i in x], score_nll, width=w, color=BLUE, label="held-out score NLL")
    ax.bar([i + w / 2 for i in x], typed_nll, width=w, color=ORANGE, label="typed-decisions NLL")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("NLL (nats)")
    ax.legend(frameon=False)

    if "ladder_14b" in labels and "tl1b" in labels:
        i0, i1 = labels.index("ladder_14b"), labels.index("tl1b")
        y0, y1 = score_nll[i0], score_nll[i1]
        ax.annotate(
            f"long-state fix + frozen-teacher KD\nscore NLL {y0:.2f} \u2192 {y1:.2f}",
            xy=(i1 - w / 2, y1), xytext=((i0 + i1) / 2, max(score_nll) * 0.75),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
            fontsize=8, ha="center",
        )

    save(
        fig, "fig_calibration",
        sources=[f"runs/{r}/results.json eval.wf_heldout_score.scaled.nll, "
                 f"eval.typed_decisions_test.scaled.nll" for r in labels],
        desc="Held-out score NLL and typed-decisions NLL across the calibration-fix "
             "sequence (ladder_14b \u2192 tl1b/tl1b_nokd/ts1b/tm1b/tm2), showing the "
             "2.87 \u2192 ~0.95 collapse fix.",
    )


# ---------------------------------------------------------------------------
# fig_hard_families — per-family JevBench-hard accuracy, 3 models
# ---------------------------------------------------------------------------
def fig_hard_families():
    models = [("ladder_14b", "jev_native_ladder_14b", BLUE),
              ("tl1b", "jev_native_tl1b", ORANGE),
              ("frozen-14B (3-shot)", "jev_zs3_14b", GREEN)]
    data = {}
    families = None
    for label, d, _ in models:
        s = jev_summary(d)
        if s is None:
            note_omitted(f"fig_hard_families: {d} summary.json unavailable")
            continue
        pf = s["hard"]["per_family"]
        data[label] = {fam: v["accuracy"] for fam, v in pf.items()}
        families = set(pf.keys()) if families is None else families & set(pf.keys())

    families = sorted(families, key=lambda f: data["tl1b"].get(f, 0))
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    y = range(len(families))
    h = 0.25
    for i, (label, _, color) in enumerate(models):
        if label not in data:
            continue
        vals = [data[label][f] for f in families]
        ax.barh([yi + (i - 1) * h for yi in y], vals, height=h, color=color, label=label)

    ax.axvline(0.336, color="#333333", linestyle="--", linewidth=1.0)
    ax.text(0.336, len(families) - 0.3, "chance (.336)", rotation=90, fontsize=7.5,
            ha="right", va="top")
    ax.set_yticks(list(y))
    ax.set_yticklabels(families)
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")

    save(
        fig, "fig_hard_families",
        sources=[
            "runs/jev_native_ladder_14b/summary.json hard.per_family.<family>.accuracy",
            "runs/jev_native_tl1b/summary.json hard.per_family.<family>.accuracy",
            "runs/jev_zs3_14b/summary.json hard.per_family.<family>.accuracy (pulled from HF hub)",
        ],
        desc="Per-family JevBench-hard accuracy (10 families) for ladder_14b, tl1b, and the "
             "frozen-14B 3-shot control, sorted by tl1b's value, with the chance line at .336. "
             "Note: ladder_14b's temporal_numeric/probability values here (.333/.50, "
             "re-derived from hard/results.jsonl) differ from REPORT.md 3ah's prose "
             "(.20/.40) -- the JSON artefact is used as the source of truth.",
    )


# ---------------------------------------------------------------------------
# fig_deltaq — the candidate-blind negative: raw accuracy vs question-dependence
# ---------------------------------------------------------------------------
def fig_deltaq():
    """Raw MMLU-Pro among-K accuracy (x) vs Delta_q^sh (y) for every readout we probed.

    Delta_q^sh = acc(real question) - acc(shuffled question), candidate set fixed.
    A point high on x but at y ~ 0 is exploiting candidate-set priors, not answering
    the question. Every candidate-blind variant sits in that band.
    """
    blind = [
        ("single vector (z1)", "probe_e3a_z1", 0.0335),
        ("8 probes (zr)", "probe_e3b_zr", -0.0325),
        ("set-conditioned (zr_set)", "probe_e3c_zr_set", -0.0045),
        ("single-set control", "probe_e3ms_ctrl", -0.0465),
        (r"multi-set + $\Delta$log-odds", "probe_e3ms_zr_set", 0.0195),
        ("8 probes, full depth", "probe_e3b_zr_tap28", -0.0185),
    ]
    aware = [
        ("letter logits (N1)", "probe_nc_n1", ORANGE, "s"),
        ("letter logits (N1, seed 1)", "probe_nc_n1_s1", ORANGE, "s"),
        ("cand.-blind semantic (N2)", "probe_nc_n2", YELLOW, "D"),
        ("contextual cand. (N3)", "probe_nc_n3", GREEN, "o"),
        ("contextual cand. (N3, seed 1)", "probe_nc_n3_s1", GREEN, "o"),
    ]

    def probe(name):
        p = REPO / "runs" / name / "results.json"
        if not p.exists():
            note_omitted(f"fig_deltaq: runs/{name}/results.json unavailable")
            return None
        e = json.loads(p.read_text())["eval"]
        try:
            a = e["mmlu_pro"]["raw"]["acc_k"]
            s = e["mmlu_shuffledq"]["raw"]["acc_k"]
        except KeyError:
            note_omitted(f"fig_deltaq: runs/{name} missing mmlu probe fields")
            return None
        return a, a - s

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhspan(-0.02, 0.02, color=GRAY, alpha=0.13, zorder=0)
    ax.axhline(0.0, color=GRAY, linewidth=1.0, zorder=1)
    ax.text(0.398, 0.021, r"$\Delta_q^{\mathrm{sh}} \approx 0$: candidate-set priors only",
            fontsize=8, color="#444444", va="bottom", ha="right")

    xs, ys = [], []
    for label, name, ty in blind:
        v = probe(name)
        if v is None:
            continue
        xs.append(v[0]); ys.append(v[1])
        ax.annotate(label, xy=v, xytext=(0.203, ty), fontsize=7.2, ha="left",
                    va="center", color="#444444",
                    arrowprops=dict(arrowstyle="-", color="#BBBBBB", linewidth=0.7,
                                    shrinkA=1, shrinkB=4))
    if xs:
        ax.scatter(xs, ys, s=58, marker="^", color=BLUE, zorder=3,
                   label="candidate-blind readouts (6 variants)")

    for label, name, color, marker in aware:
        v = probe(name)
        if v is None:
            continue
        ax.scatter([v[0]], [v[1]], s=58, marker=marker, color=color, zorder=3,
                   label=label if "seed 1" not in label else None)

    t = probe("probe_teacher_kb")
    if t is not None:
        ax.scatter([t[0]], [t[1]], s=90, marker="*", color="#333333", zorder=4,
                   label="listwise teacher (options in context)")

    ax.set_xlabel("MMLU-Pro among-$K$ accuracy (raw)")
    ax.set_ylabel(r"$\Delta_q^{\mathrm{sh}}$ (question-dependent signal)")
    ax.set_xlim(0.105, 0.40)
    ax.set_ylim(-0.055, 0.145)
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    save(
        fig, "fig_deltaq",
        sources=[f"runs/{n}/results.json eval.mmlu_pro.raw.acc_k, eval.mmlu_shuffledq.raw.acc_k"
                 for _, n, _ in blind] +
                [f"runs/{n}/results.json eval.mmlu_pro.raw.acc_k, eval.mmlu_shuffledq.raw.acc_k"
                 for _, n, _, _ in aware] +
                ["runs/probe_teacher_kb/results.json eval.mmlu_pro.raw.acc_k, "
                 "eval.mmlu_shuffledq.raw.acc_k"],
        desc="The candidate-blind negative result. Raw MMLU-Pro among-K accuracy (x) against "
             "Delta_q_sh, the drop in accuracy when the real question is replaced by a shuffled "
             "one with the candidate set held fixed (y). All six candidate-blind variants sit in "
             "the +/-.02 band around zero at 1,200 items: their above-chance accuracy is "
             "candidate-set priors. The letter readout, the contextual-candidate readout, and the "
             "listwise teacher all carry a question-dependent signal of .09-.12.",
    )


# ---------------------------------------------------------------------------
# fig_truncation — data_wf_long state-token-length histogram + long_policy accuracy
# ---------------------------------------------------------------------------
def fig_truncation():
    import random

    path = REPO / "data_wf_long" / "train.jsonl"
    if not path.exists():
        note_omitted("fig_truncation: data_wf_long/train.jsonl not found")
        return

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")

    lines = path.read_text().splitlines()
    random.seed(0)
    sample = random.sample(lines, min(2000, len(lines)))
    lengths = []
    for line in sample:
        row = json.loads(line)
        lengths.append(len(tok(row["state"], add_special_tokens=False)["input_ids"]))

    thresholds = [256, 1024, 2048, 3072]
    n = len(lengths)
    fracs = {t: sum(1 for l in lengths if l > t) / n for t in thresholds}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [2.2, 1]})
    ax1.hist(lengths, bins=60, color=BLUE, alpha=0.85)
    for t in thresholds:
        ax1.axvline(t, color=GRAY, linestyle="--", linewidth=1.0)
        ax1.text(t, ax1.get_ylim()[1] * 0.97, f"{t}\n{fracs[t]*100:.0f}% trunc.",
                  fontsize=7.2, ha="center", va="top", color="#333333")
    ax1.set_xlabel(f"state token length (Qwen3 tokenizer, n={n} sampled rows)")
    ax1.set_ylabel("count")

    lp_series = [("ladder_14b", "jev_native_ladder_14b"),
                 ("typical-medium", "jev_native_tm1b"),
                 ("tl1b", "jev_native_tl1b"),
                 ("frozen-14B (3-shot)", "jev_zs3_14b")]
    lp_labels, lp_vals, lp_sources = [], [], []
    for label, d in lp_series:
        s = jev_summary(d)
        if s is None:
            note_omitted(f"fig_truncation: {d} summary.json unavailable for long_policy accuracy")
            continue
        acc = s["hard"]["per_family"]["long_policy"]["accuracy"]
        lp_labels.append(label)
        lp_vals.append(acc)
        lp_sources.append(f"runs/{d}/summary.json hard.per_family.long_policy.accuracy")

    ax2.bar(range(len(lp_labels)), lp_vals, color=[BLUE, GREEN, ORANGE, GRAY][: len(lp_labels)])
    ax2.set_xticks(range(len(lp_labels)))
    ax2.set_xticklabels(lp_labels, rotation=30, ha="right", fontsize=7.6)
    ax2.set_ylabel("long_policy accuracy (hard tier)")
    ax2.set_ylim(0, 0.55)
    for i, v in enumerate(lp_vals):
        ax2.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)

    save(
        fig, "fig_truncation",
        sources=[
            f"data_wf_long/train.jsonl (state field, {n}/{len(lines)} rows sampled, "
            "tokenized with Qwen/Qwen3-1.7B-Base)",
        ] + lp_sources,
        desc="Left: histogram of data_wf_long state token lengths with truncation fractions at "
             "256/1024/2048/3072 tokens. Right: long_policy (hard-tier) accuracy across the "
             "long-state bug/fix sequence.",
    )


# ---------------------------------------------------------------------------
# fig_serving — before/after per-decision latency (serve_bench2)
# ---------------------------------------------------------------------------
def fig_serving():
    before = load_json(RUNS / "serve_bench2" / "before.json")
    after = load_json(RUNS / "serve_bench2" / "results.json")

    def get(entries, model, k=2, state_tokens=256):
        m = next((e for e in entries if e["model"] == model), None)
        if m is None:
            return None
        cfg = next((c for c in m["configs"]
                    if c["K"] == k and c["state_tokens"] == state_tokens), None)
        return cfg

    models = [("typical-small", "1.7B"), ("typical-medium", "4B"), ("tm2", "Qwen3.5-4B")]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    labels, before_cold, before_warm, after_cold, after_warm = [], [], [], [], []
    for model, size_label in models:
        cb, ca = get(before, model), get(after, model)
        if cb is None or ca is None:
            note_omitted(f"fig_serving: {model} missing K=2/state=256 config")
            continue
        labels.append(f"{model}\n({size_label})")
        before_cold.append(cb["cold"]["p50_ms"])
        before_warm.append(cb["warm"]["p50_ms"])
        after_cold.append(ca["cold"]["p50_ms"])
        after_warm.append(ca["warm"]["p50_ms"])

    x = range(len(labels))
    w = 0.2
    ax.bar([i - 1.5 * w for i in x], before_cold, width=w, color=ORANGE, alpha=0.5, label="before, cold")
    ax.bar([i - 0.5 * w for i in x], before_warm, width=w, color=BLUE, alpha=0.5, label="before, warm")
    ax.bar([i + 0.5 * w for i in x], after_cold, width=w, color=ORANGE, label="after, cold")
    ax.bar([i + 1.5 * w for i in x], after_warm, width=w, color=BLUE, label="after, warm")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("p50 latency per decision (ms)")
    ax.legend(frameon=False, ncol=2)

    save(
        fig, "fig_serving",
        sources=["runs/serve_bench2/before.json and results.json (== after.json), "
                 "configs[K=2, state_tokens=256].{cold,warm}.p50_ms"],
        desc="Per-decision p50 latency before/after the zero-copy prefix fix, cold vs warm, "
             "for typical-small (1.7B), typical-medium (4B) and tm2 (Qwen3.5-4B), at K=2, "
             "state_tokens=256.",
    )


# ---------------------------------------------------------------------------
def write_latex_includes():
    order = ["fig_architecture", "fig_deltaq", "fig_ladder", "fig_latency_quality", "fig_calibration",
             "fig_hard_families", "fig_truncation", "fig_serving"]
    blocks = []
    for name in order:
        if name not in manifest:
            continue
        label = "fig:" + name[len("fig_"):]
        cap = manifest[name]["description"].replace("_", "\\_")
        cap = cap.replace("→", "$\\rightarrow$").replace("–", "--")
        blocks.append(
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\linewidth]{{figures/{name}.pdf}}\n"
            f"  \\caption{{{cap}}}\n"
            f"  \\label{{{label}}}\n"
            "\\end{figure}\n"
        )
    (FIGDIR / "latex_includes.tex").write_text("\n".join(blocks))


def write_manifest():
    out = {"figures": manifest, "omitted_series": omitted}
    (FIGDIR / "figures_manifest.json").write_text(json.dumps(out, indent=2))


def main():
    fig_architecture()
    fig_deltaq()
    fig_ladder()
    fig_latency_quality()
    fig_calibration()
    fig_hard_families()
    fig_truncation()
    fig_serving()
    write_latex_includes()
    write_manifest()

    print("\n=== files written ===")
    for p in sorted(FIGDIR.glob("*")):
        print(" ", p.relative_to(REPO))
    if omitted:
        print("\n=== omitted series ===")
        for o in omitted:
            print(" -", o)


if __name__ == "__main__":
    main()
