"""Evaluation metrics. numpy float64, CPU. See PLAN.md."""
from collections import defaultdict

import numpy as np


def summarize(probs, target, label=None) -> dict:
    """
    probs, target: [N, K+1], null in the last column, caller-padded with 0.
    label: [N] int, -1 = null correct, None/NaN = no hard label (soft-only sets).
    """
    probs = np.asarray(probs, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    N, K1 = probs.shape
    K = K1 - 1

    out: dict = {}

    p_clipped = np.clip(probs, 1e-12, 1.0)
    out["nll"] = float(-(target * np.log(p_clipped)).sum(axis=-1).mean())
    out["brier"] = float(((probs - target) ** 2).sum(axis=-1).mean())

    m = np.clip(0.5 * (probs + target), 1e-12, 1.0)
    def kl(a):
        a = np.clip(a, 1e-12, 1.0)
        return (a * (np.log(a) - np.log(m))).sum(axis=-1)
    out["jsd"] = float((0.5 * kl(probs) + 0.5 * kl(target)).mean())

    if label is not None:
        label_arr = np.array([np.nan if l is None else l for l in label], dtype=np.float64)
        valid = ~np.isnan(label_arr)
    else:
        valid = np.zeros(N, dtype=bool)

    if valid.any():
        lbl = label_arr[valid].astype(np.int64)
        lbl_idx = np.where(lbl == -1, K, lbl)  # null maps to the last column
        pv = probs[valid]
        pred = pv.argmax(axis=-1)
        correct = (pred == lbl_idx).astype(np.float64)
        out["acc"] = float(correct.mean())

        k_rows = lbl >= 0
        if k_rows.any():
            pred_k = pv[k_rows, :K].argmax(axis=-1)
            out["acc_k"] = float((pred_k == lbl[k_rows]).mean())

        conf = pv.max(axis=-1)
        bins = np.linspace(0.0, 1.0, 16)
        ece = 0.0
        for i in range(15):
            lo, hi = bins[i], bins[i + 1]
            sel = (conf >= lo) & (conf <= hi if i == 14 else conf < hi)
            if sel.any():
                ece += (sel.sum() / len(conf)) * abs(correct[sel].mean() - conf[sel].mean())
        out["ece"] = float(ece)

        cutoff = int(np.ceil(0.8 * len(conf)))
        top = np.argsort(-conf)[:cutoff]
        out["sel_acc80"] = float(correct[top].mean()) if cutoff > 0 else float("nan")

        null_rows = lbl == -1
        if null_rows.any() and K > 0:
            out["conf_wrong"] = float((pv[null_rows][:, :K].max(axis=-1) > 0.9).mean())

        # PI memo: keep abstention disaggregated -- a never-abstaining model has zero false
        # abstentions and zero null recall, so neither is a substitute for the other.
        abstain = pred == K
        out["coverage"] = float(1.0 - abstain.mean())
        out["false_abstain"] = float(abstain[k_rows].mean()) if k_rows.any() else "n/a"
        out["null_recall"] = float(abstain[null_rows].mean()) if null_rows.any() else "n/a"

    null_mask = np.zeros(N, dtype=bool)
    if valid.any():
        null_mask[valid] = label_arr[valid] == -1
    is_null = null_mask | (target[:, K] == 1)
    if is_null.any() and (~is_null).any():
        from sklearn.metrics import roc_auc_score
        out["auroc_null"] = float(roc_auc_score(is_null.astype(int), probs[:, K]))
        # null detection from confidence alone (1 - max candidate prob): the only option a no-null model has
        out["auroc_conf"] = float(roc_auc_score(is_null.astype(int), 1 - probs[:, :K].max(axis=-1)))

    return out


def choice_set_effects(probs, examples) -> dict:
    """data_v4 cse_* battery (see data.py cse_variants). examples: the ordered list of eval
    rows, each carrying meta.pair (groups a base row with its variants) and meta.variant in
    {base, add_irr, remove_gold, dup, reorder, near_dup}. probs: [N, Kmax+1], null last, row
    order == examples order. A property that should hold for a well-behaved scorer: adding an
    irrelevant option, duplicating a candidate, or reordering the list shouldn't move
    probability mass around (IIA); removing the gold should push mass to null.
    Returns a flat {"variant/metric": mean over pairs that have that variant}.
    """
    probs = np.asarray(probs, dtype=np.float64)
    groups = defaultdict(dict)
    for i, ex in enumerate(examples):
        m = ex.get("meta", {})
        if "pair" in m and "variant" in m:
            groups[m["pair"]][m["variant"]] = i

    def cand_probs(i):
        return {c: float(probs[i, j]) for j, c in enumerate(examples[i]["candidates"])}

    acc = defaultdict(list)
    for g in groups.values():
        if "base" not in g:
            continue
        bi = g["base"]
        base_ex = examples[bi]
        bp = cand_probs(bi)
        lbl = base_ex.get("label")
        gold_str = base_ex["candidates"][lbl] if isinstance(lbl, int) and lbl >= 0 else None
        order = sorted(bp, key=lambda c: -bp[c])
        top1, top2 = (order + [None, None])[:2]

        if "add_irr" in g:
            ap = cand_probs(g["add_irr"])
            irr = [c for c in ap if c not in bp]
            if irr:
                acc["add_irr/p_irr"].append(ap[irr[0]])
            if top1 and top2 and min(bp[top1], bp[top2], ap.get(top1, 0), ap.get(top2, 0)) > 0:
                acc["add_irr/dlo_top2"].append(abs(
                    np.log(ap[top1] / ap[top2]) - np.log(bp[top1] / bp[top2])))

        if "remove_gold" in g:
            i = g["remove_gold"]
            acc["remove_gold/dp_null"].append(float(probs[i, -1]) - float(probs[bi, -1]))

        if "dup" in g:
            i = g["dup"]
            dex, dp = examples[i], cand_probs(g["dup"])
            src = dex.get("meta", {}).get("src")
            if src is not None and src in bp:
                slots = [float(probs[i, j]) for j, c in enumerate(dex["candidates"]) if c == src]
                if len(slots) == 2:
                    acc["dup/mass_err"].append(abs(sum(slots) - bp[src]))
                    acc["dup/slot_gap"].append(abs(slots[0] - slots[1]))
            if gold_str is not None:
                K = len(dex["candidates"])
                pred_str = dex["candidates"][int(probs[i, :K].argmax())]
                acc["dup/acc_dedup"].append(float(pred_str == gold_str))

        if "reorder" in g:
            rp = cand_probs(g["reorder"])
            diffs = [abs(rp[c] - bp[c]) for c in rp if c in bp]
            if diffs:
                acc["reorder/max_dp"].append(max(diffs))

        if "near_dup" in g:
            i = g["near_dup"]
            nex = examples[i]
            src = nex.get("meta", {}).get("src")
            new_str = nex["candidates"][-1]
            if src is not None and src in bp:
                np_ = cand_probs(i)
                combined = np_.get(new_str, 0.0) + np_.get(src, 0.0)
                acc["near_dup/cluster_mass_err"].append(abs(combined - bp[src]))

    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def counterfactual(probs, examples, teacher_probs=None) -> dict:
    """PLAN3 E3-cf battery (scripts/mmlu_counterfactual.py builds data_v4/eval/mmlu_cf.jsonl).
    examples carry meta.qid (groups a question's variants, same role as choice_set_effects'
    meta.pair) and meta.variant in {orig, remove3, add3_unrel, replace_hard, add_neardup,
    reorder}. probs: [N, Kmax+1], null last, row order == examples order.
    teacher_probs: optional {qid: {candidate_str: prob}}, aligned to probs by qid + candidate
    string (see cand_probs below) -> adds a per-variant mean KL(teacher||student).
    Returns a flat {"<variant>/<metric>": mean over questions that have that variant}, plus
    two IIA-style checks: add3_unrel/dlo_top2 (Δ log-odds between two original candidates --
    should be ~0, adding an unrelated option shouldn't move relative mass) and
    reorder/max_dp (should be ~0, shuffling order shouldn't move mass)."""
    probs = np.asarray(probs, dtype=np.float64)
    groups = defaultdict(dict)
    for i, ex in enumerate(examples):
        m = ex.get("meta", {})
        if "qid" in m and "variant" in m:
            groups[m["qid"]][m["variant"]] = i

    def cand_probs(i):
        return {c: float(probs[i, j]) for j, c in enumerate(examples[i]["candidates"])}

    acc = defaultdict(list)
    for g in groups.values():
        for variant, i in g.items():
            ex = examples[i]
            K = len(ex["candidates"])
            lbl = ex.get("label")
            if isinstance(lbl, int) and lbl >= 0:
                acc[f"{variant}/acc"].append(float(int(probs[i].argmax()) == lbl))
                acc[f"{variant}/among_k"].append(float(int(probs[i, :K].argmax()) == lbl))
            # per-row `teacher` (the teacher scored THIS variant; written by scripts/teacher_label.py on the
            # eval file) takes precedence over the per-question orig distribution in teacher_probs
            t = None
            if isinstance(ex.get("teacher"), list) and len(ex["teacher"]) >= K:
                t = dict(zip(ex["candidates"], ex["teacher"][:K]))
            elif teacher_probs and ex.get("meta", {}).get("qid") in teacher_probs:
                t = teacher_probs[ex["meta"]["qid"]]
            if t is not None:
                cp = cand_probs(i)
                pairs = [(t[c], cp[c]) for c in cp if c in t]
                if pairs:
                    tp = np.clip(np.array([p for p, _ in pairs]), 1e-12, 1.0)
                    tp = tp / tp.sum()  # renormalise over the candidates this variant kept
                    sp = np.clip(np.array([p for _, p in pairs]), 1e-12, 1.0)
                    acc[f"{variant}/kl_teacher"].append(float((tp * (np.log(tp) - np.log(sp))).sum()))

        # Δ-log-odds reproduction (PLAN3 E3-ms): for each variant with per-row teacher probs, compare the
        # student's shift in log-odds between the orig top-2 candidates with the teacher's shift.
        if "orig" in g and isinstance(examples[g["orig"]].get("teacher"), list):
            eo = examples[g["orig"]]; Ko = len(eo["candidates"])
            to = dict(zip(eo["candidates"], eo["teacher"][:Ko])); so = cand_probs(g["orig"])
            order = sorted(to, key=lambda c: -to[c]); a, b = (order + [None, None])[:2]
            for variant, j in g.items():
                ev = examples[j]
                if variant == "orig" or not isinstance(ev.get("teacher"), list):
                    continue
                tv = dict(zip(ev["candidates"], ev["teacher"][:len(ev["candidates"])])); sv = cand_probs(j)
                if a in tv and b in tv and a in sv and b in sv and min(to[a], to[b], tv[a], tv[b], so[a], so[b], sv[a], sv[b]) > 0:
                    d_t = np.log(tv[a] / tv[b]) - np.log(to[a] / to[b])
                    d_s = np.log(sv[a] / sv[b]) - np.log(so[a] / so[b])
                    acc[f"{variant}/delta_mae"].append(abs(d_s - d_t))
                    acc[f"{variant}/delta_teacher_abs"].append(abs(d_t))

        if "orig" in g and "add3_unrel" in g:
            bp, ap = cand_probs(g["orig"]), cand_probs(g["add3_unrel"])
            order = sorted(bp, key=lambda c: -bp[c])
            top1, top2 = (order + [None, None])[:2]
            if top1 and top2 and min(bp[top1], bp[top2], ap.get(top1, 0), ap.get(top2, 0)) > 0:
                acc["add3_unrel/dlo_top2"].append(abs(
                    np.log(ap[top1] / ap[top2]) - np.log(bp[top1] / bp[top2])))

        if "orig" in g and "reorder" in g:
            bp, rp = cand_probs(g["orig"]), cand_probs(g["reorder"])
            diffs = [abs(rp[c] - bp[c]) for c in rp if c in bp]
            if diffs:
                acc["reorder/max_dp"].append(max(diffs))

        if "add_neardup" in g:
            i = g["add_neardup"]
            ex = examples[i]
            gold, dup = ex["candidates"][ex["label"]], ex["candidates"][-1]
            cp = cand_probs(i)
            if gold in cp and dup in cp:
                acc["add_neardup/slot_gap"].append(abs(cp[gold] - cp[dup]))
                if "orig" in g:
                    bp = cand_probs(g["orig"])
                    if gold in bp:
                        acc["add_neardup/cluster_mass_err"].append(abs((cp[gold] + cp[dup]) - bp[gold]))

    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def ksweep(probs, examples) -> dict:
    """data_v4 ksweep_* sets (see data.py build_ksweep): examples carry meta.K/meta.u/meta.present.
    Returns P(null | absent)@K, P(null | present)@K, acc@K, null-detection AUROC@K (present vs.
    absent, scored by P(null)), and p_null_absent_range = max-min of P(null | absent) over K
    (data_v4's target: this should be small -- p_null shouldn't be a function of K)."""
    probs = np.asarray(probs, dtype=np.float64)
    by_k = defaultdict(lambda: {"absent": [], "present": [], "acc": []})
    for i, ex in enumerate(examples):
        m = ex.get("meta", {})
        if "K" not in m or "present" not in m:
            continue
        d = by_k[m["K"]]
        p_null = float(probs[i, -1])
        if m["present"]:
            d["present"].append(p_null)
            K = len(ex["candidates"])
            lbl = ex.get("label")
            if isinstance(lbl, int) and lbl >= 0 and K > 0:
                d["acc"].append(float(int(probs[i, :K].argmax()) == lbl))
        else:
            d["absent"].append(p_null)

    out, absent_means = {}, {}
    for k, d in by_k.items():
        if d["absent"]:
            absent_means[k] = out[f"K{k}/p_null_absent"] = float(np.mean(d["absent"]))
        if d["present"]:
            out[f"K{k}/p_null_present"] = float(np.mean(d["present"]))
        if d["acc"]:
            out[f"K{k}/acc"] = float(np.mean(d["acc"]))
        if d["absent"] and d["present"]:
            from sklearn.metrics import roc_auc_score
            scores = d["absent"] + d["present"]
            labels = [1] * len(d["absent"]) + [0] * len(d["present"])
            out[f"K{k}/auroc"] = float(roc_auc_score(labels, scores))
    if absent_means:
        out["p_null_absent_range"] = float(max(absent_means.values()) - min(absent_means.values()))
    return out


if __name__ == "__main__":
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    target = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    label = [0, -1]
    m = summarize(probs, target, label)
    assert abs(m["nll"] - 0.2899) < 1e-3, m["nll"]
    assert abs(m["brier"] - 0.10) < 1e-6, m["brier"]
    assert m["acc"] == 1.0, m["acc"]
    assert m["acc_k"] == 1.0, m["acc_k"]
    assert m["conf_wrong"] == 0.0, m["conf_wrong"]
    assert abs(m["auroc_null"] - 1.0) < 1e-9, m["auroc_null"]
    print("metrics.py self-check passed:", m)

    # choice_set_effects: a base row (a=.2,b=.3,c=.5, gold=c) and its 5 variants, built so a
    # "perfect" (IIA-respecting) scorer gives dlo_top2=0, mass_err=0, slot_gap=0, max_dp=0,
    # cluster_mass_err=0, and the deliberately-off `dup` row (mass grew from .5 to .8) shows up.
    cse_ex = [
        {"candidates": ["a", "b", "c"], "label": 2, "meta": {"pair": 0, "variant": "base"}},
        {"candidates": ["a", "b", "c", "irr"], "meta": {"pair": 0, "variant": "add_irr"}},
        {"candidates": ["a", "b"], "meta": {"pair": 0, "variant": "remove_gold"}},
        {"candidates": ["a", "b", "c", "c"], "meta": {"pair": 0, "variant": "dup", "src": "c"}},
        {"candidates": ["c", "a", "b"], "meta": {"pair": 0, "variant": "reorder"}},
        {"candidates": ["a", "b", "c", "A"], "meta": {"pair": 0, "variant": "near_dup", "src": "a"}},
    ]
    cse_probs = np.array([
        [.2, .3, .5, 0, 0],
        [.2, .3, .5, 0, 0],
        [.05, .05, 0, 0, .9],
        [.1, .1, .4, .4, 0],
        [.5, .2, .3, 0, 0],
        [.15, .3, .5, .05, 0],
    ])
    cse = choice_set_effects(cse_probs, cse_ex)
    assert abs(cse["add_irr/p_irr"]) < 1e-9, cse
    assert abs(cse["add_irr/dlo_top2"]) < 1e-9, cse
    assert abs(cse["remove_gold/dp_null"] - 0.9) < 1e-9, cse
    assert abs(cse["dup/mass_err"] - 0.3) < 1e-9, cse
    assert abs(cse["dup/slot_gap"]) < 1e-9, cse
    assert cse["dup/acc_dedup"] == 1.0, cse
    assert abs(cse["reorder/max_dp"]) < 1e-9, cse
    assert abs(cse["near_dup/cluster_mass_err"]) < 1e-9, cse
    print("choice_set_effects self-check passed:", cse)

    # ksweep: K=2 {present acc=1, p_null~0} vs {absent, p_null~1} perfectly separated (auroc=1);
    # K=5 present/absent p_null both ~0.5 (auroc~chance) -> p_null_absent_range should pick up
    # the K=2 vs K=5 gap in P(null | absent).
    ks_ex = [
        {"candidates": ["a", "b"], "label": 0, "meta": {"K": 2, "u": 0, "present": True}},
        {"candidates": ["a", "b"], "label": 0, "meta": {"K": 2, "u": 1, "present": True}},
        {"candidates": ["a", "b"], "meta": {"K": 2, "u": 2, "present": False}},
        {"candidates": list("abcde"), "label": 0, "meta": {"K": 5, "u": 3, "present": True}},
        {"candidates": list("abcde"), "meta": {"K": 5, "u": 4, "present": False}},
    ]
    ks_probs = np.array([
        [.9, .1, 0, 0, 0, 0],
        [.9, .1, 0, 0, 0, 0],
        [.05, .05, 0, 0, 0, .9],
        [.5, .1, .1, .1, .1, .1],
        [.1, .1, .1, .1, .1, .5],
    ])
    ks = ksweep(ks_probs, ks_ex)
    assert abs(ks["K2/p_null_absent"] - 0.9) < 1e-9, ks
    assert abs(ks["K2/p_null_present"]) < 1e-9, ks
    assert ks["K2/acc"] == 1.0, ks
    assert abs(ks["K2/auroc"] - 1.0) < 1e-9, ks
    assert abs(ks["K5/p_null_absent"] - 0.5) < 1e-9, ks
    assert abs(ks["p_null_absent_range"] - 0.4) < 1e-9, ks
    print("ksweep self-check passed:", ks)

    # counterfactual: one MMLU-cf question, per-candidate logits fixed so a softmax over
    # any subset preserves ratios between candidates present in both variants -- the
    # IIA-consistent case dlo_top2/max_dp should score as exactly 0. null_logit very
    # negative -> p_null ~ 0, so a teacher fed the same per-candidate numbers as the
    # student should give ~0 KL.
    def cf_softmax(logit_map, cands, null_logit=-50.0):
        vals = np.array([logit_map[c] for c in cands] + [null_logit])
        e = np.exp(vals - vals.max())
        return e / e.sum()

    cf_logit = {"a": 2.0, "b": 1.0, "c": 3.0, "d": 0.0,
                "irr1": -1.0, "irr2": -1.0, "irr3": -1.0, "C.": 3.0, "x": -1.0}
    cf_variants = {
        "orig": (["a", "b", "c", "d"], 2),
        "add3_unrel": (["a", "b", "c", "d", "irr1", "irr2", "irr3"], 2),
        "reorder": (["c", "a", "d", "b"], 0),
        "add_neardup": (["a", "b", "c", "d", "C."], 2),
        "remove3": (["c", "d"], 0),
        "replace_hard": (["a", "b", "c", "x"], 2),
    }
    kmax = max(len(cands) for cands, _ in cf_variants.values())
    cf_ex, cf_rows = [], []
    for variant, (cands, lbl) in cf_variants.items():
        cf_ex.append({"candidates": cands, "label": lbl, "meta": {"qid": 0, "variant": variant}})
        p = cf_softmax(cf_logit, cands)
        row = np.zeros(kmax + 1)
        row[:len(cands)] = p[:-1]
        row[-1] = p[-1]
        cf_rows.append(row)
    cf_probs = np.array(cf_rows)
    orig_cands, _ = cf_variants["orig"]
    teacher_probs = {0: dict(zip(orig_cands, cf_softmax(cf_logit, orig_cands)[:-1]))}
    cf = counterfactual(cf_probs, cf_ex, teacher_probs=teacher_probs)
    assert all(np.isfinite(v) for v in cf.values()), cf
    assert abs(cf["add3_unrel/dlo_top2"]) < 1e-9, cf
    assert abs(cf["reorder/max_dp"]) < 1e-9, cf
    assert cf["orig/acc"] == 1.0 and cf["orig/among_k"] == 1.0, cf
    assert cf["orig/kl_teacher"] < 1e-9, cf
    print("counterfactual self-check passed:", cf)


def rubric_flip(probs, examples) -> dict:
    """data_wf wf_rubric_flip* (scripts/workflow_corpus.py): rows come in pairs sharing meta.flip_pair --
    same state, same candidates, two rubrics with different gold (y1 != y2). probs: [N, Kmax+1], null
    last, row order == examples order; argmax is over the candidates only (JevBench conditions on non-∅).
    flip_rate = P[argmax(x,r1,A) != argmax(x,r2,A) | y1 != y2] -- a rubric-blind model scores ~0;
    both_correct = fraction of pairs where both rows are right; acc = plain per-row accuracy."""
    probs = np.asarray(probs, dtype=np.float64)
    pairs = defaultdict(list)
    for i, ex in enumerate(examples):
        if ex.get("meta", {}).get("flip_pair") is not None:
            pairs[ex["meta"]["flip_pair"]].append(i)
    flips, both, accs = [], [], []
    for idx in pairs.values():
        if len(idx) != 2:
            continue
        a, b = idx
        if examples[a]["label"] == examples[b]["label"]:
            continue
        pa, pb = (int(probs[i, :len(examples[i]["candidates"])].argmax()) for i in (a, b))
        ca, cb = pa == examples[a]["label"], pb == examples[b]["label"]
        flips.append(float(pa != pb)); both.append(float(ca and cb)); accs += [float(ca), float(cb)]
    return {"n_pairs": len(flips), "flip_rate": float(np.mean(flips)) if flips else float("nan"),
            "both_correct": float(np.mean(both)) if both else float("nan"),
            "acc": float(np.mean(accs)) if accs else float("nan")}


def ordinal_metrics(probs, examples) -> dict:
    """PLAN7 track C (score_A_kway / score_B_smooth / score_C_cumlink): ordinal MAE (|argmax
    level - gold level|, hard prediction) and expected-score error (|E[level] - gold|, E under
    the candidate-conditional distribution renormalised off ∅) for meta.qtype == "score" rows,
    where candidate index == ordinal level (as rendered). probs: [N, Kmax+1], null last, row
    order == examples order; only rows with a valid hard candidate label contribute."""
    probs = np.asarray(probs, dtype=np.float64)
    mae, exp_err = [], []
    for i, ex in enumerate(examples):
        lbl = ex.get("label")
        if not (isinstance(lbl, int) and lbl >= 0):
            continue
        K = len(ex["candidates"])
        pc = probs[i, :K]
        total = pc.sum()
        if total <= 0:
            continue
        pc = pc / total
        mae.append(abs(int(pc.argmax()) - lbl))
        exp_err.append(abs(float((np.arange(K) * pc).sum()) - lbl))
    return {"ordinal_mae": float(np.mean(mae)) if mae else float("nan"),
            "exp_score_err": float(np.mean(exp_err)) if exp_err else float("nan"),
            "n": len(mae)}


def noul_reversed_check(probs_a, examples_a, probs_b, examples_b) -> dict:
    """--noul_head bern positional-invariance control (PLAN7 track C noul_B_bern): examples_b is
    examples_a with candidate order reversed (["yes","no"] <-> ["no","yes"]); the Bernoulli head
    reads P(yes) off h_D alone (no candidates rendered), so it must give the identical P(yes)
    either way -- both diffs below should be ~0 for a correct implementation (the K-way Choice
    control is expected to differ, since it does read the rendered option text)."""
    diffs = []
    for pa, ea, pb, eb in zip(probs_a, examples_a, probs_b, examples_b):
        ya = next(j for j, c in enumerate(ea["candidates"]) if c.strip().lower() == "yes")
        yb = next(j for j, c in enumerate(eb["candidates"]) if c.strip().lower() == "yes")
        diffs.append(abs(float(pa[ya]) - float(pb[yb])))
    return {"max_abs_diff": float(np.max(diffs)) if diffs else float("nan"),
            "mean_abs_diff": float(np.mean(diffs)) if diffs else float("nan")}
