"""Train the decision tower + LoRA backbone and evaluate it.

uv run train.py --name X [--backbone Qwen/Qwen3-1.7B-Base] [--lora_layers 8] [--lora_r 16]
                 [--steps 12000] [--bs 64] [--lr 3e-4] [--lora_lr 1e-4] [--seed 0]
                 [--no_hybrid] [--no_cand_null] [--no_null] [--null softmax|factored] [--hard_only] [--mix full|nlionly]
                 [--eval_every 4000] [--val_every 1000] [--ckpt_every 2000] [--eval_bs 128]
                 [--device auto] [--wandb] [--hf_repo guychuk/pcdm-runs] [--data data]
                 [--smoke] [--eval_only] [--dump_logits DIR]
                 [--head mlp|z1|zr|zr_set] [--z_dim 128] [--z_probes 8] [--tiny_layers 2]
                 [--cand_encoder backbone|qwen3emb|tiny]
                 [--readout energy|mcq|native] [--nc_head n2|n3|n2n3] [--no_shuffle]
"""
import argparse
import sys
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from encode import Backbone, FeatureCache, pick_device
from model import DecisionModel, decision_loss, collate, run_batch, split_joint
from mcq import MCQHead, collate_mcq, run_batch_mcq
from native import NativeHead, run_batch_native, native_features, shuffle_options
from metrics import summarize, choice_set_effects, ksweep, counterfactual
import data as data_mod

FAMILY = getattr(data_mod, "FAMILY", None)
NLI_ONLY_TASKS = {"snli", "mnli", "anli", "snli_soft", "unli", "boolq", "squad", "clinc"}

SMOKE_STATES = [
    "A man is playing guitar on a busy city street while people walk by.",
    "The research team published their findings after months of careful analysis.",
    "A dog is chasing a ball across a green park near the river.",
    "The customer asked to cancel their subscription due to billing issues.",
    "Scientists discovered a new species of frog in the rainforest.",
]
SMOKE_QUERIES = [
    "what is the relationship between the state and the hypothesis?",
    "what is the intent behind this request?",
    "what topic does this text discuss?",
]
SMOKE_LABEL_POOL = {
    "snli": ["entailment", "neutral", "contradiction"],
    "clinc": ["cancel subscription", "book a flight", "check account balance", "play music", "set an alarm"],
}


def family_of(task):
    return FAMILY.get(task, task) if FAMILY else task


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_hard_only(examples):
    out = []
    for ex in examples:
        ex = dict(ex)
        if ex["task"] == "unli":
            ex["p_null"] = float(round(ex["p_null"]))
            ex["target"] = [1.0]
        else:
            gold = max(range(len(ex["target"])), key=lambda j: ex["target"][j])
            ex["target"] = [1.0 if j == gold else 0.0 for j in range(len(ex["target"]))]
        out.append(ex)
    return out


def apply_no_null(examples):
    out = []
    for ex in examples:
        if ex["p_null"] == 1:
            continue
        ex = dict(ex)
        ex["p_null"] = 0.0
        out.append(ex)
    return out


def make_smoke_examples(n, seed):
    rng = random.Random(seed)
    tasks = ["snli", "unli", "clinc"]
    examples = []
    for i in range(n):
        task = rng.choice(tasks)
        state = f"{rng.choice(SMOKE_STATES)} ({i})"
        query = rng.choice(SMOKE_QUERIES)
        if task == "unli":
            p = rng.random()
            examples.append({
                "state": state, "query": query, "candidates": ["the hypothesis is plausible"],
                "target": [1.0], "p_null": 1 - p, "task": task, "label": None,
            })
        else:
            pool = SMOKE_LABEL_POOL[task]
            k = rng.randint(2, len(pool))
            candidates = rng.sample(pool, k)
            gold = rng.randrange(k)
            is_null = rng.random() < 0.25
            target = [1.0 if j == gold else 0.0 for j in range(k)]
            examples.append({
                "state": state, "query": query, "candidates": candidates, "target": target,
                "p_null": 1.0 if is_null else 0.0, "task": task, "label": -1 if is_null else gold,
            })
    return examples


def prefetch(gen, cache, depth=3, readout="energy", shuffle=False, seed=0):
    """Yield (examples, collate(cache, examples)) with the collate done on a worker thread.
    shuffle (mcq/native readouts, PLAN4 sec 7): re-order every example's options first.
    ponytail: the shuffle rng is not checkpointed -- a resumed run re-draws option orders."""
    import queue, threading
    q = queue.Queue(maxsize=depth)
    rng = random.Random(seed)
    def work():
        try:
            for ex in gen:
                if shuffle:
                    ex = shuffle_options(ex, rng)
                batch = collate_mcq(ex) if readout in ("mcq", "native") else collate(cache, ex)
                q.put((ex, batch))
        except BaseException as e:  # a collate error must kill the run, not leave the main loop waiting at 0% GPU
            q.put(e)
    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if isinstance(item, BaseException):
            raise item
        yield item


def group_units(examples):
    """Rows sharing meta.ms_group (scripts/multiset.py) form one unit, in first-appearance
    order; every other row is its own unit."""
    units, by_group = [], {}
    for ex in examples:
        g = ex.get("meta", {}).get("ms_group")
        if g is None:
            units.append([ex])
        elif g in by_group:
            by_group[g].append(ex)
        else:
            by_group[g] = [ex]
            units.append(by_group[g])
    return units


def data_generator(examples, bs, seed):
    """Infinite stream of length-bucketed batches: shuffle -> chunk (bs*50 units) -> sort by
    len(state) -> pack units into batches of <= bs rows -> shuffle batch order. Cycles forever.
    PLAN3 E3-ms: a multiset group (group_units) never straddles a batch, so decision_loss can
    pair every variant with its orig row; with no groups this is exactly the old per-row stream.
    ponytail: a group larger than bs becomes one oversize batch (cap --variants, not this)."""
    rng = random.Random(seed)
    units = group_units(examples)
    chunk_size = bs * 50
    while True:
        order = units[:]
        rng.shuffle(order)
        for i in range(0, len(order), chunk_size):
            chunk = sorted(order[i:i + chunk_size], key=lambda u: len(u[0]["state"]))
            batches, cur = [], []
            for u in chunk:
                if cur and len(cur) + len(u) > bs:
                    batches.append(cur)
                    cur = []
                cur.extend(u)
            if cur:
                batches.append(cur)
            rng.shuffle(batches)
            for b in batches:
                yield b


def add_extra_data(args, train_examples, eval_sets):
    """PLAN3 E3: mix --extra_data corpora (e.g. data_kb, closed-book MCQ with teacher labels) into
    training in place; each dir's val becomes an eval set (<dir>_val) so distillation quality is
    reported separately. multiset groups (meta.ms_group) are namespaced by dir so two expanded
    corpora never pair across files."""
    for extra in (args.extra_data.split(",") if args.extra_data else []):
        ed = Path(extra)
        rows = load_jsonl(ed / "train.jsonl")
        for ex in rows:
            if "ms_group" in ex.get("meta", {}):
                ex["meta"]["ms_group"] = f"{ed.name}/{ex['meta']['ms_group']}"
        train_examples += rows
        if (ed / "val.jsonl").exists():
            eval_sets[f"{ed.name}_val"] = load_jsonl(ed / "val.jsonl")
        eval_sets.update({Path(p).stem: load_jsonl(p) for p in sorted(glob.glob(str(ed / "eval" / "*.jsonl")))})


def load_run_data(args, backbone):
    # readout=mcq never touches the cached frozen-space candidate features (candidates
    # are read straight into the prompt text) -> skip building/loading that cache entirely.
    # cand_encoder=tiny caches token ids only (TokenCandCache, built below; the encoder is
    # trained inside the model) -> no frozen-feature cache either.
    # readout=native needs the embedding VecCache only for n2 heads (n3 pools the suffix itself).
    build_cache_ = (args.readout == "energy" and args.cand_encoder != "tiny") or \
        (args.readout == "native" and "n2" in args.nc_head)
    use_vec = args.cand_encoder == "qwen3emb"
    if args.smoke:
        train_examples = make_smoke_examples(64, seed=0)
        val_examples = make_smoke_examples(16, seed=1)
        eval_sets = {"smoke_eval": make_smoke_examples(16, seed=2)}
        add_extra_data(args, train_examples, eval_sets)  # e.g. a tiny multiset file for an E3-ms smoke
        cache = None
        if build_cache_:
            all_ex = train_examples + val_examples + eval_sets["smoke_eval"]
            if use_vec:
                from encode import EmbedEncoder, VecCache, CAND_RENDER
                encoder = EmbedEncoder(device=backbone.device)
                cache = VecCache(device="cpu")
                cache.add(encoder, sorted({c for ex in all_ex for c in ex["candidates"]}), render=CAND_RENDER, max_len=32)
                cache.add(encoder, sorted({ex["state"] for ex in all_ex}), max_len=128)
                cache.add(encoder, sorted({ex["query"] for ex in all_ex}), max_len=32)
            else:
                cache = FeatureCache(device="cpu")
                cache.add(backbone, sorted({c for ex in all_ex for c in ex["candidates"]}), max_len=16)
    else:
        data_dir = Path(args.data)
        train_examples = load_jsonl(data_dir / "train.jsonl")
        val_examples = load_jsonl(data_dir / "val.jsonl")
        eval_sets = {Path(p).stem: load_jsonl(p) for p in sorted(glob.glob(str(data_dir / "eval" / "*.jsonl")))}
        add_extra_data(args, train_examples, eval_sets)

        cache = None
        if build_cache_ and use_vec:
            from encode import build_vec_cache, VecCache
            cache_path = data_dir / "veccache_qwen3emb.pt"
            if not cache_path.exists():
                build_vec_cache(str(data_dir), str(cache_path))
            cache = VecCache.load(str(cache_path), "cpu")

            # states/queries/candidates all live in this one cache -> top up with everything,
            # same "may predate new rows" reasoning as the frozen-feature cache below.
            # ... including --extra_data rows, which build_vec_cache(data_dir) would never see: embed
            # exactly the missing strings per category (candidates are rendered, states/queries raw).
            cands, states, queries = set(), set(), set()
            for exs in (train_examples, val_examples, *eval_sets.values()):
                for ex in exs:
                    cands.update(ex["candidates"]); states.add(ex["state"]); queries.add(ex["query"])
            have = cache.index.keys()
            missing = [sorted(s - have) for s in (cands, states, queries)]
            if any(missing):
                from encode import EmbedEncoder, CAND_RENDER
                enc = EmbedEncoder(device=backbone.device)
                cache.add(enc, missing[0], render=CAND_RENDER, max_len=32)
                cache.add(enc, missing[1], max_len=128)
                cache.add(enc, missing[2], max_len=32)
                cache.save(str(cache_path))
                print(f"vec cache topped up: +{len(missing[0])} candidates, +{len(missing[1])} states, +{len(missing[2])} queries")
                del enc
        elif build_cache_:
            # candidate features live in the backbone's frozen layer-L space -> one cache per backbone
            cache_path = data_dir / f"cache_{args.backbone.split('/')[-1]}_L{backbone.split_layer}.pt"
            if not cache_path.exists():
                from encode import build_cache
                build_cache(str(data_dir), str(cache_path), backbone)
            cache = FeatureCache.load(str(cache_path), "cpu")

            # the on-disk cache may predate rows added to train/val/eval since it was built;
            # top it up (build_cache merges into the existing file) rather than KeyError-ing
            # the first time collate() hits a candidate string it hasn't seen.
            all_cands = {c for ex in train_examples + val_examples for c in ex["candidates"]}
            all_cands.update(c for exs in eval_sets.values() for ex in exs for c in ex["candidates"])
            if all_cands - cache.index.keys():
                from encode import build_cache
                build_cache(str(data_dir), str(cache_path), backbone)
                cache = FeatureCache.load(str(cache_path), "cpu")

        if args.eval_limit:  # smoke runs: first N rows of every eval set
            eval_sets = {k: v[:args.eval_limit] for k, v in eval_sets.items()}
            val_examples = val_examples[:args.eval_limit]

    if args.readout == "energy" and args.cand_encoder == "tiny":
        from encode import TokenCandCache
        cache = TokenCandCache(backbone)
        cache.add(sorted({c for exs in (train_examples, val_examples, *eval_sets.values()) for ex in exs for c in ex["candidates"]}))

    if args.mix == "nlionly":
        train_examples = [ex for ex in train_examples if ex["task"] in NLI_ONLY_TASKS]
    if args.hard_only:
        train_examples = apply_hard_only(train_examples)
        val_examples = apply_hard_only(val_examples)
    if args.no_null:
        train_examples = apply_no_null(train_examples)
        val_examples = apply_no_null(val_examples)
    return cache, train_examples, val_examples, eval_sets


def probs_from_logits(logits, cmask, T=1.0, null="softmax"):
    null_col = torch.ones(cmask.shape[0], 1, dtype=torch.bool, device=cmask.device)
    valid = torch.cat([cmask, null_col], dim=1)
    if null == "factored" and T != 1.0:
        # factored null composes p_j = softmax(s/T) *inside* the model (DecisionModel.
        # _factored_logits) -- T can't be applied to the already-composed logits like the
        # plain-softmax null below. But r=P(null) only sees untempered s-statistics, so it's
        # exactly T-invariant and recoverable unchanged from these (T=1) logits; only the K
        # candidate columns need re-softmaxing at the new T. This algebraic identity lets the
        # T grid search in fit_temperature reuse one forward pass instead of one per T.
        K = logits.shape[-1] - 1
        log_r = logits[:, K]
        log_1mr = torch.log1p(-log_r.exp().clamp(max=1 - 1e-7))
        adj = (logits[:, :K] - log_1mr.unsqueeze(-1)).masked_fill(~cmask, float("-inf"))
        logp = torch.log_softmax(adj / T, dim=-1)
        cand = (log_1mr.unsqueeze(-1) + logp).exp().masked_fill(~cmask, 0.0)
        return torch.cat([cand, log_r.exp().unsqueeze(-1)], dim=-1)
    masked = logits.masked_fill(~valid, float("-inf"))
    return torch.softmax(masked / T, dim=-1)


def full_target_from(target, p_null):
    return torch.cat([(1 - p_null).unsqueeze(-1) * target, p_null.unsqueeze(-1)], dim=-1)


def run_readout(readout, backbone, model, batch, examples, joint=False, vec_cache=None, shots=0):
    """One forward through whichever readout is configured -> logits [B, Kmax+1]."""
    if readout == "mcq":
        return run_batch_mcq(backbone, batch, examples, shots=shots)
    if readout == "native":
        return run_batch_native(backbone, model, batch, examples, vec_cache=vec_cache)
    return run_batch(backbone, model, batch, joint=joint, vec_cache=vec_cache)


def forward_batches(backbone, model, cache, examples, bs, joint=False, readout="energy", vec_cache=None, shots=0):
    """(logits, cmask, target, p_null, batch) per batch, no grad, single forward each."""
    model.eval(); backbone.eval()  # LoRALinear modules default to train mode (LoRA dropout active at eval otherwise)
    out = []
    with torch.inference_mode():
        for i in range(0, len(examples), bs):
            ex_batch = examples[i:i + bs]
            batch = collate_mcq(ex_batch) if readout in ("mcq", "native") else collate(cache, ex_batch)
            logits = run_readout(readout, backbone, model, batch, ex_batch, joint=joint, vec_cache=vec_cache, shots=shots)
            # run_batch only moves state/query/C to backbone.device internally; target/p_null/cmask
            # stay on the collate()-produced CPU tensors, so callers must move them themselves.
            dev = logits.device
            out.append((logits, batch["cmask"].to(dev), batch["target"].to(dev), batch["p_null"].to(dev), ex_batch))
    model.train(); backbone.train()  # Qwen3 has no dropout, so train mode only re-enables LoRA dropout
    return out


def eval_val_loss(backbone, model, cache, examples, bs, joint=False, readout="energy", vec_cache=None, shots=0):
    total, n = 0.0, 0
    for logits, cmask, target, p_null, ex_batch in forward_batches(backbone, model, cache, examples, bs,
                                                                     joint=joint, readout=readout, vec_cache=vec_cache, shots=shots):
        loss = decision_loss(logits, target, p_null, cmask)
        total += loss.item() * len(ex_batch)
        n += len(ex_batch)
    return total / max(n, 1)


def fit_temperature(backbone, model, cache, val_examples, bs, joint=False, readout="energy", vec_cache=None, shots=0):
    # ponytail: this doesn't set model.temperature even though DecisionModel has the attribute
    # (used by null="factored"'s forward composition) -- doing so would leak into eval_dataset's
    # own forward pass right after (which wants T=1 cached logits to sweep both raw and scaled
    # metrics via probs_from_logits) and into the next training step's run_batch (which doesn't
    # go through forward_batches at all). T stays a reporting-time overlay; a deployment script
    # that wants it baked into forward() can set model.temperature = fit_temperature(...)[0] itself.
    null_mode = getattr(model, "null", "softmax")
    batches = forward_batches(backbone, model, cache, val_examples, bs, joint=joint, readout=readout, vec_cache=vec_cache, shots=shots)
    best_T, best_nll = 1.0, float("inf")
    for T in np.geomspace(0.1, 10, 60):
        total, n = 0.0, 0
        for logits, cmask, target, p_null, _ in batches:
            probs = probs_from_logits(logits, cmask, T, null=null_mode)
            tgt = full_target_from(target, p_null)
            nll = -(tgt * torch.log(probs.clamp_min(1e-12))).sum(-1)
            total += nll.sum().item()
            n += nll.shape[0]
        avg = total / n
        if avg < best_nll:
            best_nll, best_T = avg, float(T)
    return best_T, best_nll


def eval_dataset(backbone, model, cache, examples, bs, Ts=(1.0,), joint=False, readout="energy", vec_cache=None, shots=0):
    null_mode = getattr(model, "null", "softmax")
    n = len(examples)
    kmax = max(len(ex["candidates"]) for ex in examples)
    probs_all = {T: np.zeros((n, kmax + 1)) for T in Ts}
    target_all = np.zeros((n, kmax + 1))
    label_all = np.full(n, np.nan)
    idx = 0
    for logits, cmask, target, p_null, ex_batch in forward_batches(backbone, model, cache, examples, bs,
                                                                     joint=joint, readout=readout, vec_cache=vec_cache,
                                                                     shots=shots):
        tgt = full_target_from(target, p_null).cpu().numpy()
        k_local = target.shape[1]
        b = len(ex_batch)
        for T in Ts:
            probs = probs_from_logits(logits, cmask, T, null=null_mode).cpu().numpy()
            probs_all[T][idx:idx + b, :k_local] = probs[:, :k_local]
            probs_all[T][idx:idx + b, -1] = probs[:, -1]
        target_all[idx:idx + b, :k_local] = tgt[:, :k_local]
        target_all[idx:idx + b, -1] = tgt[:, -1]
        for j, ex in enumerate(ex_batch):
            label_all[idx + j] = ex["label"] if ex.get("label") is not None else np.nan
        idx += b
    return probs_all, target_all, label_all


def dump_logits(backbone, model, cache, examples, bs, joint=False, readout="energy", vec_cache=None, shots=0):
    """Like eval_dataset but keeps raw (pre-softmax, pre-T) logits -- padded with finfo.min --
    plus the real candidate count K per row, for scripts/null_bias.py to fit a post-hoc
    K-aware null-bias correction (s_null' = s_null + alpha*log(K) + beta) without retraining."""
    n = len(examples)
    kmax = max(len(ex["candidates"]) for ex in examples)
    logits_all = np.full((n, kmax + 1), np.finfo(np.float64).min)
    target_all = np.zeros((n, kmax + 1))
    label_all = np.full(n, np.nan)
    K_all = np.zeros(n, dtype=np.int64)
    idx = 0
    for logits, cmask, target, p_null, ex_batch in forward_batches(backbone, model, cache, examples, bs,
                                                                     joint=joint, readout=readout, vec_cache=vec_cache,
                                                                     shots=shots):
        tgt = full_target_from(target, p_null).cpu().numpy()
        lg = logits.cpu().numpy()
        k_local = target.shape[1]
        b = len(ex_batch)
        logits_all[idx:idx + b, :k_local] = lg[:, :k_local]
        logits_all[idx:idx + b, -1] = lg[:, -1]
        target_all[idx:idx + b, :k_local] = tgt[:, :k_local]
        target_all[idx:idx + b, -1] = tgt[:, -1]
        K_all[idx:idx + b] = cmask.sum(-1).cpu().numpy()
        for j, ex in enumerate(ex_batch):
            label_all[idx + j] = ex["label"] if ex.get("label") is not None else np.nan
        idx += b
    return logits_all, target_all, label_all, K_all


def dump_eval_logits(args, backbone, model, cache, val_examples, eval_sets):
    """--dump_logits DIR: for val and every eval set, write DIR/<set>.npz (logits/target/label/K)
    and DIR/<set>.meta.json (the ordered examples, minus the "state" text, for choice_set_effects
    /ksweep which read meta.pair/variant/K/u/present and candidates off them)."""
    vec_cache = cache if args.cand_encoder == "qwen3emb" else None
    out_dir = Path(args.dump_logits)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in {"val": val_examples, **eval_sets}.items():
        logits, target, label, K = dump_logits(backbone, model, cache, examples, args.eval_bs,
                                                 joint=args.joint, readout=args.readout, vec_cache=vec_cache,
                                                 shots=args.shots)
        np.savez(out_dir / f"{name}.npz", logits=logits, target=target, label=label, K=K)
        meta = [{k: v for k, v in ex.items() if k != "state"} for ex in examples]
        with open(out_dir / f"{name}.meta.json", "w") as f:
            json.dump(meta, f)


def run_full_eval(args, backbone, model, cache, val_examples, eval_sets, val_nll):
    vec_cache = cache if args.cand_encoder == "qwen3emb" else None
    best_T, _ = fit_temperature(backbone, model, cache, val_examples, args.eval_bs, joint=args.joint,
                                 readout=args.readout, vec_cache=vec_cache, shots=args.shots)
    results = {"T": best_T, "val_nll": val_nll, "args": vars(args), "eval": {}}  # args: so a run is reproducible from results.json alone
    for name, examples in eval_sets.items():
        probs, target, label = eval_dataset(backbone, model, cache, examples, args.eval_bs, Ts=(1.0, best_T),
                                             joint=args.joint, readout=args.readout, vec_cache=vec_cache, shots=args.shots)
        raw_m, scaled_m = summarize(probs[1.0], target, label), summarize(probs[best_T], target, label)
        if args.no_null:
            raw_m["auroc_null"] = scaled_m["auroc_null"] = "n/a"
        results["eval"][name] = {"raw": raw_m, "scaled": scaled_m}
        if name.startswith("cse_"):      # paired choice-set-effect battery (data v4)
            results["eval"][name]["cse"] = choice_set_effects(probs[best_T], examples)
        elif name.startswith("ksweep_"):  # P(null) as a function of K
            results["eval"][name]["ksweep"] = ksweep(probs[best_T], examples)
        elif name.startswith("mmlu_cf"):  # PLAN3 E3-cf counterfactual option-set battery (scripts/mmlu_counterfactual.py)
            results["eval"][name]["cf"] = counterfactual(probs[best_T], examples)
    return results


def calibrate_zscore_native(head, model, examples, seed, n=256):
    """NativeHead.mu_h/sd_h from n training rows' decision states, mu_c/sd_c from their pooled
    option spans (valid options + the null line) -- same rogue-dim reasoning as calibrate_zscore."""
    sample = random.Random(seed).sample(examples, min(n, len(examples)))
    hs, cs = [], []
    with torch.inference_mode():
        for i in range(0, len(sample), 32):
            chunk = sample[i:i + 32]
            h, C3, cmask, _ = native_features(head, [ex["state"] for ex in chunk], [ex["query"] for ex in chunk],
                                              [ex["candidates"] for ex in chunk])
            valid = torch.cat([cmask, torch.ones_like(cmask[:, :1])], 1)
            hs.append(h); cs.append(C3[valid])
    model.calibrate(torch.cat(hs), torch.cat(cs))


def calibrate_zscore(backbone, model, examples, cache, seed, joint=False, n=256, vec_cache=None):
    """mu_top/sd_top: n states + n queries (top-layer LoRA-adapted features).
    mu_fz/sd_fz: same states + queries in frozen space, plus 512 cached candidate
    vectors -- C is compared against Uf/Vf in frozen space, so it needs to be covered
    by the same stats (fixes the v1 defect: 512 STATE texts only, candidates never seen).
    vec_cache (--cand_encoder qwen3emb): frozen space IS the embedding cache, so mu_fz/sd_fz
    sample straight from it (states+queries+candidates) instead of a backbone forward pass."""
    rng = random.Random(seed)
    sample = rng.sample(examples, min(n, len(examples)))
    tops, fzs = [], []
    with torch.inference_mode():
        for i in range(0, len(sample), 64):
            chunk = sample[i:i + 64]
            states, queries = [ex["state"] for ex in chunk], [ex["query"] for ex in chunk]
            if joint:
                ids, am, ls, lq = backbone.tokenize_joint(states, queries, 256, 64)
                h_top, h_fz, mask = backbone(ids.to(backbone.device), am.to(backbone.device))
                H, hmask, Q, qmask, Hf, Qf = split_joint(h_top, h_fz, mask, ls, lq)
            else:
                ids_s, am_s = backbone.tokenize(states, 256)
                ids_q, am_q = backbone.tokenize(queries, 64)
                H, Hf, hmask = backbone(ids_s.to(backbone.device), am_s.to(backbone.device))[:3]
                Q, Qf, qmask = backbone(ids_q.to(backbone.device), am_q.to(backbone.device))[:3]
            tops.append(H[hmask]); tops.append(Q[qmask])
            if vec_cache is None:
                fzs.append(Hf[hmask]); fzs.append(Qf[qmask])
    if vec_cache is not None:
        fz_texts = rng.sample(sorted(vec_cache.index), min(n * 2 + 512, len(vec_cache.index)))
        fzs.append(torch.stack([vec_cache.pooled(t) for t in fz_texts]).to(backbone.device))
    else:
        cand_sample = rng.sample(sorted(cache.index), min(512, len(cache.index)))
        fzs.append(torch.stack([cache.pooled(c) for c in cand_sample]).to(backbone.device))
    model.calibrate(torch.cat(tops), torch.cat(fzs))


def device_memory(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated()
    if device == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def per_family_loss(logits, batch, examples):
    with torch.no_grad():
        t = full_target_from(batch["target"], batch["p_null"])
        logp = torch.log_softmax(logits, dim=-1)
        per_ex = (-(t * logp).sum(-1)).tolist()
    fam = {}
    for ex, l in zip(examples, per_ex):
        fam.setdefault(family_of(ex["task"]), []).append(l)
    return {k: sum(v) / len(v) for k, v in fam.items()}


def build_optimizer(model, backbone, args):
    opt = AdamW([
        {"params": model.parameters(), "lr": args.lr, "weight_decay": 0.01},
        {"params": backbone.trainable_parameters(), "lr": args.lora_lr, "weight_decay": 0.0},
    ])
    warmup_steps = min(500, max(1, args.steps // 10))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, args.steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return opt, LambdaLR(opt, lr_lambda)


def rng_state_dict():
    d = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        d["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        d["mps"] = torch.mps.get_rng_state()
    return d


def restore_rng(d):
    # the CPU default generator needs a CPU ByteTensor even if the checkpoint was
    # loaded with map_location="mps"/"cuda" (which moves every tensor in the file).
    torch.set_rng_state(d["torch"].cpu())
    random.setstate(d["python"])
    if "cuda" in d and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in d["cuda"]])
    if "mps" in d and torch.backends.mps.is_available():
        torch.mps.set_rng_state(d["mps"].cpu())


def save_checkpoint(path, model, backbone, step, opt, sched, best_val, args, tower_lora_only=False):
    ckpt = {"tower": model.state_dict(), "lora": backbone.lora_state_dict(), "step": step,
            "best_val": best_val, "args": vars(args)}
    if not tower_lora_only:
        ckpt.update({"opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng_state_dict()})
    torch.save(ckpt, path)


def load_checkpoint(path, model, backbone, opt, sched, device):
    # weights_only=False: trusted, self-produced checkpoint (opt/sched/rng state, not third-party input)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["tower"])
    backbone.load_lora_state_dict(ckpt["lora"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    restore_rng(ckpt["rng"])
    return ckpt["step"], ckpt["best_val"]


def fmt(v, width=8, prec=3):
    if v is None:
        return f"{'-':>{width}}"
    if v == "n/a":
        return f"{'n/a':>{width}}"
    return f"{v:>{width}.{prec}f}"


def print_table(results):
    print(f"\nT={results['T']:.3f}  val_nll={results['val_nll']:.4f}")
    print(f"{'set':<20}{'acc':>8}{'acc_k':>8}{'nll':>8}{'brier':>8}{'ece':>8}{'auroc_null':>12}")
    for name, ev in results["eval"].items():
        m = ev["scaled"]
        print(f"{name:<20}{fmt(m.get('acc'))}{fmt(m.get('acc_k'))}{fmt(m.get('nll'))}"
              f"{fmt(m.get('brier'))}{fmt(m.get('ece'))}{fmt(m.get('auroc_null'), width=12)}")


def init_wandb(args):
    import wandb
    wandb.init(project="pcdm", name=args.name, id=args.name, resume="allow", config=vars(args))
    return wandb


def log_eval_wandb(wb, results, step):
    if not wb:
        return
    log = {"T": results["T"]}
    for name, ev in results["eval"].items():
        for k, v in ev["scaled"].items():
            if v != "n/a":
                log[f"eval/{name}/{k}"] = v
        for k, v in ev["raw"].items():
            if v != "n/a":
                log[f"eval_raw/{name}/{k}"] = v
        for extra in ("cse", "ksweep"):
            for k, v in ev.get(extra, {}).items():
                if isinstance(v, (int, float)):
                    log[f"{extra}/{name}/{k}"] = v
    wb.log(log, step=step)


def finish_wandb(wb, results):
    if not wb:
        return
    cols = ["set", "acc", "acc_k", "nll", "brier", "ece", "auroc_null"]
    num = lambda v: float(v) if isinstance(v, (int, float)) else float("nan")  # wandb.Table needs one type per column
    rows = [[name] + [num(ev["scaled"].get(c)) for c in cols[1:]] for name, ev in results["eval"].items()]
    wb.log({"report": wb.Table(columns=cols, data=rows)})
    wb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--backbone", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--lora_layers", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--bs", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora_lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_hybrid", action="store_true")
    p.add_argument("--no_cand_null", action="store_true")
    p.add_argument("--no_null", action="store_true")
    p.add_argument("--null", choices=["softmax", "factored"], default="softmax",
                    help="factored: r=sigmoid(g(z)) over O(K) candidate-score set statistics, "
                         "P(a_j)=(1-r)*softmax(s)_j, P(null)=r (exact IIA); softmax: current behavior")
    p.add_argument("--hard_only", action="store_true")
    p.add_argument("--mix", choices=["full", "nlionly"], default="full")
    p.add_argument("--eval_every", type=int, default=None)
    p.add_argument("--val_every", type=int, default=None)
    p.add_argument("--ckpt_every", type=int, default=None)
    p.add_argument("--eval_bs", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--hf_repo", default=None)
    p.add_argument("--data", default="data")
    p.add_argument("--extra_data", default=None, help="comma-separated extra data dirs mixed into train (their val/eval become eval sets)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--eval_limit", type=int, default=0, help="cap each eval set (smoke runs)")
    p.add_argument("--tap_layer", type=int, default=0, help="tower memory from layer T (0 = last layer)")
    p.add_argument("--zscore", action="store_true", help="per-dim standardise backbone features (stats from 512 train states)")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--dump_logits", default=None,
                    help="with --eval_only: dump raw per-set logits/target/label/K + meta to DIR "
                         "for scripts/null_bias.py (post-hoc K-aware null-bias fit)")
    p.add_argument("--tower_d", type=int, default=512)
    p.add_argument("--tower_layers", type=int, default=2)
    p.add_argument("--tower_heads", type=int, default=8)
    p.add_argument("--joint", action="store_true", help="encode query as a causal prefix continuation of state (one backbone pass)")
    p.add_argument("--listwise", action="store_true", help="set-attention mixer over [h; candidates] before scoring (zero-init)")
    p.add_argument("--extra_tap", type=int, default=0, help="also concatenate layer E's state onto the top state ([h_E; h_top]); needs --cand_encoder qwen3emb")
    p.add_argument("--init_from", default=None, help="warm-start tower+lora from a checkpoint (tower loaded strict=False, e.g. adding the listwise mixer)")
    p.add_argument("--readout", choices=["energy", "mcq", "native"], default="energy",
                    help="mcq: options enumerated in the suffix, answer read from next-token letter logits (see mcq.py); "
                         "native: same suffix + terminal decision token, direct scorer over candidates (see native.py)")
    p.add_argument("--nc_head", choices=["n2", "n3", "n2n3"], default="n2n3",
                    help="--readout native candidates: n2 = Qwen3-Embedding vectors, n3 = pooled option spans "
                         "from the suffix, n2n3 = both (PLAN4 sec 12)")
    p.add_argument("--no_shuffle", action="store_true",
                    help="mcq/native: keep the given option order in training (default: random order per example per step)")
    p.add_argument("--zero_shot", action="store_true", help="eval only, no checkpoint (e.g. a frozen --lora_r 0 backbone)")
    p.add_argument("--shots", type=int, default=0, help="--readout mcq: prepend N fixed exemplars (mcq.build_shots) so a base model picks up the letter-answer format")
    p.add_argument("--cand_encoder", choices=["backbone", "qwen3emb", "tiny"], default="backbone",
                    help="qwen3emb: candidate/state/query frozen-space features come from a separate "
                         "Qwen3-Embedding-0.6B cache instead of the backbone's own frozen layer; "
                         "tiny: token-level candidates from the backbone's frozen input-embedding table "
                         "+ --tiny_layers trained blocks inside the model (PLAN3 E3; cold-capable)")
    p.add_argument("--head", choices=["mlp", "z1", "zr", "zr_set"], default="mlp",
                    help="PLAN3 E3 students: z1 = one decision vector Z from h, bilinear late interaction; "
                         "zr = R probe vectors over the query tokens + token MaxSim; zr_set = zr + one "
                         "cross-attention from the probes over all candidate tokens (O(K), IIA-breaking)")
    p.add_argument("--z_dim", type=int, default=128, help="d' of the z heads")
    p.add_argument("--z_probes", type=int, default=8, help="R probes for zr/zr_set")
    p.add_argument("--tiny_layers", type=int, default=2, help="--cand_encoder tiny: trained blocks over the embedding table (0 = table only)")
    p.add_argument("--distill_alpha", type=float, default=1.0, help="weight on the existing gold soft-CE")
    p.add_argument("--distill_beta", type=float, default=0.0,
                    help="PLAN3 E3-T: weight on beta*T^2*KL(teacher_T || student_T) for rows with a "
                         "`teacher` field (scripts/teacher_label.py); 0 = off (default, no behavior change)")
    p.add_argument("--distill_T", type=float, default=2.0, help="temperature for the distillation KL term")
    p.add_argument("--delta_gamma", type=float, default=0.0,
                    help="PLAN3 E3-ms: weight on the Huber log-odds-shift term over meta.delta_t triples "
                         "(scripts/ms_targets.py); 0 = off (default, loss unchanged)")
    args = p.parse_args()

    if args.readout == "native":
        # n2 needs the embedding cache; n3 needs nothing beyond the backbone
        args.cand_encoder = "qwen3emb" if "n2" in args.nc_head else "backbone"
    if args.smoke:
        args.backbone, args.lora_layers, args.lora_r = "Qwen/Qwen3-0.6B-Base", 4, 8
        if args.wandb:
            os.environ["WANDB_MODE"] = "offline"
    defaults = {"steps": 20, "bs": 8, "val_every": 10, "eval_every": 20, "ckpt_every": 10} if args.smoke else \
        {"steps": 12000, "bs": 64, "val_every": 1000, "eval_every": 4000, "ckpt_every": 2000}  # 12k = 1 epoch of ~758k
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def main():
    sys.stdout.reconfigure(line_buffering=True)  # live logs when piped/tee'd (pod, nohup)
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path, results_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "results.json"

    ckpt = None
    if args.eval_only and not args.zero_shot:
        # reconstruct the architecture from the checkpoint's own args, not whatever flags
        # this invocation happened to be called with -- otherwise a mismatched
        # --lora_layers/--tap_layer/--joint silently loads weights into the wrong shapes.
        ckpt = torch.load(best_path, map_location=device, weights_only=False)  # trusted, self-produced
        for k in ("backbone", "tap_layer", "lora_layers", "lora_r", "no_hybrid", "no_cand_null", "joint", "tower_d", "tower_layers", "tower_heads", "listwise", "readout", "cand_encoder", "extra_tap", "null", "head", "z_dim", "z_probes", "tiny_layers", "nc_head"):
            if k in ckpt.get("args", {}):
                setattr(args, k, ckpt["args"][k])

    if args.readout == "mcq":
        assert not args.zscore and not args.joint and not args.listwise, \
            "--zscore/--joint/--listwise are energy-only (mcq mode has no tower)"
        backbone = MCQHead(args.backbone, lora_layers=args.lora_layers, lora_r=args.lora_r, device=device,
                           tap_layer=args.tap_layer)
        model = nn.Module()  # no tower: optimiser/checkpoint code below stays generic (empty param group / state dict)
    elif args.readout == "native":
        assert not args.joint and not args.listwise and not args.shots, "--joint/--listwise/--shots are not native options"
        backbone = MCQHead(args.backbone, lora_layers=args.lora_layers, lora_r=args.lora_r, device=device,
                           tap_layer=args.tap_layer)  # same backbone + LoRA as mcq; its lm_head is simply unused
        model = NativeHead(backbone.backbone.d, nc_head=args.nc_head, null=args.null).to(device)
    else:
        assert not args.extra_tap or args.cand_encoder == "qwen3emb", "--extra_tap doubles the top width; candidates must come from the embedder"
        backbone = Backbone(args.backbone, lora_layers=args.lora_layers, lora_r=args.lora_r, device=device,
                            tap_layer=args.tap_layer, extra_tap=args.extra_tap)
        from encode import EMBED_DIM
        d_cand = {"qwen3emb": EMBED_DIM, "tiny": backbone.model.config.hidden_size}.get(args.cand_encoder)
        model = DecisionModel(d_in=backbone.d, d=args.tower_d, num_layers=args.tower_layers, nhead=args.tower_heads,
                              dim_feedforward=2 * args.tower_d, no_null=args.no_null, cand_null=not args.no_cand_null,
                              hybrid=not args.no_hybrid, mps_safe=(device == "mps"), listwise=args.listwise,
                              d_cand=d_cand, null=args.null, head=args.head, z_dim=args.z_dim, z_probes=args.z_probes,
                              tiny_layers=args.tiny_layers if args.cand_encoder == "tiny" else 0).to(device)
    cache, train_examples, val_examples, eval_sets = load_run_data(args, backbone)
    vec_cache = cache if args.cand_encoder == "qwen3emb" else None
    if args.zscore and args.readout == "native":
        calibrate_zscore_native(backbone, model, train_examples, args.seed)
    elif args.zscore:
        calibrate_zscore(backbone, model, train_examples, cache, args.seed, joint=args.joint, vec_cache=vec_cache)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)  # trusted, self-produced
        model.load_state_dict(ck["tower"], strict=False)
        backbone.load_lora_state_dict(ck["lora"])
    wb = init_wandb(args) if args.wandb else None

    if args.zero_shot:
        val_nll = eval_val_loss(backbone, model, cache, val_examples, args.eval_bs, joint=args.joint,
                                 readout=args.readout, vec_cache=vec_cache, shots=args.shots)
        results = run_full_eval(args, backbone, model, cache, val_examples, eval_sets, val_nll)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print_table(results)
        finish_wandb(wb, results)
        return

    if args.eval_only:
        model.load_state_dict(ckpt["tower"])
        backbone.load_lora_state_dict(ckpt["lora"])
        val_nll = eval_val_loss(backbone, model, cache, val_examples, args.eval_bs, joint=args.joint,
                                 readout=args.readout, vec_cache=vec_cache, shots=args.shots)
        results = run_full_eval(args, backbone, model, cache, val_examples, eval_sets, val_nll)
        if args.dump_logits:
            dump_eval_logits(args, backbone, model, cache, val_examples, eval_sets)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print_table(results)
        finish_wandb(wb, results)
        return

    opt, sched = build_optimizer(model, backbone, args)
    all_params = list(model.parameters()) + list(backbone.trainable_parameters())

    start_step, best_val = 0, float("inf")
    if last_path.exists() and not results_path.exists():
        start_step, best_val = load_checkpoint(last_path, model, backbone, opt, sched, device)
        print(f"resumed from step {start_step}")

    gen = data_generator(train_examples, args.bs, args.seed)
    for _ in range(start_step):
        next(gen)
    batches = prefetch(gen, cache, readout=args.readout, seed=args.seed,
                       shuffle=args.readout in ("mcq", "native") and not args.no_shuffle)  # collate on a background thread

    results = None
    for step in range(start_step + 1, args.steps + 1):
        t0 = time.time()
        examples, batch = next(batches)
        logits = run_readout(args.readout, backbone, model, batch, examples, joint=args.joint, vec_cache=vec_cache, shots=args.shots)
        # run_batch moves state/query/C to backbone.device but leaves target/p_null/cmask/
        # teacher/has_teacher/delta_* on the CPU tensors collate()/collate_mcq() built; move
        # them here so loss math matches logits' device (delta_* only exist for collate()).
        for k in ("target", "p_null", "cmask", "teacher", "has_teacher", "delta_idx", "delta_tgt"):
            if k in batch:
                batch[k] = batch[k].to(logits.device)
        loss = decision_loss(logits, batch["target"], batch["p_null"], batch["cmask"],
                              teacher=batch["teacher"], has_teacher=batch["has_teacher"],
                              alpha=args.distill_alpha, beta=args.distill_beta, T=args.distill_T,
                              delta_idx=batch.get("delta_idx"), delta_tgt=batch.get("delta_tgt"),
                              gamma=args.delta_gamma)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        sched.step()
        step_time = time.time() - t0

        n_tokens = batch.get("n_tokens", 0)
        log = {
            "train/loss": loss.item(), "train/step_time": step_time,
            "train/tokens_per_s": n_tokens / step_time if step_time > 0 else 0.0,
            "train/lr_tower": opt.param_groups[0]["lr"], "train/lr_lora": opt.param_groups[1]["lr"],
            "train/mem": device_memory(device),
        }
        if step % 50 == 0:
            fam = per_family_loss(logits, batch, examples)
            log.update({f"train/family_loss/{k}": v for k, v in fam.items()})
            print(f"step {step} loss {loss.item():.4f} step_time {step_time:.2f}s "
                  f"tok/s {log['train/tokens_per_s']:.0f} lr_tower {log['train/lr_tower']:.2e} "
                  f"lr_lora {log['train/lr_lora']:.2e}")
        if wb:
            wb.log(log, step=step)

        if step % args.val_every == 0 or step == args.steps:
            val_nll = eval_val_loss(backbone, model, cache, val_examples, args.eval_bs, joint=args.joint,
                                     readout=args.readout, vec_cache=vec_cache, shots=args.shots)
            print(f"step {step} val_nll {val_nll:.4f}")
            if wb:
                wb.log({"val/nll": val_nll}, step=step)
            if val_nll < best_val:
                best_val = val_nll
                save_checkpoint(best_path, model, backbone, step, opt, sched, best_val, args, tower_lora_only=True)

        if step % args.eval_every == 0 and step != args.steps:  # final eval below uses best.pt
            results = run_full_eval(args, backbone, model, cache, val_examples, eval_sets, best_val)
            log_eval_wandb(wb, results, step)

        if step % args.ckpt_every == 0 or step == args.steps:
            save_checkpoint(last_path, model, backbone, step, opt, sched, best_val, args)

    # final eval on the best-by-val checkpoint (as v0), not the last step
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["tower"])
        backbone.load_lora_state_dict(ckpt["lora"])
    results = run_full_eval(args, backbone, model, cache, val_examples, eval_sets, best_val)
    log_eval_wandb(wb, results, args.steps)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print_table(results)
    finish_wandb(wb, results)

    if args.hf_repo:
        try:
            import huggingface_hub
            huggingface_hub.upload_folder(folder_path=str(run_dir), repo_id=args.hf_repo,
                                           path_in_repo=args.name, repo_type="model", ignore_patterns=["last.pt"])
        except Exception as e:
            print(f"warning: hf upload failed: {e}")


if __name__ == "__main__":
    main()
