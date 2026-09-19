"""Baselines B (prompted log-prob, KV-cached prefix) and C (LoRA cross-encoder
+ per-task linear heads). See PLAN2.md "Baselines".

uv run baselines.py B [--backbone Qwen/Qwen3-1.7B-Base] [--limit N] [--no_kv_cache] [--check]
uv run baselines.py C [--backbone ...] [--lora_layers 8] [--lora_r 16] [--steps 3000] [--bs 32] [--train_limit N]
"""
import argparse
import glob
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from encode import Backbone, pick_device
from metrics import summarize

# data.py is being rewritten concurrently; FAMILY may not exist yet in older checkouts.
try:
    from data import FAMILY
except ImportError:
    FAMILY = {}
try:
    from data import CAND3, PARAPHRASES, TRAIN_NAMES
except ImportError:
    CAND3 = ("entailment", "neutral", "contradiction")
    PARAPHRASES, TRAIN_NAMES = [], []

# every wording of entailment/neutral/contradiction -> its canonical 0/1/2 slot
NLI_NAME2IDX = {n: i for names in [CAND3, *PARAPHRASES, *TRAIN_NAMES] for i, n in enumerate(names)}

NEG = float("-inf")  # pad sentinel; softmax(-inf)=0 exactly, never leaks into results.json (only derived metrics are written)
METRIC_KEYS = ["nll", "brier", "jsd", "acc", "acc_k", "ece", "sel_acc80", "conf_wrong", "auroc_null", "auroc_conf"]

SNLI_DEMO = (
    "Premise: A man is playing guitar on stage.\nHypothesis: A person is performing music.\nRelation: entailment\n\n"
    "Premise: A woman is reading a book in the park.\nHypothesis: A woman is sleeping at home.\nRelation: contradiction\n\n"
    "Premise: Two dogs run across a field.\nHypothesis: The dogs are chasing a ball.\nRelation: neutral\n\n"
)
CLS_DEMO = (
    "Text: What is the weather like today?\nLabel: weather\n\n"
    "Text: I want to cancel my subscription.\nLabel: cancel subscription\n\n"
    "Text: This movie was absolutely wonderful.\nLabel: positive\n\n"
)

# ---------------------------------------------------------------- shared ----

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def load_eval_sets(data_dir):
    return {Path(p).stem: load_jsonl(p) for p in sorted(glob.glob(f"{data_dir}/eval/*.jsonl"))}

def subsample(examples, n, rng=None):
    rng = rng or random.Random(0)
    return examples if len(examples) <= n else rng.sample(examples, n)

def cap_for(examples, limit):
    if limit:
        return limit
    kmax = max(len(ex["candidates"]) for ex in examples)
    return 300 if kmax >= 50 else 1000

def strip_query_prefix(query):
    for p in ("Relation of the text to: ", "Is this true given the text? "):
        if query.startswith(p):
            return query[len(p):]
    return query

def get_hyp(ex):
    """The hypothesis/question text for an example. Prefers data.py's meta dict
    (hyp/question), falls back to stripping the v0 query-prefix convention."""
    meta = ex.get("meta") or {}
    if "hyp" in meta:
        return meta["hyp"]
    if "question" in meta:
        return meta["question"]
    return strip_query_prefix(ex["query"])

def family_of(task):
    """task -> family. Prefers data.FAMILY; falls back to a prefix heuristic
    covering every v0/v1 task name (nli/prop/boolq/qa families, else classification)."""
    if task in FAMILY:
        return FAMILY[task]
    if task.startswith(("snli", "mnli", "anli", "chaos")):
        return "nli"
    if task.startswith("unli"):
        return "prop"
    if task.startswith("boolq"):
        return "boolq"
    if task.startswith("squad"):
        return "qa"
    return "cls"

def pad_stack(rows):
    """rows: list of (scores[K+1], target[K+1], label). -> [N,Kmax+1] arrays, null last."""
    kmax = max(len(s) - 1 for s, _, _ in rows)
    n = len(rows)
    scores = np.full((n, kmax + 1), NEG)
    target = np.zeros((n, kmax + 1))
    label = np.full(n, np.nan)
    for i, (s, t, l) in enumerate(rows):
        assert any(v > NEG for v in s), "row has no finite (unmasked) score"
        k = len(s) - 1
        scores[i, :k], scores[i, kmax] = s[:k], s[k]
        target[i, :k], target[i, kmax] = t[:k], t[k]
        label[i] = np.nan if l is None else l
    return scores, target, label

def softmax(x):
    # x may contain -inf (padding or "this model has no null column"); exp(-inf)=0
    # exactly and never nan as long as every row has >=1 finite entry (pad_stack asserts this).
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)

def nll_at(scores, target, T):
    p = softmax(scores / T)
    return float(-(target * np.log(np.clip(p, 1e-12, 1.0))).sum(-1).mean())

def fit_temperature(scores, target):
    best_T, best_nll = 1.0, nll_at(scores, target, 1.0)
    for T in np.geomspace(0.1, 10, 60):
        n = nll_at(scores, target, T)
        if n < best_nll:
            best_nll, best_T = n, float(T)
    return best_T

def fill_na(m):
    return {k: m.get(k, "n/a") for k in METRIC_KEYS}

NA_ROW = {k: "n/a" for k in METRIC_KEYS}

def evaluate(name, val_examples, eval_sets, score_fn, val_limit=2000):
    """score_fn(examples) -> (scores[N,Kmax+1], target[N,Kmax+1], label[N], has_null) or
    None if the whole set can't be scored (e.g. no head for this task).
    Writes runs/<name>/results.json in train.py's format; undefined cells are "n/a"."""
    val = subsample(val_examples, val_limit)
    v = score_fn(val)
    assert v is not None, f"[{name}] val.jsonl must be scorable"
    vs, vt, _, _ = v
    val_nll = nll_at(vs, vt, 1.0)
    T = fit_temperature(vs, vt)
    results = {"T": T, "val_nll": val_nll, "eval": {}}
    print(f"\n[{name}] T={T:.3f} val_nll={val_nll:.4f}")
    for set_name, examples in eval_sets.items():
        t0 = time.perf_counter()
        r = score_fn(examples)
        dt = (time.perf_counter() - t0) / max(len(examples), 1)
        if r is None:
            results["eval"][set_name] = {"raw": dict(NA_ROW), "scaled": dict(NA_ROW)}
            print(f"  {set_name:<22}n={len(examples):<6}{dt*1000:7.1f} ms/ex  n/a (no head)")
            continue
        s, t, l, has_null = r
        raw = fill_na(summarize(softmax(s), t, l))
        scaled = fill_na(summarize(softmax(s / T), t, l))
        if not has_null:
            raw["auroc_null"] = scaled["auroc_null"] = "n/a"
        results["eval"][set_name] = {"raw": raw, "scaled": scaled}
        acc = scaled.get("acc", "n/a")
        nll = scaled.get("nll", "n/a")
        acc_s = f"{acc:.3f}" if isinstance(acc, float) else acc
        nll_s = f"{nll:.3f}" if isinstance(nll, float) else nll
        print(f"  {set_name:<22}n={len(examples):<6}{dt*1000:7.1f} ms/ex  acc={acc_s}  nll={nll_s}")
    out_dir = Path("runs") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_dir / "results.json", "w"), indent=2)
    return results

# -------------------------------------------------------------- baseline B --

def b_prompt(ex):
    """-> (prefix ending right before the candidate continuation, null candidate literal).
    data.TEMPLATES[family] is a list of *query*-wording variants (what data.py used to
    build ex["query"]), not an LM prompt format -- not reusable here. family_of() still
    uses data.FAMILY for routing; the prompt shape per family is hardcoded below."""
    fam = family_of(ex["task"])
    state, hyp = ex["state"], get_hyp(ex)
    if fam == "nli":
        return f"{SNLI_DEMO}Premise: {state}\nHypothesis: {hyp}\nRelation:", "none of the above"
    if fam == "prop":
        return f"Premise: {state}\nHypothesis: {hyp}\nTrue or false:", "no"
    if fam == "boolq":
        return f"Passage: {state}\nQuestion: {hyp}\nAnswer:", "no"
    if fam == "qa":
        return f"Context: {state}\nQuestion: {hyp}\nAnswer:", "none of the above"
    return f"{CLS_DEMO}Text: {state}\nLabel:", "none of the above"

def score_chunk_full(lm, tok, device, pairs, batch_size=48):
    """pairs: list of (prefix, candidate) -> length-normalised candidate log-prob per pair.
    No KV reuse: every pair re-encodes prefix+candidate from scratch (fallback / --check baseline)."""
    out = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        pre_ids = tok([p for p, _ in chunk], add_special_tokens=True)["input_ids"]
        cand_ids = tok([" " + c for _, c in chunk], add_special_tokens=False)["input_ids"]
        seqs = [p + c for p, c in zip(pre_ids, cand_ids)]
        maxlen = max(len(s) for s in seqs)
        pad_id = tok.pad_token_id
        ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            attn[i, :len(s)] = 1
        ids, attn = ids.to(device), attn.to(device)
        rows, cols, tgts = [], [], []
        for i, (p, c) in enumerate(zip(pre_ids, cand_ids)):
            for k, t in enumerate(c):
                rows.append(i); cols.append(len(p) - 1 + k); tgts.append(t)
        with torch.inference_mode():
            hidden = lm.model(input_ids=ids, attention_mask=attn).last_hidden_state
            sel = hidden[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]
            logp = torch.log_softmax(lm.lm_head(sel).float(), dim=-1)
            tok_lp = logp.gather(-1, torch.tensor(tgts, device=device).unsqueeze(-1)).squeeze(-1).cpu()
        j = 0
        for p, c in zip(pre_ids, cand_ids):
            n = len(c)
            out.append(NEG if n == 0 else tok_lp[j:j + n].sum().item() / n)
            j += n
    return out

_warned_no_batch_repeat = False

def score_example_kv(lm, tok, device, prefix, cands):
    """Run the prefix once with use_cache=True, then continue all len(cands) candidates
    from the cached past_key_values (DynamicCache.batch_repeat_interleave). Falls back to
    score_chunk_full (no KV reuse) if that API is unavailable in the installed transformers."""
    global _warned_no_batch_repeat
    pre_ids = tok(prefix, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    with torch.inference_mode():
        out = lm(input_ids=pre_ids, use_cache=True)
    cache = out.past_key_values
    if not hasattr(cache, "batch_repeat_interleave"):
        if not _warned_no_batch_repeat:
            print("[B] DynamicCache.batch_repeat_interleave unavailable; falling back to "
                  "per-candidate re-encoding (no KV reuse, slower).")
            _warned_no_batch_repeat = True
        return score_chunk_full(lm, tok, device, [(prefix, c) for c in cands])

    n = len(cands)
    cand_ids = tok([" " + c for c in cands], add_special_tokens=False)["input_ids"]
    cache.batch_repeat_interleave(n)
    maxlen = max(len(c) for c in cand_ids)
    pad_id = tok.pad_token_id
    ids = torch.full((n, maxlen), pad_id, dtype=torch.long, device=device)
    cur_mask = torch.zeros((n, maxlen), dtype=torch.long, device=device)
    for i, c in enumerate(cand_ids):
        ids[i, :len(c)] = torch.tensor(c, device=device)
        cur_mask[i, :len(c)] = 1
    past_len = pre_ids.shape[1]
    attn = torch.cat([torch.ones(n, past_len, dtype=torch.long, device=device), cur_mask], dim=1)
    cache_position = torch.arange(past_len, past_len + maxlen, device=device)
    with torch.inference_mode():
        out2 = lm(input_ids=ids, attention_mask=attn, past_key_values=cache,
                   cache_position=cache_position, use_cache=False)
    logp_cont = torch.log_softmax(out2.logits.float(), dim=-1)          # [n, maxlen, V]
    logp_first = torch.log_softmax(out.logits[:, -1].float(), dim=-1)  # [1, V], shared prefix

    scores = []
    for i, c in enumerate(cand_ids):
        lp = logp_first[0, c[0]].item()
        for k in range(1, len(c)):
            lp += logp_cont[i, k - 1, c[k]].item()
        scores.append(lp / len(c))
    return scores

def b_score_examples(lm, tok, device, examples, use_kv=True):
    rows = []
    for ex in examples:
        prefix, null_lit = b_prompt(ex)
        cands = list(ex["candidates"]) + [null_lit]
        if use_kv:
            scores = score_example_kv(lm, tok, device, prefix, cands)
        else:
            scores = score_chunk_full(lm, tok, device, [(prefix, c) for c in cands])
        s = np.array(scores)
        t = np.array([(1 - ex["p_null"]) * x for x in ex["target"]] + [ex["p_null"]])
        rows.append((s, t, ex.get("label")))
    scores, target, label = pad_stack(rows)
    return scores, target, label, True

def run_b(args):
    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.backbone, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(args.backbone, dtype=torch.bfloat16).to(device).eval()

    val_examples = load_jsonl(f"{args.data}/val.jsonl")
    eval_sets = load_eval_sets(args.data)
    rng = random.Random(0)
    capped = {name: subsample(exs, cap_for(exs, args.limit), rng) for name, exs in eval_sets.items()}
    use_kv = not args.no_kv_cache

    if args.check:
        # ponytail: verify in fp32, not the production bf16 weights. bf16 + MPS's
        # cached-attention kernel vs full-recompute kernel take different codepaths and
        # disagree by up to ~1 nat after 28 layers (verified: matches to 2e-5 in fp32,
        # and matches bit-for-bit on CPU in bf16 too) -- a bf16/MPS precision gap, not a
        # KV-cache logic bug. This isolates the algorithm from that hardware gap.
        lm_check = AutoModelForCausalLM.from_pretrained(args.backbone, dtype=torch.float32).to(device).eval()
        chk = subsample(val_examples, 8, rng)
        a, _, _, _ = b_score_examples(lm_check, tok, device, chk, use_kv=True)
        b, _, _, _ = b_score_examples(lm_check, tok, device, chk, use_kv=False)
        finite = np.isfinite(a) & np.isfinite(b)
        max_diff = float(np.abs(a[finite] - b[finite]).max())
        assert max_diff < 1e-3, f"KV-cache scores diverge from full recompute by {max_diff}"
        print(f"[B] --check: KV-cache vs full recompute (fp32) max diff {max_diff:.2e} < 1e-3 OK")
        del lm_check

    evaluate(args.name or "B", val_examples, capped, lambda exs: b_score_examples(lm, tok, device, exs, use_kv=use_kv))

# -------------------------------------------------------------- baseline C --

def masked_mean(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp_min(1e-6)

def c_joint_string(ex):
    if family_of(ex["task"]) == "cls":
        return ex["state"]
    return f'{ex["state"]} [SEP] {get_hyp(ex)}'

def encode_pool(backbone, texts, max_len=256):
    ids, am = backbone.tokenize(texts, max_len)
    ids, am = ids.to(backbone.device), am.to(backbone.device)
    h_top, _, mask = backbone(ids, am)
    return masked_mean(h_top, mask)

def head_key_of(ex):
    """Which head this example trains/scores against. nli and boolq are single
    family-wide heads (like unli) -- their train/eval task *names* don't share a string
    prefix (e.g. train "boolq_pair" vs eval "boolq_val"), so family, not task name, is the
    only reliable link. Everything else (intent/topic/emotion/...) gets its own per-task
    head, since those genuinely have different, unrelated label vocabularies."""
    fam = family_of(ex["task"])
    if fam == "nli":
        return "nli"
    if fam == "prop":
        return "unli"
    if fam == "boolq":
        return "boolq"
    return ex["task"]

def resolve_head_key(task, heads):
    """Eval-only task names extend a trained task name with a suffix (clinc_test ->
    clinc, hwu64_test -> hwu64): try the full name, then progressively shorter
    underscore-prefixes, against the trained head keys."""
    parts = task.split("_")
    for i in range(len(parts), 0, -1):
        cand = "_".join(parts[:i])
        if cand in heads:
            return cand
    return None

def build_vocabs(train_examples):
    """head_key -> (sorted vocab, max_k) for cls/boolq heads. vocab is the union of every
    candidate string seen among that head's widest (max_k) rows -- some datasets (e.g.
    clinc) draw a different random subset of the full label pool per "full" row rather
    than repeating one fixed set, so the union can exceed any single row's K; max_k
    (not len(vocab)) is what identifies a full/non-null-synth-reduced row.
    ponytail: boolq's vocab similarly grows with every yes/no paraphrase wording seen
    (e.g. 8, not a literal 2) since there's no canonical name->{yes,no} table like NLI's;
    scoring still works fine at any width (name lookup + renormalise), just a wider head
    than PLAN2's "2-way" framing suggests. Add a canonical mapping if that head needs to
    be exactly binary."""
    by_key = defaultdict(list)
    for ex in train_examples:
        if family_of(ex["task"]) in ("nli", "prop", "qa"):
            continue
        by_key[head_key_of(ex)].append(ex)
    out = {}
    for key, exs in by_key.items():
        max_k = max(len(ex["candidates"]) for ex in exs)
        vocab = sorted({c for ex in exs if len(ex["candidates"]) == max_k for c in ex["candidates"]})
        out[key] = (vocab, max_k)
    return out

def build_train_pool(examples, vocabs):
    """-> list of (joint_string, head_key, loss_kind, target). Only full-candidate-set,
    non-null-synth rows: K==len(vocab) (or K==3 for nli) and p_null==0. "prop" (unli) is the
    one exception: its p_null IS the graded regression target (1-p_null = P(true)), not a
    null-synth flag, so it's exempt from the p_null==0 filter."""
    pool = []
    for ex in examples:
        fam = family_of(ex["task"])
        if fam == "qa":
            continue
        text = c_joint_string(ex)
        if fam == "prop":
            pool.append((text, "unli", "bce", 1.0 - ex["p_null"]))
            continue
        if ex["p_null"] > 0:
            continue
        if fam == "nli":
            if len(ex["candidates"]) != 3:
                continue
            try:
                tgt = [0.0, 0.0, 0.0]
                for c, w in zip(ex["candidates"], ex["target"]):
                    tgt[NLI_NAME2IDX[c]] = w
            except KeyError:
                continue
            pool.append((text, "nli", "softmax", tgt))
        else:  # cls (own per-task head) or boolq (shared family head)
            key = head_key_of(ex)
            vocab, max_k = vocabs.get(key, ([], 0))
            if not vocab or len(ex["candidates"]) != max_k:
                continue
            name2idx = {n: j for j, n in enumerate(vocab)}
            try:
                tgt = [0.0] * len(vocab)
                for c, w in zip(ex["candidates"], ex["target"]):
                    tgt[name2idx[c]] = w
            except KeyError:
                continue
            pool.append((text, key, "softmax", tgt))
    return pool

def build_heads(vocabs, train_pool, d):
    keys = {b[1] for b in train_pool}
    heads = nn.ModuleDict()
    if "nli" in keys:
        heads["nli"] = nn.Linear(d, 3)
    if "unli" in keys:
        heads["unli"] = nn.Linear(d, 1)
    for task, (vocab, _max_k) in vocabs.items():
        if task in keys:
            heads[task] = nn.Linear(d, len(vocab))
    return heads

def c_batch_loss(backbone, heads, batch, device):
    texts = [b[0] for b in batch]
    pooled = encode_pool(backbone, texts)
    loss = 0.0
    for key in {b[1] for b in batch}:
        idxs = [i for i, b in enumerate(batch) if b[1] == key]
        x = pooled[idxs]
        kind = batch[idxs[0]][2]
        head = heads[key]
        if kind == "bce":
            y = torch.tensor([batch[i][3] for i in idxs], dtype=torch.float32, device=device)
            l = F.binary_cross_entropy_with_logits(head(x).squeeze(-1), y)
        else:
            y = torch.tensor([batch[i][3] for i in idxs], dtype=torch.float32, device=device)
            l = -(y * F.log_softmax(head(x), dim=-1)).sum(-1).mean()
        loss = loss + l * (len(idxs) / len(batch))
    return loss

def train_c(backbone, heads, train_pool, val_pool, steps, bs, head_lr, lora_lr, device):
    opt = AdamW([
        {"params": backbone.trainable_parameters(), "lr": lora_lr},
        {"params": heads.parameters(), "lr": head_lr},
    ])
    rng = random.Random(0)
    val_fixed = subsample(val_pool, 500, rng) if val_pool else []
    best_val, best_state = float("inf"), None
    last_loss = float("nan")
    for step in range(steps):
        batch = rng.choices(train_pool, k=bs)
        loss = c_batch_loss(backbone, heads, batch, device)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(backbone.trainable_parameters()) + list(heads.parameters()), 1.0)
        opt.step()
        last_loss = loss.item()
        if step % 500 == 0 or step == steps - 1:
            if val_fixed:
                with torch.no_grad():
                    vloss = sum(c_batch_loss(backbone, heads, val_fixed[i:i + bs], device).item() * len(val_fixed[i:i + bs])
                                for i in range(0, len(val_fixed), bs)) / len(val_fixed)
            else:
                vloss = last_loss
            print(f"[C] step {step} train_loss={last_loss:.4f} val_loss={vloss:.4f}")
            if vloss <= best_val:
                best_val = vloss
                best_state = ({k: v.clone() for k, v in heads.state_dict().items()}, backbone.lora_state_dict())
    if best_state is not None:
        heads.load_state_dict(best_state[0])
        backbone.load_lora_state_dict(best_state[1])

def name_lookup_scores(probs, candidates, name2idx):
    """probs: softmax over the head's vocab. Some candidates (e.g. clinc_oos distractors
    drawn from held-out intents) may not be in name2idx -- v0-style: renormalise probability
    mass over only the present candidates, 0 for the rest. None if none are present at all."""
    present = [(j, name2idx[c]) for j, c in enumerate(candidates) if c in name2idx]
    if not present:
        return None
    cols, idx = zip(*present)
    k_probs = probs[list(idx)]
    s = k_probs.sum()
    k_probs = k_probs / s if s > 0 else np.full(len(idx), 1.0 / len(idx))
    out = np.zeros(len(candidates))
    for col, p in zip(cols, k_probs):
        out[col] = p
    return out

def c_score_examples(backbone, heads, vocabs, examples, batch_size=64):
    if not examples:
        return None
    fam0 = family_of(examples[0]["task"])
    has_null = fam0 == "prop"
    texts = [c_joint_string(ex) for ex in examples]
    pooled_chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            pooled_chunks.append(encode_pool(backbone, texts[i:i + batch_size]))
    pooled = torch.cat(pooled_chunks) if pooled_chunks else torch.empty(0, backbone.d, device=backbone.device)

    rows = []
    with torch.no_grad():
        for i, ex in enumerate(examples):
            fam = family_of(ex["task"])
            x = pooled[i:i + 1]
            if fam == "qa":
                continue
            if fam == "nli":
                if "nli" not in heads:
                    continue
                probs = F.softmax(heads["nli"](x), dim=-1).squeeze(0).cpu().numpy()
                k_probs = name_lookup_scores(probs, ex["candidates"], NLI_NAME2IDX)
            elif fam == "prop":
                if "unli" not in heads:
                    continue
                p = torch.sigmoid(heads["unli"](x)).item()
                scores = np.log(np.clip([p, 1 - p], 1e-12, 1.0))
                t = np.array([(1 - ex["p_null"]) * v for v in ex["target"]] + [ex["p_null"]])
                rows.append((scores, t, ex.get("label")))
                continue
            elif fam == "boolq":
                if "boolq" not in heads:
                    continue
                vocab, _max_k = vocabs.get("boolq", ([], 0))
                name2idx = {n: j for j, n in enumerate(vocab)}
                probs = F.softmax(heads["boolq"](x), dim=-1).squeeze(0).cpu().numpy()
                k_probs = name_lookup_scores(probs, ex["candidates"], name2idx)
            else:
                key = resolve_head_key(ex["task"], heads)
                if key is None:
                    continue
                vocab, _max_k = vocabs.get(key, ([], 0))
                if not vocab:
                    continue
                name2idx = {n: j for j, n in enumerate(vocab)}
                probs = F.softmax(heads[key](x), dim=-1).squeeze(0).cpu().numpy()
                k_probs = name_lookup_scores(probs, ex["candidates"], name2idx)
            if k_probs is None:
                continue
            scores = np.concatenate([np.log(np.clip(k_probs, 1e-12, 1.0)), [NEG]])  # no null column
            t = np.array([(1 - ex["p_null"]) * v for v in ex["target"]] + [ex["p_null"]])
            rows.append((scores, t, ex.get("label")))
    if not rows:
        return None
    scores, target, label = pad_stack(rows)
    return scores, target, label, has_null

def run_c(args):
    device = pick_device(args.device)
    backbone = Backbone(name=args.backbone, lora_layers=args.lora_layers, lora_r=args.lora_r, device=device)
    train_examples = load_jsonl(f"{args.data}/train.jsonl")
    val_examples = load_jsonl(f"{args.data}/val.jsonl")
    eval_sets = load_eval_sets(args.data)
    rng = random.Random(0)
    if args.train_limit:
        train_examples = subsample(train_examples, args.train_limit, rng)

    vocabs = build_vocabs(train_examples)
    train_pool = build_train_pool(train_examples, vocabs)
    val_pool = build_train_pool(val_examples, vocabs)
    heads_seen = sorted({b[1] for b in train_pool})
    print(f"[C] train pool: {len(train_pool)} rows over heads {heads_seen}")
    assert train_pool, "no trainable (full-candidate-set, non-null) rows found in train.jsonl"

    heads = build_heads(vocabs, train_pool, backbone.d).to(device)
    train_c(backbone, heads, train_pool, val_pool, args.steps, args.bs, head_lr=1e-3, lora_lr=1e-4, device=device)

    capped = {name: subsample(exs, cap_for(exs, args.limit), rng) for name, exs in eval_sets.items()}
    evaluate(args.name or "C", val_examples, capped, lambda exs: c_score_examples(backbone, heads, vocabs, exs))

# ------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["B", "C"])
    ap.add_argument("--name", default=None, help="runs/<name>/results.json (default: mode, i.e. B or C)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backbone", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_kv_cache", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lora_layers", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--train_limit", type=int, default=None)
    args = ap.parse_args()
    {"B": run_b, "C": run_c}[args.mode](args)

if __name__ == "__main__":
    main()
