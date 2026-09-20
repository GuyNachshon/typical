"""native_choice_v1 (PLAN4 sec 11-12): one option-aware suffix pass, direct probability readout.

Prompt = mcq.py's exactly ([eos] + state + query + lettered options + "none of the above" +
"Answer:") plus one terminal decision token (the tokenizer's eos id, appended as an id so no
text-level special-token parsing is involved). h_D = the terminal token's last-layer (normed)
hidden state; the letters stay in the text but are never read out.

Candidates (--nc_head):
  n2   Qwen3-Embedding vectors (VecCache; EmbedEncoder on misses = cold path)
  n3   mean of each option's OWN suffix hidden states over its rendered span (mcq._render
       records the char spans; token spans come from the tokenizer's offsets). The null line
       is pooled the same way and feeds the null head.
  n2n3 each source projected to d' and concatenated.
Scorer: u = W_h h_D, v_j = W_c c_j, s_j = u.v_j/sqrt(d_v) + w.(u*v_j) + MLP(u*v_j).
Null: --null factored -> model.factored_null_logits (r from set statistics of s + [u; v_null]);
      --null softmax  -> one extra score from [u; v_null]. Either way logits are [B, Kmax+1],
null last, pads = finfo.min, so decision_loss/probs_from_logits/summarize are unchanged.

K: no letter cap. A row whose suffix exceeds max_suffix tokens is scored in chunks that fit
(consecutive options, re-lettered from A) and composed hierarchically by mcq._score_chunked
(stage 1 per chunk, stage 2 over chunk winners). ponytail: h_D is then per chunk, not per
full set -- fine as a fallback; with max_suffix=1024 every current eval set (K<=150) fits.

native_v2 (PLAN5 sec 2, --nc_render tags): letter-free rendering -- each option inside a plain
"<choice>\n...\n</choice>" block, no null line (the null is the head's, not a rendered option:
C3's null row is a zero vector, so v_null is a learned constant and the null head reads h_D +
set statistics). Candidate identity is only its span, so option order carries no signal
beyond position; --perm_lambda adds perm_consistency_loss to push F(pi(A)) = pi(F(A)).
"""
import bisect
import copy
import math

import torch
import torch.nn as nn

from encode import EMBED_DIM, CAND_RENDER
from mcq import _label, _render, _ids, _pack, _score_chunked, collate_mcq
from model import factored_null_logits, _vecs_or_embed

MAX_SUFFIX = 1024
CHOICE_OPEN, CHOICE_CLOSE = "<choice>\n", "\n</choice>\n"


def _render_tags(query: str, cand_texts: list[str]):
    """native_v2: query, then one "<choice>\ntext\n</choice>" block per option; no letters, no
    null line, no "Answer:" (the terminal decision token follows). spans[k] = (start, end) of
    option k's text (tags excluded) -- len(spans) == K, unlike _render's K+1."""
    text, spans = query + "\n", []
    for c in cand_texts:
        text += CHOICE_OPEN
        spans.append((len(text), len(text) + len(c)))
        text += c + CHOICE_CLOSE
    return text, spans


def _render_letters_nonull(query: str, cand_texts: list[str]):
    """native_v3: lettered options (slot identity kept -> Δ_q, REPORT §3p) but NO rendered "none of the
    above" line and no "Answer:" -- ∅ is purely the head's decision (the null fix of nc_v2). K spans."""
    text, spans = query + "\n", []
    for i, c in enumerate(cand_texts):
        text += f"{_label(i)}. "
        spans.append((len(text), len(text) + len(c)))
        text += c + "\n"
    return text, spans


RENDERS = {"letters": _render, "tags": _render_tags, "letters_nonull": _render_letters_nonull}


class NativeHead(nn.Module):
    def __init__(self, d: int, nc_head: str = "n2n3", null: str = "factored", d_proj: int = 256,
                 d_emb: int = EMBED_DIM, render: str = "letters"):
        super().__init__()
        assert nc_head in ("n2", "n3", "n2n3"), nc_head
        assert null in ("softmax", "factored"), null
        assert render in RENDERS, render
        self.nc_head, self.null, self.render = nc_head, null, render  # render: restored from args, not state_dict
        self.use2, self.use3 = "n2" in nc_head, "n3" in nc_head
        dv = d_proj * (self.use2 + self.use3)
        self.dv = dv
        # z-score buffers for LM-space vectors (h_D and the n3 pooled spans); identity until calibrate().
        self.register_buffer("mu_h", torch.zeros(d)); self.register_buffer("sd_h", torch.ones(d))
        self.register_buffer("mu_c", torch.zeros(d)); self.register_buffer("sd_c", torch.ones(d))
        self.proj_h = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, dv))
        if self.use2:
            self.proj_2 = nn.Sequential(nn.LayerNorm(d_emb), nn.Linear(d_emb, d_proj))
            self.null_emb = nn.Parameter(torch.randn(d_emb) * 0.02)  # learned "none of the above" vector
        if self.use3:
            self.proj_3 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_proj))
        self.w = nn.Linear(dv, 1)
        self.res = nn.Sequential(nn.Linear(dv, dv), nn.GELU(), nn.Linear(dv, 1))
        # null head input: [u; v_null] (2*dv); factored gate also gets 4 set stats + mean/var of v (2*dv)
        if null == "factored":
            self.null_gate = nn.Sequential(nn.Linear(4 + 4 * dv, 256), nn.GELU(), nn.Linear(256, 1))
        else:
            self.score_null = nn.Sequential(nn.Linear(2 * dv, dv // 2), nn.GELU(), nn.Linear(dv // 2, 1))
        self.temperature = 1.0  # see DecisionModel.temperature / train.fit_temperature

    @torch.no_grad()
    def calibrate(self, h, c3):
        """h [N, d] decision states, c3 [N, d] pooled option spans, from a sample of training rows."""
        self.mu_h.copy_(h.mean(0)); self.sd_h.copy_(h.std(0).clamp_min(1e-3))
        self.mu_c.copy_(c3.mean(0)); self.sd_c.copy_(c3.std(0).clamp_min(1e-3))

    def forward(self, h, cmask, c3=None, c2=None):
        """h [B, d]; cmask [B, Kmax]; c3 [B, Kmax+1, d] (null line last); c2 [B, Kmax, d_emb].
        -> logits [B, Kmax+1], null last, pads finfo.min."""
        B = h.shape[0]
        u = self.proj_h((h - self.mu_h) / self.sd_h)                              # [B, dv]
        vs = []
        if self.use2:
            vs.append(self.proj_2(torch.cat([c2, self.null_emb.expand(B, 1, -1)], 1)))
        if self.use3:
            vs.append(self.proj_3((c3 - self.mu_c) / self.sd_c))
        v = torch.cat(vs, -1)                                                     # [B, Kmax+1, dv]
        vc, vn = v[:, :-1], v[:, -1]
        uv = u.unsqueeze(1) * vc                                                  # [B, Kmax, dv]
        s = uv.sum(-1) * self.dv ** -0.5 + self.w(uv).squeeze(-1) + self.res(uv).squeeze(-1)
        s = s.masked_fill(~cmask, torch.finfo(s.dtype).min)
        hn = torch.cat([u, vn], -1)
        if self.null == "factored":
            return factored_null_logits(self.null_gate, s, vc, hn, cmask, self.temperature)
        return torch.cat([s, self.score_null(hn)], dim=-1)


def _pool_matrix(spans, offsets, base, Kmax, T, K=None):
    """[Kmax+1, T] mean-pooling rows: option k < K over the suffix tokens overlapping its char
    span; spans[K] (present for letters only) is the null line -> row Kmax; with tags (K spans)
    the null row stays zero. K defaults to len(spans) - 1 (letters). Tokens overlapping
    [cs, ce) are contiguous, found by bisection."""
    K = len(spans) - 1 if K is None else K
    starts, ends = [o[0] for o in offsets], [o[1] for o in offsets]
    pool = torch.zeros(Kmax + 1, T)
    for k, (cs, ce) in enumerate(spans):
        j0, j1 = bisect.bisect_right(ends, cs), bisect.bisect_left(starts, ce)
        pool[Kmax if k == K else k, base + j0:base + j1] = 1.0
    return pool / pool.sum(-1, keepdim=True).clamp_min(1.0)


def native_features(head, states, queries, cand_lists, max_state=256, max_suffix=MAX_SUFFIX, render="letters"):
    """One suffix pass. -> (h_D [B, d], C3 [B, Kmax+1, d] (null line last; zero for tags), cmask
    [B, Kmax], n_tokens). Suffixes are assumed to fit max_suffix (run_batch_native chunks the
    rest); a truncated option pools to a zero vector."""
    tok, dev = head.backbone.tokenizer, head.device
    rendered = [RENDERS[render](q, c) for q, c in zip(queries, cand_lists)]
    s_ids = _ids(tok, states, max_state)
    x_ids, offs = _ids(tok, [r[0] for r in rendered], max_suffix, offsets=True)
    input_ids, am, lengths = _pack(tok, s_ids, x_ids, tail=[tok.eos_token_id])
    B, T = input_ids.shape
    Kmax = max(len(c) for c in cand_lists)
    pool = torch.stack([_pool_matrix(r[1], o, 1 + len(s), Kmax, T, K=len(c))
                        for r, o, s, c in zip(rendered, offs, s_ids, cand_lists)])
    cmask = torch.zeros(B, Kmax, dtype=torch.bool)
    for i, c in enumerate(cand_lists):
        cmask[i, :len(c)] = True

    out = head.backbone.model(input_ids=input_ids.to(dev), attention_mask=am.to(dev))
    H = out.last_hidden_state.float()                                         # [B, T, d], final norm applied
    h = H[torch.arange(B, device=dev), (lengths - 1).to(dev)]
    C3 = torch.bmm(pool.to(dev), H)
    return h, C3, cmask.to(dev), int(am.sum().item())


def _n2_vectors(head, cand_lists, vec_cache, dev):
    """[B, Kmax, d_emb] Qwen3-Embedding vectors: VecCache hits, EmbedEncoder (lazily built on
    the head) for misses -- the cold path."""
    uniq = list(dict.fromkeys(c for cl in cand_lists for c in cl))
    missing = [c for c in uniq if vec_cache is None or c not in vec_cache.index]
    if missing and getattr(head, "encoder", None) is None:
        from encode import EmbedEncoder
        head.encoder = EmbedEncoder(device=dev)
    vecs = _vecs_or_embed(vec_cache, getattr(head, "encoder", None), uniq, dev, render=CAND_RENDER)
    idx = {c: i for i, c in enumerate(uniq)}
    Kmax = max(len(c) for c in cand_lists)
    C2 = torch.zeros(len(cand_lists), Kmax, vecs.shape[-1], device=dev)
    for i, cl in enumerate(cand_lists):
        C2[i, :len(cl)] = vecs[[idx[c] for c in cl]]
    return C2


def _fit_chunks(tok, query, cand_texts, max_suffix, render="letters"):
    """Consecutive option index chunks whose rendered suffix fits max_suffix tokens (per-line
    counts are exact for single-token letters; +2/option slack covers numeric labels)."""
    per = f"{CHOICE_OPEN}{{}}{CHOICE_CLOSE}" if render == "tags" else "A. {}\n"
    lens = [len(x) for x in tok([per.format(c) for c in cand_texts], add_special_tokens=False)["input_ids"]]
    overhead = len(tok(query + "\n", add_special_tokens=False)["input_ids"]) + 2
    if render == "letters":
        overhead += len(tok("A. none of the above\nAnswer:", add_special_tokens=False)["input_ids"])
    chunks, cur, tot = [], [], overhead
    for k, l in enumerate(lens):
        if cur and tot + l + 2 > max_suffix:
            chunks.append(cur); cur, tot = [], overhead
        cur.append(k); tot += l + 2
    chunks.append(cur)
    return chunks


def run_batch_native(head, model, batch, examples, max_state: int = 256, max_suffix: int = MAX_SUFFIX,
                     vec_cache=None):
    """-> logits [B, Kmax+1] (Kmax = batch["cmask"].shape[1]); sets batch["n_tokens"].
    head: MCQHead (backbone + LoRA); model: NativeHead; vec_cache: VecCache for n2 heads."""
    tok, dev = head.backbone.tokenizer, head.device
    states = [ex["state"] for ex in examples]
    queries = [ex["query"] for ex in examples]
    cand_lists = [ex["candidates"] for ex in examples]
    Kmax = batch["cmask"].shape[1]
    logits = torch.full((len(examples), Kmax + 1), torch.finfo(torch.float32).min, device=dev)
    n_tokens = 0

    render = getattr(model, "render", "letters")

    def score(sts, qs, cls):
        h, C3, cm, n_tok = native_features(head, sts, qs, cls, max_state, max_suffix, render=render)
        C2 = _n2_vectors(head, cls, vec_cache, dev) if model.use2 else None
        return model(h, cm, C3, C2), n_tok

    n_suf = [len(x) for x in tok([RENDERS[render](q, c)[0] for q, c in zip(queries, cand_lists)],
                                 add_special_tokens=False)["input_ids"]]
    direct_i = [i for i, n in enumerate(n_suf) if n <= max_suffix]
    if direct_i:
        sub, n_tok = score([states[i] for i in direct_i], [queries[i] for i in direct_i], [cand_lists[i] for i in direct_i])
        n_tokens += n_tok
        for row, i in enumerate(direct_i):
            k = len(cand_lists[i])
            logits[i, :k] = sub[row, :k]
            logits[i, -1] = sub[row, -1]
    for i in (i for i, n in enumerate(n_suf) if n > max_suffix):
        chunks = _fit_chunks(tok, queries[i], cand_lists[i], max_suffix, render=render)
        row, n_tok = _score_chunked(lambda cl: score([states[i]] * len(cl), [queries[i]] * len(cl), cl),
                                    cand_lists[i], chunks)
        n_tokens += n_tok
        k = len(cand_lists[i])
        logits[i, :k] = row[:k]
        logits[i, -1] = row[-1]
    batch["n_tokens"] = n_tokens
    return logits


def _causal_pad_mask(attn2d, q_len, dtype):
    """Explicit additive (batch,1,q_len,kv_len) mask: causal by physical slot order
    within [past|current] (order-based, so identical across rows regardless of padding)
    AND-ed with the real/pad mask `attn2d`.
    ponytail: transformers' automatic 2D-attention_mask + past_key_values path silently
    mis-handles a right-padded batch sharing one KV cache (verified empirically: this
    explicit mask + explicit position_ids reproduces the un-batched per-row computation
    exactly; the automatic path does not, off by ~0.3-0.8 nat per row). Building this
    ourselves sidesteps that rather than chasing it through transformers' mask_utils."""
    batch, kv_len = attn2d.shape
    past_len = kv_len - q_len
    q_idx = torch.arange(q_len, device=attn2d.device).unsqueeze(1)
    kv_idx = torch.arange(kv_len, device=attn2d.device).unsqueeze(0)
    causal = kv_idx <= (past_len + q_idx)  # [q_len, kv_len]
    allowed = causal.unsqueeze(0) & attn2d.bool().unsqueeze(1)  # [batch, q_len, kv_len]
    mask = torch.zeros(batch, 1, q_len, kv_len, dtype=dtype, device=attn2d.device)
    return mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)


@torch.inference_mode()
def native_kv_decide(head, model, state, queries, chunk: int = 32, max_state: int = 256,
                     max_suffix: int = MAX_SUFFIX, vec_cache=None):
    """Serving path (PLAN5 sec 1): [eos] + state encoded ONCE into a KV cache on the native
    backbone (incl. its LoRA); every query's suffix (query + rendered options (+ null line +
    "Answer:" for letters) + terminal eos) then runs against that cached prefix, `chunk` queries per forward
    (right-padded, explicit dense mask + position_ids -- see _causal_pad_mask). h_D and the
    option spans are pooled exactly as native_features. queries = [(query, candidates)] ->
    list of [K_i + 1] probability vectors (null last): the same contract as model.decide,
    checked against run_batch_native in tests. Suffixes must fit max_suffix (no hierarchical
    fallback on this path: raise max_suffix instead)."""
    tok, dev, lm = head.backbone.tokenizer, head.device, head.backbone.model
    prefix = torch.tensor([[tok.eos_token_id] + _ids(tok, [state], max_state)[0]], device=dev)
    state_cache = lm(input_ids=prefix, use_cache=True).past_key_values
    Ls = prefix.shape[1]
    out = []
    for start in range(0, len(queries), chunk):
        qs = queries[start:start + chunk]
        m = len(qs)
        rendered = [RENDERS[getattr(model, "render", "letters")](q, c) for q, c in qs]
        enc = tok([r[0] for r in rendered], add_special_tokens=False, return_offsets_mapping=True)
        x_ids, offs = enc["input_ids"], enc["offset_mapping"]
        assert max(len(x) for x in x_ids) <= max_suffix, f"suffix > max_suffix={max_suffix}; raise it"
        input_ids, am, lengths = _pack(tok, [[]] * m, x_ids, tail=[tok.eos_token_id], sink=False)
        T = input_ids.shape[1]
        Kmax = max(len(c) for _, c in qs)
        pool = torch.stack([_pool_matrix(r[1], o, 0, Kmax, T, K=len(c)) for r, o, (_, c) in zip(rendered, offs, qs)])
        cmask = torch.zeros(m, Kmax, dtype=torch.bool)
        for i, (_, c) in enumerate(qs):
            cmask[i, :len(c)] = True

        cache = copy.deepcopy(state_cache)
        cache.batch_repeat_interleave(m)
        attn = torch.cat([torch.ones(m, Ls, dtype=torch.long), am], dim=1).to(dev)
        position_ids = (torch.arange(T) + Ls).expand(m, -1).to(dev)
        H = lm(input_ids=input_ids.to(dev), attention_mask=_causal_pad_mask(attn, T, lm.dtype),
               position_ids=position_ids, past_key_values=cache, use_cache=False).last_hidden_state.float()
        h = H[torch.arange(m, device=dev), (lengths - 1).to(dev)]
        C3 = torch.bmm(pool.to(dev), H)
        C2 = _n2_vectors(head, [c for _, c in qs], vec_cache, dev) if model.use2 else None
        probs = torch.softmax(model(h, cmask.to(dev), C3, C2), dim=-1)
        for i, (_, c) in enumerate(qs):
            out.append(torch.cat([probs[i, :len(c)], probs[i, -1:]]))
    return out


def shuffle_options(examples, rng):
    """PLAN4 sec 7: a fresh random option order per example (training only; eval keeps the given
    order). target/teacher/label are remapped so the gold candidate is unchanged -- letters are
    positional, so the model must read the option text. Rows with K < 2 pass through.
    Not for energy-readout multiset rows: meta.delta_t indexes the original order."""
    out = []
    for ex in examples:
        K = len(ex["candidates"])
        if K < 2:
            out.append(ex)
            continue
        perm = list(range(K))
        rng.shuffle(perm)
        ex = dict(ex, candidates=[ex["candidates"][p] for p in perm], target=[ex["target"][p] for p in perm])
        if ex.get("teacher") is not None:
            ex["teacher"] = [ex["teacher"][p] for p in perm] + list(ex["teacher"][K:])
        lab = ex.get("label")
        if isinstance(lab, int) and 0 <= lab < K:
            ex["label"] = perm.index(lab)
        out.append(ex)
    return out


def perm_consistency_loss(score_fn, logits, examples, rng, frac: float = 0.25):
    """PLAN5 sec 2 --perm_lambda: mean KL(P_1 || P_2) over a random ceil(frac*B) subset of rows
    between the candidate distributions (softmax over candidate logits, null excluded) of the
    batch's rendering (logits, already computed) and a fresh random option order of the same
    rows (score_fn(examples2, collate_mcq(examples2)) -> logits2, a second forward pass in the
    same step). P_2 is realigned to P_1's order by identity, so the term is 0 iff the scorer
    is order-equivariant, F(pi(A)) = pi(F(A)). Both passes receive gradient.
    ponytail: the subset is re-scored in full (+frac of the batch's tokens); K < 2 rows add 0."""
    B = len(examples)
    rows = sorted(rng.sample(range(B), max(1, math.ceil(B * frac))))
    perms, ex2 = [], []
    for i in rows:
        ex = examples[i]
        perm = list(range(len(ex["candidates"])))
        rng.shuffle(perm)
        perms.append(perm)
        ex2.append(dict(ex, candidates=[ex["candidates"][p] for p in perm], target=[ex["target"][p] for p in perm]))
    logits2 = score_fn(ex2, collate_mcq(ex2))
    kl = logits.new_zeros(())
    for r, (i, perm) in enumerate(zip(rows, perms)):
        K = len(perm)
        lp1 = torch.log_softmax(logits[i, :K], -1)
        lp2 = torch.log_softmax(logits2[r, :K], -1)
        inv = torch.empty(K, dtype=torch.long, device=logits.device)
        inv[torch.tensor(perm, device=logits.device)] = torch.arange(K, device=logits.device)  # lp2[inv[j]] scored original j
        kl = kl + (lp1.exp() * (lp1 - lp2[inv])).sum()
    return kl / len(rows)
