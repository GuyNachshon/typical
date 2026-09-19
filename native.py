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
"""
import bisect

import torch
import torch.nn as nn

from encode import EMBED_DIM, CAND_RENDER
from mcq import _render, _ids, _pack, _score_chunked
from model import factored_null_logits, _vecs_or_embed

MAX_SUFFIX = 1024


class NativeHead(nn.Module):
    def __init__(self, d: int, nc_head: str = "n2n3", null: str = "factored", d_proj: int = 256,
                 d_emb: int = EMBED_DIM):
        super().__init__()
        assert nc_head in ("n2", "n3", "n2n3"), nc_head
        assert null in ("softmax", "factored"), null
        self.nc_head, self.null = nc_head, null
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


def _pool_matrix(spans, offsets, base, Kmax, T):
    """[Kmax+1, T] mean-pooling rows: option k over the suffix tokens overlapping its char span
    (row Kmax = the null line). Tokens overlapping [cs, ce) are contiguous, found by bisection."""
    starts, ends = [o[0] for o in offsets], [o[1] for o in offsets]
    pool = torch.zeros(Kmax + 1, T)
    for k, (cs, ce) in enumerate(spans):
        j0, j1 = bisect.bisect_right(ends, cs), bisect.bisect_left(starts, ce)
        pool[Kmax if k == len(spans) - 1 else k, base + j0:base + j1] = 1.0
    return pool / pool.sum(-1, keepdim=True).clamp_min(1.0)


def native_features(head, states, queries, cand_lists, max_state=256, max_suffix=MAX_SUFFIX):
    """One suffix pass. -> (h_D [B, d], C3 [B, Kmax+1, d] (null line last), cmask [B, Kmax],
    n_tokens). Suffixes are assumed to fit max_suffix (run_batch_native chunks the rest); a
    truncated option pools to a zero vector."""
    tok, dev = head.backbone.tokenizer, head.device
    rendered = [_render(q, c) for q, c in zip(queries, cand_lists)]
    s_ids = _ids(tok, states, max_state)
    x_ids, offs = _ids(tok, [r[0] for r in rendered], max_suffix, offsets=True)
    input_ids, am, lengths = _pack(tok, s_ids, x_ids, tail=[tok.eos_token_id])
    B, T = input_ids.shape
    Kmax = max(len(c) for c in cand_lists)
    pool = torch.stack([_pool_matrix(r[1], o, 1 + len(s), Kmax, T) for r, o, s in zip(rendered, offs, s_ids)])
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


def _fit_chunks(tok, query, cand_texts, max_suffix):
    """Consecutive option index chunks whose rendered suffix fits max_suffix tokens (per-line
    counts are exact for single-token letters; +2/option slack covers numeric labels)."""
    lens = [len(x) for x in tok([f"A. {c}\n" for c in cand_texts], add_special_tokens=False)["input_ids"]]
    overhead = len(tok(query + "\n", add_special_tokens=False)["input_ids"]) \
        + len(tok("A. none of the above\nAnswer:", add_special_tokens=False)["input_ids"]) + 2
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

    def score(sts, qs, cls):
        h, C3, cm, n_tok = native_features(head, sts, qs, cls, max_state, max_suffix)
        C2 = _n2_vectors(head, cls, vec_cache, dev) if model.use2 else None
        return model(h, cm, C3, C2), n_tok

    n_suf = [len(x) for x in tok([_render(q, c)[0] for q, c in zip(queries, cand_lists)],
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
        chunks = _fit_chunks(tok, queries[i], cand_lists[i], max_suffix)
        row, n_tok = _score_chunked(lambda cl: score([states[i]] * len(cl), [queries[i]] * len(cl), cl),
                                    cand_lists[i], chunks)
        n_tokens += n_tok
        k = len(cand_lists[i])
        logits[i, :k] = row[:k]
        logits[i, -1] = row[-1]
    batch["n_tokens"] = n_tokens
    return logits


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
