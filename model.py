"""Decision tower over backbone features (frozen + LoRA-adapted). See PLAN2.md."""
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from encode import TokenCandCache

D_IN = 2048
D = 512


class _MPSSafeDecoderLayer(nn.TransformerDecoderLayer):
    # ponytail: MPS's fused scaled_dot_product_attention kernel rejects dropout_p>0
    # (torch 2.14, "does not support dropout"). need_weights=True makes
    # MultiheadAttention take its manual (softmax+dropout) path instead of the
    # fused SDPA kernel, which works on every backend. Slightly slower than SDPA
    # on CUDA; irrelevant at this model's size. Drop this override once MPS SDPA
    # supports dropout.
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                            is_causal=is_causal, need_weights=True)[0]
        return self.dropout1(x)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal=False):
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                                 is_causal=is_causal, need_weights=True)[0]
        return self.dropout2(x)


class SetMixer(nn.Module):
    """One pre-norm self-attention block over the listwise token set [h; c_j + sim_j].
    Zero-init (attn.out_proj, last FF linear) so the block is the IDENTITY at init --
    a listwise model loaded with strict=False from a non-listwise checkpoint reproduces
    that checkpoint's logits exactly until the block trains away from init."""
    def __init__(self, d, nhead, dim_feedforward, dropout, mps_safe: bool = False, zero_init: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, dim_feedforward), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim_feedforward, d))
        if zero_init:
            nn.init.zeros_(self.attn.out_proj.weight); nn.init.zeros_(self.attn.out_proj.bias)
            nn.init.zeros_(self.ff[-1].weight); nn.init.zeros_(self.ff[-1].bias)
        # ponytail: attn dropout is 0 so MPS's fused-SDPA-with-dropout bug (see
        # _MPSSafeDecoderLayer above) shouldn't fire; mps_safe just reuses that same
        # manual-softmax (need_weights=True) fallback path in case it ever does.
        self.need_weights = mps_safe

    def forward(self, x, key_padding_mask):
        y = self.ln1(x)
        x = x + self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=self.need_weights)[0]
        return x + self.ff(self.ln2(x))


class DecisionModel(nn.Module):
    def __init__(self, d_in: int = D_IN, d: int = D, nhead: int = 8,
                 dim_feedforward: int = 1024, num_layers: int = 2, dropout: float = 0.1,
                 no_null: bool = False, cand_null: bool = True, hybrid: bool = True,
                 mps_safe: bool = False, listwise: bool = False, d_cand: int = None,
                 null: str = "softmax", head: str = "mlp", z_dim: int = 128, z_probes: int = 8,
                 tiny_layers: int = 0):
        super().__init__()
        assert null in ("softmax", "factored"), null
        assert head in ("mlp", "z1", "zr", "zr_set"), head
        self.null, self.head, self.need_weights = null, head, mps_safe
        # cand_null: null energy sees the best-matching candidate (score-weighted pool),
        # so it means "no candidate fits" rather than "this state looks unfamiliar".
        self.cand_null = cand_null
        self.hybrid = hybrid
        self.no_null = no_null  # ablation D: force K-way softmax (no null option)
        # d_cand: candidate/frozen-space width, when it differs from the backbone's d_in
        # (e.g. --cand_encoder qwen3emb, whose embeddings live in their own 1024-d space).
        d_cand = d_cand or d_in
        # per-dimension standardisation of backbone features (rogue dims hold ~30% of every
        # token's norm; per-token LayerNorm cannot remove them). Identity until calibrate() is called.
        self.register_buffer("mu_top", torch.zeros(d_in)); self.register_buffer("sd_top", torch.ones(d_in))
        self.register_buffer("mu_fz", torch.zeros(d_cand)); self.register_buffer("sd_fz", torch.ones(d_cand))
        self.proj_h = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.proj_q = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.proj_c = nn.Sequential(nn.LayerNorm(d_cand), nn.Linear(d_cand, d))
        self.ln_f = nn.LayerNorm(d_cand)  # shared norm for hybrid frozen-space features
        self.proj_sim = nn.Linear(2 * d_cand, d)
        self.slot = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.slot, std=0.02)
        Layer = _MPSSafeDecoderLayer if mps_safe else nn.TransformerDecoderLayer
        layer = Layer(d, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                      batch_first=True, norm_first=True)
        self.tower = nn.TransformerDecoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d)) if num_layers > 0 else None
        self.mixer = SetMixer(d, nhead, dim_feedforward, dropout, mps_safe=mps_safe) if listwise else None
        if head == "mlp":
            self.score = nn.Sequential(nn.Linear(5 * d + 2, d), nn.GELU(), nn.Linear(d, 1))
            dh = d
        else:
            # PLAN3 E3 student heads (see _z_forward). Candidate tokens [N, L, d_cand] (unique per
            # batch) -> proj_ct -> tiny_layers TRAINED pre-norm transformer blocks -> d' tokens.
            # tiny_layers=0: embedding table only. The null/factored heads below run in d'.
            dh = z_dim
            self.proj_ct = nn.Sequential(nn.LayerNorm(d_cand), nn.Linear(d_cand, z_dim))
            self.tiny = nn.ModuleList([SetMixer(z_dim, 4, 4 * z_dim, dropout, mps_safe=mps_safe, zero_init=False)
                                       for _ in range(tiny_layers)])
            if head == "z1":
                self.to_z = nn.Linear(d, z_dim)  # Z = linear projection of the pooled query state h
            else:
                self.probes = nn.Parameter(torch.randn(z_probes, z_dim) * 0.02)  # R learned probe queries
                self.probe_attn = nn.MultiheadAttention(z_dim, 4, kdim=d, vdim=d, batch_first=True)
                self.probe_ln = nn.LayerNorm(z_dim)
            if head == "zr_set":
                # one O(K) set-conditioning step: probes attend over ALL candidate tokens of the set.
                # zero-init out_proj -> zr_set starts as zr (loads a zr checkpoint with strict=False).
                self.set_attn = nn.MultiheadAttention(z_dim, 4, batch_first=True)
                nn.init.zeros_(self.set_attn.out_proj.weight); nn.init.zeros_(self.set_attn.out_proj.bias)
                self.set_ln = nn.LayerNorm(z_dim)
            # factorized pair scorer: s_j = Z.v_j + g^T(Z*v_j) + res(Z*v_j)
            self.g = nn.Linear(z_dim, 1)
            self.res = nn.Sequential(nn.Linear(z_dim, z_dim), nn.GELU(), nn.Linear(z_dim, 1))
        d_null = 5 * dh if cand_null else dh
        self.score_null = nn.Sequential(nn.Linear(d_null, dh // 2), nn.GELU(), nn.Linear(dh // 2, 1))
        # null="factored": r = sigmoid(g(z)), z = [max_j s, top2 margin, mean_j s, lse_j s - logK]
        # (4 scalars) + h + candidate-set mean/var (masked) -- see _factored_logits.
        # built only when used, so checkpoints from --null softmax load strictly in both directions
        self.null_gate = nn.Sequential(nn.Linear(4 + 3 * dh, 256), nn.GELU(), nn.Linear(256, 1)) if null == "factored" else None
        # temperature: only used by the factored null (see _factored_logits) to scale the
        # candidate softmax before composition. Left at 1.0 by training/eval (see train.py's
        # fit_temperature ponytail comment) -- a knob for a caller that wants it baked into forward().
        self.temperature = 1.0

    @torch.no_grad()
    def calibrate(self, h_top, h_frozen):
        """h_top/h_frozen: [N, d_in] token features from a sample of training text."""
        self.mu_top.copy_(h_top.mean(0)); self.sd_top.copy_(h_top.std(0).clamp_min(1e-3))
        self.mu_fz.copy_(h_frozen.mean(0)); self.sd_fz.copy_(h_frozen.std(0).clamp_min(1e-3))

    def forward(self, H, hmask, Q, qmask, C, cmask, Uf=None, Vf=None, ctok=None, ctmask=None, cidx=None):
        """ctok/ctmask/cidx (heads z1/zr/zr_set, --cand_encoder tiny): token-level candidates as
        [N, L, d_cand] unique candidates + [N, L] mask + [B, Kmax] index of each slot's unique row.
        None -> each candidate is one "token" = its pooled vector C (backbone / qwen3emb tiers)."""
        H, Q = (H - self.mu_top) / self.sd_top, (Q - self.mu_top) / self.sd_top
        C = (C - self.mu_fz) / self.sd_fz
        if Uf is not None:
            Uf, Vf = (Uf - self.mu_fz) / self.sd_fz, (Vf - self.mu_fz) / self.sd_fz
        B = H.shape[0]
        qp = self.proj_q(Q)  # [B, Lq, d]

        if self.tower is None:  # ablation: no cross-attention slot; h = mean of projected query tokens
            m = qmask.unsqueeze(-1).float()
            h = (qp * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            tgt = torch.cat([self.slot.expand(B, 1, -1), qp], dim=1)  # [B, 1+Lq, d]
            tgt_kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=H.device), ~qmask], dim=1)
            h = self.tower(
                tgt, self.proj_h(H),
                tgt_key_padding_mask=tgt_kpm,
                memory_key_padding_mask=~hmask,
            )[:, 0]  # [B, d]

        if self.head != "mlp":
            return self._z_forward(h, qp, qmask, C, cmask, ctok, ctmask, cidx)

        c = self.proj_c(C)  # [B, Kmax, d] -- pooled before projection (C is already mean-pooled)

        if self.hybrid:
            assert Uf is not None and Vf is not None, "hybrid=True requires Uf, Vf"
            Kmax = C.shape[1]
            Cn = self.ln_f(C)
            Ufn = self.ln_f(Uf).unsqueeze(1).expand(-1, Kmax, -1)
            Vfn = self.ln_f(Vf).unsqueeze(1).expand(-1, Kmax, -1)
            sim = self.proj_sim(torch.cat([Ufn * Cn, Vfn * Cn], dim=-1))  # [B, Kmax, d]
            cos_u = F.cosine_similarity(Uf.unsqueeze(1), C, dim=-1).unsqueeze(-1)
            cos_v = F.cosine_similarity(Vf.unsqueeze(1), C, dim=-1).unsqueeze(-1)
        else:
            sim = torch.zeros_like(c)
            cos_u = torch.zeros(B, C.shape[1], 1, device=H.device, dtype=c.dtype)
            cos_v = torch.zeros_like(cos_u)

        if self.mixer is not None:  # listwise: one set-attention pass over [h; c_j + sim_j]
            x = torch.cat([h.unsqueeze(1), c + sim], dim=1)  # [B, 1+Kmax, d]
            kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=H.device), ~cmask], dim=1)
            delta = self.mixer(x, kpm) - x
            h = h + delta[:, 0]
            c = c + delta[:, 1:]

        h_exp = h.unsqueeze(1).expand_as(c)
        feats = torch.cat([h_exp, c, h_exp * c, (h_exp - c).abs(), sim, cos_u, cos_v], dim=-1)
        hid = self.score[0](feats)                              # scorer hidden (pre-GELU), [B, Kmax, d]
        s = self.score[2](self.score[1](hid)).squeeze(-1)       # [B, Kmax]
        return self._null_logits(s, c, h, hid, cmask)

    def _z_forward(self, h, qp, qmask, C, cmask, ctok, ctmask, cidx):
        """PLAN3 E3 students. Z = g(state, query) never sees candidate identities (z1: one vector
        from h; zr/zr_set: R probes cross-attending over the query tokens). Candidates arrive as
        tokens, encoded by the trainable tiny stack, and are scored by late interaction:
        s_j = Z.v_j + g^T(Z*v_j) + res(Z*v_j) on pooled vectors (+ ColBERT MaxSim
        mean_r max_t z_r.c_jt for zr/zr_set). zr_set adds ONE cross-attention from the probes over
        all candidate tokens of the set before scoring -- O(K*R*d'), the only IIA-breaking step."""
        B, Kmax = cmask.shape
        if ctok is None:  # pooled encoders: one "token" per candidate = its (z-scored) pooled vector
            ctok, ctmask = C.reshape(B * Kmax, 1, -1), cmask.reshape(B * Kmax, 1)
            cidx = torch.arange(B * Kmax, device=C.device).view(B, Kmax)
        t = self.proj_ct(ctok)                                                   # [N, L, d']
        kpm = torch.cat([torch.zeros_like(ctmask[:, :1]), ~ctmask[:, 1:]], 1)  # never all-masked (pad slots) -> no NaN
        for layer in self.tiny:
            t = layer(t, kpm)
        tm = ctmask[cidx] & cmask.unsqueeze(-1)                                  # [B, Kmax, L]
        tk = t[cidx].masked_fill(~tm.unsqueeze(-1), 0.0)                         # [B, Kmax, L, d']
        vk = tk.sum(2) / tm.sum(2, keepdim=True).clamp_min(1)                    # [B, Kmax, d'] pooled

        if self.head == "z1":
            Z = self.to_z(h).unsqueeze(1)                                        # [B, 1, d']
        else:
            p = self.probes.unsqueeze(0).expand(B, -1, -1)                       # [B, R, d']
            Z = self.probe_ln(p + self.probe_attn(p, qp, qp, key_padding_mask=~qmask,
                                                  need_weights=self.need_weights)[0])
        if self.head == "zr_set":
            keys = tk.flatten(1, 2)                                              # [B, Kmax*L, d']
            Z = self.set_ln(Z + self.set_attn(Z, keys, keys, key_padding_mask=~tm.flatten(1, 2),
                                              need_weights=self.need_weights)[0])
        scale = Z.shape[-1] ** -0.5
        zbar = Z.mean(1)                                                         # [B, d']
        zv = zbar.unsqueeze(1) * vk                                              # [B, Kmax, d']
        hid = self.res[0](zv)
        s = zv.sum(-1) * scale + self.g(zv).squeeze(-1) + self.res[2](self.res[1](hid)).squeeze(-1)
        if self.head != "z1":  # MaxSim, one einsum over the padded token set (mean over probes: O(1) logit scale at init)
            sims = torch.einsum("brd,bkld->bkrl", Z, tk) * scale
            s = s + sims.masked_fill(~tm[:, :, None, :], -1e4).max(-1).values.mean(-1)
        return self._null_logits(s, vk, zbar, hid, cmask)

    def _null_logits(self, s, c, h, hid, cmask):
        """Shared tail: pad-mask s, then the candidate-aware null (score_null over the
        score-weighted best candidate / scorer hidden) or the factored null."""
        s = s.masked_fill(~cmask, torch.finfo(s.dtype).min)
        if self.null == "factored":  # no_null/cand_null don't apply -- see _factored_logits
            return self._factored_logits(s, c, h, cmask)

        if self.cand_null:
            w = torch.softmax(s, dim=-1).unsqueeze(-1)          # [B, Kmax, 1], pads already ~0
            c_best = (w * c).sum(1)                              # [B, d]
            hid_best = (w * hid).sum(1)                          # pooled scorer hidden, [B, d]
            s_null = self.score_null(torch.cat([h, c_best, h * c_best, (h - c_best).abs(), hid_best], -1)).squeeze(-1)
        else:
            s_null = self.score_null(h).squeeze(-1)  # [B]
        if self.no_null:
            s_null = torch.full_like(s_null, torch.finfo(s_null.dtype).min)

        return torch.cat([s, s_null.unsqueeze(-1)], dim=-1)  # [B, Kmax+1]

    def _factored_logits(self, s, c, h, cmask):
        return factored_null_logits(self.null_gate, s, c, h, cmask, self.temperature)


def factored_null_logits(gate, s, c, h, cmask, temperature: float = 1.0):
    """r = sigmoid(gate(z)): z is O(K) permutation-invariant set statistics over the valid
    candidate scores s (max, top-2 margin, mean, logsumexp-avg) + rep h + masked
    candidate-set mean/var of c (gate input width = 4 + h.shape[-1] + 2*c.shape[-1]).
    p_j = softmax(s/T) over candidates only (exact IIA: p_j/p_k doesn't depend on the rest
    of the set). Composed as logits so softmax(logits) gives P(a_j) = (1-r)*p_j and
    P(null) = r exactly:  logits_j = log p_j + log(1-r), logits_null = log r.
    Shared by DecisionModel (--null factored) and native.NativeHead."""
    K = cmask.sum(-1)                       # [B]
    Kf = K.float().clamp_min(1)
    top2 = torch.topk(s, k=min(2, s.shape[1]), dim=-1).values
    margin = top2[:, 0] - top2[:, 1] if top2.shape[1] == 2 else torch.zeros_like(top2[:, 0])
    margin = torch.where(K > 1, margin, torch.zeros_like(margin))  # K=1 -> 0 (no second candidate)
    mean_s = s.masked_fill(~cmask, 0.0).sum(-1) / Kf
    lse = torch.logsumexp(s, dim=-1) - torch.log(Kf)
    c0 = c.masked_fill(~cmask.unsqueeze(-1), 0.0)
    mean_c = c0.sum(1) / Kf.unsqueeze(-1)
    var_c = (c0 - mean_c.unsqueeze(1)).pow(2).masked_fill(~cmask.unsqueeze(-1), 0.0).sum(1) / Kf.unsqueeze(-1)
    z = torch.cat([top2[:, :1], margin.unsqueeze(-1), mean_s.unsqueeze(-1), lse.unsqueeze(-1),
                   h, mean_c, var_c], dim=-1)
    r_logit = gate(z).squeeze(-1)
    log_1mr, log_r = -F.softplus(r_logit), -F.softplus(-r_logit)

    # ponytail: temperature tempers only the candidate softmax (r is a separate head over
    # untempered s-statistics, so it's exactly T-invariant) -- zero pad, scale, THEN re-mask,
    # since scaling an already-finfo.min pad value by 1/T<1 can overflow to -inf.
    s_t = (s.masked_fill(~cmask, 0.0) / temperature).masked_fill(~cmask, torch.finfo(s.dtype).min)
    logp = F.log_softmax(s_t, dim=-1)
    cand_logits = (logp + log_1mr.unsqueeze(-1)).masked_fill(~cmask, torch.finfo(s.dtype).min)
    return torch.cat([cand_logits, log_r.unsqueeze(-1)], dim=-1)


def decision_loss(logits, target, p_null, cmask, teacher=None, has_teacher=None,
                   alpha: float = 1.0, beta: float = 0.0, T: float = 2.0,
                   delta_idx=None, delta_tgt=None, gamma: float = 0.0):
    """Soft-CE against the hard/soft gold target (weight alpha), plus PLAN3 E3-T's
    distillation term beta * T^2 * KL(P_T^T || P_S^T) for rows carrying a `teacher`
    field (scripts/teacher_label.py) -- see collate_teacher. beta=0 (default) reproduces
    the original loss exactly, so every existing call site/test is unaffected.
    teacher/has_teacher: [B, Kmax+1] / [B], from collate_teacher (padded with 0 / False).
    gamma > 0 (PLAN3 E3-ms): + gamma * mean Huber(D^S_ij - D^T_ij) over collate_delta's triples,
    D^S_ij = [s_i - s_j](variant row) - [s_i - s_j](orig row) from the candidate logits (the
    per-row null composition cancels in the difference). Candidate-blind heads (z1/zr) have
    D^S == 0 -> constant, no gradient; only set-conditioned heads (zr_set) can fit it."""
    t = torch.cat([(1 - p_null).unsqueeze(-1) * target, p_null.unsqueeze(-1)], dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    loss = alpha * -(t * logp).sum(-1)
    if beta > 0 and teacher is not None:
        valid = torch.cat([cmask, torch.ones(cmask.shape[0], 1, dtype=torch.bool, device=cmask.device)], dim=-1)
        neg = torch.finfo(logits.dtype).min
        logp_s = F.log_softmax(logits.masked_fill(~valid, neg) / T, dim=-1)
        # ponytail: we only ever persist the teacher's T=1 probabilities (not logits), so
        # re-tempering to T is log(p) -> /T -> softmax again -- exact for T_orig=1 (the
        # lse(z) constant introduced by log-then-exp cancels in the softmax), see PLAN3.md E3-T.
        logt = torch.log(teacher.clamp_min(1e-12)).masked_fill(~valid, neg)
        logp_t = F.log_softmax(logt / T, dim=-1)
        kl = (logp_t.exp() * (logp_t - logp_s)).sum(-1)
        mask = has_teacher.to(kl.dtype) if has_teacher is not None else torch.ones_like(kl)
        loss = loss + beta * (T ** 2) * kl * mask
    loss = loss.mean()
    if gamma > 0 and delta_idx is not None and len(delta_idx):
        s = logits[:, :-1]
        r, i, j, ro, io, jo = delta_idx.unbind(1)
        d_s = (s[r, i] - s[r, j]) - (s[ro, io] - s[ro, jo])
        loss = loss + gamma * F.huber_loss(d_s, delta_tgt)
    return loss


def collate_teacher(examples, Kmax):
    """[B, Kmax+1] teacher probs (candidates... + null last, zero-padded) + [B] has_teacher
    mask, from each example's optional `teacher` field (scripts/teacher_label.py: K+1 floats,
    K = len(example["candidates"])). Shared by model.collate and mcq.collate_mcq."""
    B = len(examples)
    teacher = torch.zeros(B, Kmax + 1, dtype=torch.float32)
    has_teacher = torch.zeros(B, dtype=torch.bool)
    for i, ex in enumerate(examples):
        tvec = ex.get("teacher")
        if tvec is None:
            continue
        K = len(ex["candidates"])
        teacher[i, :K] = torch.tensor(tvec[:K], dtype=torch.float32)
        teacher[i, Kmax] = tvec[K]  # null is always the last of the K+1 teacher floats
        has_teacher[i] = True
    return teacher, has_teacher


def collate_delta(examples):
    """PLAN3 E3-ms (scripts/ms_targets.py): every row's meta.delta_t triples [i, j, D^T_ij] whose orig
    row (same meta.ms_group, ms_variant == "orig") is in this batch -> [M, 6] long
    (variant row, i, j, orig row, i_orig, j_orig; orig indices aligned by candidate string) +
    [M] float D^T. Rows without delta_t, or whose orig isn't in the batch, contribute nothing."""
    orig = {ex["meta"]["ms_group"]: r for r, ex in enumerate(examples) if ex.get("meta", {}).get("ms_variant") == "orig"}
    idx, tgt = [], []
    for r, ex in enumerate(examples):
        meta = ex.get("meta", {})
        ro = orig.get(meta.get("ms_group"))
        if ro is None or not meta.get("delta_t"):
            continue
        oc = examples[ro]["candidates"]
        for i, j, d in meta["delta_t"]:
            idx.append([r, i, j, ro, oc.index(ex["candidates"][i]), oc.index(ex["candidates"][j])])
            tgt.append(d)
    return torch.tensor(idx, dtype=torch.long).view(-1, 6), torch.tensor(tgt, dtype=torch.float32)


def collate(cache, examples):
    """examples: list of JSONL-schema dicts. No state/query tokenisation here -- they stay strings.
    cache: FeatureCache/VecCache -> pooled "C" [B, Kmax, d]; TokenCandCache (--cand_encoder tiny)
    -> candidate token ids "ctok_ids" [N, L] + "ctmask" [N, L] + "cidx" [B, Kmax] (embedded in
    run_batch -- the encoder is trainable and runs inside the model)."""
    Kmax = max(len(ex["candidates"]) for ex in examples)

    C_list, cmask_list, target_list, p_null_list = [], [], [], []
    for ex in examples:
        K = len(ex["candidates"])
        if not isinstance(cache, TokenCandCache):
            feats = [cache.pooled(c) for c in ex["candidates"]]
            d_in = feats[0].shape[0]  # works for FeatureCache (frozen dim) and VecCache (embed dim)
            feats += [torch.zeros(d_in, dtype=torch.float32, device=cache.device)] * (Kmax - K)
            C_list.append(torch.stack(feats))
        cmask_list.append([i < K for i in range(Kmax)])
        target_list.append(list(ex["target"]) + [0.0] * (Kmax - K))
        p_null_list.append(ex["p_null"])

    cmask = torch.tensor(cmask_list, dtype=torch.bool)
    assert cmask.any(-1).all(), "every example needs at least one candidate slot"
    teacher, has_teacher = collate_teacher(examples, Kmax)
    batch = {
        "state": [ex["state"] for ex in examples],
        "query": [ex["query"] for ex in examples],
        "cmask": cmask,
        "target": torch.tensor(target_list, dtype=torch.float32),
        "p_null": torch.tensor(p_null_list, dtype=torch.float32),
        "task": [ex["task"] for ex in examples],
        "teacher": teacher,
        "has_teacher": has_teacher,
    }
    batch["delta_idx"], batch["delta_tgt"] = collate_delta(examples)
    if isinstance(cache, TokenCandCache):
        batch["ctok_ids"], batch["ctmask"], batch["cidx"], _ = cache.batch([ex["candidates"] for ex in examples])
    else:
        batch["C"] = torch.stack(C_list)
    return batch


def _embed_cand_tokens(backbone, ids, tmask, cidx, dev):
    """--cand_encoder tiny: frozen input-embedding lookup of the unique candidate tokens
    [N, L, hidden] + per-slot pooled C [B, Kmax, hidden] (masked mean; used by the mlp head,
    hybrid sim and zscore calibration). Returns (C, forward kwargs for the token-level heads)."""
    ids, tmask, cidx = ids.to(dev), tmask.to(dev), cidx.to(dev)
    with torch.no_grad():
        ctok = backbone.model.embed_tokens(ids).float()
    pooled = (ctok * tmask.unsqueeze(-1)).sum(1) / tmask.sum(1, keepdim=True).clamp_min(1)
    return pooled[cidx], {"ctok": ctok, "ctmask": tmask, "cidx": cidx}


def _batch_cands(backbone, batch, dev):
    """(C, cmask, token kwargs) for run_batch from a collate() batch of either cache type."""
    if "ctok_ids" in batch:
        C, tok = _embed_cand_tokens(backbone, batch["ctok_ids"], batch["ctmask"], batch["cidx"], dev)
    else:
        C, tok = batch["C"].to(dev), {}
    return C, batch["cmask"].to(dev), tok


def _masked_mean(h_frozen, mask):
    return (h_frozen * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)


def split_joint(h_top, h_frozen, mask, ls, lq):
    """Split a joint [eos]+state+query pass back into state/query blocks.
    -> (H, hmask, Q, qmask, Hf, Qf), each [B, L*_max, d] / [B, L*_max] bool."""
    B, T, d = h_top.shape
    dev = h_top.device
    ls, lq = ls.to(dev), lq.to(dev)
    Ls_max, Lq_max = int(ls.max()), int(lq.max())

    idx_s = torch.arange(Ls_max, device=dev)[None, :].expand(B, -1)
    hmask = idx_s < ls[:, None]
    idx_s = idx_s.clamp(max=T - 1).unsqueeze(-1)
    H, Hf = torch.gather(h_top, 1, idx_s.expand(-1, -1, d)), torch.gather(h_frozen, 1, idx_s.expand(-1, -1, h_frozen.shape[-1]))

    idx_q = ls[:, None] + torch.arange(Lq_max, device=dev)[None, :]
    qmask = torch.arange(Lq_max, device=dev)[None, :] < lq[:, None]
    idx_q = idx_q.clamp(max=T - 1).unsqueeze(-1)
    Q, Qf = torch.gather(h_top, 1, idx_q.expand(-1, -1, d)), torch.gather(h_frozen, 1, idx_q.expand(-1, -1, h_frozen.shape[-1]))

    return H, hmask, Q, qmask, Hf, Qf


def run_batch(backbone, model, batch, max_state: int = 256, max_query: int = 64, joint: bool = False,
              vec_cache=None):
    """vec_cache: when set (--cand_encoder qwen3emb), Uf/Vf come from the embedding cache
    instead of masked-mean frozen backbone features -- everything else is unchanged."""
    dev = backbone.device
    if joint:
        input_ids, attention_mask, ls, lq = backbone.tokenize_joint(batch["state"], batch["query"], max_state, max_query)
        input_ids, attention_mask = input_ids.to(dev), attention_mask.to(dev)
        h_top, h_frozen, mask = backbone(input_ids, attention_mask)
        H, hmask, Q, qmask, Hf, Qf = split_joint(h_top, h_frozen, mask, ls, lq)
        if vec_cache is not None:
            Uf = torch.stack([vec_cache.pooled(s) for s in batch["state"]]).to(dev)
            Vf = torch.stack([vec_cache.pooled(q) for q in batch["query"]]).to(dev)
        else:
            Uf, Vf = _masked_mean(Hf, hmask), _masked_mean(Qf, qmask)

        C, cmask, tok = _batch_cands(backbone, batch, dev)
        batch["n_tokens"] = int(attention_mask.sum().item())
        return model(H, hmask, Q, qmask, C, cmask, Uf, Vf, **tok)

    ids_s, am_s = backbone.tokenize(batch["state"], max_state)
    ids_q, am_q = backbone.tokenize(batch["query"], max_query)
    ids_s, am_s = ids_s.to(dev), am_s.to(dev)
    ids_q, am_q = ids_q.to(dev), am_q.to(dev)

    h_top_s, h_frozen_s, mask_s = backbone(ids_s, am_s)
    h_top_q, h_frozen_q, mask_q = backbone(ids_q, am_q)
    if vec_cache is not None:
        Uf = torch.stack([vec_cache.pooled(s) for s in batch["state"]]).to(dev)
        Vf = torch.stack([vec_cache.pooled(q) for q in batch["query"]]).to(dev)
    else:
        Uf = _masked_mean(h_frozen_s, mask_s)
        Vf = _masked_mean(h_frozen_q, mask_q)

    C, cmask, tok = _batch_cands(backbone, batch, dev)
    batch["n_tokens"] = int(am_s.sum().item() + am_q.sum().item())

    return model(h_top_s, mask_s, h_top_q, mask_q, C, cmask, Uf, Vf, **tok)


def _decide_candidates(backbone, cache, dev, qs, encoder=None):
    """encoder: EmbedEncoder, for candidates missing from a VecCache (--cand_encoder
    qwen3emb) -- mirrors the frozen_features fallback used for a plain FeatureCache.
    Unique candidates are resolved once (cache lookups stacked on CPU, one transfer; misses
    embedded in one batched call), then scattered into [m, Kmax, d] -- O(K) Python work
    per chunk was the bench's K-scaling artifact.
    -> (C, cmask, Ks, token kwargs for the model). TokenCandCache: ids come from the cache or
    are tokenised on the fly (cold path), then one embedding lookup."""
    Ks = [len(c) for _, c in qs]
    if isinstance(cache, TokenCandCache):
        ids, tmask, cidx, cmask = cache.batch([c for _, c in qs])
        C, tok = _embed_cand_tokens(backbone, ids, tmask, cidx, dev)
        return C, cmask.to(dev), Ks, tok
    Kmax = max(Ks)
    uniq = list(dict.fromkeys(c for _, cands in qs for c in cands))
    vecs = {}
    cached = [c for c in uniq if cache is not None and c in cache.index]
    if cached:
        stacked = torch.stack([cache.pooled(c) for c in cached]).float().to(dev)
        vecs.update(zip(cached, stacked))
    missing = [c for c in uniq if c not in vecs]
    if missing:
        if encoder is not None:
            from encode import CAND_RENDER
            emb = encoder.embed(missing, render=CAND_RENDER).to(dev)
        else:
            emb = torch.stack([backbone.frozen_features([c], 16)[0].float().mean(0) for c in missing]).to(dev)
        vecs.update(zip(missing, emb))
    d = next(iter(vecs.values())).shape[-1]
    C = torch.zeros(len(qs), Kmax, d, dtype=torch.float32, device=dev)
    cmask = torch.zeros(len(qs), Kmax, dtype=torch.bool, device=dev)
    for i, (_, cands) in enumerate(qs):
        C[i, :len(cands)] = torch.stack([vecs[c] for c in cands])
        cmask[i, :len(cands)] = True
    return C, cmask, Ks, {}


def _vecs_or_embed(cache, encoder, texts, dev, render=None):
    """[len(texts), d]: cached vectors where available, one batched embed call for the rest
    (render=CAND_RENDER for candidates, matching how the cache was built)."""
    out = [cache.pooled(t).to(dev) if cache is not None and t in cache.index else None for t in texts]
    missing = [i for i, v in enumerate(out) if v is None]
    if missing:
        vecs = encoder.embed([texts[i] for i in missing], render=render).to(dev)
        for j, i in enumerate(missing):
            out[i] = vecs[j]
    return torch.stack(out)


@torch.inference_mode()
def decide(backbone, model, cache, state: str, queries: list[tuple[str, list[str]]], chunk: int = 64,
           joint: bool = False, encoder=None, max_state: int = 256, max_query: int = 64):
    """Encode state once; process queries in chunks, expanding the state to each chunk size.
    encoder: EmbedEncoder, used in place of the backbone's frozen features for Uf/Vf/C when
    cache is a VecCache (--cand_encoder qwen3emb); falls back to embedding on the fly for
    any state/query/candidate string not already in cache."""
    dev = backbone.device

    if joint:
        # state pass via backbone.model directly (not backbone.forward): the KV cache
        # must KEEP the sink token (position 0) for later attention, while forward()
        # drops it from its *returned* tensors -- so we tap hidden_states ourselves and
        # drop the sink manually, mirroring forward()'s [:, 1:] slicing.
        ids_s, am_s = backbone.tokenize([state], max_state)
        ids_s, am_s = ids_s.to(dev), am_s.to(dev)
        out_s = backbone.model(input_ids=ids_s, attention_mask=am_s, use_cache=True, output_hidden_states=True)
        H = backbone.top(out_s)[:, 1:].float()
        Hf = backbone.model.norm(out_s.hidden_states[backbone.split_layer])[:, 1:].float()
        hmask = am_s.bool()[:, 1:]
        Uf = _vecs_or_embed(cache, encoder, [state], dev) if encoder is not None \
            else _masked_mean(Hf, hmask)
        Ls_total, Lh = ids_s.shape[1], H.shape[1]

        out = []
        for start in range(0, len(queries), chunk):
            qs = queries[start:start + chunk]
            m = len(qs)
            ids_q, am_q = backbone.tokenize_plain([q for q, _ in qs], max_query)
            ids_q, am_q = ids_q.to(dev), am_q.to(dev)
            Lq = ids_q.shape[1]

            cache_copy = copy.deepcopy(out_s.past_key_values)
            cache_copy.batch_repeat_interleave(m)
            attn = torch.cat([torch.ones(m, Ls_total, dtype=am_q.dtype, device=dev), am_q], dim=1)
            cache_position = torch.arange(Ls_total, Ls_total + Lq, device=dev)
            out_q = backbone.model(input_ids=ids_q, attention_mask=attn, past_key_values=cache_copy,
                                    cache_position=cache_position, output_hidden_states=True, use_cache=False)
            Q = backbone.top(out_q).float()  # no sink in the query pass -- nothing to drop
            Qf = backbone.model.norm(out_q.hidden_states[backbone.split_layer]).float()
            qmask = am_q.bool()
            Vf = _vecs_or_embed(cache, encoder, [q for q, _ in qs], dev) if encoder is not None \
                else _masked_mean(Qf, qmask)

            C, cmask, Ks, tok = _decide_candidates(backbone, cache, dev, qs, encoder=encoder)
            logits = model(H.expand(m, Lh, -1), hmask.expand(m, Lh), Q, qmask, C, cmask, Uf.expand(m, -1), Vf, **tok)
            probs = F.softmax(logits, dim=-1)
            Kmax = C.shape[1]
            for i, K in enumerate(Ks):
                out.append(torch.cat([probs[i, :K], probs[i, Kmax:Kmax + 1]]))
        return out

    ids_s, am_s = backbone.tokenize([state], max_state)
    h_top_s, h_frozen_s, mask_s = backbone(ids_s.to(dev), am_s.to(dev))
    Uf = _vecs_or_embed(cache, encoder, [state], dev) if encoder is not None \
        else _masked_mean(h_frozen_s, mask_s)
    Lh = h_top_s.shape[1]

    out = []
    for start in range(0, len(queries), chunk):
        qs = queries[start:start + chunk]
        m = len(qs)
        ids_q, am_q = backbone.tokenize([q for q, _ in qs], max_query)
        h_top_q, h_frozen_q, mask_q = backbone(ids_q.to(dev), am_q.to(dev))
        Vf = _vecs_or_embed(cache, encoder, [q for q, _ in qs], dev) if encoder is not None \
            else _masked_mean(h_frozen_q, mask_q)

        C, cmask, Ks, tok = _decide_candidates(backbone, cache, dev, qs, encoder=encoder)
        Kmax = C.shape[1]
        logits = model(h_top_s.expand(m, Lh, -1), mask_s.expand(m, Lh), h_top_q, mask_q, C, cmask,
                       Uf.expand(m, -1), Vf, **tok)
        probs = F.softmax(logits, dim=-1)
        for i, K in enumerate(Ks):
            out.append(torch.cat([probs[i, :K], probs[i, Kmax:Kmax + 1]]))
    return out


if __name__ == "__main__":
    from encode import Backbone, FeatureCache, pick_device

    device = pick_device("auto")
    bb = Backbone(name="Qwen/Qwen3-0.6B-Base", lora_layers=4, lora_r=8, device=device)

    cands = ["entailment", "neutral", "contradiction", "yes", "no", "maybe"]
    cache = FeatureCache(device="cpu")
    cache.add(bb, cands, max_len=16)

    random.seed(0)
    examples = []
    for K in (1, 2, 3, 5):
        chosen = random.sample(cands, K)
        tgt = [1.0] + [0.0] * (K - 1)
        examples.append({
            "state": f"This is example state number {K}.",
            "query": f"What is the answer for K={K}?",
            "candidates": chosen,
            "target": tgt,
            "p_null": 0.1,
            "task": "smoke",
        })

    batch = collate(cache, examples)
    model = DecisionModel(d_in=bb.d, mps_safe=(device == "mps")).to(device)

    n_tower = sum(p.numel() for p in model.parameters())
    n_lora = sum(p.numel() for p in bb.trainable_parameters())
    print(f"params: tower={n_tower:,} lora={n_lora:,}")

    logits = run_batch(bb, model, batch)
    loss = decision_loss(logits, batch["target"].to(device), batch["p_null"].to(device), batch["cmask"].to(device))
    loss.backward()
    assert torch.isfinite(loss).all(), "loss not finite"

    lora_grad = all(p.grad is not None for p in bb.trainable_parameters())
    base_grad_none = all(p.grad is None for n, p in bb.named_parameters() if not n.endswith((".A", ".B")))
    tower_grad = any(p.grad is not None for p in model.parameters())
    assert lora_grad, "some LoRA params missing grads"
    assert base_grad_none, "base backbone params received grads"
    assert tower_grad, "tower got no grads"
    print(f"loss={loss.item():.4f}; grads OK (lora + tower present, base none)")

    model.zero_grad()
    bb.zero_grad()
    model.eval()
    perm = [2, 0, 3, 1]
    batch_perm = collate(cache, [examples[i] for i in perm])
    with torch.no_grad():
        logits_orig = run_batch(bb, model, batch)
        logits_perm = run_batch(bb, model, batch_perm)
    assert torch.allclose(logits_perm, logits_orig[perm], atol=1e-3), "batch permutation not equivariant"
    print("permutation equivariance: OK")

    queries = [(f"query {i}", random.sample(cands, random.randint(1, 4))) for i in range(5)]
    dists = decide(bb, model, cache, "some state text", queries, chunk=2)
    assert len(dists) == 5
    for d in dists:
        assert torch.allclose(d.sum(), torch.tensor(1.0), atol=1e-3)
    print("decide(): OK, 5 distributions correctly sized and sum to 1")

    model_nh = DecisionModel(d_in=bb.d, hybrid=False, mps_safe=(device == "mps")).to(device)
    logits_nh = run_batch(bb, model_nh, batch)
    assert torch.isfinite(logits_nh).all()
    print("hybrid=False path: OK")

    model_lw = DecisionModel(d_in=bb.d, listwise=True, mps_safe=(device == "mps")).to(device)
    n_mixer = sum(p.numel() for p in model_lw.mixer.parameters())
    logits_lw = run_batch(bb, model_lw, batch)
    assert torch.isfinite(logits_lw).all()
    print(f"listwise=True path: OK (mixer params={n_mixer:,})")

    print("model.py smoke test passed")
