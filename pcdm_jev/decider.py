"""One JevBench decision -> PCDM probabilities over the exact label set.

Mapping (jevbench's local_openjev rule): query = instructions + rubric (never dropped);
candidates = the task's labels verbatim ("no"/"yes" for noul, "0".."n" for score); the model's
null mass is dropped and the rest renormalised over the labels -- p_null is reported in the
runtime block, not scored. Modes: energy (model.decide, KV-cached state), native
(native.native_kv_decide), compose (PLAN5 sec 3: P(null) = 1 - r_energy, P(a_j) = r_energy *
P_native(a_j | answerable)), mcq_zero_shot (PLAN7 track A: frozen --backbone, no checkpoint,
no LoRA -- next-token logits over the option letters via pcdm/mcq.py's rendering, restricted to
the option set; ∅ is never rendered, so p_null = 0). Cold path: no candidate cache, every
label embedded on the fly.

mcq_zero_shot's `prompt_style` ("ours" default vs "semif") swaps only the rendering + answer-slot
convention; the decide() control flow, null handling and to_labels renormalisation are shared.
"semif" mirrors github.com/TheoLeeCJ/SemIf (MIT) `src/semif_phase1/{core,direct}.py`'s `direct`
readout: chat-template system+user JSON payload, next-token logits over bare uppercase letter
tokens (no leading space, no rendered null, no exemplars in their `direct` mode).
"""
import json
import time
from pathlib import Path

import torch

import bench  # load_ours / fresh_empty_cache: the checkpoint -> (backbone, model) contract lives there
from encode import pick_device
from mcq import MAXK_DIRECT, MCQHead, _ids, _label, _pack
from model import decide as energy_decide
from native import NativeHead, native_kv_decide

MAX_QUERY = 64  # the energy checkpoints TRAINED with 64-token queries; decide() now accepts max_query, so longer rubrics are in-context but out-of-distribution
MAX_STATE = 256  # both model.decide and native_kv_decide default max_state=256 -- the trained length, independent of the --max_state extrapolation cap
MODES = ("energy", "native", "compose", "mcq_zero_shot")
PROMPT_STYLES = ("ours", "semif")

# Verbatim from SemIf's src/semif_phase1/core.py (DIRECT_SYSTEM) / direct.py (LETTERS).
SEMIF_SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
                "Respond with only its uppercase letter, with no explanation or reasoning.")
SEMIF_LETTERS = "ABCDEFGHIJKLMNOP"


def _semif_messages(state: str, query: str, labels: list[str]) -> list[dict]:
    """SemIf's direct_messages: system instruction + a JSON user payload {evidence, criterion,
    options: [{letter, description}]}. `state` here is already a plain string (decide() dumps
    non-string state to JSON text before this is called), so it lands as one JSON string value
    rather than SemIf's original nested object -- a ponytail simplification, not a behavior gap
    for our (already-string) states."""
    payload = {"evidence": state, "criterion": query,
               "options": [{"letter": SEMIF_LETTERS[i], "description": lab} for i, lab in enumerate(labels)]}
    return [{"role": "system", "content": SEMIF_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _semif_slot_ids(tok, count: int) -> list[int]:
    """Bare uppercase letters (no leading space, unlike mcq.LETTERS' " A" convention) -- SemIf's
    _slot_ids, with the same single-token round-trip check."""
    ids = []
    for letter in SEMIF_LETTERS[:count]:
        enc = tok.encode(letter, add_special_tokens=False)
        if len(enc) != 1 or tok.decode(enc) != letter:
            raise ValueError(f"semif prompt_style: answer slot {letter!r} is not one exact round-trip token")
        ids.append(enc[0])
    return ids


def _render_zero_shot(query: str, labels: list[str]) -> str:
    """query + lettered options + "Answer:" -- no rendered null line (unlike mcq._render):
    mcq_zero_shot never scores a null option, ∅ is hardcoded 0 by the caller."""
    text = query + "\n"
    for i, c in enumerate(labels):
        text += f"{_label(i)}. {c}\n"
    return text + "Answer:"


def _data_shots(n: int, seed: int = 0, path: str = "data_v5/val.jsonl") -> str:
    """PLAN7 track A: n worked examples sampled from data_v5's val split (never JevBench or
    any W eval set), same rendering as _render_zero_shot + the gold letter, so a base model
    picks up the "Answer: <letter>" format the way mcq.build_shots's generic-trivia prefix
    does for --readout mcq training. Fixed seed -> byte-identical exemplars at every ladder
    size (n<=0 -> "" so the existing shots=0 zero-shot path is unchanged). Answerable rows only
    (p_null < .5, a clear one-hot-ish target) so every demonstration has an unambiguous gold letter."""
    if n <= 0:
        return ""
    import random

    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pool = [r for r in rows if r.get("p_null", 0) < 0.5 and max(r["target"]) > 0.5]
    picked = random.Random(seed).sample(pool, min(n, len(pool)))
    blocks = []
    for ex in picked:
        state = ex["state"] if isinstance(ex["state"], str) else json.dumps(ex["state"], ensure_ascii=False)
        gold = ex["target"].index(max(ex["target"]))
        suffix = _render_zero_shot(ex["query"], ex["candidates"])
        blocks.append(f"{state}\n{suffix} {_label(gold)}")
    return "\n\n".join(blocks) + "\n\n"


def query_text(question: dict) -> str:
    """instructions + criteria: `yes: .. no: ..` (noul), `0: .. 1: ..` (score), `label: desc` (choice)."""
    q, crit = question["instructions"], question.get("criteria")
    if not crit:
        return q
    if question["type"] == "noul":
        rubric = f"yes: {crit.get('true', 'Yes')}  no: {crit.get('false', 'No')}"
    elif isinstance(crit, list):
        rubric = "  ".join(f"{i}: {c}" for i, c in enumerate(crit))
    else:
        rubric = "  ".join(f"{k}: {v or k}" for k, v in crit.items())
    return f"{q}\n{rubric}"


def to_labels(p: torch.Tensor, labels: list[str]) -> tuple[dict, float]:
    """[K+1] (null last) -> (probs renormalised over labels, p_null). Zero label mass -> all
    zeros, which jevbench's validate_probs marks invalid (sum != 1) rather than NaN -- never
    repaired here."""
    p = p.detach().float().cpu()
    if not torch.isfinite(p).all():
        raise RuntimeError("non-finite model output")
    s = p[:-1].sum()
    lab = p[:-1] / s if s > 0 else torch.zeros_like(p[:-1])
    return dict(zip(labels, lab.tolist())), float(p[-1])


class PCDMDecider:
    def __init__(self, run_dir: str | None = None, device: str = "auto", mode: str = "energy",
                 energy_run: str | None = None, max_state: int = 4096, max_query: int = 256,
                 backbone: str | None = None, tap_layer: int = 0, shots: int = 0,
                 prompt_style: str = "ours"):
        assert mode in MODES, mode
        assert prompt_style in PROMPT_STYLES, prompt_style
        if mode == "compose" and not energy_run:
            raise ValueError("compose needs energy_run (the support-gate checkpoint)")
        if mode == "mcq_zero_shot" and not backbone:
            raise ValueError("mcq_zero_shot needs backbone (frozen, no checkpoint)")
        if prompt_style == "semif" and shots:
            raise ValueError("prompt_style=semif has no exemplar mechanism (SemIf's `direct` "
                              "readout is 0-shot only) -- run with --shots 0")
        self.device, self.mode, self.max_state, self.max_query = pick_device(device), mode, max_state, max_query
        self.prompt_style = prompt_style
        self.energy = self.native = self.zero = None
        if mode == "mcq_zero_shot":
            self.zero = MCQHead(backbone, lora_layers=0, lora_r=0, device=self.device, tap_layer=tap_layer).eval()
            self.tok = self.zero.backbone.tokenizer
            self.shots_prefix = _data_shots(shots, seed=0)
            self.shots_prefix_tokens = len(self.tok(self.shots_prefix, add_special_tokens=False)["input_ids"]) \
                if self.shots_prefix else 0
            return
        for kind, rd in {"energy": [("energy", run_dir)], "native": [("native", run_dir)],
                         "compose": [("energy", energy_run), ("native", run_dir)]}[mode]:
            setattr(self, kind, self._load(rd, kind))
        self.tok = (self.native or self.energy)["tok"]

    def _load(self, run_dir, kind):
        if not Path(run_dir, "best.pt").exists():
            raise FileNotFoundError(f"{run_dir}/best.pt")
        bb, m = bench.load_ours(run_dir, None, self.device)
        m.eval()
        if isinstance(m, NativeHead) != (kind == "native"):
            raise ValueError(f"{run_dir} is not a {kind} checkpoint")
        if kind == "native":
            return {"bb": bb, "m": m, "tok": bb.backbone.tokenizer}
        return {"bb": bb, "m": m, "tok": bb.tokenizer, "joint": bench.JOINT, "enc": bench.ENCODER}

    def _probs(self, kind, state, query, labels) -> torch.Tensor:
        """[K+1] probabilities, null last, from one checkpoint (cold candidate path)."""
        r = getattr(self, kind)
        if kind == "native":
            return native_kv_decide(r["bb"], r["m"], state, [(query, labels)], max_state=self.max_state)[0]
        return energy_decide(r["bb"], r["m"], bench.fresh_empty_cache(r["bb"]), state, [(query, labels)],
                             joint=r["joint"], encoder=r["enc"], max_state=self.max_state, max_query=self.max_query)[0]

    def _probs_zero_shot(self, state, query, labels) -> torch.Tensor:
        """[K+1] next-token letter logits over exactly `labels`, softmax-normalised, p_null
        hardcoded 0 (no null option is ever rendered). ponytail: direct-scoring only (no
        mcq._score_chunked fallback) -- JevBench label sets never exceed MAXK_DIRECT."""
        head = self.zero
        assert len(labels) <= MAXK_DIRECT, f"mcq_zero_shot: K={len(labels)} > {MAXK_DIRECT} (not chunked)"
        tok = head.backbone.tokenizer
        suffix = _render_zero_shot(query, labels)
        # shots_prefix is fixed text prepended to every state; widen the budget by its own token
        # count so the real state keeps its full max_state (mirrors mcq.run_batch_mcq's --shots).
        state = self.shots_prefix + state
        s_ids = _ids(tok, [state], self.max_state + self.shots_prefix_tokens)
        x_ids = _ids(tok, [suffix], self.max_query)
        input_ids, attention_mask, lengths = _pack(tok, s_ids, x_ids)
        input_ids, attention_mask = input_ids.to(head.device), attention_mask.to(head.device)
        out = head.backbone.model(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state[0, lengths[0] - 1].float()
        letter_ids = head.letter_ids[:len(labels)].to(head.device)
        w = head.lm_head_weight.to(head.device)[letter_ids].float()
        probs = torch.softmax(last @ w.T, dim=-1)
        return torch.cat([probs, probs.new_zeros(1)])

    def _probs_zero_shot_semif(self, state, query, labels) -> torch.Tensor:
        """[K+1] next-token logits over exactly `labels`, SemIf's `direct` readout: chat-template
        system+user JSON prompt, bare-letter answer slots (no leading space), no rendered null
        (p_null hardcoded 0, matching _probs_zero_shot). No --shots support (see __init__)."""
        head = self.zero
        assert len(labels) <= len(SEMIF_LETTERS), f"semif prompt_style: K={len(labels)} > {len(SEMIF_LETTERS)}"
        tok = head.backbone.tokenizer
        messages = _semif_messages(state, query, labels)
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        except TypeError:  # template doesn't accept enable_thinking (non-Qwen3-style template)
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(prompt, add_special_tokens=False)
        slot_ids = _semif_slot_ids(tok, len(labels))
        input_ids = torch.tensor([ids], dtype=torch.long, device=head.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        out = head.backbone.model(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state[0, -1].float()
        w = head.lm_head_weight.to(head.device)[torch.tensor(slot_ids, device=head.device)].float()
        probs = torch.softmax(last @ w.T, dim=-1)
        return torch.cat([probs, probs.new_zeros(1)])

    def decide(self, state, question: dict, labels: list[str]) -> tuple[dict, dict]:
        """-> (probs over exactly `labels`, runtime block)."""
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        query = query_text(question)
        n_state = len(self.tok(state, add_special_tokens=False)["input_ids"])
        n_query = len(self.tok(query, add_special_tokens=False)["input_ids"])
        t0 = time.perf_counter()
        with torch.inference_mode():
            if self.mode == "compose":
                r = 1.0 - self._probs("energy", state, query, labels)[-1]
                pn = self._probs("native", state, query, labels)[:-1]
                p = torch.cat([r * pn / pn.sum(), (1.0 - r).reshape(1)])
            elif self.mode == "mcq_zero_shot":
                p = (self._probs_zero_shot_semif(state, query, labels) if self.prompt_style == "semif"
                     else self._probs_zero_shot(state, query, labels))
            else:
                p = self._probs(self.mode, state, query, labels)
        probs, p_null = to_labels(p, labels)  # includes the .cpu() sync -- latency below covers the real wall-clock cost
        latency = time.perf_counter() - t0
        return probs, {"mode": self.mode, "device": self.device, "latency_s": latency, "p_null": p_null,
                       "state_tokens": n_state, "max_state_tokens": self.max_state,
                       "state_truncated": n_state > self.max_state,
                       "trained_max_state": MAX_STATE, "state_beyond_train_len": n_state > MAX_STATE,
                       "query_tokens": n_query,
                       "query_truncated": bool(self.energy) and n_query > self.max_query,
                       "trained_max_query": MAX_QUERY, "query_beyond_train_len": bool(self.energy) and n_query > MAX_QUERY,
                       "probability_origin": (f"mcq-zero-shot-softmax-{self.prompt_style}"
                                               if self.mode == "mcq_zero_shot" and self.prompt_style == "semif"
                                               else "mcq-zero-shot-softmax" if self.mode == "mcq_zero_shot"
                                               else "native-softmax")}
