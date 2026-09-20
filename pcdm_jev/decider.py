"""One JevBench decision -> PCDM probabilities over the exact label set.

Mapping (jevbench's local_openjev rule): query = instructions + rubric (never dropped);
candidates = the task's labels verbatim ("no"/"yes" for noul, "0".."n" for score); the model's
null mass is dropped and the rest renormalised over the labels -- p_null is reported in the
runtime block, not scored. Modes: energy (model.decide, KV-cached state), native
(native.native_kv_decide), compose (PLAN5 sec 3: P(null) = 1 - r_energy, P(a_j) = r_energy *
P_native(a_j | answerable)). Cold path: no candidate cache, every label embedded on the fly.
"""
import json
import time
from pathlib import Path

import torch

import bench  # load_ours / fresh_empty_cache: the checkpoint -> (backbone, model) contract lives there
from encode import pick_device
from model import decide as energy_decide
from native import NativeHead, native_kv_decide

MAX_QUERY = 64  # the energy checkpoints TRAINED with 64-token queries; decide() now accepts max_query, so longer rubrics are in-context but out-of-distribution
MAX_STATE = 256  # both model.decide and native_kv_decide default max_state=256 -- the trained length, independent of the --max_state extrapolation cap
MODES = ("energy", "native", "compose")


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
    def __init__(self, run_dir: str, device: str = "auto", mode: str = "energy",
                 energy_run: str | None = None, max_state: int = 4096, max_query: int = 256):
        assert mode in MODES, mode
        if mode == "compose" and not energy_run:
            raise ValueError("compose needs energy_run (the support-gate checkpoint)")
        self.device, self.mode, self.max_state, self.max_query = pick_device(device), mode, max_state, max_query
        self.energy = self.native = None
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
                       "probability_origin": "native-softmax"}
