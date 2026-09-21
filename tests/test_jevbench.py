"""pcdm_jev: label mapping / renormalisation, rubric, adapter contract; one real CPU pass on
runs/mini_emb over 3 public tasks (skipped when the checkpoint or /tmp/jevbench is missing)."""
import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch

JEV = Path("/tmp/jevbench")
if JEV.exists():
    sys.path.insert(0, str(JEV))
pytestmark = pytest.mark.skipif(not JEV.exists(), reason="jevbench clone missing")

from pcdm_jev.decider import _data_shots, query_text, to_labels  # noqa: E402

DATA_V5_VAL = Path("data_v5/val.jsonl")


def test_to_labels_drops_null_and_renormalises():
    probs, p_null = to_labels(torch.tensor([0.2, 0.6, 0.2]), ["a", "b"])
    assert list(probs) == ["a", "b"] and p_null == pytest.approx(0.2)
    assert probs["a"] == pytest.approx(0.25) and probs["b"] == pytest.approx(0.75)


def test_to_labels_zero_mass_is_all_zero_never_nan():
    from jevbench.scoring import InvalidDistribution, validate_probs
    probs, p_null = to_labels(torch.tensor([0.0, 0.0, 1.0]), ["a", "b"])
    assert probs == {"a": 0.0, "b": 0.0} and p_null == pytest.approx(1.0)
    assert not any(math.isnan(v) for v in probs.values())
    json.dumps(probs, allow_nan=False)  # must not raise -- jevbench's own writers use allow_nan=False
    with pytest.raises(InvalidDistribution):  # sums to 0, not 1 -- the harness marks it invalid, not repaired
        validate_probs(probs, ["a", "b"])


def test_to_labels_raises_on_non_finite():
    with pytest.raises(RuntimeError):
        to_labels(torch.tensor([float("nan"), 0.5, 0.5]), ["a", "b"])


def test_compose_label_distribution_equals_native_on_fake_decider():
    """PLAN5's compose = r * P_native(label) renormalised; to_labels renormalises again, so r
    cancels out exactly -- compose and native must agree on every label, whatever r is."""
    from pcdm_jev.decider import PCDMDecider
    energy_p = torch.tensor([0.1, 0.2, 0.7])  # mostly abstains: r is far from 1, so a bug would show up
    native_p = torch.tensor([0.3, 0.5, 0.2])

    def fake_probs(kind, *_a, **_k):
        return energy_p if kind == "energy" else native_p

    def make(mode):
        d = PCDMDecider.__new__(PCDMDecider)  # ponytail: skip __init__, no checkpoints in a unit test
        d.mode, d.device, d.max_state, d.max_query = mode, "cpu", 4096, 256
        d.energy, d.native = True, True
        d.tok = lambda s, add_special_tokens=False: {"input_ids": [0] * len(s)}
        d._probs = fake_probs
        return d

    labels, question = ["a", "b"], {"type": "noul", "instructions": "?"}
    compose_probs, _ = make("compose").decide("state", question, labels)
    native_probs, _ = make("native").decide("state", question, labels)
    for k in labels:
        assert compose_probs[k] == pytest.approx(native_probs[k])


@pytest.mark.skipif(not DATA_V5_VAL.exists(), reason="data_v5/val.jsonl missing")
def test_data_shots_deterministic_and_never_empty_for_zero():
    assert _data_shots(0) == ""
    a = _data_shots(3, seed=0)
    b = _data_shots(3, seed=0)
    assert a == b and a.count("Answer: ") == 3  # fixed seed -> byte-identical exemplars, one gold letter each
    assert a != _data_shots(3, seed=1)  # a different seed picks different rows


@pytest.mark.skipif(not DATA_V5_VAL.exists(), reason="data_v5/val.jsonl missing")
def test_data_shots_gold_letter_matches_target_argmax():
    import random as _random
    with open(DATA_V5_VAL) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pool = [r for r in rows if r.get("p_null", 0) < 0.5 and max(r["target"]) > 0.5]
    prefix = _data_shots(1, seed=0)
    ex = _random.Random(0).sample(pool, 1)[0]
    gold = ex["target"].index(max(ex["target"]))
    letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[gold] if gold < 26 else str(gold)
    assert prefix.strip().endswith(f"Answer: {letter}")


def test_query_text_keeps_rubric():
    q = query_text({"type": "noul", "instructions": "Permitted?", "criteria": {"true": "T", "false": "F"}})
    assert q == "Permitted?\nyes: T  no: F"
    q = query_text({"type": "score", "instructions": "Rate.", "criteria": ["none", "minor"]})
    assert q == "Rate.\n0: none  1: minor"
    q = query_text({"type": "choice", "instructions": "Which?", "criteria": {"x": "desc", "y": None}})
    assert q == "Which?\nx: desc  y: y"
    assert query_text({"type": "choice", "instructions": "Which?"}) == "Which?"


class FakeDecider:
    def decide(self, state, question, labels):
        p = torch.full((len(labels) + 1,), 1.0 / (len(labels) + 1))
        return to_labels(p, labels)[0], {"p_null": 0.0, "state_truncated": False, "state_tokens": 1}


def test_adapter_returns_valid_result_on_fake_decider():
    from jevbench.scoring import validate_probs
    from jevbench.tasks import load_jsonl
    from pcdm_jev.adapter import LocalPCDMAdapter
    tasks = load_jsonl(str(JEV / "datasets/public/original.jsonl"))
    score = next(t for t in tasks if t.question["type"] == "score")
    ad = LocalPCDMAdapter(model="nowhere")
    ad._decider = FakeDecider()
    res = ad.run(score)
    assert res.ok and res.probs_source == "native" and res.raw["runtime"]["p_null"] == 0.0
    assert list(res.probs) == score.labels == ["0", "1", "2", "3"]  # ordinal labels, in order
    validate_probs(res.probs, score.labels)
    assert ad.price_input_per_m is None and ad.reserve_estimate(score) == 0.0


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="downloads/loads the real Qwen3-1.7B-Base "
                     "backbone; set RUN_SLOW=1 to run")
def test_mcq_zero_shot_real_backbone_two_items():
    """PLAN7 track A: frozen backbone, no checkpoint, no LoRA -- next-token letter logits
    restricted to the option set (no rendered null), p_null hardcoded 0. Two items: a
    noul (K=2) and a score (K=4) task from the public set, run through both the decider
    directly and the adapter contract."""
    from jevbench.scoring import validate_probs
    from jevbench.tasks import load_jsonl
    from pcdm_jev.adapter import LocalPCDMAdapter
    from pcdm_jev.decider import PCDMDecider

    dec = PCDMDecider(mode="mcq_zero_shot", backbone="Qwen/Qwen3-1.7B-Base", tap_layer=0, device="cpu")
    tasks = load_jsonl(str(JEV / "datasets/public/original.jsonl"))
    picks = [next(t for t in tasks if t.question["type"] == k) for k in ("noul", "score")]
    for t in picks:
        probs, rt = dec.decide(t.state, t.question, list(t.labels))
        clean = validate_probs(probs, t.labels)  # sums to 1, no null column to drop twice
        assert list(clean) == list(t.labels) and math.isclose(sum(clean.values()), 1.0, abs_tol=1e-6)
        assert rt["p_null"] == 0.0 and rt["mode"] == "mcq_zero_shot"
        assert rt["probability_origin"] == "mcq-zero-shot-softmax"

    ad = LocalPCDMAdapter(mode="mcq_zero_shot", backbone="Qwen/Qwen3-1.7B-Base", device="cpu")
    res = ad.run(picks[0])
    assert res.ok, res.error
    validate_probs(res.probs, picks[0].labels)
    assert res.raw["runtime"]["p_null"] == 0.0


def test_semif_prompt_style_rejects_shots_without_loading_a_backbone():
    """The ValueError fires before MCQHead loads any weights -- no RUN_SLOW/network needed."""
    from pcdm_jev.decider import PCDMDecider
    with pytest.raises(ValueError, match="0-shot only"):
        PCDMDecider(mode="mcq_zero_shot", backbone="Qwen/Qwen3-1.7B-Base", device="cpu",
                    prompt_style="semif", shots=3)


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="downloads/loads the real Qwen3-1.7B-Base "
                     "backbone; set RUN_SLOW=1 to run")
def test_mcq_zero_shot_semif_prompt_style_real_backbone():
    """prompt_style=semif (SemIf's chat-template `direct` readout): same decide() contract as the
    default style -- valid renormalised probs, p_null hardcoded 0 -- just a different rendering.
    Qwen3-1.7B-Base's tokenizer may lack a chat_template; if apply_chat_template raises, that is
    itself the diagnostic this flag exists to surface, so this test only checks the happy path."""
    from jevbench.scoring import validate_probs
    from jevbench.tasks import load_jsonl
    from pcdm_jev.decider import PCDMDecider

    dec = PCDMDecider(mode="mcq_zero_shot", backbone="Qwen/Qwen3-1.7B-Base", tap_layer=0, device="cpu",
                       prompt_style="semif")
    tasks = load_jsonl(str(JEV / "datasets/public/original.jsonl"))
    t = next(x for x in tasks if x.question["type"] == "noul")
    probs, rt = dec.decide(t.state, t.question, list(t.labels))
    clean = validate_probs(probs, t.labels)
    assert list(clean) == list(t.labels) and math.isclose(sum(clean.values()), 1.0, abs_tol=1e-6)
    assert rt["p_null"] == 0.0 and rt["probability_origin"] == "mcq-zero-shot-softmax-semif"


@pytest.mark.skipif(not Path("runs/mini_emb/best.pt").exists(), reason="runs/mini_emb missing")
def test_adapter_real_mini_emb_cpu():
    from jevbench.scoring import validate_probs
    from jevbench.tasks import load_jsonl
    from pcdm_jev.adapter import LocalPCDMAdapter
    tasks = load_jsonl(str(JEV / "datasets/public/original.jsonl"))
    picks = [next(t for t in tasks if t.question["type"] == k) for k in ("noul", "choice", "score")]
    ad = LocalPCDMAdapter(model="runs/mini_emb", mode="energy", device="cpu", max_state=64)
    for t in picks:
        res = ad.run(t)
        assert res.ok, res.error
        clean = validate_probs(res.probs, t.labels)
        assert list(clean) == t.labels and math.isclose(sum(clean.values()), 1.0, abs_tol=1e-6)
        rt = res.raw["runtime"]
        assert 0.0 <= rt["p_null"] <= 1.0 and rt["state_tokens"] > 0 and rt["state_truncated"] is False
        assert rt["latency_s"] > 0 and res.latency_s >= rt["latency_s"]
    assert ad.run(picks[0]).raw["runtime"]["query_truncated"] is False
