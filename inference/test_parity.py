"""Parity check: inference/typical/ vs the training repo's pcdm_jev.decider.PCDMDecider
(mode="native"), same checkpoint (guychuk/pcdm-runs/typical-small/best.pt), same device,
same 6 items. Downloads a 1.7B checkpoint -> gated like the repo's other real-checkpoint
tests (tests/test_pipeline.py, tests/test_jevbench.py): RUN_SLOW=1 required.

uv run --no-sync pytest inference/test_parity.py -v   # (with RUN_SLOW=1 set)
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(INFERENCE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

CKPT_REPO = "guychuk/pcdm-runs"
CKPT_FILE = "typical-small/best.pt"

STATE = (
    "Ticket #4821: Customer says their package arrived damaged. They want a replacement "
    "shipped overnight, not a refund. Order was placed 3 days ago, still under warranty."
)

ITEMS = [
    ("What does the customer want?", ["refund", "replacement", "repair"]),
    ("Is the order still under warranty?", ["no", "yes"]),
    ("How urgent is this ticket?", ["0", "1", "2", "3"]),
    ("What is the customer's sentiment?", ["positive", "neutral", "negative"]),
    ("Should this be escalated to a manager?", ["no", "yes"]),
    ("Which department should own this ticket?", ["billing", "shipping", "support"]),
]

NOUL_ITEMS = [
    "Is the order still under warranty?",
    "Should this be escalated to a manager?",
    "Was the package damaged?",
]


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="downloads the real typical-small "
                     "checkpoint (Qwen3-1.7B-Base + LoRA); set RUN_SLOW=1 to run")
def test_typical_matches_pcdm_decider():
    from huggingface_hub import hf_hub_download

    from pcdm_jev.decider import PCDMDecider
    from typical import Typical, to_labels

    local_best_pt = hf_hub_download(CKPT_REPO, CKPT_FILE)
    run_dir = os.path.dirname(local_best_pt)

    orig = PCDMDecider(run_dir=run_dir, mode="native", device="auto")
    mine = Typical.from_pretrained(CKPT_REPO, filename=CKPT_FILE, device=orig.device,
                                   max_state=orig.max_state)
    assert orig.device == mine.device

    max_diff = 0.0
    for query, labels in ITEMS:
        p_orig = orig._probs("native", STATE, query, labels)
        p_mine = mine._raw(STATE, query, labels)
        diff = (p_orig.float().cpu() - p_mine.float().cpu()).abs().max().item()
        max_diff = max(max_diff, diff)
        assert diff < 1e-4, f"{query!r} {labels}: max abs diff {diff}"
    print(f"[parity] {len(ITEMS)} items, max abs prob diff = {max_diff:.2e}")

    # noul reversed-label control: the checkpoint's noul_head is bern (query-only, position-
    # invariant by construction -- the rendered suffix never contains the candidate text, so
    # P(yes) under ["no","yes"] and ["yes","no"] is the SAME forward pass mathematically).
    # ponytail: exact float equality is verified on CPU (checked separately, bit-identical);
    # on MPS the two calls can differ by ~1e-7 from non-deterministic reduction order, so this
    # check uses a tight epsilon rather than `==` to stay device-agnostic.
    eps = 1e-6
    for q in NOUL_ITEMS:
        p_yes_forward = mine.noul(STATE, q)
        probs_rev, _ = to_labels(mine._raw(STATE, q, ["yes", "no"]), ["yes", "no"])
        assert abs(probs_rev["yes"] - p_yes_forward) < eps, (q, probs_rev["yes"], p_yes_forward)

        orig_fwd, _ = to_labels(orig._probs("native", STATE, q, ["no", "yes"]), ["no", "yes"])
        orig_rev, _ = to_labels(orig._probs("native", STATE, q, ["yes", "no"]), ["yes", "no"])
        assert abs(orig_fwd["yes"] - orig_rev["yes"]) < eps, (q, orig_fwd["yes"], orig_rev["yes"])
        assert abs(orig_fwd["yes"] - p_yes_forward) < 1e-4, (q, orig_fwd["yes"], p_yes_forward)
    print(f"[parity] noul reversed-label check OK over {len(NOUL_ITEMS)} items (eps={eps})")


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="downloads the real "
                     "typical-small-preview checkpoint; set RUN_SLOW=1 to run")
def test_typical_noul_via_choice_path_on_preview():
    """typical-small-preview trains noul_head="choice" (no Bernoulli head): m.noul() must
    still work by scoring the plain K-way ["no","yes"] choice and reading off P(yes) -- and
    must still match the original decider exactly (order-invariance is a bern-only guarantee,
    not required here)."""
    from huggingface_hub import hf_hub_download

    from pcdm_jev.decider import PCDMDecider
    from typical import Typical, to_labels

    local_best_pt = hf_hub_download(CKPT_REPO, "typical-small-preview/best.pt")
    run_dir = os.path.dirname(local_best_pt)

    orig = PCDMDecider(run_dir=run_dir, mode="native", device="auto")
    assert orig.native["m"].noul_head == "choice"
    mine = Typical.from_pretrained(CKPT_REPO, filename="typical-small-preview/best.pt",
                                   device=orig.device, max_state=orig.max_state)
    assert mine.model.noul_head == "choice"

    for q in NOUL_ITEMS:
        p_yes = mine.noul(STATE, q)
        assert 0.0 <= p_yes <= 1.0
        orig_probs, _ = to_labels(orig._probs("native", STATE, q, ["no", "yes"]), ["no", "yes"])
        assert abs(orig_probs["yes"] - p_yes) < 1e-4, (q, orig_probs["yes"], p_yes)
    print(f"[parity] preview noul() via K-way choice path OK over {len(NOUL_ITEMS)} items")


if __name__ == "__main__":
    if not os.environ.get("RUN_SLOW"):
        os.environ["RUN_SLOW"] = "1"
    test_typical_matches_pcdm_decider()
    test_typical_noul_via_choice_path_on_preview()
