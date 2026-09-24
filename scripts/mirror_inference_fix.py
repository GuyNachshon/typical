"""Mirror the Phase A serving-speedup fix (inference/typical/native.py,
inference/test_parity.py) into the three public HF model repos, after local parity has
passed. One-shot script, not meant to be a general mirroring tool -- see PLAN item 5.

uv run --no-sync python scripts/mirror_inference_fix.py
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd

REPO_ROOT = Path(__file__).resolve().parent.parent
REPOS = ["OzLabs/typical-small-preview", "OzLabs/typical-small", "OzLabs/typical-medium"]
FILES = ["inference/typical/native.py", "inference/test_parity.py"]

if __name__ == "__main__":
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    api = HfApi()
    for repo in REPOS:
        ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(REPO_ROOT / f)) for f in FILES]
        commit = api.create_commit(
            repo_id=repo, operations=ops,
            commit_message="Phase A serving speedup: zero-copy state-cache expand (no deepcopy per decision)",
        )
        print(repo, "->", commit)
