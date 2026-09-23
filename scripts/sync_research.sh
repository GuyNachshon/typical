#!/usr/bin/env bash
# Pull the latest research artefacts into the site worktree before touching demos or results.
# Sources: origin/gpu-runpod-full-experiment (committed) + the buffalo worktree (untracked run JSONs).
set -euo pipefail
cd "$(dirname "$0")/.."
BR=gpu-runpod-full-experiment
BUFFALO=/Users/guynachshon/conductor/workspaces/typical/buffalo

git fetch -q origin
echo "branch: $(git log -1 --format='%h %ad %s' --date=short "origin/$BR")"
if git rev-parse -q --verify "$BR" >/dev/null; then
  git branch -q -f "$BR" "origin/$BR" 2>/dev/null || true
fi

# committed artefacts (same blobs -> merge-friendly)
git checkout "origin/$BR" -- inference releases PLAN7.md 2>/dev/null
git show "origin/$BR:REPORT.md" > .context/REPORT.md 2>/dev/null || true
# every small JSON under runs/ (results/summary/bench/eval_wf), never .pt/.jsonl
git ls-tree -r --name-only "origin/$BR" -- runs \
  | grep -E '/(results|summary|bench|eval_wf|eval_wf_full|manifest)\.json$' \
  | xargs -r git checkout "origin/$BR" --

# untracked JSONs in the research worktree (benches run after the last commit)
if [ -d "$BUFFALO/runs" ]; then
  (cd "$BUFFALO" && git status --porcelain runs | awk '$1=="??"{print $2}') | while read -r p; do
    find "$BUFFALO/$p" -name '*.json' -size -2M 2>/dev/null | while read -r f; do
      rel=${f#"$BUFFALO/"}; mkdir -p "$(dirname "$rel")"; cp "$f" "$rel"
    done
  done
fi

uv run precompute.py >/dev/null && uv run scripts/precompute_research.py >/dev/null
echo "--- changed:"; git status --short inference releases runs site/data | head -40
echo "REPORT sections newer than the site? grep '^## ' .context/REPORT.md | tail -3"
