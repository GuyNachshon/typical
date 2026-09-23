#!/usr/bin/env bash
# Rebuild the site's data from the research artefacts before touching demos or results.
# Since the branches were consolidated into main (2026-09-23) the committed artefacts
# (inference/, releases/, runs/**/*.json, REPORT.md) already live in this tree; the only
# thing still outside git is whatever the buffalo worktree has benched since its last commit.
set -euo pipefail
cd "$(dirname "$0")/.."
BUFFALO=/Users/guynachshon/conductor/workspaces/typical/buffalo

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
echo "--- newest REPORT sections:"; grep '^## ' REPORT.md | tail -3
