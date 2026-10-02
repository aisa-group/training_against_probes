#!/usr/bin/env bash
# Fetch the Safety Polytope reference code (MIT, github.com/lasgroup/SafetyPolytope)
# at the commit used for the paper's SafeFlow baseline. Only
# src/safety_polytope/polytope/safe_rep_model.py is imported, by putting
# third_party/SafetyPolytope/src on sys.path; no package install is needed.
# Alternatively point SAFETY_POLYTOPE_SRC at the src/ dir of an existing checkout.
set -euo pipefail
COMMIT=137096bb9f683842ff0f58754e980cb8e3824bd5
DEST="$(cd "$(dirname "$0")" && pwd)/SafetyPolytope"
if [ ! -d "$DEST/.git" ]; then
    git clone https://github.com/lasgroup/SafetyPolytope.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$COMMIT"
echo "SafetyPolytope at $(git -C "$DEST" rev-parse HEAD) in $DEST"
