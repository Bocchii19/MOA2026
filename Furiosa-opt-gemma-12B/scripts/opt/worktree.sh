#!/bin/bash
# worktree.sh <name> [base]  -- fresh worktree of <base> (default: the repo's checked-out branch) at $OPT_WORK/wt_<name>
set -eu
. "$(dirname "$0")/env.sh"
NAME="$1"; BASE="${2:-$(git -C "$OPT_REPO" symbolic-ref --short HEAD)}"; WT="$OPT_WORK/wt_$NAME"
cd "$OPT_REPO"; git worktree prune
[ -d "$WT" ] && git worktree remove --force "$WT" 2>/dev/null || true
git branch -D "exp/$NAME" >/dev/null 2>&1 || true
git worktree add -q -b "exp/$NAME" "$WT" "$BASE"
echo "$WT"
