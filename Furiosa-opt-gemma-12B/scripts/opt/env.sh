# Shared settings for the scripts/opt tools. Work products (worktrees, binaries, logs) live in
# $OPT_WORK, which defaults to the session scratchpad; the scripts themselves live in the repo.
export OPT_REPO="${OPT_REPO:-/home/hanet/MinhDuc/gm/projects/furiosa-opt-gemma4-12B}"
export OPT_WORK="${OPT_WORK:-/tmp/claude-1000/-home-hanet/4821056b-9897-4ed6-8763-75b74fff1ae9/scratchpad/opt}"
export CARGO_TARGET_DIR="$OPT_WORK/target"
export PATH="$HOME/.cargo/bin:$PATH"
mkdir -p "$OPT_WORK"
