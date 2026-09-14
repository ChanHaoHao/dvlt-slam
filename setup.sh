#!/bin/bash
set -euo pipefail

# Clones the local source trees that are deliberately absent from uv.lock, and
# installs them editable with --no-deps.
#
# PREREQUISITE: run `uv sync` FIRST. It builds .venv from the lock file and
# installs this project (vggt_slam + evals) editable. It also PRUNES anything
# not in the lock, so re-running it later silently uninstalls everything below.
#
# This script does NOT `pip install -r requirements.txt`: that file is upstream's
# and pins torch==2.3.1, which would clobber the locked torch 2.5.1+cu124 that
# DVLT requires. It does NOT install this project either -- `uv sync` did.
#
# --no-deps is load-bearing. third_party/vggt declares numpy<2 while the lock
# resolves numpy 2.4.4; letting pip resolve would downgrade it.
#
# Usage:  ./setup.sh            salad + vggt (everything the default path needs)
#         ./setup.sh --with-os  also SAM 3 + Perception Encoder, for --run_os

cd "$(dirname "$0")"
REPO="$PWD"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found. See https://docs.astral.sh/uv/" >&2
    exit 1
fi
if [ ! -d "$REPO/.venv" ]; then
    echo "error: no .venv. Run 'uv sync' first." >&2
    exit 1
fi
if [ ! -e "$REPO/dvlt/pyproject.toml" ]; then
    echo "error: dvlt submodule is empty. Run 'git submodule update --init'." >&2
    exit 1
fi

WITH_OS=0
[ "${1:-}" = "--with-os" ] && WITH_OS=1

mkdir -p third_party

# clone <url> <dest>  -- idempotent; leaves an existing tree alone
clone() {
    if [ -d "$2/.git" ]; then
        echo "  $2 already present, skipping clone"
    else
        git clone "$1" "$2"
    fi
}

echo "Cloning SALAD (loop-closure place recognition)..."
clone https://github.com/Dominic101/salad.git third_party/salad

echo "Cloning VGGT (the --backbone vggt baseline)..."
clone https://github.com/MIT-SPARK/VGGT_SPARK.git third_party/vggt

TREES=(./dvlt ./third_party/salad ./third_party/vggt)

if [ "$WITH_OS" -eq 1 ]; then
    echo "Cloning Perception Encoder and SAM 3 (--run_os only)..."
    clone https://github.com/facebookresearch/perception_models.git third_party/perception_models
    clone https://github.com/facebookresearch/sam3.git third_party/sam3
    TREES+=(./third_party/perception_models ./third_party/sam3)
else
    echo "Skipping SAM 3 / Perception Encoder; they are only imported behind"
    echo "--run_os. Re-run as './setup.sh --with-os' if you need them."
fi

echo "Installing local trees editable, without dependency resolution..."
ARGS=()
for t in "${TREES[@]}"; do ARGS+=(-e "$t"); done
uv pip install --no-deps "${ARGS[@]}"

cat <<'NOTE'

Installation complete.

Still needed before a run: the SALAD checkpoint, which this script does not fetch.
  curl -L --create-dirs -o ~/.cache/torch/hub/checkpoints/dino_salad.ckpt \
    https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
NOTE
