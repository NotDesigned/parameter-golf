#!/usr/bin/env bash
set -euo pipefail

# ── speedrun.sh ──────────────────────────────────────────────────────
# One-shot setup + data download + training for parameter-golf.
# Designed for a fresh remote GPU box (RunPod / Lambda / etc.)
# with CUDA + NCCL already present.
#
# Usage:
#   bash speedrun.sh              # auto-detect GPUs
#   NGPU=1 bash speedrun.sh       # force single GPU
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NGPU="${NGPU:-1}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_gpt_sota.py}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"

echo "=== speedrun.sh ==="
echo "  NGPU:          $NGPU"
echo "  TRAIN_SCRIPT:  $TRAIN_SCRIPT"
echo "  TRAIN_SHARDS:  $TRAIN_SHARDS"
echo ""

# ── 1. Install Python dependencies ──────────────────────────────────
echo "[1/4] Installing Python dependencies ..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q zstandard

# Flash Attention 3 (Hopper) — needed by train_gpt_sota.py
if ! python -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "  Installing flash-attn (Hopper/FA3) ..."
    pip install -q flash-attn --no-build-isolation
fi

echo "  Done."
echo ""

# ── 2. Download data & tokenizer ────────────────────────────────────
echo "[2/4] Downloading dataset (sp1024, ${TRAIN_SHARDS} shards) ..."
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$TRAIN_SHARDS"
echo "  Done."
echo ""

# ── 3. Train ────────────────────────────────────────────────────────
echo "[3/4] Training with $NGPU GPU(s) ..."
echo "  Script: $TRAIN_SCRIPT"
echo ""

torchrun --standalone --nproc_per_node="$NGPU" "$TRAIN_SCRIPT"

echo ""
echo "[4/4] Training finished."

# ── 4. Summary ──────────────────────────────────────────────────────
# Print final results from the log
LOGDIR="logs"
if [ -d "$LOGDIR" ]; then
    LATEST_LOG=$(ls -t "$LOGDIR"/*.txt 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo ""
        echo "=== Results (from $LATEST_LOG) ==="
        grep -E "final_int6_roundtrip_exact|final_int6_sliding_window_exact|final_int8_zlib_roundtrip_exact|DIAGNOSTIC|stopping_early|Total submission size" "$LATEST_LOG" || true
    fi
fi

echo ""
echo "=== Done ==="
