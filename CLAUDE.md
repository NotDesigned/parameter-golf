# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Challenge Overview

Parameter Golf is a competition to train the best language model that:
- Fits in a 16MB artifact
- Trains in under 10 minutes on 8×H100s
- Evaluated by compression on FineWeb validation set (bits per byte, tokenizer-agnostic)

This is L(N) optimization: minimize loss given fixed parameters, unconstrained by data, compute, steps, or architecture.

## Key Commands

### Setup and Data Download
```bash
# Quick start: setup + download + train
bash speedrun.sh

# Download dataset only (default: 80 train shards = 8B tokens)
python data/cached_challenge_fineweb.py --variant sp1024

# Download more training data
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 180
```

### Training
```bash
# Single GPU
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Multi-GPU (8×H100 for leaderboard submissions)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# With custom hyperparameters (all configurable via env vars)
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Testing Single Changes
```bash
# Run with specific seed
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Quick test run (fewer iterations)
ITERATIONS=1000 MAX_WALLCLOCK_SECONDS=60 torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Architecture

### Training Scripts
- `train_gpt.py` - Baseline training script for newcomers (~1100 lines, must stay under 1500)
- `train_gpt_mlx.py` - Apple MLX variant for Mac training
- `train_gpt_sota.py` - Advanced SOTA techniques (~1400 lines)
- All scripts are self-contained with model, optimizer, and training loop

### Core Components (train_gpt.py)
- `Hyperparameters` class: All hyperparameters configurable via environment variables
- `Muon` optimizer: Newton-Schulz orthogonalization-based optimizer (from modded-nanogpt)
- `GPT` model: Standard transformer with RMSNorm, RoPE, GQA, tied embeddings
- Tokenizer-agnostic evaluation: BPB (bits-per-byte) instead of loss
- Distributed training: DDP with NCCL backend, gradient accumulation

### Model Architecture (Baseline)
- 9 transformer blocks, 512 width
- 8 attention heads, 4 KV heads (GQA)
- 2× MLP expansion
- Vocab size 1024, sequence length 1024
- Tied embeddings (input/output share weights)
- RoPE positional encoding
- RMSNorm instead of LayerNorm

### Data Pipeline
- Dataset: FineWeb 10B tokens (train) + validation split
- Tokenizer: SentencePiece BPE (1024 vocab by default)
- Sharded binary format: `fineweb_train_*.bin`, `fineweb_val_*.bin`
- Each shard: 100M tokens
- Data path: `./data/datasets/fineweb10B_sp1024/`

### Evaluation
- Tokenizer-agnostic: measures compression (bits-per-byte) not loss
- Validation on full FineWeb validation split
- Common techniques: sliding window eval, int6/int8 quantization roundtrip
- Artifact size: model weights compressed (zlib/zstd) must fit in 16MB

## Submission Structure

Each record in `records/track_10min_16mb/YYYY-MM-DD_Name/` contains:
- `README.md` - Detailed explanation, results, ablations
- `submission.json` - Metadata (author, score, GitHub ID)
- `train_gpt.py` - Self-contained training script
- `train_seed*.log` - Training logs (typically 3 seeds for statistical significance)

## Common Techniques (from SOTA submissions)

### Architecture Innovations
- XSA (Cross-layer Self-Attention): attention on last N layers only
- Parameter tying/banking: share weights across layers
- Mixed precision: FP16 embeddings, quantized MLPs
- Partial RoPE: apply RoPE to subset of dimensions
- BigramHash: hash-based bigram embeddings for compression

### Training Techniques
- Muon optimizer with momentum warmup
- EMA (Exponential Moving Average) + SWA (Stochastic Weight Averaging)
- Warmdown: reduce LR in final iterations
- QAT (Quantization-Aware Training): train with quantization in mind
- TTT (Test-Time Training): adapt on validation chunks (score-first protocol)

### Compression
- GPTQ-lite: post-training quantization
- Int6/Int8 quantization with STE (Straight-Through Estimator)
- zstd-22 or lzma compression for final artifact

## Development Notes

- `train_gpt.py` and `train_gpt_mlx.py` are for newcomers, not SOTA configs
- Competitive submissions go in `/records` folder
- Each submission must compile and run successfully
- Logs written to `logs/{run_id}.txt`
- Distributed training assumes 8 GPUs (world_size must divide 8)
- All hyperparameters exposed via environment variables for easy experimentation
