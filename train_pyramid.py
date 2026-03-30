"""
Pyramid Transformer training script for Parameter Golf.

Single-pass hierarchical architecture with causal attention and information bottleneck.
Adapted from train_gpt.py baseline.
"""

from __future__ import annotations

import glob
import math
import os
import time
import uuid

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Training
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 65536))  # Very small for testing
    base_seq_len = int(os.environ.get("BASE_SEQ_LEN", 256))  # Level 0 sequence length
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 65536))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))

    # Model - match baseline config
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))  # Match baseline tokenizer
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    d_ff = int(os.environ.get("D_FF", 2048))
    n_levels = int(os.environ.get("N_LEVELS", 3))
    n_blocks = int(os.environ.get("N_BLOCKS", 6))

    # Noise control
    noise_mode = os.environ.get("NOISE_MODE", "none")  # "none" | "vp" | "dropout"
    level_noise_std = tuple(float(x) for x in os.environ.get("LEVEL_NOISE_STD", "0.0,0.1,0.2").split(","))

    # Optimizer
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """Orthogonalize a 2D update matrix with Newton-Schulz iteration."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss

# -----------------------------
# PYRAMID UTILITIES
# -----------------------------

def level_seq_len(base_seq_len: int, level: int) -> int:
    return base_seq_len // (2 ** level)

def build_flat_index(base_seq_len: int, n_levels: int):
    """Build interleaved position index sorted by (tau, -level)"""
    positions = []
    for level in range(n_levels):
        L_i = level_seq_len(base_seq_len, level)
        for k in range(L_i):
            tau = k * (2 ** level)
            positions.append((tau, level, k))

    positions.sort(key=lambda x: (x[0], -x[1]))

    flat_tau = torch.tensor([p[0] for p in positions], dtype=torch.long)
    flat_level = torch.tensor([p[1] for p in positions], dtype=torch.long)
    flat_k = torch.tensor([p[2] for p in positions], dtype=torch.long)

    level_offsets = {level: [] for level in range(n_levels)}
    for idx, (tau, level, k) in enumerate(positions):
        level_offsets[level].append(idx)

    return flat_level, flat_k, flat_tau, level_offsets, len(positions)

# -----------------------------
# MODEL
# -----------------------------

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if (self._cos_cached is None or self._seq_len_cached != seq_len or
            self._cos_cached.device != device):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (B, H, N, Dh), cos/sin: (1, 1, N, Dh)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_half = cos[..., :half]
    sin_half = sin[..., :half]
    return torch.cat((x1 * cos_half - x2 * sin_half, x1 * sin_half + x2 * cos_half), dim=-1)


class PyramidPositionalEncoding(nn.Module):
    """RoPE on tau coordinate only - level differentiation via noise"""
    def __init__(self, d_model: int, n_levels: int, base_seq_len: int, rope_base: float = 10000.0):
        super().__init__()
        self.rotary = Rotary(d_model, base=rope_base)
        max_tau = base_seq_len * (2 ** (n_levels - 1))
        self.max_tau = max_tau

    def forward(self, flat_level: Tensor, flat_tau: Tensor) -> Tensor:
        # RoPE will be applied in attention, just return zeros here
        return torch.zeros(flat_tau.size(0), self.rotary.inv_freq.size(0) * 2,
                          device=flat_tau.device)

class GQA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.Wk = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.Wv = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.Wo = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, N, _ = x.shape
        H, Hkv, Dh = self.n_heads, self.n_kv_heads, self.d_head

        Q = self.Wq(x).view(B, N, H, Dh).transpose(1, 2)
        K = self.Wk(x).view(B, N, Hkv, Dh).transpose(1, 2)
        V = self.Wv(x).view(B, N, Hkv, Dh).transpose(1, 2)

        # Apply RoPE
        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)
        out = out.transpose(1, 2).contiguous().view(B, N, H * Dh)
        return self.Wo(out)

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff, bias=False)
        self.proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.gelu(self.fc(x)))

class PyramidBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def _apply_level_noise(self, x: Tensor, flat_level: Tensor, noise_mode: str,
                          level_noise_std: tuple, n_levels: int, training: bool) -> Tensor:
        if noise_mode == "none":
            return x

        for level in range(1, n_levels):
            if level >= len(level_noise_std):
                continue
            level_mask = (flat_level == level)

            if noise_mode == "vp":
                std = level_noise_std[level]
                signal_scale = math.sqrt(1 - std * std)
                noise = torch.randn_like(x[:, level_mask]) * std
                x[:, level_mask] = signal_scale * x[:, level_mask] + noise
            elif noise_mode == "dropout" and training:
                dropout_p = level_noise_std[level]
                x[:, level_mask] = F.dropout(x[:, level_mask], p=dropout_p, training=True)

        return x

    def forward(self, x: Tensor, flat_level: Tensor, noise_mode: str,
                level_noise_std: tuple, n_levels: int, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = self._apply_level_noise(x, flat_level, noise_mode, level_noise_std, n_levels, self.training)
        x = x + self.mlp(self.norm2(x))
        x = self._apply_level_noise(x, flat_level, noise_mode, level_noise_std, n_levels, self.training)
        return x

class PyramidLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_kv_heads: int,
                 d_ff: int, n_levels: int, base_seq_len: int, n_blocks: int,
                 noise_mode: str, level_noise_std: tuple):
        super().__init__()
        self.noise_mode = noise_mode
        self.level_noise_std = level_noise_std
        self.n_levels = n_levels

        flat_level, flat_k, flat_tau, level_offsets, N = build_flat_index(base_seq_len, n_levels)
        self.register_buffer('flat_level', flat_level)
        self.register_buffer('flat_k', flat_k)
        self.register_buffer('flat_tau', flat_tau)
        self.level_offsets = level_offsets
        self.N = N

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PyramidPositionalEncoding(d_model, n_levels, base_seq_len)
        self.blocks = nn.ModuleList([PyramidBlock(d_model, n_heads, n_kv_heads, d_ff) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _build_grid(self, token_ids: Tensor) -> Tensor:
        B, L = token_ids.shape
        x = torch.zeros(B, self.N, self.tok_emb.weight.size(1),
                       device=token_ids.device, dtype=self.tok_emb.weight.dtype)

        level0_indices = self.level_offsets[0]
        for i, idx in enumerate(level0_indices[:L]):
            x[:, idx] = self.tok_emb(token_ids[:, i])

        return x

    def forward(self, token_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._build_grid(token_ids)

        # Get RoPE cos/sin for all tau positions
        cos, sin = self.pos_enc.rotary(self.N, x.device, x.dtype)
        # Index by flat_tau to get correct RoPE for each position
        cos = cos[:, :, self.flat_tau, :]
        sin = sin[:, :, self.flat_tau, :]

        for block in self.blocks:
            x = block(x, self.flat_level, self.noise_mode, self.level_noise_std, self.n_levels, cos, sin)
        x = self.ln_f(x)

        # Extract level 0 logits
        level0_indices = self.level_offsets[0]
        L = token_ids.shape[1]
        logits_list = []
        for idx in level0_indices[:L]:
            logits_list.append(x[:, idx:idx+1, :])
        logits = torch.cat(logits_list, dim=1)
        logits = self.lm_head(logits)

        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

# -----------------------------
# DATA LOADING
# -----------------------------

def load_tokens(filename: str) -> np.ndarray:
    """Load tokens from binary file, skipping 256-int32 header"""
    header_bytes = 256 * 4  # 256 int32s
    with open(filename, 'rb') as f:
        f.seek(header_bytes)
        return np.fromfile(f, dtype=np.uint16)

class DataLoader:
    def __init__(self, files_pattern: str, batch_tokens: int, seq_len: int,
                 world_size: int, rank: int, seed: int):
        self.files = sorted(glob.glob(files_pattern))
        if not self.files:
            raise ValueError(f"No files found matching {files_pattern}")
        self.batch_tokens = batch_tokens
        self.seq_len = seq_len
        self.world_size = world_size
        self.rank = rank
        self.rng = np.random.RandomState(seed + rank)
        self.current_shard_idx = 0
        self.current_position = 0
        self.tokens = load_tokens(self.files[0])

    def next_batch(self):
        B = self.batch_tokens // self.seq_len
        batch_tokens = []
        batch_targets = []

        for _ in range(B):
            if self.current_position + self.seq_len + 1 > len(self.tokens):
                self.current_shard_idx = (self.current_shard_idx + 1) % len(self.files)
                self.tokens = load_tokens(self.files[self.current_shard_idx])
                self.current_position = 0

            chunk = self.tokens[self.current_position:self.current_position + self.seq_len + 1]
            batch_tokens.append(chunk[:-1])
            batch_targets.append(chunk[1:])
            self.current_position += self.seq_len

        return (torch.from_numpy(np.array(batch_tokens)).long(),
                torch.from_numpy(np.array(batch_targets)).long())

# -----------------------------
# EVALUATION
# -----------------------------

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)

    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))

    return (
        torch.from_numpy(base_bytes_np).to(device),
        torch.from_numpy(has_leading_space_np).to(device),
    )

@torch.no_grad()
def evaluate_bpb(model: PyramidLM, val_loader: DataLoader, sp: spm.SentencePieceProcessor,
                device: torch.device, max_batches: int = 100) -> float:
    model.eval()
    base_bytes, _ = build_sentencepiece_luts(sp, model.lm_head.out_features, device)

    total_bits = 0.0
    total_bytes = 0.0

    for _ in range(max_batches):
        tokens, targets = val_loader.next_batch()
        tokens, targets = tokens.to(device), targets.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Forward through model
            x = model._build_grid(tokens)

            # Get RoPE
            cos, sin = model.pos_enc.rotary(model.N, x.device, x.dtype)
            cos = cos[:, :, model.flat_tau, :]
            sin = sin[:, :, model.flat_tau, :]

            for block in model.blocks:
                x = block(x, model.flat_level, model.noise_mode, model.level_noise_std, model.n_levels, cos, sin)
            x = model.ln_f(x)

            # Extract level 0 logits
            level0_indices = model.level_offsets[0]
            L = tokens.shape[1]
            logits_list = []
            for idx in level0_indices[:L]:
                logits_list.append(x[:, idx:idx+1, :])
            logits = torch.cat(logits_list, dim=1)
            logits = model.lm_head(logits)

        log_probs = F.log_softmax(logits.float(), dim=-1)
        target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        bits = -target_log_probs.sum().item() / math.log(2)
        bytes_count = base_bytes[targets].sum().item()

        total_bits += bits
        total_bytes += bytes_count

    model.train()
    return total_bits / total_bytes if total_bytes > 0 else float('inf')

# -----------------------------
# MAIN
# -----------------------------

def main():
    args = Hyperparameters()

    # Distributed setup
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend="nccl")

    master_process = rank == 0

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Model
    model = PyramidLM(
        vocab_size=args.vocab_size,
        d_model=args.model_dim,
        n_heads=args.num_heads,
        n_kv_heads=args.num_kv_heads,
        d_ff=args.d_ff,
        n_levels=args.n_levels,
        base_seq_len=args.base_seq_len,
        n_blocks=args.n_blocks,
        noise_mode=args.noise_mode,
        level_noise_std=args.level_noise_std,
    ).to(device)

    # Compile model
    compiled_model = torch.compile(model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank]) if distributed else compiled_model

    raw_model = compiled_model.module if distributed else compiled_model

    # Optimizer - split params like baseline
    matrix_params = []
    scalar_params = []

    for name, param in raw_model.named_parameters():
        if 'tok_emb' in name or 'lm_head' in name:
            continue  # Handle separately
        elif param.ndim >= 2:
            matrix_params.append(param)
        else:
            scalar_params.append(param)

    optimizer_tok = torch.optim.AdamW(
        [{"params": [raw_model.tok_emb.weight], "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True
    )
    optimizer_head = torch.optim.AdamW(
        [{"params": [raw_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True
    )
    optimizers = [optimizer_tok, optimizer_head, optimizer_muon, optimizer_scalar]

    # Data
    train_loader = DataLoader(
        args.train_files,
        args.train_batch_tokens // world_size,
        args.base_seq_len,
        world_size,
        rank,
        args.seed,
    )

    val_loader = DataLoader(
        args.val_files,
        args.val_batch_size // world_size,
        args.base_seq_len,
        world_size,
        rank,
        args.seed + 1,
    )

    # Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_path)

    # Training loop
    if master_process:
        print(f"Training Pyramid Transformer")
        print(f"  Levels: {args.n_levels}, Base seq len: {args.base_seq_len}")
        print(f"  Noise mode: {args.noise_mode}")
        print(f"  Total positions: {raw_model.N}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    start_time = time.time()
    training_time_ms = 0.0

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def lr_schedule(step: int, elapsed_ms: float) -> float:
        """Warmdown schedule"""
        if args.warmdown_iters <= 0:
            return 1.0
        if args.max_wallclock_seconds <= 0:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        # Time-based warmdown
        max_wallclock_ms = 1000.0 * args.max_wallclock_seconds
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(args.iterations):
        # Check time limit
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.max_wallclock_seconds > 0 and elapsed_ms >= 1000.0 * args.max_wallclock_seconds:
            if master_process:
                print(f"Reached time limit at step {step}")
            break

        # Get batch
        tokens, targets = train_loader.next_batch()
        tokens, targets = tokens.to(device), targets.to(device)

        # Forward
        zero_grad_all()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(tokens, targets)

        # Backward
        loss.backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip_norm)

        # Update learning rates with warmdown
        scale = lr_schedule(step, elapsed_ms)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        # Optimizer step
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # Logging
        should_log = (step <= 10 or step % args.train_log_every == 0)
        if master_process and should_log:
            print(f"Step {step:5d} | Loss {loss.item():.4f} | LR scale {scale:.4f} | {elapsed_ms/1000:.1f}s")

        # Validation
        if step % args.val_loss_every == 0 and step > 0:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            bpb = evaluate_bpb(raw_model, val_loader, sp, device)
            if master_process:
                print(f"Step {step:5d} | Val BPB {bpb:.4f} | Train time {training_time_ms/1000:.1f}s")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

    # Final eval
    torch.cuda.synchronize()
    training_time_ms += 1000.0 * (time.perf_counter() - t0)
    if master_process:
        final_bpb = evaluate_bpb(raw_model, val_loader, sp, device, max_batches=200)
        print(f"\nFinal validation BPB: {final_bpb:.4f}")
        print(f"Total time: {training_time_ms/1000:.1f}s")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
