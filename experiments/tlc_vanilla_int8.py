from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

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
    # Using a plain class (not @dataclass) so field defaults can reference each other
    # correctly when DATA_PATH is overridden via env var.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = 1337

    val_batch_size = 524288
    val_loss_every = 100000
    train_log_every = 200

    iterations = 20000
    warmdown_iters = 1000
    warmup_steps = 20
    train_batch_tokens = 393216
    train_seq_len = 4096
    max_wallclock_seconds = 600.0
    qk_gain_init = 1.0

    vocab_size = 1024
    num_layers = 3  # Unique blocks; looped num_loops times for num_layers*num_loops effective layers
    num_loops = 3   # 3 blocks × 3 loops = 9 effective layers
    num_kv_heads = 6
    model_dim = 816
    num_heads = 12
    mlp_mult = 2.5
    tie_embeddings = True
    rope_base = 10000.0
    logit_softcap = 30.0

    embed_lr = 0.5
    head_lr = 0.01
    tied_embed_lr = 0.04
    tied_embed_init_std = 0.006
    matrix_lr = 0.04
    scalar_lr = 0.03
    muon_momentum = 0.95
    muon_backend_steps = 10
    muon_momentum_warmup_start = 0.85
    muon_momentum_warmup_steps = 800
    beta1 = 0.9
    beta2 = 0.95
    adam_eps = 1e-8
    grad_clip_norm = 0.0  # Disabled: higher NS steps + resid_mix stabilise gradients sufficiently

# -----------------------------
# MUON OPTIMIZER
# -----------------------------
@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
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
        distributed = dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank:
                    state = self.state[p]
                    if "mom" not in state: state["mom"] = torch.zeros_like(p.grad)
                    buf = state["mom"]
                    buf.mul_(group["momentum"]).add_(p.grad)
                    g = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                p.add_(updates_flat[curr : curr + p.numel()].view_as(p).to(p.dtype), alpha=-group["lr"])
                curr += p.numel()

# -----------------------------
# QUANTIZATION
# -----------------------------
CONTROL_TENSOR_NAME_PATTERNS = ("q_gain", "k_gain", "attn_scale", "mlp_scale", "resid_mix", "loop_embeds", "loop_skip")
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Per-row scale with percentile clipping — robust to weight outliers.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    # Vectors/scalars: per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        # Keep embeddings in fp16 — quantising the tied weight hurts both input and output paths.
        if "tok_emb" in name:
            kept = t.to(dtype=torch.float16).contiguous()
            passthrough[name] = kept
            passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        # Small tensors: keep as float rather than pay int8+scale overhead.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out

# -----------------------------
# CASTED LINEAR
# -----------------------------

class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)

# -----------------------------
# DATA LOADING
# -----------------------------
def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    tokens = np.fromfile(file, dtype="<u2", offset=1024)
    return torch.from_numpy(tokens.astype(np.uint16))

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.files = sorted([Path(p) for p in glob.glob(pattern)])
        self.rank, self.world_size, self.device = rank, world_size, device
        self.file_idx, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum: int) -> tuple[Tensor, Tensor]:
        n = (global_tokens // (self.world_size * grad_accum * seq_len)) * seq_len
        needed = (n + 1) * self.world_size
        chunks = []
        while needed > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.file_idx = (self.file_idx + 1) % len(self.files)
                self.tokens = load_data_shard(self.files[self.file_idx])
                self.pos = 0
                continue
            k = min(needed, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            needed -= k
        chunk = torch.cat(chunks)
        local = chunk[self.rank * (n + 1) : (self.rank + 1) * (n + 1)].to(torch.long)
        return local[:-1].view(-1, seq_len).to(self.device), local[1:].view(-1, seq_len).to(self.device)

# -----------------------------
# EVALUATION UTILS
# -----------------------------
def build_sentencepiece_luts(sp, vocab_size, device):
    base_bytes = np.zeros((vocab_size,), dtype=np.int16)
    has_space = np.zeros((vocab_size,), dtype=np.bool_)
    is_boundary = np.ones((vocab_size,), dtype=np.bool_)
    for i in range(sp.vocab_size()):
        if i >= vocab_size or sp.is_control(i): continue
        is_boundary[i] = False
        if sp.is_byte(i): base_bytes[i] = 1; continue
        p = sp.id_to_piece(i)
        if p.startswith("▁"): has_space[i] = True; p = p[1:]
        base_bytes[i] = len(p.encode("utf-8"))
    return torch.tensor(base_bytes, device=device), torch.tensor(has_space, device=device), torch.tensor(is_boundary, device=device)

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    tokens = torch.cat([load_data_shard(Path(p)) for p in sorted(glob.glob(pattern))])
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

def eval_val(args, model, rank, world_size, device, grad_accum, val_tokens, b_lut, s_lut, bnd_lut):
    model.eval()
    total_loss, total_tokens, total_bytes = 0.0, 0.0, 0.0
    seqs = (val_tokens.numel() - 1) // args.train_seq_len
    per_rank = seqs // world_size
    with torch.no_grad():
        for i in range(rank * per_rank, (rank + 1) * per_rank):
            raw = val_tokens[i * args.train_seq_len : (i + 1) * args.train_seq_len + 1].to(device).long()
            x, y = raw[:-1].view(1, -1), raw[1:].view(1, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            t_bytes = b_lut[y] + (s_lut[y] & ~bnd_lut[x]).short()
            total_bytes += t_bytes.sum().item()
    
    stats = torch.tensor([total_loss, total_tokens, total_bytes], device=device)
    if dist.is_initialized(): dist.all_reduce(stats)
    v_loss = stats[0] / stats[1]
    v_bpb = (v_loss / math.log(2.0)) * (stats[1] / stats[2])
    model.train()
    return v_loss.item(), v_bpb.item()

# -----------------------------
# TRANSFORMER BLOCKS
# -----------------------------
class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv", inv, persistent=False)
    def forward(self, n, device):
        t = torch.arange(n, device=device).float()
        freqs = torch.outer(t, self.inv)
        return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]

def apply_rope(x, c, s):
    x1, x2 = x.chunk(2, -1)
    return torch.cat((x1 * c + x2 * s, x1 * -s + x2 * c), -1)

class Block(nn.Module):
    def __init__(self, h: Hyperparameters):
        super().__init__()
        self.h, self.hd = h, h.model_dim // h.num_heads
        self.an, self.mn = nn.RMSNorm(h.model_dim), nn.RMSNorm(h.model_dim)
        self.qkv = nn.Linear(h.model_dim, (h.num_heads + 2 * h.num_kv_heads) * self.hd, bias=False)
        self.proj = nn.Linear(h.model_dim, h.model_dim, bias=False)
        self.fc = nn.Linear(h.model_dim, int(h.model_dim * h.mlp_mult), bias=False)
        self.mlp_out = nn.Linear(int(h.model_dim * h.mlp_mult), h.model_dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((h.num_heads,), h.qk_gain_init))
        self.k_gain = nn.Parameter(torch.ones(h.num_kv_heads))
        # Per-dim learned output gates — let each sub-layer control its residual contribution.
        # Especially important in recursive models where the same block runs at 3 effective depths.
        self.attn_scale = nn.Parameter(torch.ones(h.model_dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(h.model_dim, dtype=torch.float32))
        # resid_mix: learned per-dim blend of current x and initial embedding x0.
        # Lets each block decide how much to "refresh" from the original token embedding.
        # Especially valuable in recursive models where x drifts far from x0 over 9 loops.
        # Initialised as [1, 0] so it starts as identity (no change to existing behaviour).
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(h.model_dim), torch.zeros(h.model_dim)]).float())

    def forward(self, x, x0, cos, sin):
        # Blend current representation with initial embedding — learned per-dim mix.
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        r = self.an(x)
        qkv = self.qkv(r)
        q, k, v = qkv.split([self.h.model_dim, self.h.num_kv_heads * self.hd, self.h.num_kv_heads * self.hd], -1)
        q = q.view(x.size(0), x.size(1), self.h.num_heads, self.hd).transpose(1, 2)
        k = k.view(x.size(0), x.size(1), self.h.num_kv_heads, self.hd).transpose(1, 2)
        v = v.view(x.size(0), x.size(1), self.h.num_kv_heads, self.hd).transpose(1, 2)
        q = F.rms_norm(q, (self.hd,)) * self.q_gain[None, :, None, None]
        k = F.rms_norm(k, (self.hd,)) * self.k_gain[None, :, None, None]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.h.num_kv_heads != self.h.num_heads))
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.proj(y.transpose(1, 2).reshape(x.shape))
        r = self.mn(x)
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp_out(F.relu(self.fc(r)).square())
        return x

class GPT(nn.Module):
    def __init__(self, h: Hyperparameters):
        super().__init__()
        self.h = h
        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)
        self.loop_embeds = nn.Parameter(torch.zeros(h.num_loops, h.model_dim))
        # Cross-cycle skip: passes cycle-0 output directly into cycle-(num_loops-1) input.
        # Analogous to U-net encoder→decoder skip, gives the last loop direct access to
        # early representations without relying on cycle-1 to route them through.
        self.loop_skip_weight = nn.Parameter(torch.zeros(h.model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([Block(h) for _ in range(h.num_layers)])
        self.final_norm = nn.RMSNorm(h.model_dim)
        self.rope = Rotary(h.model_dim // h.num_heads, h.rope_base)
        self.lm_head = nn.Linear(h.model_dim, h.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        nn.init.normal_(self.tok_emb.weight, std=h.tied_embed_init_std)

    def forward(self, idx, targets=None):
        cos, sin = self.rope(idx.size(1), idx.device)
        x = F.rms_norm(self.tok_emb(idx), (self.h.model_dim,))
        x0 = x
        x_cycle0 = None
        for cycle in range(self.h.num_loops):
            x = x + self.loop_embeds[cycle]
            # Last loop: inject skip from end of cycle 0 for direct early→late signal.
            if cycle == self.h.num_loops - 1 and x_cycle0 is not None:
                x = x + self.loop_skip_weight.to(x.dtype)[None, None, :] * x_cycle0
            for b in self.blocks: x = b(x, x0, cos, sin)
            if cycle == 0:
                x_cycle0 = x
        logits = self.h.logit_softcap * torch.tanh(self.lm_head(self.final_norm(x)) / self.h.logit_softcap)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else logits

def restore_low_dim_params_to_fp32(model):
    for n, p in model.named_parameters():
        if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS):
            p.data = p.data.float()

# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    grad_accum_steps = max(4, 32 // world_size)
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(True)
    enable_math_sdp(True)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}_seed{args.seed}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    
    # We collect all parameters not handled by special optimizers (tok_emb, lm_head)
    # and split them into matrix (2D) and scalar/vector (<2D or control) groups.
    # Only route block parameters through Muon / scalar Adam; tok_emb and lm_head
    # get their own optimizers below so they must NOT appear here as well.
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    # loop_embeds live outside blocks — add them to the scalar group.
    scalar_params.append(base_model.loop_embeds)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    # Only create a separate head optimizer when embeddings are NOT tied.
    # When tie_embeddings=True the lm_head weight IS tok_emb.weight, so adding
    # it to a second optimizer would cause double-updates on the same tensor.
    if not args.tie_embeddings:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=True math=True")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
