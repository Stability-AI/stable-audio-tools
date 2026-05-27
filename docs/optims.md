# Optimizers

Stable Audio Tools supports several optimizers for training diffusion models. Optimizers are configured in the model config JSON under `training.optimizer_configs`, with each training component (e.g. `diffusion`) getting its own optimizer and optional scheduler.

## Configuration

The optimizer config has two parts: an `optimizer` section and an optional `scheduler` section.

```json
"training": {
    "optimizer_configs": {
        "diffusion": {
            "optimizer": {
                "type": "AdamW",
                "config": {
                    "lr": 5e-5,
                    "betas": [0.9, 0.95],
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "InverseLR",
                "config": {
                    "inv_gamma": 1000000,
                    "power": 0.5,
                    "warmup": 0.995
                }
            }
        }
    }
}
```

The `type` field selects the optimizer class, and `config` is passed as keyword arguments to its constructor. Any optimizer available in `torch.optim` can be used by name (e.g. `AdamW`, `SGD`, `Adam`), in addition to the custom and third-party optimizers described below.

# Supported Optimizers

## `AdamW`

The standard PyTorch AdamW optimizer. This is the default baseline for most training runs.

| Key | Default | Description |
|-----|---------|-------------|
| `lr` | required | Learning rate |
| `betas` | `[0.9, 0.999]` | Momentum coefficients (beta1, beta2) |
| `eps` | `1e-8` | Numerical stability constant |
| `weight_decay` | `0.01` | Decoupled weight decay |

### Example config
```json
{
    "type": "AdamW",
    "config": {
        "lr": 5e-5,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.01
    }
}
```

## `MuonAdamW`

A hybrid optimizer that applies Muon (Newton-Schulz orthogonalized momentum) to 2D+ weight matrices and AdamW to everything else (biases, norms, scalars). Parameters are automatically split by dimensionality: `ndim >= 2` goes to Muon, `ndim < 2` goes to AdamW.

This is a single optimizer with two parameter groups, so learning rate schedulers work correctly with per-group base learning rates.

| Key | Default | Description |
|-----|---------|-------------|
| `muon_lr` | `0.02` | Learning rate for Muon (2D+ weights) |
| `muon_momentum` | `0.95` | Momentum coefficient for Muon |
| `muon_nesterov` | `true` | Use Nesterov momentum |
| `muon_weight_decay` | `0.0` | Weight decay for Muon params |
| `ns_steps` | `5` | Newton-Schulz iteration count |
| `adam_lr` | `3e-4` | Learning rate for AdamW (1D params) |
| `adam_betas` | `[0.9, 0.95]` | AdamW momentum coefficients |
| `adam_eps` | `1e-8` | AdamW epsilon |
| `adam_weight_decay` | `0.01` | Weight decay for AdamW params |
| `fused_layer_patterns` | `null` | Glob patterns for fused layers (see below) |

### Example config
```json
{
    "type": "MuonAdamW",
    "config": {
        "muon_lr": 0.001,
        "muon_momentum": 0.95,
        "adam_lr": 5e-5,
        "adam_betas": [0.9, 0.95],
        "adam_weight_decay": 0.01,
        "fused_layer_patterns": ["*.to_qkv.*", "*.to_kv.*", "*.to_q.*", "*.ff.*.proj.*"]
    }
}
```

## `CAdamW`

A custom AdamW variant with gradient agreement masking. On each step, a mask is computed where the momentum and current gradient agree in sign. The update is scaled by this mask (normalized by its mean), so parameters where momentum and gradient disagree receive smaller updates.

| Key | Default | Description |
|-----|---------|-------------|
| `lr` | `1e-3` | Learning rate |
| `betas` | `[0.9, 0.999]` | Momentum coefficients |
| `eps` | `1e-6` | Numerical stability constant |
| `weight_decay` | `0.0` | Decoupled weight decay |
| `correct_bias` | `true` | Apply Adam bias correction |

### Example config
```json
{
    "type": "CAdamW",
    "config": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 0.01
    }
}
```

## `CLion`

A sign-based momentum optimizer inspired by Lion. Computes the update as `sign(beta1 * momentum + (1 - beta1) * grad)`, masked by gradient agreement. Only maintains a single momentum buffer per parameter, making it more memory-efficient than Adam-family optimizers.

| Key | Default | Description |
|-----|---------|-------------|
| `lr` | `1e-4` | Learning rate |
| `betas` | `[0.9, 0.99]` | Momentum coefficients (beta1 for update, beta2 for EMA decay) |
| `weight_decay` | `0.0` | Decoupled weight decay |

### Example config
```json
{
    "type": "CLion",
    "config": {
        "lr": 1e-4,
        "betas": [0.9, 0.99],
        "weight_decay": 0.01
    }
}
```

## `AdamW8bit`

8-bit quantized AdamW from the `bitsandbytes` library. Stores optimizer state in 8-bit precision, reducing optimizer VRAM by ~75% compared to standard AdamW. Requires the `bitsandbytes` package.

### Example config
```json
{
    "type": "AdamW8bit",
    "config": {
        "lr": 5e-5,
        "betas": [0.9, 0.95],
        "weight_decay": 0.01
    }
}
```

## `FusedAdam`

Fused kernel Adam optimizer from DeepSpeed. Uses a custom CUDA kernel that fuses the Adam update into a single kernel launch, reducing overhead. Requires the `deepspeed` package.

### Example config
```json
{
    "type": "FusedAdam",
    "config": {
        "lr": 5e-5,
        "betas": [0.9, 0.95],
        "weight_decay": 0.01
    }
}
```

## Other PyTorch Optimizers

Any optimizer available in `torch.optim` can be used by setting `type` to the class name. For example, `SGD`, `Adam`, `RMSprop`, etc. The `config` dict is passed directly to the constructor.

# MuonAdamW Details

## Automatic Parameter Splitting

MuonAdamW automatically splits parameters into two groups based on tensor dimensionality:

- **Muon group** (`ndim >= 2`): Weight matrices in attention projections, feed-forward layers, embeddings. These get Newton-Schulz orthogonalized momentum updates.
- **AdamW group** (`ndim < 2`): Biases, layer norm weights, scalar parameters. These get standard AdamW updates.

For a typical 2.7B DiT model, ~99.9% of parameters are in the Muon group. The optimizer prints the split at initialization:

```
MuonAdamW: 248 Muon params (2787.1M), 318 AdamW params (1.1M)
```

## Newton-Schulz Orthogonalization

The core of Muon is Newton-Schulz (NS) orthogonalization, which computes the polar factor of the momentum update. For a matrix `G = U @ S @ V^T` (SVD), NS approximates `U @ V^T` -- the nearest orthogonal matrix. This implements steepest descent under the spectral norm, which provides better-conditioned updates for weight matrices compared to element-wise methods like Adam.

NS uses a quintic polynomial with optimized coefficients and runs in bfloat16 for tensor core efficiency. Five iterations are sufficient for convergence in practice.

## Fused Layer Chunking

Many transformer architectures use fused linear layers where multiple projections are packed into a single weight matrix for compute efficiency. For example, a fused `to_qkv` layer has shape `[3 * dim, dim]`, combining Q, K, and V projections.

Running NS on the entire fused matrix is problematic because it couples the orthogonalization across functionally independent sub-projections and distorts the effective learning rate via aspect ratio scaling (up to 2.83x for SwiGLU layers).

The `fused_layer_patterns` config accepts a list of `fnmatch` glob patterns that identify fused layers by parameter name. Matched layers are chunked into square sub-blocks along dim 0 before NS, with each chunk orthogonalized independently:

| Layer | Shape | Chunks | Sub-block |
|-------|-------|--------|-----------|
| `to_qkv` (differential) | [10240, 2048] | 5 | [2048, 2048] |
| `GLU.proj` (SwiGLU) | [16384, 2048] | 8 | [2048, 2048] |
| `to_kv` (differential) | [6144, 2048] | 3 | [2048, 2048] |
| `to_q` (differential) | [4096, 2048] | 2 | [2048, 2048] |

Non-fused tall matrices (like `project_in` [2048, 256]) are not chunked and use the standard full-matrix NS with aspect ratio scaling.

When `fused_layer_patterns` is set, the optimizer expects `(name, param)` tuples instead of bare parameters. This is handled automatically by `configure_optimizers` when the optimizer type is `MuonAdamW`.

## FSDP Compatibility

MuonAdamW supports FSDP training. NS orthogonalization requires full (unsharded) gradient matrices, so the optimizer uses `summon_full_params` to temporarily materialize each FSDP module's parameters during the Muon step. Modules are processed one at a time to limit peak VRAM.

FSDP modules are automatically detected and registered with the optimizer in `configure_optimizers`. This requires `use_orig_params=True` in the FSDP config so the optimizer sees individual parameters rather than flattened `FlatParameter` blobs.

## Hyperparameter Guidance

Muon learning rates are typically much higher than AdamW learning rates because NS orthogonalization normalizes the update direction. A reasonable starting point:

- `muon_lr`: 0.001 - 0.02 (for 2D weight matrices)
- `adam_lr`: Match your previous AdamW LR (e.g. 5e-5) for 1D params
- `muon_momentum`: 0.95 (standard)
- `adam_betas`: [0.9, 0.95] (can match previous config)

## Memory Comparison

MuonAdamW stores one momentum buffer per Muon parameter vs. two buffers (exp_avg + exp_avg_sq) for AdamW. For a 2.7B model this saves ~11 GB of optimizer state VRAM.

| Optimizer | State per param | Total state (2.7B model) |
|-----------|----------------|--------------------------|
| AdamW | 2 buffers (exp_avg + exp_avg_sq) | ~21.6 GB |
| MuonAdamW | 1 buffer (momentum) for 2D, 2 buffers for 1D | ~11.2 GB |
| AdamW8bit | 2 buffers in 8-bit | ~5.4 GB |

# Learning Rate Scheduler

## `InverseLR`

An inverse polynomial decay scheduler with exponential warmup.

| Key | Default | Description |
|-----|---------|-------------|
| `inv_gamma` | `1.0` | Steps to decay to ~50% of initial LR (with power=1) |
| `power` | `1.0` | Decay exponent |
| `warmup` | `0.0` | Exponential warmup factor in [0, 1); 0 disables warmup |
| `final_lr` | `0.0` | Minimum learning rate floor |

The learning rate at step `t` is computed as:

```
warmup_factor = 1 - warmup^(t + 1)
decay = (1 + t / inv_gamma) ^ -power
lr = warmup_factor * max(final_lr, base_lr * decay)
```

With `warmup=0.995`, the warmup reaches ~95% of the base LR after ~600 steps and >99% after ~1000 steps. With `inv_gamma=1000000` and `power=0.5`, the decay is very slow (square-root inverse schedule).

The scheduler runs per-step (not per-epoch) and respects per-param-group base learning rates, which is important for MuonAdamW where the Muon and AdamW groups have different base LRs.

### Example config
```json
{
    "type": "InverseLR",
    "config": {
        "inv_gamma": 1000000,
        "power": 0.5,
        "warmup": 0.995
    }
}
```

# Implementation Reference

- `stable_audio_tools/training/optims.py` -- CAdamW, CLion, MuonAdamW, and `zeropower_via_newtonschulz5`
- `stable_audio_tools/training/utils.py` -- `create_optimizer_from_config`, `InverseLR`
- `stable_audio_tools/training/diffusion.py` -- `configure_optimizers` integration, named params for MuonAdamW
