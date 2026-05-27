# LoRA in Stable Audio Tools

LoRA (Low-Rank Adaptation) enables parameter-efficient fine-tuning of large diffusion models. Instead of updating all model weights during training, LoRA freezes the pre-trained weights and injects small, trainable low-rank matrices into each layer. This dramatically reduces the number of trainable parameters and the size of saved checkpoints while still allowing the model to learn new behaviors.

Stable Audio Tools supports three adapter types with increasing parameter efficiency:

| Adapter | Trainable Params per Layer | Use Case |
|---------|--------------------------|----------|
| **LoRA** | `rank * (fan_in + fan_out)` | General-purpose fine-tuning. Good balance of expressiveness and efficiency. |
| **DoRA** | `rank * (fan_in + fan_out) + fan_in` | When you need more expressive fine-tuning. Separates direction and magnitude updates. |
| **LoRA-XS** | `rank * rank` | Maximum parameter efficiency. Only a tiny core matrix is trainable. |

LoRA is applied to both the diffusion backbone (DiT) and the conditioner (only trainable parameters like the `seconds_total` conditioner), and is supported in both standard diffusion training and ARC training.

# Adapter Types

## LoRA (Standard)

Standard LoRA decomposes each weight update into two low-rank matrices:

```
W' = W + (alpha/rank) * B @ A
```

Where `W` is the frozen pre-trained weight, `A` is a `(rank, fan_in)` matrix, and `B` is a `(fan_out, rank)` matrix. Only `A` and `B` are trained. The `alpha/rank` ratio controls the scaling of the LoRA update relative to the original weight.

`A` is initialized with Kaiming uniform initialization and `B` is initialized to zeros, so the LoRA update starts at zero and the model begins training from its pre-trained behavior.

## DoRA

DoRA (Weight-Decomposed Low-Rank Adaptation) extends standard LoRA by separating weight updates into direction and magnitude components:

```
V = W + (alpha/rank) * B @ A       (low-rank update, same as LoRA)
V_hat = V / ||V||_col               (column-normalized direction)
W' = V_hat * magnitude              (scale by per-column magnitude)
```

The `magnitude` vector is initialized from the column norms of the original pre-trained weight: `||W||_col`. This means the model starts with the same effective weights as the pre-trained model, but can independently adjust the direction and magnitude of each column during training.

DoRA trains the same `A` and `B` matrices as LoRA, plus an additional `magnitude` vector of size `fan_in`. This is slightly more parameters than standard LoRA but can be more expressive for certain fine-tuning tasks.

## LoRA-XS

LoRA-XS takes parameter efficiency to the extreme by freezing the low-rank bases and only training a tiny core matrix:

```
U, S, V^T = SVD(W)                  (SVD of pre-trained weight)
W' = W + (alpha/rank) * U[:, :r] @ M_xs @ V[:, :r]^T
```

`U` and `V` are frozen buffers derived from the SVD of the original weight. Only `M_xs`, a `(rank, rank)` matrix, is trainable. For a typical rank of 8, this means only 64 trainable parameters per layer, compared to thousands for standard LoRA.

Because LoRA-XS computes SVD during initialization, placing the model on GPU before adding LoRA enables GPU-accelerated SVD, which is significantly faster for large models. The inference code does this automatically.

# Training Configuration

## Standard Diffusion Training

Add a `lora_config` section to your training configuration:

```json
{
    "model_type": "diffusion_cond",
    "training": {
        "learning_rate": 1e-4,
        "lora_config": {
            "rank": 8,
            "alpha": 8,
            "adapter_type": "lora"
        }
    }
}
```

## ARC Training

For ARC training, the LoRA config goes inside the `arc` section. You can configure LoRA independently for the generator and discriminator:

```json
{
    "model_type": "diffusion_cond",
    "training": {
        "arc": {
            "lora": {
                "rank": 8,
                "alpha": 8,
                "adapter_type": "lora",
                "dropout": 0.0
            },
            "discriminator": {
                "lora": {
                    "rank": 4,
                    "alpha": 4,
                    "adapter_type": "lora"
                }
            }
        }
    }
}
```

The generator LoRA is applied to both the diffusion model and conditioner. The discriminator LoRA is applied only to the discriminator's backbone Linear layers.

## Config Keys

| Key | Default | Description |
|-----|---------|-------------|
| `rank` | 8 | LoRA rank. Lower = fewer parameters, higher = more expressive. |
| `alpha` | same as `rank` | Scaling factor. The effective scaling is `alpha / rank`. Setting `alpha = rank` gives a scaling of 1.0. |
| `adapter_type` | `"lora"` | One of `"lora"`, `"dora"`, or `"lora-xs"`. |
| `dropout` | 0.0 | Dropout probability applied to LoRA inputs during training. Only used in ARC config. |
| `include` | `null` (all layers) | List of substring patterns. Only modules whose name contains at least one pattern get LoRA. Supports bracket expansion. |
| `exclude` | `null` (no exclusions) | List of substring patterns. Modules matching any pattern are skipped, even if they match an include pattern. Supports bracket expansion. |

## Layer Filtering

You can control which layers receive LoRA using `include` and `exclude` in the LoRA config. Both accept lists of substring patterns with bracket expansion support (e.g., `layers[0-11]`).

When `include` is specified, only layers whose fully-qualified name contains at least one of the include substrings are candidates for LoRA. When `exclude` is specified, any layer matching an exclude pattern is skipped, even if it also matches an include pattern. When neither is specified, all layers of matching types (Linear, Conv1d) receive LoRA, which is the default behavior.

### Examples

Exclude the `seconds_total` conditioner (prevents conditioner hijacking on small datasets):
```json
{
    "rank": 8,
    "alpha": 8,
    "adapter_type": "lora-xs",
    "exclude": ["seconds_total"]
}
```

Only apply LoRA to transformer layers:
```json
{
    "rank": 8,
    "include": ["transformer.layers"]
}
```

Only the first 12 transformer layers:
```json
{
    "rank": 8,
    "include": ["layers[0-11]"]
}
```

Everything except local embedding and seconds_total conditioner:
```json
{
    "rank": 8,
    "exclude": ["to_local_embed", "seconds_total"]
}
```

Layer names are matched against the module's fully-qualified name relative to the submodel root. For the diffusion backbone these look like `transformer.layers.0.self_attn.to_qkv`, `to_timestep_embed.0`, etc. For the conditioner: `conditioners.seconds_total.embedder.embedding.1`, etc.

The `include`/`exclude` config is persisted in the checkpoint and automatically applied when loading the LoRA at inference time.

## Resuming LoRA Training

To resume training from a saved LoRA checkpoint, add `lora_ckpt_path` to your training config:

```json
{
    "training": {
        "lora_ckpt_path": "/path/to/lora_checkpoint.ckpt",
        "lora_config": {
            "rank": 8,
            "alpha": 8,
            "adapter_type": "lora"
        }
    }
}
```

This loads the LoRA weights from the checkpoint into the freshly-created LoRA layers, allowing you to continue training from where you left off. The `lora_config` must match the config used when the checkpoint was originally saved.

Note: LoRA checkpoints do not contain PyTorch Lightning training state (optimizer, scheduler, epoch, etc.) - only the LoRA weights and config are saved. Resuming via `lora_ckpt_path` starts training with a fresh optimizer state.

# How LoRA Training Works

When LoRA is enabled:

1. **Base weights are frozen.** Both the diffusion model and conditioner are set to `eval()` mode with `requires_grad_(False)`. No base model weights are updated during training.

2. **LoRA layers are added.** `add_lora()` uses PyTorch's `nn.utils.parametrize` API to add LoRA parametrizations to `Linear` and `Conv1d` layers in both the model and conditioner. If `include` or `exclude` filters are specified in the config, only matching layers receive LoRA (see [Layer Filtering](#layer-filtering)).

3. **EMA is disabled.** EMA (Exponential Moving Average) is not compatible with LoRA training because the base weights are frozen, so it is automatically disabled.

4. **Only LoRA params are optimized.** The optimizer receives only the LoRA parameters (collected via `get_lora_params()`), not the full model parameters.

5. **Checkpoints save only LoRA.** On save, the checkpoint is cleared of all default PyTorch Lightning state and replaced with just the LoRA state dict and the `lora_config`. 

## ARC Discriminator LoRA

ARC training supports LoRA on the discriminator independently. When discriminator LoRA is enabled:

- LoRA is applied to both `Linear` and `Conv1d` layers in the discriminator backbone (same layer types as the generator)
- Base weights of LoRA layers are explicitly frozen, but norms and embeddings remain trainable
- Discriminator LoRA is used only during training and is **not saved** in checkpoints - only the generator LoRA is persisted

# Checkpoint Format

LoRA checkpoints can be saved in two formats. Both are transparently supported when loading - the inference code automatically detects the format from the file extension.

## PyTorch Format (`.ckpt`)

The default format produced by PyTorch Lightning during training. Contains two keys:

```python
{
    "state_dict": {
        # LoRA parameters from both model and conditioner
        # Keys follow the pattern: *.parametrizations.weight.0.lora_A
        # For LoRA: lora_A, lora_B
        # For DoRA: lora_A, lora_B, magnitude
        # For LoRA-XS: M_xs
    },
    "lora_config": {
        "rank": 8,
        "alpha": 8,
        "adapter_type": "lora",
        # ... other config keys
    }
}
```

## Safetensors Format (`.safetensors`)

The recommended format for distribution and sharing. The LoRA tensors are stored as standard safetensors entries, and the `lora_config` dict is JSON-serialized into the safetensors file metadata under the key `"lora_config"`.

To convert an existing checkpoint:

```python
from stable_audio_tools.models.lora import convert_lora_ckpt_to_safetensors

convert_lora_ckpt_to_safetensors("lora.ckpt")  # produces lora.safetensors
```

Or export directly from a training wrapper:

```python
training_wrapper.export_lora_safetensors("lora.safetensors")
```

## Notes

The `lora_config` is essential for correctly reconstructing the LoRA parametrization at inference time. Without it, the system falls back to heuristic rank inference from the tensor shapes, which cannot determine `alpha` or `adapter_type`, and will default to standard LoRA - breaking DoRA and LoRA-XS checkpoints.

The model and conditioner LoRA parameters are saved together in a single `state_dict`. Because the model backbone and conditioner have different internal architectures, their keys don't collide. When loading, the combined dict is passed to each submodule with `strict=False`, and each picks up only its own keys.

# Inference with LoRA

## Loading LoRA Checkpoints

Use the `--lora-ckpt-path` argument when running the Gradio interface. You can load one or multiple LoRAs:

```bash
# Single LoRA
python run_gradio.py --model-config config.json --ckpt-path model.ckpt --lora-ckpt-path lora.safetensors

# Multiple LoRAs
python run_gradio.py --model-config config.json --ckpt-path model.ckpt \
    --lora-ckpt-path style_a.safetensors style_b.safetensors
```

The loading process:
1. The base model is created and loaded normally
2. The model is moved to GPU (important for LoRA-XS GPU-accelerated SVD)
3. For each LoRA checkpoint, LoRA parametrizations are added based on the config embedded in the checkpoint
4. LoRA weights are loaded with `strict=False`

When multiple LoRAs are loaded, they are stacked using PyTorch's native `nn.utils.parametrize` API. Each call to `register_parametrization` appends to the `ParametrizationList` on each weight, and the forward chains them additively. Each LoRA is assigned a unique `lora_index` (0, 1, 2, ...) that enables independent control.

Multiple LoRAs can use different adapter types (e.g., one standard LoRA and one DoRA), different ranks, and different layer filters. They are all applied simultaneously during inference.

## Gradio UI Controls

When LoRA checkpoints are loaded, the Gradio interface shows per-LoRA controls. Each LoRA gets its own collapsible accordion with independent settings:

### LoRA DiT Strength
Controls the strength of the LoRA effect on the diffusion model backbone. Default is 1.0 (full effect). Setting to 0 disables the LoRA entirely; values above 1.0 amplify the effect. Range: 0.0 - 10.0.

### LoRA Conditioner Strength
Controls the LoRA effect on the text conditioner independently from the DiT. Default is 1.0. This allows you to, for example, keep the LoRA effect on the conditioner while reducing it on the backbone.

### LoRA Interval
Controls when LoRA is active during the sampling process based on the noise level (sigma). The interval is specified as `[min, max]` where both are in the range 0.0 to 1.0.

- `[0.0, 1.0]` (default): LoRA is active at all noise levels
- `[0.0, 0.95]`: LoRA is active only during the low-noise (late) part of sampling
- `[0.95, 1.0]`: LoRA is active only during the high-noise (early) part of sampling

This can be useful for controlling which aspects of generation the LoRA influences - early steps affect global structure while later steps affect fine details.

### LoRA Layer Filter
A text field that selectively disables LoRA on specific layers by name. Comma-separated substrings are matched against layer names (logical OR). Bracket notation supports ranges.

Examples:
- `.to_global_embed` - disables LoRA on the global embedding projection
- `.transformer.layers[0-5]` - disables LoRA on transformer layers 0 through 5
- `.transformer.layers[0-10], .to_global_embed` - disables both

Layers matching any filter substring are disabled; all other LoRA layers remain active.

### Multi-LoRA Example

With two LoRAs loaded, you could configure them independently:

- **LoRA 1 (style)**: DiT strength 1.0, interval `[0.0, 1.0]` (active everywhere)
- **LoRA 2 (detail)**: DiT strength 0.5, interval `[0.0, 0.5]` (active only in later denoising steps)

Each LoRA's interval and layer filter are evaluated independently at each sampling step. A LoRA is enabled for a step only if the current sigma falls within its interval.

# Advanced Features

## Strength Control

The `set_lora_strength()` function adjusts the LoRA contribution at runtime without modifying the LoRA weights:

```python
from stable_audio_tools.models.lora import set_lora_strength

set_lora_strength(model, 0.5)   # Half-strength on all LoRAs
set_lora_strength(model, 0.0)   # Effectively disable all LoRAs
set_lora_strength(model, 2.0)   # Double-strength on all LoRAs

# With multiple LoRAs, target a specific one by index:
set_lora_strength(model, 1.0, lora_index=0)  # Full strength on first LoRA
set_lora_strength(model, 0.5, lora_index=1)  # Half strength on second LoRA
```

This works by scaling the `lora_strength` buffer in each LoRA layer, which multiplies the low-rank delta before adding it to the base weight. When `lora_index` is `None` (the default), all LoRAs are affected.

## Multiple LoRA Merging

You can merge multiple LoRA checkpoints with different weights into a single base model:

```python
from stable_audio_tools.models.lora.utils import merge_loras_into_base_model

lora_configurations = [
    {
        'name': 'style_a',
        'state_dict': lora_sd_a,
        'application_weight': 0.7
    },
    {
        'name': 'style_b',
        'state_dict': lora_sd_b,
        'application_weight': 0.3
    }
]

merge_loras_into_base_model(model, lora_configurations)
```

This computes the weighted sum of LoRA deltas and applies them directly to the base weights. After merging, the LoRA parametrizations are disabled (the effect is baked into the base weights). This works with all adapter types (LoRA, DoRA, LoRA-XS).

## Weight Tying

For models where the input embedding and output projection share weights, LoRA supports weight tying:

```python
from stable_audio_tools.models.lora.utils import tie_weights, untie_weights

tie_weights(linear_layer, embedding_layer)   # Share LoRA params
untie_weights(linear_layer, embedding_layer) # Create independent copies
```

This is only supported for standard LoRA (not DoRA or LoRA-XS).

# Implementation Reference

The LoRA implementation lives in `stable_audio_tools/models/lora/`:

- `model.py` - `LoRAParametrization` class with all three adapter types, `add_lora()`, `merge_lora()`, `remove_lora()`, `set_lora_strength()`. Each parametrization carries a `lora_index` for multi-LoRA targeting.
- `utils.py` - Parameter collection (`get_lora_params`, `get_lora_state_dict`), layer filtering, multi-LoRA support (`get_lora_count`, `remap_lora_state_dict`), rank inference, merging utilities
- `__init__.py` - Public API exports

Training integration:
- `training/diffusion.py` - `DiffusionCondTrainingWrapper` with LoRA support
- `training/arc.py` - `ARCTrainingWrapper` with independent generator/discriminator LoRA
- `training/factory.py` - Config parsing and `lora_ckpt_path` loading

Inference integration:
- `interface/gradio.py` - Multi-LoRA checkpoint loading and stacking
- `interface/interfaces/diffusion_cond.py` - Per-LoRA Gradio UI controls for strength, interval, and filtering
- `models/dit.py` - `lora_configs` (multi-LoRA) and legacy `lora_interval`/`lora_layer_filter` logic in the DiT forward pass
