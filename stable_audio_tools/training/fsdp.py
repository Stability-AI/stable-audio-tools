import datetime
import functools
import os
import re

import pytorch_lightning
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies.fsdp import FSDPStrategy
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy, enable_wrap, wrap
from typing import List, Set, Callable


# =============================================================================
# Utilities for selective FSDP wrapping and gradient synchronization
# =============================================================================

def get_fsdp_wrapped_params(module: torch.nn.Module) -> Set[torch.nn.Parameter]:
    """
    Get parameters that are inside FSDP-wrapped submodules.

    Args:
        module: The root module to search

    Returns:
        Set of parameters managed by FSDP instances
    """
    wrapped_params = set()
    for child in module.modules():
        if isinstance(child, FullyShardedDataParallel):
            wrapped_params.update(child.parameters())
    return wrapped_params


def get_non_fsdp_trainable_params(module: torch.nn.Module) -> List[torch.nn.Parameter]:
    """
    Get trainable parameters that are NOT managed by FSDP.

    These parameters need explicit gradient synchronization via sync_non_fsdp_gradients().

    Args:
        module: The root module to search

    Returns:
        List of trainable parameters not in any FSDP wrapper
    """
    wrapped = get_fsdp_wrapped_params(module)
    return [p for p in module.parameters() if p.requires_grad and p not in wrapped]


def sync_non_fsdp_gradients(params: List[torch.nn.Parameter]):
    """
    All-reduce gradients for non-FSDP trainable parameters.

    Call this after backward() but before optimizer.step() to ensure
    gradient synchronization for parameters not managed by FSDP.

    Args:
        params: List of parameters whose gradients need synchronization
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return

    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


def sync_non_fsdp_params(params: List[torch.nn.Parameter]):
    """
    Broadcast non-FSDP trainable parameters from rank 0 to all other ranks.

    Call this after optimizer.step() to ensure parameter synchronization
    for parameters not managed by FSDP. This is critical because non-FSDP
    parameters don't get synchronized by FSDP's all-gather mechanism.

    Args:
        params: List of parameters that need synchronization
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return

    for param in params:
        # Broadcast from rank 0 to ensure all ranks have identical parameters
        dist.broadcast(param.data, src=0)


def update_ema_fsdp(
    ema,
    online_model: torch.nn.Module,
) -> None:
    """
    FSDP-aware EMA update using summon_full_params.

    The ema_pytorch library captures parameter names at initialization. When the
    online model is wrapped with FSDP, parameter names change, causing EMA's
    get_params_iter() to skip all wrapped parameters. This function bypasses
    that issue by directly iterating parameters (matched by order, not name).

    Requirements:
    - EMA model must NOT be FSDP-wrapped (keep it unsharded)
    - Online model may be FSDP-wrapped

    Args:
        ema: The ema_pytorch.EMA instance
        online_model: The FSDP-wrapped online model to copy parameters from
    """
    step = ema.step.item()
    ema.step += 1

    if (step % ema.update_every) != 0:
        return

    current_decay = ema.get_current_decay()

    # summon_full_params temporarily materializes the full sharded parameters
    # rank0_only=False ensures all ranks can access (needed since EMA is replicated)
    with FullyShardedDataParallel.summon_full_params(online_model, writeback=False, rank0_only=False):
        with torch.no_grad():
            if step <= ema.update_after_step or not ema.initted.item():
                # Copy for initialization
                for (_, ema_param), (_, online_param) in zip(
                    ema.ema_model.named_parameters(),
                    online_model.named_parameters()
                ):
                    ema_param.data.copy_(online_param.data)
                if step > ema.update_after_step:
                    ema.initted.data.fill_(True)
            else:
                # EMA lerp update
                for (_, ema_param), (_, online_param) in zip(
                    ema.ema_model.named_parameters(),
                    online_model.named_parameters()
                ):
                    ema_param.data.lerp_(online_param.data, 1. - current_decay)


def recursive_wrap(
    module: torch.nn.Module,
    wrap_policy: Callable[[torch.nn.Module, str], bool],
    parent_name: str = ""
):
    """
    Recursively wrap modules matching the given policy.

    Uses depth-first traversal to wrap child modules before their parents.
    Must be called within an enable_wrap() context from PreWrapCallback.

    Args:
        module: The module to process
        wrap_policy: Function(child_module, full_name) -> bool that returns True
                     for modules that should be wrapped with FSDP
        parent_name: Current path in the module tree (for nested calls)

    Example:
        def my_policy(module, name):
            from stable_audio_tools.models.transformer import TransformerBlock
            return isinstance(module, TransformerBlock)

        recursive_wrap(self.diffusion.model.model, my_policy)
    """
    for name, child in list(module.named_children()):
        full_name = f"{parent_name}.{name}" if parent_name else name

        # Recurse into children first (depth-first)
        recursive_wrap(child, wrap_policy, full_name)

        # Check if this child should be wrapped
        if wrap_policy(child, full_name):
            setattr(module, name, wrap(child))


# =============================================================================
# Strategy and callback setup
# =============================================================================

def process_common_options(pl_module, config):
 
    supported_wrap_policies = [
        "auto_wrap_policy",
        "module_wrap_policy",
        "size_wrap_policy",
        "lambda_wrap_policy",
    ]

    if "module_wrap_policy" in config:
        # module wrap policy with list of classes specified as strings
        for i in range(len(config["module_wrap_policy"])):
            config["module_wrap_policy"][i] = get_obj_from_str(
                config["module_wrap_policy"][i]
            )
        config["auto_wrap_policy"] = ModuleWrapPolicy(config.pop("module_wrap_policy"))

    if "size_wrap_policy" in config:
        config["auto_wrap_policy"] = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=int(config.pop("size_wrap_policy")),
        )

    if "sharding_strategy" in config:
        config["sharding_strategy"] = getattr(
            ShardingStrategy, config["sharding_strategy"]
        )

    ignored_states = []
    ignored_modules_set = set()
    if "ignored_modules" in config:
        # TODO subject to deprecation see https://pytorch.org/docs/main/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel
        ignored_modules_set = set(config["ignored_modules"])
        # replace attribute names of the module with the actual attributes
        for i in range(len(config["ignored_modules"])):
            ignored_states.append(getattr(pl_module, config["ignored_modules"][i]))

        del config["ignored_modules"]

    ignored_parameters_set = set()
    if "ignored_parameters" in config:
        # match parameters via regexes
        for name, param in pl_module.named_parameters():
            for pattern in config["ignored_parameters"]:
                if re.match(pattern, name):
                    ignored_states.append(param)
                    ignored_parameters_set.add(pattern)
                    break
        del config["ignored_parameters"]

    config["ignored_states"] = ignored_states

    # HYBRID SHARD options
    if config["sharding_strategy"] == "HYBRID_SHARD":
        # NCCL_CROSS_NIC=1 is mentioned as a potential speedup for hybrid sharding
        # in the fsdp docs https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel
        os.environ["NCCL_CROSS_NIC"] = "1"
        # NCCL_P2P_LEVEl=NVL limits p2p communication to those gpus
        # connected via nvlink, see https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-level
        os.environ["NCCL_P2P_LEVEL"] = "NVL"

    if "timeout" in config:
        config["timeout"] = datetime.timedelta(seconds=config["timeout"])

    return ignored_modules_set, ignored_parameters_set

class PreWrapCallback(Callback):

    # Keys that are FSDPStrategy-specific and should not be passed to FullyShardedDataParallel
    STRATEGY_ONLY_KEYS = {"state_dict_type", "auto_wrap_policy"}

    def __init__(self, fsdp_config):
        self.fsdp_config = fsdp_config

    def setup(self, trainer, pl_module, stage=None):

        device = pl_module.device

        # Move the module to the device. PyTorch Lightning sets the device property for the `setup` hook, but doesn't move the module until later.
        # With the FSDP strategy, we need to move the module before wrapping it, or non-FSDP parameters will end up on the CPU.
        pl_module.to(device=device)

        # Check if pl_module has a method 'wrap_fsdp'
        if hasattr(pl_module, "wrap_fsdp"):
            # Filter out strategy-only keys that shouldn't be passed to FullyShardedDataParallel
            fsdp_wrap_config = {
                k: v for k, v in self.fsdp_config.items()
                if k not in self.STRATEGY_ONLY_KEYS
            }

            # Enable wrapping context, so all wrap() calls in pl_module.wrap_fsdp() have the correct config
            with enable_wrap(
                wrapper_cls=FullyShardedDataParallel,
                device_id=pl_module.device,
                **fsdp_wrap_config
            ):
                pl_module = pl_module.wrap_fsdp()

def create_fsdp_strategy_and_callback(
    pl_module: pytorch_lightning.LightningModule,
    precision: str = None,
    **config,
):
    # Set a use_fsdp attribute on the pl_module to indicate FSDP is being used
    # Could be needed for things like EMA to check for FSDP-specific behavior
    pl_module.use_fsdp = True

    # If precision is specified and mixed_precision is not already configured,
    # set up mixed precision with fp32 gradient reduction for numerical stability.
    # bf16/fp16 gradient reduction loses precision during allreduce operations.
    # See: https://main-horse.github.io/posts/reduction-precision/
    if precision is not None and "mixed_precision" not in config:
        if precision == "bf16-mixed":
            # To match DDP behavior with autocast:
            # - param_dtype=None keeps weights in fp32 (like DDP)
            # - reduce_dtype=fp32 keeps gradient reduction in fp32 (like DDP)
            # - buffer_dtype=bf16 for buffers
            # Autocast handles bf16 computation during forward/backward
            config["mixed_precision"] = MixedPrecision(
                param_dtype=None,  # Keep weights in fp32, use autocast for bf16 compute
                reduce_dtype=torch.float32,  # Keep gradient reduction in fp32
                buffer_dtype=torch.bfloat16,
            )
        elif precision == "16-mixed":
            config["mixed_precision"] = MixedPrecision(
                param_dtype=None,  # Keep weights in fp32, use autocast for fp16 compute
                reduce_dtype=torch.float32,  # Keep gradient reduction in fp32
                buffer_dtype=torch.float16,
            )

    process_common_options(pl_module, config)

    # Use full state dict type for checkpoint saving with selective wrapping.
    # With selective FSDP wrapping (only some submodules wrapped), we need to ensure
    # consistent checkpoint behavior. "full" gathers all FSDP module state to rank 0.
    # Non-FSDP modules already have full state on each rank.
    if "state_dict_type" not in config:
        config["state_dict_type"] = "full"

    strategy = FSDPStrategy(**config)

    # We need to manually wrap sub-modules of the PL module to avoid FSDPStrategy
    # trying to wrap the entire training wrapper itself.
    # This avoids issues with the FSDP context flattening parameters, breaking manual optimization, and causing demos to fail.
    pre_wrap_callback = PreWrapCallback(config)

    return strategy, pre_wrap_callback
