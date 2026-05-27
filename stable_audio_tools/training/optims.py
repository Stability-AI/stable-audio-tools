# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple, Optional

import torch
from torch import nn
from torch.optim import Optimizer


class CAdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        self.init_lr = lr

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # apply weight decay
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                
                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                mask = (exp_avg * grad > 0).to(grad.dtype)
                # mask = mask * (mask.numel() / (mask.sum() + 1)) ## original implementation, leaving it here for record
                mask.div_(mask.mean().clamp_(min=1e-3)) # https://huggingface.co/rwightman/timm-optim-caution found this implementation is more favoarable in many cases
                norm_grad = (exp_avg * mask) / denom
                p.add_(norm_grad, alpha=-step_size)
        return loss


def exists(val):
    return val is not None

# update functions

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # weight update
    update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1).sign_()
    mask = (update * grad > 0).to(grad.dtype)
    mask = mask * (mask.numel() / (mask.sum() + 1))
    p.add_(update * mask, alpha = -lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)

# class

class CLion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2
                )

        return loss


# =============================================================================
# Muon optimizer (Newton-Schulz orthogonalized momentum)
# Reference: https://github.com/KellerJordan/Muon
# =============================================================================

def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the polar factor of G.

    Produces approximately U @ V^T where G = U @ S @ V^T is the SVD.
    Uses a quintic polynomial whose coefficients maximize convergence speed.
    Runs in bfloat16 for efficiency on tensor cores.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonAdamW(Optimizer):
    """
    Combined Muon + AdamW optimizer.

    Applies Muon (Newton-Schulz orthogonalized momentum) to 2D+ weight matrices
    and AdamW to everything else (biases, norms, embeddings, scalars).

    Parameters are auto-split by ndim: >= 2 goes to Muon, < 2 goes to AdamW.
    This is a single optimizer with two param groups, so LR schedulers work
    correctly (each group maintains its own base_lr).

    For FSDP compatibility, set fsdp_modules after construction. The Muon step
    will be wrapped in summon_full_params to get full gradient matrices for
    Newton-Schulz orthogonalization.

    Args:
        params: Iterable of parameters to optimize.
        muon_lr: Learning rate for Muon (2D+ weights). Default: 0.02.
        muon_momentum: Momentum coefficient for Muon. Default: 0.95.
        muon_nesterov: Use Nesterov momentum for Muon. Default: True.
        muon_weight_decay: Weight decay for Muon params. Default: 0.0.
        ns_steps: Newton-Schulz iteration count. Default: 5.
        adam_lr: Learning rate for AdamW (1D params). Default: 3e-4.
        adam_betas: AdamW beta coefficients. Default: (0.9, 0.95).
        adam_eps: AdamW epsilon. Default: 1e-8.
        adam_weight_decay: Weight decay for AdamW params. Default: 0.01.
    """

    def __init__(
        self,
        params,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_weight_decay: float = 0.0,
        ns_steps: int = 5,
        adam_lr: float = 3e-4,
        adam_betas: Tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        adam_weight_decay: float = 0.01,
        fused_layer_patterns: list = None,
    ):
        params_list = list(params)

        # Support (name, param) tuples for fused layer pattern matching
        if params_list and isinstance(params_list[0], tuple):
            param_names = {p: name for name, p in params_list}
            params_list = [p for _, p in params_list]
        else:
            param_names = {}

        # Build set of fused params that need chunked NS (2D+ only)
        self._fused_params = set()
        if fused_layer_patterns and param_names:
            import fnmatch
            for p, name in param_names.items():
                if p.ndim >= 2 and any(fnmatch.fnmatch(name, pat) for pat in fused_layer_patterns):
                    self._fused_params.add(p)

        muon_params = [p for p in params_list if p.ndim >= 2]
        adam_params = [p for p in params_list if p.ndim < 2]

        # defaults must be a superset of all keys used in any param group
        defaults = dict(
            lr=muon_lr, momentum=muon_momentum, nesterov=muon_nesterov,
            weight_decay=muon_weight_decay, ns_steps=ns_steps, use_muon=True,
            betas=adam_betas, eps=adam_eps,
        )

        super().__init__([
            {
                'params': muon_params,
                'lr': muon_lr,
                'momentum': muon_momentum,
                'nesterov': muon_nesterov,
                'weight_decay': muon_weight_decay,
                'ns_steps': ns_steps,
                'use_muon': True,
            },
            {
                'params': adam_params,
                'lr': adam_lr,
                'betas': adam_betas,
                'eps': adam_eps,
                'weight_decay': adam_weight_decay,
                'use_muon': False,
            },
        ], defaults)

        self.fsdp_modules = []

        n_muon = sum(p.numel() for p in muon_params)
        n_adam = sum(p.numel() for p in adam_params)
        print(f"MuonAdamW: {len(muon_params)} Muon params ({n_muon/1e6:.1f}M), "
              f"{len(adam_params)} AdamW params ({n_adam/1e6:.1f}M)")

        if self._fused_params:
            n_fused = sum(p.numel() for p in self._fused_params)
            print(f"MuonAdamW: {len(self._fused_params)} fused params ({n_fused/1e6:.1f}M) will use chunked NS")

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group['use_muon']:
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group):
        if not self.fsdp_modules:
            for p in group['params']:
                if p.grad is not None:
                    self._muon_update_single(p, group)
            return

        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

        # Build param → FSDP module mapping (cached after first call)
        if not hasattr(self, '_fsdp_param_map'):
            self._fsdp_param_map = {}
            all_fsdp_params = set()
            for m in self.fsdp_modules:
                for p in m.parameters():
                    self._fsdp_param_map[p] = m
                    all_fsdp_params.add(p)
            self._non_fsdp_muon_params = [
                p for p in group['params'] if p not in all_fsdp_params
            ]

        # Process non-FSDP params directly (always full-sized)
        for p in self._non_fsdp_muon_params:
            if p.grad is not None:
                self._muon_update_single(p, group)

        # Group params by their FSDP module
        module_to_params = {}
        for p in group['params']:
            if p in self._fsdp_param_map:
                m = self._fsdp_param_map[p]
                module_to_params.setdefault(m, []).append(p)

        # Process one FSDP module at a time to limit peak VRAM
        for module, params in module_to_params.items():
            with FSDP.summon_full_params(module, writeback=True, with_grads=True):
                for p in params:
                    if p.grad is not None:
                        self._muon_update_single(p, group)

    def _muon_update_single(self, p, group):
        lr = group['lr']
        beta = group['momentum']
        wd = group['weight_decay']
        ns_steps = group['ns_steps']
        nesterov = group['nesterov']

        grad = p.grad
        state = self.state[p]

        if len(state) == 0:
            state['momentum_buffer'] = torch.zeros_like(grad)

        buf = state['momentum_buffer']

        # Handle shape mismatch (e.g. first step after FSDP summon_full_params)
        if buf.shape != grad.shape:
            state['momentum_buffer'] = torch.zeros_like(grad)
            buf = state['momentum_buffer']

        # Momentum: buf = beta * buf + (1 - beta) * grad
        buf.lerp_(grad, 1 - beta)

        # Nesterov blend
        if nesterov:
            update = grad.lerp_(buf, beta)
        else:
            update = buf.clone()

        # Flatten conv weights to 2D for NS
        orig_shape = update.shape
        if update.ndim == 4:
            update = update.view(update.size(0), -1)
        elif update.ndim < 2:
            # 1D param that ended up in Muon group — skip NS
            if wd > 0:
                p.mul_(1 - lr * wd)
            p.add_(update, alpha=-lr)
            return

        # Newton-Schulz orthogonalization
        if p in self._fused_params and update.size(0) > update.size(1):
            # Fused layer: chunk into square sub-blocks, orthogonalize each independently
            cols = update.size(1)
            chunks = update.split(cols, dim=0)
            ns_chunks = [zeropower_via_newtonschulz5(c, steps=ns_steps) for c in chunks]
            update = torch.cat(ns_chunks, dim=0)
        else:
            update = zeropower_via_newtonschulz5(update, steps=ns_steps)
            # Scale by aspect ratio (for non-fused tall/wide matrices)
            update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

        # Reshape back
        update = update.reshape(orig_shape)

        # Decoupled weight decay
        if wd > 0:
            p.mul_(1 - lr * wd)

        # Apply update
        p.add_(update.to(p.dtype), alpha=-lr)

    def _adamw_step(self, group):
        lr = group['lr']
        beta1, beta2 = group['betas']
        eps = group['eps']
        wd = group['weight_decay']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(grad)
                state['exp_avg_sq'] = torch.zeros_like(grad)

            state['step'] += 1
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            # Bias correction
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            # Decoupled weight decay
            if wd > 0:
                p.mul_(1 - lr * wd)

            # Adam update
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

            p.addcdiv_(exp_avg, denom, value=-step_size)