import torch
import os

def get_rank():
    """Get rank of current process."""
    
    print(os.environ.keys())

    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    
    return torch.distributed.get_rank()

def copy_state_dict(model, state_dict):
    """Load state_dict to model, but only for keys that match exactly.

    Args:
        model (nn.Module): model to load state_dict.
        state_dict (OrderedDict): state_dict to load.
    """
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
            if isinstance(state_dict[key], torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                state_dict[key] = state_dict[key].data
            model_state_dict[key] = state_dict[key]
        
    model.load_state_dict(model_state_dict, strict=False)

def create_optimizer_from_config(optimizer_config, parameters):
    """Create optimizer from config.

    Args:
        parameters (iterable): parameters to optimize.
        optimizer_config (dict): optimizer config.

    Returns:
        torch.optim.Optimizer: optimizer.
    """
    optimizer_fn = getattr(torch.optim, optimizer_config["type"])
    optimizer = optimizer_fn(parameters, **optimizer_config["config"])
    return optimizer

def create_scheduler_from_config(scheduler_config, optimizer):
    """Create scheduler from config.

    Args:
        scheduler_config (dict): scheduler config.
        optimizer (torch.optim.Optimizer): optimizer.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: scheduler.
    """
    scheduler_fn = getattr(torch.optim.lr_scheduler, scheduler_config["type"])
    scheduler = scheduler_fn(optimizer, **scheduler_config["config"])
    return scheduler