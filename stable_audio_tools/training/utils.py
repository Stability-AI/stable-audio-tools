from pytorch_lightning.loggers import WandbLogger, CometLogger
from ..interface.aeiou import pca_point_cloud

import wandb
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

class InverseLR(torch.optim.lr_scheduler._LRScheduler):
    """Implements an inverse decay learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    inv_gamma is the number of steps/epochs required for the learning rate to decay to
    (1 / 2)**power of its original value.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        inv_gamma (float): Inverse multiplicative factor of learning rate decay. Default: 1.
        power (float): Exponential factor of learning rate decay. Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        final_lr (float): The final learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., final_lr=0.,
                 last_epoch=-1, verbose=False):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.final_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]

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

    optimizer_type = optimizer_config["type"]

    if optimizer_type == "FusedAdam":
        from deepspeed.ops.adam import FusedAdam
        optimizer = FusedAdam(parameters, **optimizer_config["config"])
    else:
        optimizer_fn = getattr(torch.optim, optimizer_type)
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
    if scheduler_config["type"] == "InverseLR":
        scheduler_fn = InverseLR
    else:
        scheduler_fn = getattr(torch.optim.lr_scheduler, scheduler_config["type"])
    scheduler = scheduler_fn(optimizer, **scheduler_config["config"])
    return scheduler

def logger_project_name(logger) -> str:
    if isinstance(logger, WandbLogger):
        return logger.experiment.project
    elif isinstance(logger, CometLogger):
        return logger.name

def log_metric(logger, key, value, step=None):
    from pytorch_lightning.loggers import WandbLogger, CometLogger
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: value})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_metrics({key: value}, step=step)

def log_audio(logger, key, audio_path, sample_rate, caption=None):
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: wandb.Audio(audio_path, sample_rate=sample_rate, caption=caption)})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_audio(audio_path, file_name=key, sample_rate=sample_rate)

def log_image(logger, key, img_data):
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: wandb.Image(img_data)})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_image(img_data, name=key)

def log_point_cloud(logger, key, tokens, caption=None):
    if isinstance(logger, WandbLogger):
        point_cloud = pca_point_cloud(tokens)
        logger.experiment.log({key: point_cloud})
    elif isinstance(logger, CometLogger):
        point_cloud = pca_point_cloud(tokens, rgb_float=True, output_type="points")
        #logger.experiment.log_points_3d(scene_name=key, points=point_cloud)
