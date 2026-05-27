# Disable HuggingFace progress bars BEFORE any imports
# This must be at the very top to take effect
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
import json
import pytorch_lightning as pl

from typing import Dict, Optional, Union
from prefigure.prefigure import get_all_args, push_wandb_config
from pytorch_lightning.strategies import DDPStrategy
from stable_audio_tools.data.dataset import create_dataloader_from_config, fast_scandir
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.fsdp import create_fsdp_strategy_and_callback

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def main():
    # Removed: torch.multiprocessing.set_sharing_strategy('file_system')
    # 'file_system' creates named files in /dev/shm that accumulate over long runs,
    # causing gradual slowdown and eventual "Shared memory manager connection has
    # timed out" crashes. The default 'file_descriptor' strategy uses kernel-managed
    # FDs with automatic cleanup. Requires ulimit -n 65536 in SLURM script.
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision('high')
    args = get_all_args()
    seed = args.seed


    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    pl.seed_everything(seed, workers=True)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    if model_config["training"].get("lora_config"):
        print("Lora Config", model_config["training"]["lora_config"])

    model = create_model_from_config(model_config)

    # Extract tokenizers from conditioners for pre-tokenization in DataLoader workers
    tokenizers = {}
    if hasattr(model, 'conditioner'):
        for key, cond in model.conditioner.conditioners.items():
            if hasattr(cond, 'tokenizer') and hasattr(cond, 'max_length'):
                tokenizers[key] = (cond.tokenizer, cond.max_length)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        tokenizers=tokenizers if tokenizers else None,
    )

    val_dl = None
    val_dataset_config = None

    if args.val_dataset_config:
        with open(args.val_dataset_config) as f:
            val_dataset_config = json.load(f)

        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            shuffle=True
        )

    if args.pretrained_ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        print(f"Loading pretransform weights from {args.pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))

    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    exc_callback = ExceptionCallback()

    if args.logger == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.name)
        logger.watch(training_wrapper)
    
        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.experiment.project, logger.experiment.id, "checkpoints") 
        else:
            checkpoint_dir = None
    elif args.logger == 'comet':
        logger = pl.loggers.CometLogger(project=args.name)
        if args.save_dir and isinstance(logger.version, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.name, logger.version, "checkpoints") 
        else:
            print(f"No save_dir specified, using {args.save_dir if args.save_dir else None}.")
            checkpoint_dir = args.save_dir if args.save_dir else None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None
        
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    if args.val_dataset_config:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=val_dl)
    else:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)

    callbacks = [ckpt_callback, exc_callback, save_model_config_callback, demo_callback]

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    args_dict.update({"val_dataset_config": val_dataset_config})

    if args.logger == 'wandb':
        push_wandb_config(logger, args_dict)
    elif args.logger == 'comet':
        logger.log_hyperparams(args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2,
                                        contiguous_gradients=True,
                                        overlap_comm=True,
                                        reduce_scatter=True,
                                        reduce_bucket_size=5e8,
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True)
        elif args.strategy == "fsdp":
            strategy, pre_wrap_callback = create_fsdp_strategy_and_callback(
                training_wrapper,
                precision=args.precision,
                sharding_strategy="FULL_SHARD",
                limit_all_gathers=True,
                use_orig_params=True,
            )
            callbacks.append(pre_wrap_callback)
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"

    if strategy == 'ddp_find_unused_parameters_true' or (strategy == 'ddp' and args.num_nodes > 1):
        strategy = DDPStrategy(
            bucket_cap_mb=150,
            gradient_as_bucket_view=True,
            find_unused_parameters = True if strategy == 'ddp_find_unused_parameters_true' else False,
            static_graph = not model_config.get("training", {}).get("quantize_dropout", False)
        )

    val_args = {}
    
    if args.val_every > 0:
        val_args.update({
            "check_val_every_n_epoch": None,
            "val_check_interval": args.val_every,
        })

    if not hasattr(args, 'gradient_clip_val') or args.gradient_clip_val == 0:
        args.gradient_clip_val = None

    summary = pl.callbacks.ModelSummary(max_depth=2)
    callbacks.append(summary)
    
    if model_config["training"].get("metrics"):
        from stable_audio_tools.training import create_metrics_callback_from_config
        metrics_callback = create_metrics_callback_from_config(model_config)
        callbacks.append(metrics_callback)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0,
        num_sanity_val_steps=0, # If you need to debug validation, change this line
        **val_args      
    )

    trainer.fit(training_wrapper, train_dl, val_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()
