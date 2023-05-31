from prefigure.prefigure import get_all_args, push_wandb_config
import json
import torch
import pytorch_lightning as pl

from harmonai.data.dataset import create_dataloader_from_configs_and_args
from harmonai.models import create_model_from_config
from harmonai.training import create_training_wrapper_from_config_and_args, create_demo_callback_from_config_and_args

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_configs_and_args(model_config, args, dataset_config)

    model = create_model_from_config(model_config)
    
    training_wrapper = create_training_wrapper_from_config_and_args(model_config, args, model)

    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)

    demo_callback = create_demo_callback_from_config_and_args(model_config, args, demo_dl=train_dl)

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(training_wrapper)

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update(model_config)
    args_dict.update(dataset_config)
    push_wandb_config(wandb_logger, args_dict)

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy='ddp',
        precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir
    )

    trainer.fit(training_wrapper, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()