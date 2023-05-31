from prefigure.prefigure import get_all_args, push_wandb_config
import json
import torch
from torch.utils import data
import pytorch_lightning as pl

from harmonai.data.dataset import SampleDataset, S3WebDataLoader, S3DatasetConfig
from harmonai.models.diffusion import DiffusionAttnUnet1D
from harmonai.training.diffusion import UncondDiffusionTrainingWrapper, UncondDemoCallback

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

    train_set = SampleDataset(
        [args.training_dir],
        sample_rate=args.sample_rate,
        sample_size=model_config["sample_size"],
        random_crop=True
    )

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)

    # dataset_configs = [
    #     S3DatasetConfig(
    #         id="my_audio",
    #         s3_path="s3://path/to/audio",
    #     )
    # ]

    # train_dl = S3WebDataLoader(
    #     dataset_configs,
    #     sample_rate=args.sample_rate,
    #     batch_size=args.batch_size,
    #     sample_size=model_config["sample_size"],
    #     random_crop=True,
    #     num_workers=args.num_workers,
    #     persistent_workers=True,
    # ).data_loader

    diffusion_model = DiffusionAttnUnet1D(**model_config['diffusion'])
    training_wrapper = UncondDiffusionTrainingWrapper(diffusion_model)

    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = UncondDemoCallback(
        demo_every=args.demo_every, 
        num_demos=args.num_demos, 
        sample_size=model_config["sample_size"], 
        demo_steps=args.demo_steps, 
        sample_rate=args.sample_rate)

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(diffusion_model)
    push_wandb_config(wandb_logger, args)

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