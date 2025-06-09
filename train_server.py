# file: train_server.py
#!/usr/bin/env python3
import os
import json
import argparse

import torch
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.training.factory import create_training_wrapper_from_config

def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    audio, sr = torchaudio.load(path)
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    # [C, T] -> [1, C, T]
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    return audio

def save_audio(audio: torch.Tensor, path: str, sample_rate: int):
    # [1, C, T] -> [C, T]
    audio = audio.squeeze(0)
    torchaudio.save(path, audio, sample_rate)

def recon(autoencoder, input_path: str, output_path: str, device: torch.device, sample_rate: int):
    print(f"[Recon] {input_path} -> {output_path}")
    
    # Skip reconstruction if output_path is None, empty, or /dev/null
    if not output_path or output_path == "/dev/null":
        print(f"[Recon] Skipping reconstruction (output_path: {output_path})")
        return
    
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    x = load_audio(input_path, sample_rate).to(device)
    with torch.no_grad():
        latents, _ = autoencoder.encode(x, return_info=True)
        y = autoencoder.decode(latents)
    y = y.clamp(-1, 1).cpu()
    save_audio(y, output_path, sample_rate)

def main():
    parser = argparse.ArgumentParser(
        description="First recon, then train 50 epochs, then recon again"
    )
    parser.add_argument("--model-config",     required=True,
                        help="Model configuration JSON with dithered_fsq + lm_config")
    parser.add_argument("--data-dir",         required=True,
                        help="Training data directory (audio_dir format)")
    parser.add_argument("--input-audio",      required=True,
                        help="Input audio file path for reconstruction")
    # parser.add_argument("--pre-output",       required=True,
    #                     help="Pre-training reconstruction output path")
    parser.add_argument("--post-output",      required=True,
                        help="Post-training reconstruction output path after 50 epochs")
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--num-workers",  type=int, default=6)
    parser.add_argument("--max-epochs",   type=int, default=50)
    parser.add_argument("--precision",    type=str, default="16-mixed")
    parser.add_argument("--accelerator",  type=str, default="auto")
    parser.add_argument("--devices",      type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="stable_audio_tools", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    args = parser.parse_args()

    # 1. Load config
    with open(args.model_config) as f:
        model_cfg = json.load(f)

    # 1.5 Setup W&B logger
    wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_run_name)

    # 2. Construct dataset_config for DataLoader
    data_cfg = {
        "dataset_type": "audio_dir",
        "datasets": [{"id": "train", "path": args.data_dir}],
        "random_crop": True,
        "drop_last": True
    }
    sample_rate = model_cfg["sample_rate"]

    # 3. DataLoader
    train_dl = create_dataloader_from_config(
        data_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=sample_rate,
        sample_size=model_cfg["sample_size"],
        audio_channels=model_cfg.get("audio_channels", 2),
    )

    # 4. Model + Wrapper
    model = create_model_from_config(model_cfg)                      # AudioAutoencoder
    wrapper = create_training_wrapper_from_config(model_cfg, model)  # LightningModule

    device = torch.device("cuda" if torch.cuda.is_available() and args.devices>0 else "cpu")

    # 5. Pre-training reconstruction
    recon(wrapper.autoencoder, args.input_audio, args.post_output, device, sample_rate)

    # 6. Training
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        log_every_n_steps=50,
        logger=wandb_logger,
    )
    trainer.fit(wrapper, train_dl)

    # 7. Post-training reconstruction
    recon(wrapper.autoencoder, args.input_audio, args.post_output, device, sample_rate)

if __name__ == "__main__":
    main()