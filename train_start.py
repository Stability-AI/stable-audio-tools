# file: train_server.py
#!/usr/bin/env python3
import os
import json
import argparse
import math

import torch
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

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

def laplace_cdf(x, expectation, scale):
    shifted_x = x - expectation
    return 0.5 - 0.5 * (shifted_x).sign() * torch.expm1(-(shifted_x).abs() / scale)

class EpochReconCallback(pl.Callback):
    def __init__(self, input_audio_path: str, output_dir: str, sample_rate: int, device: torch.device):
        self.input_audio_path = input_audio_path
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.device = device
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  # 1-indexed
        output_path = os.path.join(self.output_dir, f"post_output_epoch_{epoch}.mp3")
        recon(pl_module.autoencoder, self.input_audio_path, output_path, self.device, self.sample_rate)

        # Only compute bitrate if language model exists
        if hasattr(pl_module, 'lm') and pl_module.lm is not None:
            # 1) load the same audio
            wav = load_audio(self.input_audio_path, self.sample_rate).to(self.device)  # [1, C, N]

            # 2) AE encode + LM > compute rate
            with torch.no_grad():
                latents, _ = pl_module.autoencoder.encode(wav, return_info=True)
                mu, log_b = pl_module.lm(latents)
                scale = log_b.exp().clamp(min=1e-6)
                
                # z are the quantized latents (assuming they're rounded)
                z = latents.round()
                
                # Compute probability using Laplace CDF
                p = torch.clamp_min(
                    laplace_cdf(z + 0.5, mu, scale)
                    - laplace_cdf(z - 0.5, mu, scale),
                    min=2**-16, 
                )
                
                # Compute rate in bits
                rate = -torch.log2(p)
                
                # Debug information
                print(f"latents range: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
                print(f"mu range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
                print(f"scale range: [{scale.min().item():.4f}, {scale.max().item():.4f}]")
                print(f"rate sum: {rate.sum().item():.4f}")
                
            # 3) compute bitrate
            num_samples = wav.shape[-1]
            seconds = num_samples / self.sample_rate
            kbps = rate.sum() / seconds / 1024
            bits_per_sample = rate.sum() / num_samples

            # 4) print and upload
            print(f"[Epoch {epoch}] bitrate = {kbps:.1f} kbit/s  ({bits_per_sample:.4f} bits/sample)")
            # use Lightning's log interface, WandBLogger will handle it
            pl_module.log("train/bitrate_kbps", kbps, on_step=False, on_epoch=True)
            pl_module.log("train/bits_per_sample", bits_per_sample, on_step=False, on_epoch=True)
        else:
            print(f"[Epoch {epoch}] No language model found, skipping bitrate calculation")

def main():
    parser = argparse.ArgumentParser(
        description="Train with checkpoints and reconstruction after each epoch"
    )
    parser.add_argument("--model-config",     required=True,
                        help="Model configuration JSON with dithered_fsq + lm_config")
    parser.add_argument("--data-dir",         required=True,
                        help="Training data directory (audio_dir format)")
    parser.add_argument("--input-audio",      required=True,
                        help="Input audio file path for reconstruction")
    parser.add_argument("--output-dir",       required=True,
                        help="Directory to save reconstruction outputs and checkpoints")
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--num-workers",  type=int, default=6)
    parser.add_argument("--max-epochs",   type=int, default=50)
    parser.add_argument("--precision",    type=str, default="16-mixed")
    parser.add_argument("--accelerator",  type=str, default="auto")
    parser.add_argument("--devices",      type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default="stable_audio_tools", help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--ckpt-path", type=str, default=None, help="resume from this checkpoint")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    lm_cfg = model_cfg["model"].get("lm", None)
    lm_weight = model_cfg["model"].get("lm_weight", 1.0)
    if lm_cfg is not None:
        from stable_audio_tools.models.lm_continuous import LaplaceLanguageModel
        wrapper.lm = LaplaceLanguageModel(wrapper.autoencoder.latent_dim, lm_cfg)
        wrapper.lm_weight = lm_weight
        wrapper.lm_config = lm_cfg

    device = torch.device("cuda" if torch.cuda.is_available() and args.devices>0 else "cpu")

    # 5. Setup callbacks
    # Checkpoint callback - save only the best performing checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="best_checkpoint",
        monitor="train/loss",
        mode="min",
        save_top_k=1,
        save_last=False
    )
    
    # Reconstruction callback - recon after every epoch
    recon_callback = EpochReconCallback(
        input_audio_path=args.input_audio,
        output_dir=args.output_dir,
        sample_rate=sample_rate,
        device=device
    )

    # 6. Training
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="ddp_find_unused_parameters_true",
        precision=args.precision,
        max_epochs=args.max_epochs,
        log_every_n_steps=50,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, recon_callback]
    )
    trainer.fit(wrapper, train_dl, ckpt_path=args.ckpt_path)

    # 7. Final reconstruction (optional, since we already have one from last epoch)
    final_output = os.path.join(args.output_dir, "post_output_final.mp3")
    recon(wrapper.autoencoder, args.input_audio, final_output, device, sample_rate)

if __name__ == "__main__":
    main()