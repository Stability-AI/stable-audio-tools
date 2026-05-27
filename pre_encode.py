import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict


def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, model_half=False):
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))

    model.eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)

    print("Done loading model")

    return model, model_config


class PreEncodedLatentsInferenceWrapper(pl.LightningModule):
    def __init__(
        self,
        model,
        output_path,
        is_discrete=False,
        model_half=False,
        model_config=None,
        dataset_config=None,
        sample_size=1920000,
        no_pad=False,
        args_dict=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.output_path = Path(output_path)

    def prepare_data(self):
        # runs on rank 0
        self.output_path.mkdir(parents=True, exist_ok=True)
        details_path = self.output_path / "details.json"
        if not details_path.exists():  # Only save if it doesn't exist
            details = {
                "model_config": self.hparams.model_config,
                "dataset_config": self.hparams.dataset_config,
                "sample_size": self.hparams.sample_size,
                "variable_length": self.hparams.args_dict.get("no_pad", False),
                "args": self.hparams.args_dict
            }
            details_path.write_text(json.dumps(details))

    def setup(self, stage=None):
        # runs on each device
        process_dir = self.output_path / str(self.global_rank)
        process_dir.mkdir(parents=True, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        audio, metadata = batch

        if audio.ndim == 4 and audio.shape[0] == 1:
            audio = audio[0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if self.global_rank == 0:
            try:
                silence_path = self.output_path / "silence.npy"
                if not silence_path.exists():
                    print("Saving silence latent")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    silence_audio = torch.zeros(1, self.model.io_channels, self.hparams.sample_size).to(audio.device)
                    if self.hparams.model_half:
                        silence_audio = silence_audio.to(torch.float16)

                    with torch.no_grad():
                        if not self.hparams.is_discrete:
                            silence_latent = self.model.encode(silence_audio)
                        else:
                            _, info = self.model.encode(silence_audio, return_info=True)
                            silence_latent = info[self.model.bottleneck.tokens_id]

                    silence_latent = silence_latent.cpu().numpy()

                    with open(silence_path, "wb") as f:
                        np.save(f, silence_latent)
            except Exception as e:
                print(f"Error saving silence latent: {e}")
                raise e


        # Ensure audio length is a multiple of the model's minimum length
        min_len = self.model.min_length
        audio_len = audio.shape[-1]
        if audio_len % min_len != 0:
            pad_to = ((audio_len // min_len) + 1) * min_len
            audio = F.pad(audio, (0, pad_to - audio_len))

        if self.hparams.model_half:
            audio = audio.to(torch.float16)

        with torch.no_grad():
            if not self.hparams.is_discrete:
                latents = self.model.encode(audio)
            else:
                _, info = self.model.encode(audio, return_info=True)
                latents = info[self.model.bottleneck.tokens_id]

        latents = latents.cpu().numpy()
        
        # Save each sample in the batch
        for i, latent in enumerate(latents):
            latent_id = f"{self.global_rank:03d}{batch_idx:06d}{i:04d}"

            md = metadata[i]
            padding_mask = F.interpolate(
                md["padding_mask"][0].unsqueeze(0).unsqueeze(1).float(),
                size=latent.shape[1],
                mode="nearest"
            ).squeeze(0).squeeze(0).int()

            if self.hparams.no_pad:
                # Trim latent to valid length (remove trailing silence padding)
                padding_np = padding_mask.cpu().numpy()
                valid_indices = np.where(padding_np == 1)[0]
                if len(valid_indices) > 0:
                    valid_length = valid_indices[-1] + 1
                    latent = latent[:, :valid_length]
                    padding_mask = padding_mask[:valid_length]

            # Save latent as numpy file (after trimming if applicable)
            latent_path = self.output_path / str(self.global_rank) / f"{latent_id}.npy"
            with open(latent_path, "wb") as f:
                np.save(f, latent)

            md["padding_mask"] = padding_mask.cpu().numpy().tolist()

            # Convert tensors in md to serializable types
            for k, v in md.items():
                if isinstance(v, torch.Tensor):
                    md[k] = v.cpu().numpy().tolist()

            # Save metadata to json file
            metadata_path = self.output_path / str(self.global_rank) / f"{latent_id}.json"
            with open(metadata_path, "w") as f:
                json.dump(md, f)

    def configure_optimizers(self):
        return None

    def on_validation_batch_end(self, *args, **kwargs):
        torch.distributed.barrier()


def main(args):
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    model, model_config = load_model(
        model_config=model_config,
        model_ckpt_path=args.ckpt_path,
        model_half=args.model_half
    )

    data_loader = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=args.sample_size,
        audio_channels=model_config.get("audio_channels", 2),
        shuffle=args.shuffle,
        pad=not args.no_pad,
    )

    pl_module = PreEncodedLatentsInferenceWrapper(
        model=model,
        output_path=args.output_path,
        is_discrete=args.is_discrete,
        model_half=args.model_half,
        model_config=args.model_config,
        dataset_config=args.dataset_config,
        sample_size=args.sample_size,
        no_pad=args.no_pad,
        args_dict=vars(args)
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        num_nodes = args.num_nodes,
        strategy=args.strategy,
        precision="16-true" if args.model_half else "32",
        max_steps=args.limit_batches if args.limit_batches else -1,
        logger=False,  # Disable logging since we're just doing inference
        enable_checkpointing=False,
    )
    trainer.validate(pl_module, data_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode audio dataset to VAE latents using PyTorch Lightning')
    parser.add_argument('--model-config', type=str, help='Path to model config', required=False)
    parser.add_argument('--ckpt-path', type=str, help='Path to unwrapped autoencoder model checkpoint', required=False)
    parser.add_argument('--model-half', action='store_true', help='Whether to use half precision')
    parser.add_argument('--dataset-config', type=str, help='Path to dataset config file', required=True)
    parser.add_argument('--output-path', type=str, help='Path to output folder', required=True)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=1)
    parser.add_argument('--sample-size', type=int, help='Number of audio samples to pad/crop to', default=1320960)
    parser.add_argument('--is-discrete', action='store_true', help='Whether the model is discrete')
    parser.add_argument('--num-nodes', type=int, help='Number of GPU nodes', default=1)
    parser.add_argument('--num-workers', type=int, help='Number of dataloader workers', default=4)
    parser.add_argument('--strategy', type=str, help='PyTorch Lightning strategy', default='auto')
    parser.add_argument('--limit-batches', type=int, help='Limit number of batches (optional)', default=None)
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset')
    parser.add_argument('--no-pad', action='store_true', help='Do not pad short audio to sample_size before encoding (variable-length mode)')
    args = parser.parse_args()

    if args.no_pad and args.batch_size > 1:
        parser.error("--no-pad requires --batch-size 1 (variable-length samples cannot be batched)")

    main(args)