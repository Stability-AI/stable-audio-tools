"""Conditioner distance metric callback.

Compares conditioning features re-extracted from generated demo audio
against the original conditioning features used for generation. Measures
how well the model's output adheres to its audio conditioning inputs.

Relies on DiffusionCondDemoCallback storing intermediate results on the
training module (module._demo_fakes_raw, module._demo_conditioning) so
this callback can run without duplicating the expensive generation step.
"""

import torch
import typing as tp
import pytorch_lightning as pl

from ..utils import log_metric


class ConditionerDistanceCallback(pl.Callback):
    """Log MSE and cosine similarity between original and re-extracted conditioning features.

    This callback runs after the demo callback on the same batch end, reading
    the demo's cached fakes and conditioning from the module. It only fires on
    rank 0 and only at the demo frequency.

    Args:
        demo_every: Must match the demo callback's demo_every.
        conditioner_types: Tuple of conditioner class names to measure.
            Defaults to audio conditioners (MIR, Melody, BeatThis).
    """

    def __init__(
        self,
        demo_every: int = 2000,
        conditioner_types: tp.Optional[tp.Tuple[str, ...]] = None,
    ):
        super().__init__()
        self.demo_every = demo_every
        self.last_step = -1
        self.conditioner_types = conditioner_types
        self._audio_cond_classes = None

    def _get_audio_cond_classes(self):
        """Lazy-import conditioner classes to avoid circular imports."""
        if self._audio_cond_classes is None:
            from stable_audio_tools.models.audio_conditioners import (
                MIRConditioner, MelodyConditioner, BeatThisConditioner,
                ContentConditioner,
            )
            if self.conditioner_types is not None:
                # Filter to requested types
                class_map = {
                    'MIRConditioner': MIRConditioner,
                    'MelodyConditioner': MelodyConditioner,
                    'BeatThisConditioner': BeatThisConditioner,
                    'ContentConditioner': ContentConditioner,
                }
                self._audio_cond_classes = tuple(
                    class_map[name] for name in self.conditioner_types
                    if name in class_map
                )
            else:
                self._audio_cond_classes = (
                    MIRConditioner, MelodyConditioner, BeatThisConditioner,
                    ContentConditioner,
                )
        return self._audio_cond_classes

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_step == trainer.global_step:
            return
        # Use PL's strategy-aware rank check (works with DDP, FSDP, DeepSpeed).
        # Assumes conditioners are NOT FSDP-wrapped (replicated across ranks).
        if not trainer.is_global_zero:
            return

        # Check that the demo callback has cached its outputs
        fakes_raw = getattr(module, '_demo_fakes_raw', None)
        conditioning = getattr(module, '_demo_conditioning', None)
        if fakes_raw is None or conditioning is None:
            return

        self.last_step = trainer.global_step

        audio_cond_classes = self._get_audio_cond_classes()

        # Conditioner lives on module.diffusion (ConditionedDiffusionModelWrapper),
        # NOT on the EMA model (which wraps only the DiT core).
        if not hasattr(module.diffusion, 'conditioner'):
            return

        for cond_key, conditioner in module.diffusion.conditioner.conditioners.items():
            if not isinstance(conditioner, audio_cond_classes):
                continue

            # Get original conditioning features from the forward pass
            if cond_key not in conditioning:
                continue
            orig_features = conditioning[cond_key][0]  # [B, C, T]

            # Re-extract features from generated audio
            batch_size = fakes_raw.shape[0]
            gen_audio_list = [fakes_raw[i] for i in range(batch_size)]
            was_training = conditioner.training
            conditioner.eval()
            gen_features, _ = conditioner(gen_audio_list, module.device)
            conditioner.train(was_training)

            # Align time dimension
            min_t = min(orig_features.shape[-1], gen_features.shape[-1])
            orig_f = orig_features[..., :min_t].float()
            gen_f = gen_features[..., :min_t].float()

            # MSE distance
            mse = torch.nn.functional.mse_loss(gen_f, orig_f).item()
            log_metric(trainer.logger, f'cond_distance/{cond_key}_mse', mse)

            # Cosine similarity (flatten to [B, -1])
            orig_flat = orig_f.reshape(orig_f.shape[0], -1)
            gen_flat = gen_f.reshape(gen_f.shape[0], -1)
            cos_sim = torch.nn.functional.cosine_similarity(
                orig_flat, gen_flat, dim=-1
            ).mean().item()
            log_metric(trainer.logger, f'cond_distance/{cond_key}_cos_sim', cos_sim)

        # Clean up cached data
        module._demo_fakes_raw = None
        module._demo_conditioning = None
