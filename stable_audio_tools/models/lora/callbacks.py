import os
import pytorch_lightning as pl


class LoRASafetensorsCheckpoint(pl.callbacks.ModelCheckpoint):
    """ModelCheckpoint subclass that saves .safetensors instead of .ckpt for LoRA runs."""

    def _save_checkpoint(self, trainer, filepath):
        if filepath.endswith(".ckpt") and hasattr(trainer.lightning_module, 'export_lora_safetensors'):
            st_path = filepath[:-5] + ".safetensors"
            try:
                os.makedirs(os.path.dirname(st_path), exist_ok=True)
                trainer.lightning_module.export_lora_safetensors(st_path)
            except Exception as e:
                print(f"[safetensors] Failed to save {st_path}: {e}")
                super()._save_checkpoint(trainer, filepath)
        else:
            super()._save_checkpoint(trainer, filepath)


class StepOffsetCallback(pl.Callback):
    """Offset trainer.global_step for LoRA resume (checkpoints don't contain trainer state)."""
    def __init__(self, offset):
        self.offset = offset

    def on_train_start(self, trainer, pl_module):
        if self.offset > 0:
            trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = self.offset
            print(f"Step offset applied: global_step set to {trainer.global_step}")
