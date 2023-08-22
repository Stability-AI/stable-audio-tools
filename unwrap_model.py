import argparse
import json
from torch.nn.parameter import Parameter
from harmonai_tools.models import create_model_from_config

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model-config', type=str, default=None)
    args.add_argument('--ckpt-path', type=str, default=None)
    args.add_argument('--name', type=str, default='exported_model')

    args = args.parse_args()

    with open(args.model_config) as f:
        model_config = json.load(f)
    
    model = create_model_from_config(model_config)
    
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)

    if model_type == 'autoencoder':
        from harmonai_tools.training.autoencoders import AutoencoderTrainingWrapper
        
        ema_copy = None

        if training_config.get("use_ema", False):
            from harmonai_tools.models.factory import create_model_from_config
            ema_copy = create_model_from_config(model_config)
            ema_copy = create_model_from_config(model_config) # I don't know why this needs to be called twice but it broke when I called it once
        
            # Copy each weight to the ema copy
            for name, param in model.state_dict().items():
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                ema_copy.state_dict()[name].copy_(param)

        use_ema = training_config.get("use_ema", False)

        training_wrapper = AutoencoderTrainingWrapper.load_from_checkpoint(
            args.ckpt_path, 
            autoencoder=model, 
            strict=False,
            loss_config=training_config["loss_configs"],
            use_ema=training_config["use_ema"],
            ema_copy=ema_copy if use_ema else None
        )
    elif model_type == 'diffusion_uncond':
        from harmonai_tools.training.diffusion import DiffusionUncondTrainingWrapper
        training_wrapper = DiffusionUncondTrainingWrapper.load_from_checkpoint(args.ckpt_path, model=model, strict=False)
    elif model_type == 'diffusion_autoencoder':
        from harmonai_tools.training.diffusion import DiffusionAutoencoderTrainingWrapper
        training_wrapper = DiffusionAutoencoderTrainingWrapper.load_from_checkpoint(args.ckpt_path, model=model, strict=False)
    elif model_type == 'diffusion_cond':
        from harmonai_tools.training.diffusion import DiffusionCondTrainingWrapper
        training_wrapper = DiffusionCondTrainingWrapper.load_from_checkpoint(args.ckpt_path, model=model, strict=False)
    elif model_type == 'diffusion_cond_inpaint':
        from harmonai_tools.training.diffusion import DiffusionCondInpaintTrainingWrapper
        training_wrapper = DiffusionCondInpaintTrainingWrapper.load_from_checkpoint(args.ckpt_path, model=model, strict=False)
    else:
        raise ValueError(f"Unknown model type {model_type}")

    
    print(f"Loaded model from {args.ckpt_path}")

    training_wrapper.export_model(f"{args.name}.ckpt")

    print(f"Exported model to {args.name}.ckpt")