import argparse
import json
from torch.nn.parameter import Parameter
from stable_audio_tools.models import create_model_from_config

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model-config', type=str, default=None)
    args.add_argument('--ckpt-path', type=str, default=None)
    args.add_argument('--name', type=str, default='exported_model')
    args.add_argument('--use-safetensors', action='store_true')

    args = args.parse_args()

    with open(args.model_config) as f:
        model_config = json.load(f)
    
    model = create_model_from_config(model_config)
    
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)

    if model_type == 'autoencoder':
        from stable_audio_tools.training.autoencoders import AutoencoderTrainingWrapper
        
        ema_copy = None

        if training_config.get("use_ema", False):
            from stable_audio_tools.models.factory import create_model_from_config
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
        from stable_audio_tools.training.diffusion import DiffusionUncondTrainingWrapper
        training_wrapper = DiffusionUncondTrainingWrapper.load_from_checkpoint(args.ckpt_path, model=model, strict=False)

    elif model_type == 'diffusion_autoencoder':
        from stable_audio_tools.training.diffusion import DiffusionAutoencoderTrainingWrapper

        ema_copy = create_model_from_config(model_config)
        
        for name, param in model.state_dict().items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            ema_copy.state_dict()[name].copy_(param)

        training_wrapper = DiffusionAutoencoderTrainingWrapper.load_from_checkpoint(args.ckpt_path, model=model, ema_copy=ema_copy, strict=False)
    elif model_type == 'diffusion_cond':
        from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper
        
        use_ema = training_config.get("use_ema", True)
        
        training_wrapper = DiffusionCondTrainingWrapper.load_from_checkpoint(
            args.ckpt_path, 
            model=model, 
            use_ema=use_ema, 
            lr=training_config.get("learning_rate", None),
            optimizer_configs=training_config.get("optimizer_configs", None),
            strict=False
        )
    elif model_type == 'diffusion_cond_inpaint':
        from stable_audio_tools.training.diffusion import DiffusionCondInpaintTrainingWrapper
        use_ema = training_config.get("use_ema", True)
        training_wrapper = DiffusionCondInpaintTrainingWrapper.load_from_checkpoint(args.ckpt_path, model=model, strict=False, use_ema=use_ema)
    elif model_type == 'diffusion_prior':
        from stable_audio_tools.training.diffusion import DiffusionPriorTrainingWrapper

        ema_copy = create_model_from_config(model_config)
        
        for name, param in model.state_dict().items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            ema_copy.state_dict()[name].copy_(param)

        training_wrapper = DiffusionPriorTrainingWrapper.load_from_checkpoint(args.ckpt_path, model=model, strict=False, ema_copy=ema_copy)
    elif model_type == 'lm':
        from stable_audio_tools.training.lm import AudioLanguageModelTrainingWrapper

        ema_copy = None

        if training_config.get("use_ema", False):

            ema_copy = create_model_from_config(model_config)

            for name, param in model.state_dict().items():
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                ema_copy.state_dict()[name].copy_(param)

        training_wrapper = AudioLanguageModelTrainingWrapper.load_from_checkpoint(
            args.ckpt_path, 
            model=model, 
            strict=False, 
            ema_copy=ema_copy,
            optimizer_configs=training_config.get("optimizer_configs", None)
        )

    else:
        raise ValueError(f"Unknown model type {model_type}")
    
    print(f"Loaded model from {args.ckpt_path}")

    if args.use_safetensors:
        ckpt_path = f"{args.name}.safetensors"
    else:
        ckpt_path = f"{args.name}.ckpt"

    training_wrapper.export_model(ckpt_path, use_safetensors=args.use_safetensors)

    print(f"Exported model to {ckpt_path}")