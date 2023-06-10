def create_training_wrapper_from_config_and_args(model_config, args, model):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    if model_type == 'autoencoder':
        from .autoencoders import AutoencoderTrainingWrapper
        return AutoencoderTrainingWrapper(
            model, 
            lr=args.lr,
            warmup_steps=args.warmup_steps, 
            sample_rate=model_config["sample_rate"]
        )
    elif model_type == 'diffusion_uncond':
        from .diffusion import DiffusionUncondTrainingWrapper
        return DiffusionUncondTrainingWrapper(
            model, 
            lr=args.lr
        )
    elif model_type == 'diffusion_autoencoder':
        from .diffusion import DiffusionAutoencoderTrainingWrapper
        return DiffusionAutoencoderTrainingWrapper(
            model,
            lr=args.lr,
        )
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

def create_demo_callback_from_config_and_args(model_config, args, **kwargs):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    if model_type == 'autoencoder':
        from .autoencoders import AutoencoderDemoCallback
        return AutoencoderDemoCallback(
            demo_every=args.demo_every, 
            sample_size=model_config["sample_size"], 
            sample_rate=model_config["sample_rate"],
            **kwargs
        )
    elif model_type == 'diffusion_uncond':
        from .diffusion import DiffusionUncondDemoCallback
        return DiffusionUncondDemoCallback(
            demo_every=args.demo_every, 
            demo_steps=args.demo_steps,
            sample_size=model_config["sample_size"], 
            sample_rate=model_config["sample_rate"]
        )
    elif model_type == "diffusion_autoencoder":
        from .diffusion import DiffusionAutoencoderDemoCallback
        return DiffusionAutoencoderDemoCallback(
            demo_every=args.demo_every,
            sample_size=model_config["sample_size"],
            sample_rate=model_config["sample_rate"],
            **kwargs
        )
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')