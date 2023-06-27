def create_training_wrapper_from_configs(model_config, training_config, model):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    if model_type == 'autoencoder':
        from .autoencoders import AutoencoderTrainingWrapper
        return AutoencoderTrainingWrapper(
            model, 
            lr=training_config["learning_rate"],
            warmup_steps=training_config["warmup_steps"], 
            sample_rate=model_config["sample_rate"],
            loss_config=training_config["loss_configs"]
        )
    elif model_type == 'diffusion_uncond':
        from .diffusion import DiffusionUncondTrainingWrapper
        return DiffusionUncondTrainingWrapper(
            model, 
            lr=training_config["learning_rate"]
        )
    elif model_type == 'diffusion_cond':
        from .diffusion import DiffusionCondTrainingWrapper
        return DiffusionCondTrainingWrapper(
            model, 
            lr=training_config["learning_rate"]
        )
    elif model_type == 'diffusion_autoencoder':
        from .diffusion import DiffusionAutoencoderTrainingWrapper
        return DiffusionAutoencoderTrainingWrapper(
            model,
            lr=training_config["learning_rate"]
        )
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

def create_demo_callback_from_configs(model_config, training_config, **kwargs):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    demo_config = training_config.get("demo", {})

    if model_type == 'autoencoder':
        from .autoencoders import AutoencoderDemoCallback
        return AutoencoderDemoCallback(
            demo_every=demo_config.get("demo_every", 2000), 
            sample_size=model_config["sample_size"], 
            sample_rate=model_config["sample_rate"],
            **kwargs
        )
    elif model_type == 'diffusion_uncond':
        from .diffusion import DiffusionUncondDemoCallback
        return DiffusionUncondDemoCallback(
            demo_every=demo_config.get("demo_every", 2000), 
            demo_steps=demo_config.get("demo_steps", 250), 
            sample_size=model_config["sample_size"], 
            sample_rate=model_config["sample_rate"]
        )
    elif model_type == "diffusion_autoencoder":
        from .diffusion import DiffusionAutoencoderDemoCallback
        return DiffusionAutoencoderDemoCallback(
            demo_every=demo_config.get("demo_every", 2000), 
            sample_size=model_config["sample_size"],
            sample_rate=model_config["sample_rate"],
            **kwargs
        )
    elif model_type == "diffusion_cond":
        from .diffusion import DiffusionCondDemoCallback

        return DiffusionCondDemoCallback(
            demo_every=demo_config.get("demo_every", 2000), 
            sample_size=model_config["sample_size"],
            sample_rate=model_config["sample_rate"],
            demo_steps=demo_config.get("demo_steps", 250), 
            num_demos=demo_config["num_demos"],
            demo_cfg_scales=demo_config["demo_cfg_scales"],
            demo_conditioning=demo_config["demo_cond"],
        )
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')