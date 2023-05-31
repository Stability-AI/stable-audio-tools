def create_model_from_config(model_config):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    if model_type == 'autoencoder':
        from .autoencoders import create_autoencoder_from_config
        return create_autoencoder_from_config(model_config["model"])
    elif model_type == 'diffusion_uncond':
        from .diffusion import create_diffusion_uncond_from_config
        return create_diffusion_uncond_from_config(model_config["model"])
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')