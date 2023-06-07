def create_model_from_config(model_config):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    if model_type == 'autoencoder':
        from .autoencoders import create_autoencoder_from_config
        return create_autoencoder_from_config(model_config["model"])
    elif model_type == 'diffusion_uncond':
        from .diffusion import create_diffusion_from_config
        return create_diffusion_from_config(model_config["model"])
    elif model_type == 'diffusion_autoencoder':
        from .autoencoders import create_diffAE_from_config
        return create_diffAE_from_config(model_config["model"])
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')
    
def create_pretransform_from_config(pretransform_config):
    pretransform_type = pretransform_config.get('type', None)

    assert pretransform_type is not None, 'type must be specified in pretransform config'

    if pretransform_type == 'autoencoder':
        from .autoencoders import create_autoencoder_from_config
        from .pretransforms import AutoencoderPretransform

        autoencoder_config = pretransform_config["config"]
        autoencoder = create_autoencoder_from_config(autoencoder_config)

        pretransform = AutoencoderPretransform(autoencoder)
    else:
        raise NotImplementedError(f'Unknown pretransform type: {pretransform_type}')
    
    return pretransform