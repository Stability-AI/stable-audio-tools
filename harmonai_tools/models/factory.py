def create_model_from_config(model_config):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    if model_type == 'autoencoder':
        from .autoencoders import create_autoencoder_from_config
        return create_autoencoder_from_config(model_config["model"])
    elif model_type == 'diffusion_uncond':
        from .diffusion import create_diffusion_uncond_from_config
        return create_diffusion_uncond_from_config(model_config["model"])
    elif model_type == 'diffusion_cond':
        from .diffusion import create_diffusion_cond_from_config
        return create_diffusion_cond_from_config(model_config["model"])
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

        pretransform = AutoencoderPretransform(autoencoder, scale=pretransform_config.get("scale", 1.0))
    elif pretransform_type == 'wavelet':
        from .pretransforms import WaveletPretransform

        wavelet_config = pretransform_config["config"]
        channels = wavelet_config["channels"]
        levels = wavelet_config["levels"]
        wavelet = wavelet_config["wavelet"]

        pretransform = WaveletPretransform(channels, levels, wavelet)
    else:
        raise NotImplementedError(f'Unknown pretransform type: {pretransform_type}')
    
    requires_grad = pretransform_config.get('requires_grad', False)
    pretransform.requires_grad = requires_grad

    return pretransform

def create_bottleneck_from_config(bottleneck_config):
    bottleneck_type = bottleneck_config.get('type', None)

    assert bottleneck_type is not None, 'type must be specified in bottleneck config'

    if bottleneck_type == 'tanh':
        from .bottleneck import TanhBottleneck
        return TanhBottleneck()
    elif bottleneck_type == 'vae':
        from .bottleneck import VAEBottleneck
        return VAEBottleneck()
    elif bottleneck_type == 'rvq':
        from .bottleneck import RVQBottleneck

        quantizer_params = {
            "dim": 128,
            "codebook_size": 1024,
            "num_quantizers": 8,
            "decay": 0.99,
            "kmeans_init": True,
            "kmeans_iters": 50,
            "threshold_ema_dead_code": 2,
        }

        quantizer_params.update(bottleneck_config["config"])

        return RVQBottleneck(**quantizer_params)
    elif bottleneck_type == "dac_rvq":
        from .bottleneck import DACRVQBottleneck

        return DACRVQBottleneck(**bottleneck_config["config"])
    
    elif bottleneck_type == 'rvq_vae':
        from .bottleneck import RVQVAEBottleneck

        quantizer_params = {
            "dim": 128,
            "codebook_size": 1024,
            "num_quantizers": 8,
            "decay": 0.99,
            "kmeans_init": True,
            "kmeans_iters": 50,
            "threshold_ema_dead_code": 2,
        }

        quantizer_params.update(bottleneck_config["config"])

        return RVQVAEBottleneck(**quantizer_params)
        
    elif bottleneck_type == 'dac_rvq_vae':
        from .bottleneck import DACRVQVAEBottleneck

        return DACRVQVAEBottleneck(**bottleneck_config["config"])
    elif bottleneck_type == 'memcodes':
        from .bottleneck import MemcodesBottleneck
        return MemcodesBottleneck(**bottleneck_config["config"])
    else:
        raise NotImplementedError(f'Unknown bottleneck type: {bottleneck_type}')
