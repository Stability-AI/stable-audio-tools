import torch
from torch.nn import Parameter
from ..models.factory import create_model_from_config

def create_training_wrapper_from_config(model_config, model):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'

    if model_type == 'autoencoder':
        from .autoencoders import AutoencoderTrainingWrapper

        ema_copy = None

        if training_config.get("use_ema", False):
            ema_copy = create_model_from_config(model_config)
            ema_copy = create_model_from_config(model_config) # I don't know why this needs to be called twice but it broke when I called it once
            # Copy each weight to the ema copy
            for name, param in model.state_dict().items():
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                ema_copy.state_dict()[name].copy_(param)

        use_ema = training_config.get("use_ema", False)

        latent_mask_ratio = training_config.get("latent_mask_ratio", 0.0)

        teacher_model = training_config.get("teacher_model", None)
        if teacher_model is not None:
            teacher_model = create_model_from_config(teacher_model)
            teacher_model = teacher_model.eval().requires_grad_(False)

            teacher_model_ckpt = training_config.get("teacher_model_ckpt", None)
            if teacher_model_ckpt is not None:
                teacher_model.load_state_dict(torch.load(teacher_model_ckpt)["state_dict"])
            else:
                raise ValueError("teacher_model_ckpt must be specified if teacher_model is specified")

        return AutoencoderTrainingWrapper(
            model, 
            lr=training_config.get("learning_rate", None),
            warmup_steps=training_config.get("warmup_steps", 0), 
            encoder_freeze_on_warmup=training_config.get("encoder_freeze_on_warmup", False),
            sample_rate=model_config["sample_rate"],
            loss_config=training_config.get("loss_configs", None),
            eval_loss_config=training_config.get("eval_loss_configs", None),
            optimizer_configs=training_config.get("optimizer_configs", None),
            use_ema=use_ema,
            ema_copy=ema_copy if use_ema else None,
            force_input_mono=training_config.get("force_input_mono", False),
            latent_mask_ratio=latent_mask_ratio,
            teacher_model=teacher_model
        )
    elif model_type == 'diffusion_uncond':
        from .diffusion import DiffusionUncondTrainingWrapper
        return DiffusionUncondTrainingWrapper(
            model, 
            lr=training_config["learning_rate"],
            pre_encoded=training_config.get("pre_encoded", False),
        )
    elif model_type == 'diffusion_cond':
        from .diffusion import DiffusionCondTrainingWrapper
        return DiffusionCondTrainingWrapper(
            model, 
            lr=training_config.get("learning_rate", None),
            mask_padding=training_config.get("mask_padding", False),
            mask_padding_dropout=training_config.get("mask_padding_dropout", 0.0),
            use_ema = training_config.get("use_ema", True),
            log_loss_info=training_config.get("log_loss_info", False),
            optimizer_configs=training_config.get("optimizer_configs", None),
            pre_encoded=training_config.get("pre_encoded", False),
            cfg_dropout_prob = training_config.get("cfg_dropout_prob", 0.1),
            timestep_sampler = training_config.get("timestep_sampler", "uniform"),
        )
    elif model_type == 'diffusion_prior':
        from .diffusion import DiffusionPriorTrainingWrapper
        from ..models.diffusion_prior import PriorType

        ema_copy = create_model_from_config(model_config)
        
        # Copy each weight to the ema copy
        for name, param in model.state_dict().items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            ema_copy.state_dict()[name].copy_(param)

        prior_type = training_config.get("prior_type", "mono_stereo")

        if prior_type == "mono_stereo":
            prior_type_enum = PriorType.MonoToStereo
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")

        return DiffusionPriorTrainingWrapper(
            model, 
            lr=training_config["learning_rate"],
            ema_copy=ema_copy,
            prior_type=prior_type_enum,
            log_loss_info=training_config.get("log_loss_info", False),
            use_reconstruction_loss=training_config.get("use_reconstruction_loss", False),
        )
    elif model_type == 'diffusion_cond_inpaint':
        from .diffusion import DiffusionCondInpaintTrainingWrapper
        return DiffusionCondInpaintTrainingWrapper(
            model, 
            lr=training_config.get("learning_rate", None),
            max_mask_segments = training_config.get("max_mask_segments", 10),
            log_loss_info=training_config.get("log_loss_info", False),
            optimizer_configs=training_config.get("optimizer_configs", None),
            use_ema=training_config.get("use_ema", True),
            pre_encoded=training_config.get("pre_encoded", False),
            cfg_dropout_prob = training_config.get("cfg_dropout_prob", 0.1),
            timestep_sampler = training_config.get("timestep_sampler", "uniform")
        )
    elif model_type == 'diffusion_autoencoder':
        from .diffusion import DiffusionAutoencoderTrainingWrapper

        ema_copy = create_model_from_config(model_config)
        
        # Copy each weight to the ema copy
        for name, param in model.state_dict().items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            ema_copy.state_dict()[name].copy_(param)

        return DiffusionAutoencoderTrainingWrapper(
            model,
            ema_copy=ema_copy,
            lr=training_config["learning_rate"],
            use_reconstruction_loss=training_config.get("use_reconstruction_loss", False)
        )
    elif model_type == 'lm':
        from .lm import AudioLanguageModelTrainingWrapper

        ema_copy = create_model_from_config(model_config)

        for name, param in model.state_dict().items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            ema_copy.state_dict()[name].copy_(param)

        return AudioLanguageModelTrainingWrapper(
            model,
            ema_copy=ema_copy,
            lr=training_config.get("learning_rate", None),
            use_ema=training_config.get("use_ema", False),
            optimizer_configs=training_config.get("optimizer_configs", None),
            pre_encoded=training_config.get("pre_encoded", False),
        )

    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

def create_demo_callback_from_config(model_config, **kwargs):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'

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
            sample_rate=model_config["sample_rate"]
        )
    elif model_type == "diffusion_autoencoder":
        from .diffusion import DiffusionAutoencoderDemoCallback
        return DiffusionAutoencoderDemoCallback(
            demo_every=demo_config.get("demo_every", 2000), 
            demo_steps=demo_config.get("demo_steps", 250),
            sample_size=model_config["sample_size"],
            sample_rate=model_config["sample_rate"],
            **kwargs
        )
    elif model_type == "diffusion_prior":
        from .diffusion import DiffusionPriorDemoCallback
        return DiffusionPriorDemoCallback(
            demo_every=demo_config.get("demo_every", 2000), 
            demo_steps=demo_config.get("demo_steps", 250),
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
            demo_conditioning=demo_config.get("demo_cond", {}),
            demo_cond_from_batch=demo_config.get("demo_cond_from_batch", False),
            display_audio_cond=demo_config.get("display_audio_cond", False),
            cond_display_configs=demo_config.get("cond_display_configs", None),
        )
    elif model_type == "diffusion_cond_inpaint":
        from .diffusion import DiffusionCondInpaintDemoCallback

        return DiffusionCondInpaintDemoCallback(
            demo_every=demo_config.get("demo_every", 2000), 
            sample_size=model_config["sample_size"],
            sample_rate=model_config["sample_rate"],
            demo_steps=demo_config.get("demo_steps", 250),
            demo_cfg_scales=demo_config["demo_cfg_scales"],
            num_demos=demo_config.get("num_demos", 8),
            **kwargs
        )
    
    elif model_type == "lm":
        from .lm import AudioLanguageModelDemoCallback

        return AudioLanguageModelDemoCallback(
            demo_every=demo_config.get("demo_every", 2000), 
            sample_size=model_config["sample_size"],
            sample_rate=model_config["sample_rate"],
            demo_cfg_scales=demo_config.get("demo_cfg_scales", [1]),
            demo_conditioning=demo_config.get("demo_cond", None),
            num_demos=demo_config.get("num_demos", 8),
            **kwargs
        )
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')