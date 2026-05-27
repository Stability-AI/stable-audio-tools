import json
import torch
from torch.nn import Parameter
from ..models.factory import create_model_from_config
from ..models.utils import load_ckpt_state_dict, copy_state_dict

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
            warmup_mode=training_config.get("warmup_mode", "adv"),
            encoder_freeze_on_warmup=training_config.get("encoder_freeze_on_warmup", False),
            sample_rate=model_config["sample_rate"],
            loss_config=training_config.get("loss_configs", None),
            eval_loss_config=training_config.get("eval_loss_configs", None),
            optimizer_configs=training_config.get("optimizer_configs", None),
            use_ema=use_ema,
            ema_copy=ema_copy if use_ema else None,
            force_input_mono=training_config.get("force_input_mono", False),
            latent_mask_ratio=latent_mask_ratio,
            teacher_model=teacher_model,
            clip_grad_norm = training_config.get("clip_grad_norm", 0.0),
            decoder_finetune = training_config.get("decoder_finetune", False),
            decoder_loss = training_config.get("decoder_loss", False),
            num_synthetic_chirps = training_config.get("num_synthetic_chirps", 0),
            stride_curriculum = training_config.get("stride_curriculum", None),
            log_every_n_steps = training_config.get("log_every_n_steps", 10),
            tail_masking_max_patches = training_config.get("tail_masking_max_patches", 0)
        )
    elif model_type == 'diffusion_uncond':
        from .diffusion import DiffusionUncondTrainingWrapper
        return DiffusionUncondTrainingWrapper(
            model, 
            lr=training_config["learning_rate"],
            pre_encoded=training_config.get("pre_encoded", False),
            lora_config=training_config.get("lora_config", None)
        )
    elif model_type in ['diffusion_cond', 'diffusion_cond_inpaint']:
       
        diffusion_objective = model.diffusion_objective

        if "arc" in training_config:

            arc_config = training_config["arc"]

            reuse_discriminator_as_teacher = arc_config.get("reuse_discriminator_as_teacher", False)

            # Build teacher model (skip if reusing discriminator as teacher)
            if not reuse_discriminator_as_teacher:
                teacher_model_config = arc_config.get("teacher_model", None)
                teacher_model_config_path = arc_config.get("teacher_model_config_path", None)

                if teacher_model_config is None:
                    if arc_config.get("use_model_as_teacher", False):
                        teacher_model_config = model_config
                    elif teacher_model_config_path is not None:
                        with open(teacher_model_config_path) as f:
                            teacher_model_config = json.load(f)

                if teacher_model_config is not None:
                    teacher_model = create_model_from_config(teacher_model_config)

                    teacher_model_ckpt = arc_config.get("teacher_model_ckpt", None)
                    if teacher_model_ckpt is not None:
                        copy_state_dict(teacher_model, load_ckpt_state_dict(teacher_model_ckpt))
                    else:
                        raise ValueError("teacher_model_ckpt must be specified if teacher_model is specified")

                    teacher_model = teacher_model.eval().requires_grad_(False)

                    del teacher_model.pretransform  # Remove pretransform to save memory
                else:
                    teacher_model = None
            else:
                teacher_model = None  # will be set after discriminator creation

            # Build discriminator
            if "discriminator" in arc_config:

                model_self_discriminator = arc_config.get("model_self_discriminator", False)
                if model_self_discriminator:
                    discriminator = model
                else:
                    discriminator_model_config = arc_config.get("discriminator_base_model", None)
                    discriminator_model_config_path = arc_config.get("discriminator_base_model_config_path", None)

                    if discriminator_model_config is None:
                        if arc_config.get("use_model_as_discriminator", False):
                            discriminator_model_config = model_config
                        elif discriminator_model_config_path is not None:
                            with open(discriminator_model_config_path) as f:
                                discriminator_model_config = json.load(f)

                    if discriminator_model_config is not None:
                        discriminator = create_model_from_config(discriminator_model_config)

                        discriminator_model_ckpt = arc_config.get("discriminator_base_ckpt", None)
                        if discriminator_model_ckpt is not None:
                            copy_state_dict(discriminator, load_ckpt_state_dict(discriminator_model_ckpt))

                    del discriminator.pretransform # Remove pretransform to save memory
            else:
                discriminator = None

            # Reuse discriminator as teacher for ODE warmup
            if reuse_discriminator_as_teacher:
                assert discriminator is not None, "reuse_discriminator_as_teacher requires a discriminator config"
                assert arc_config.get("discriminator_base_ckpt", None) is not None, \
                    "reuse_discriminator_as_teacher requires discriminator_base_ckpt (used as teacher weights)"
                teacher_model = discriminator

            if "clap_config" in arc_config:
                clap_config = arc_config["clap_config"]

                from ..models.clap import create_clap_from_config

                use_model_pretransform = clap_config.pop("use_model_pretransform", False)

                if use_model_pretransform:
                    assert model.pretransform is not None, "Pretransform must be provided if use_model_pretransform is True"

                clap_model = create_clap_from_config(clap_config, pretransform=model.pretransform if use_model_pretransform else None)

                clap_ckpt_path = clap_config.get("ckpt_path", None)

                if clap_ckpt_path is not None:
                    copy_state_dict(clap_model, load_ckpt_state_dict(clap_ckpt_path))
            else:
                clap_model = None

            from .arc import ARCTrainingWrapper

            lora_ckpt_path = training_config.get("lora_ckpt_path", None)
            lora_state_dict = None
            if lora_ckpt_path is not None:
                lora_state_dict = load_ckpt_state_dict(lora_ckpt_path)

            return ARCTrainingWrapper(
                model=model,
                teacher_model=teacher_model,
                discriminator=discriminator,
                arc_config=arc_config,
                clap_model=clap_model,
                optimizer_configs=training_config.get("optimizer_configs", None),
                use_ema=training_config.get("use_ema", True),
                pre_encoded=training_config.get("pre_encoded", False),
                cfg_dropout_prob=training_config.get("cfg_dropout_prob", 0.1),
                timestep_sampler=training_config.get("timestep_sampler", "uniform"),
                clip_grad_norm=training_config.get("clip_grad_norm", 0.0),
                trim_config=training_config.get("trim_config", None),
                inpainting_config=training_config.get("inpainting", None),
                clap_loss_type=arc_config.get("clap_loss_type", "audio_cosine_sim"),
                mask_padding_attention=training_config.get("mask_padding_attention", False),
                silence_extension_scale_seconds=training_config.get("silence_extension_scale_seconds", 0.0),
                mask_loss_weight=training_config.get("mask_loss_weight", 0.0),
                sample_rate=model_config.get("sample_rate", 44100),
                sample_size=model_config.get("sample_size"),
                use_effective_length_for_schedule=training_config.get("use_effective_length_for_schedule", False),
                log_every_n_steps=training_config.get("log_every_n_steps", 10),
                loss_normalization=training_config.get("loss_normalization", "none"),
                loss_norm_eps=training_config.get("loss_norm_eps", 1e-6),
                lora_state_dict=lora_state_dict,
            )

        from .diffusion import DiffusionCondTrainingWrapper

        lora_ckpt_path = training_config.get("lora_ckpt_path", None)

        lora_state_dict = None
        if lora_ckpt_path is not None:
            lora_state_dict = load_ckpt_state_dict(lora_ckpt_path)

        return DiffusionCondTrainingWrapper(
            model,
            lr=training_config.get("learning_rate", None),
            mask_loss_weight=training_config.get("mask_loss_weight", 0.0),
            mask_padding_attention=training_config.get("mask_padding_attention", False),
            silence_extension_scale_seconds=training_config.get("silence_extension_scale_seconds", 0.0),
            use_ema = training_config.get("use_ema", True),
            log_loss_info=training_config.get("log_loss_info", False),
            optimizer_configs=training_config.get("optimizer_configs", None),
            pre_encoded=training_config.get("pre_encoded", False),
            cfg_dropout_prob = training_config.get("cfg_dropout_prob", 0.1),
            timestep_sampler = training_config.get("timestep_sampler", "uniform"),
            timestep_sampler_options = training_config.get("timestep_sampler_options", {}),
            p_one_shot=training_config.get("p_one_shot", 0.0),
            inpainting_config = training_config.get("inpainting", None),
            use_effective_length_for_schedule=training_config.get("use_effective_length_for_schedule", False),
            sample_rate=model_config.get("sample_rate", 44100),
            sample_size=model_config.get("sample_size"),
            loss_normalization=training_config.get("loss_normalization", "none"),
            loss_norm_eps=training_config.get("loss_norm_eps", 1e-6),
            lora_config=training_config.get("lora_config", None),
            lora_state_dict=lora_state_dict,
            svd_bases_path=model_config.get("svd_bases_path", None),
            log_every_n_steps=training_config.get("log_every_n_steps", 10),
            ot_coupling=training_config.get("ot_coupling", False),
            base_precision=training_config.get("base_precision", None)
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
    elif model_type == 'clap':

        from .clap import CLAPTrainingWrapper
    
        return CLAPTrainingWrapper(
            model,
            lr=training_config.get("learning_rate", None),
            optimizer_configs=training_config.get("optimizer_configs", None),
            loss_config=training_config.get("loss_config", None),
            htsat_dataset=training_config.get("htsat_dataset", False),
            pre_encoded=training_config.get("pre_encoded", False),
            mask_padding_attention=training_config.get("mask_padding_attention", False)
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
            demo_conditioning=demo_config.get("demo_cond", None),
            inpaint_demo_config=demo_config.get("inpaint_demos", None),
            num_demos=demo_config.get("num_demos", 0),
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
    elif model_type == "clap":
        from .clap import CLAPValidationCallback
        
        return CLAPValidationCallback(
            demo_every=demo_config.get("demo_every", 2000),
            **kwargs
        )

    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

def create_metrics_callback_from_config(model_config, **kwargs):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'

    metrics_config = training_config.get("metrics", {})

    if model_type == "diffusion_cond_inpaint" or model_type == "diffusion_cond":
        from .metrics.fad_metrics import DiffusionMetricsCallbackDistributed

        return DiffusionMetricsCallbackDistributed(
            metrics_every=metrics_config.get("metrics_every", 1000),
            cfg_scale=metrics_config.get("cfg_scale", 7),
            sampling_steps=metrics_config.get("sampling_steps", 50),
            num_generations=metrics_config.get("num_generations", 10000),
            min_length=metrics_config.get("min_length", 5),
            max_length=metrics_config.get("max_length", 190),
            prompts_type=metrics_config.get("prompts_type", 'song_describer-nosinging'),
            ref_audios_path=metrics_config.get("ref_audios_path", False),
            ref_audios_ext=metrics_config.get("ref_audios_ext", '.mp3'),
            clap_model=metrics_config.get("clap_model", 'music_speech_audioset_epoch_15_esc_89.98.pt'),
            embedding_types=metrics_config.get("embedding_types", ('clap',)),
            show_progress=metrics_config.get("show_progress", False),
            skip_first=metrics_config.get("skip_first", True),
            decode_batch_size=metrics_config.get("decode_batch_size", 4),
            **kwargs
        )
    elif model_type == "autoencoder":
        from .metrics.fad_metrics import AutoencoderMetricsCallback

        return AutoencoderMetricsCallback(
            metrics_every=metrics_config.get("metrics_every", 1000),
            ref_audios_path=metrics_config.get("ref_audios_path", False),
            ref_audios_ext=metrics_config.get("ref_audios_ext", '.wav'),
            embedding_types=metrics_config.get("embedding_types", ('clap',)),
            clap_model=metrics_config.get("clap_model", 'music_speech_audioset_epoch_15_esc_89.98.pt'),
            show_progress=metrics_config.get("show_progress", False),
            skip_first=metrics_config.get("skip_first", True),
            max_samples=metrics_config.get("max_samples", 0),
            **kwargs
        )
    else:
        raise NotImplementedError(f'No metrics callback implemented for model type: {model_type}')