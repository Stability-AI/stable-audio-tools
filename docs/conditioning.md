# Conditioning
Conditioning, in the context of `stable-audio-tools` is the use of additional signals in a model that are used to add an additional level of control over the model's behavior. For example, we can condition the outputs of a diffusion model on a text prompt, creating a text-to-audio model.

# Conditioning types
There are a few different kinds of conditioning depending on the conditioning signal being used.

## Cross attention
Cross attention is a type of conditioning that allows us to find correlations between two sequences of potentially different lengths. For example, cross attention allows us to find correlations between a sequence of features from a text encoder and a sequence of high-level audio features.

Signals used for cross-attention conditioning should be of the shape `[batch, sequence, channels]`.

## Global conditioning
Global conditioning is the use of a single n-dimensional tensor to provide conditioning information that pertains to the whole sequence being conditioned. For example, this could be the single embedding output of a CLAP model, or a learned class embedding.

Signals used for global conditioning should be of the shape `[batch, channels]`.

## Input concatenation
Input concatenation applies a spatial conditioning signal to the model that correlates in the sequence dimension with the model's input, and is of the same length. The conditioning signal will be concatenated with the model's input data along the channel dimension. This can be used for things like inpainting information, melody conditioning, or for creating a diffusion autoencoder.

Signals used for input concatenation conditioning should be of the shape `[batch, channels, sequence]` and must be the same length as the model's input.

## Local addition
Local addition is similar to input concatenation, but instead of concatenating the conditioning signal along the channel dimension, it is added directly to the model's latent representations. This is useful when the conditioning signal has the same channel dimension as the model's latent space.

Signals used for local addition conditioning should be of the shape `[batch, channels, sequence]` and must be the same length as the model's input.

## Modular local conditioning
Modular local conditioning is a variant of local conditioning where each conditioning signal is kept as a separate entry in a dictionary, rather than being concatenated together. This allows the model to handle each conditioning signal independently through dedicated processing modules.

Each signal should be of the shape `[batch, channels, sequence]`.

## Prepend conditioning
Prepend conditioning works by prepending the conditioning signal to the beginning of the model's input sequence. This is useful for conditioning signals that have a sequence structure but should not be mixed with the model's spatial input.

Signals used for prepend conditioning should be of the shape `[batch, sequence, channels]`.

# Conditioners and conditioning configs
`stable-audio-tools` uses Conditioner modules to translate human-readable metadata such as text prompts or a number of seconds into tensors that the model can take as input.

Each conditioner has a corresponding `id` that it expects to find in the conditioning dictionary provided during training or inference. Each conditioner takes in the relevant conditioning data and returns a tuple containing the corresponding tensor and a mask.

The ConditionedDiffusionModelWrapper manages the translation between the user-provided metadata dictionary (e.g. `{"prompt": "a beautiful song", "seconds_start": 22, "seconds_total": 193}`) and the dictionary of different conditioning types that the model uses (e.g. `{"cross_attn_cond": ...}`).

To apply conditioning to a model, you must provide a `conditioning` configuration in the model's config. At the moment, we only support conditioning diffusion models though the `diffusion_cond` model type.

The `conditioning` configuration should contain a `configs` array, which allows you to define multiple conditioning signals.

Each item in `configs` array should define the `id` for the corresponding metadata, the type of conditioner to be used, and the config for that conditioner.

The `cond_dim` property is used to enforce the same dimension on all conditioning inputs, however that can be overridden with an explicit `output_dim` property on any of the individual configs.

The optional `default_keys` property is a dictionary that maps conditioner IDs to fallback metadata keys. If a conditioner's ID is not found in the batch metadata, it will look for the key specified in `default_keys` instead.

The optional `pre_encoded_keys` property is a list of conditioner IDs whose inputs are already encoded tensors. These skip the conditioner's forward pass and are stacked directly.

## Example config
```json
"conditioning": {
    "configs": [
        {
            "id": "prompt",
            "type": "t5",
            "config": {
                "t5_model_name": "t5-base",
                "max_length": 77,
                "project_out": true
            }
        },
        {
            "id": "seconds_start",
            "type": "int",
            "config": {
                "min_val": 0,
                "max_val": 512
            }
        },
        {
            "id": "seconds_total",
            "type": "number",
            "config": {
                "min_val": 0,
                "max_val": 512
            }
        }
    ],
    "cond_dim": 768,
    "default_keys": {},
    "pre_encoded_keys": []
}
```

# Conditioners

## Text encoders

### `t5`
This uses a frozen [T5](https://huggingface.co/docs/transformers/model_doc/t5) text encoder from the `transformers` library to encode text prompts into a sequence of text features.

The `t5_model_name` property determines which T5 model is loaded from the `transformers` library.

The `max_length` property determines the maximum number of tokens that the text encoder will take in, as well as the sequence length of the output text features.

If you set `enable_grad` to `true`, the T5 model will be un-frozen and saved with the model checkpoint, allowing you to fine-tune the T5 model.

The `padding_mode` property controls how padding tokens are handled: `"zero"` (default) zeros out padding positions, `"none"` leaves raw embeddings, and `"learned"` replaces padding positions with a learned embedding.

T5 encodings are only compatible with cross attention conditioning.

#### Example config
```json
{
    "id": "prompt",
    "type": "t5",
    "config": {
        "t5_model_name": "t5-base",
        "max_length": 77,
        "project_out": true
    }
}
```

### `t5gemma`
This uses a [T5Gemma](https://huggingface.co/google/t5gemma-b-b-ul2) encoder from the `transformers` library to encode text prompts into a sequence of text features.

The `model_name` property determines which T5Gemma model is loaded (e.g. `"google/t5gemma-b-b-ul2"`).

The `max_length` property determines the maximum number of tokens to encode.

If you set `enable_grad` to `true`, the model will be un-frozen and saved with the model checkpoint.

The `padding_mode` property controls padding handling (see `t5` above).

T5Gemma encodings are compatible with cross attention conditioning.

#### Example config
```json
{
    "id": "prompt",
    "type": "t5gemma",
    "config": {
        "model_name": "google/t5gemma-b-b-ul2",
        "max_length": 128
    }
}
```

### `causal_lm`
This uses a causal language model (e.g. Gemma-2) from the `transformers` library to encode text prompts into a sequence of features.

The `model_name` property determines which model is loaded (e.g. `"google/gemma-2-2b"`).

The `max_length` property determines the maximum number of tokens to encode.

If you set `enable_grad` to `true`, the model will be un-frozen and saved with the checkpoint.

If you set `learned_scale` to `true`, a learned scaling parameter will be applied to the output embeddings.

The `padding_mode` property controls padding handling (see `t5` above).

Causal LM encodings are compatible with cross attention conditioning.

#### Example config
```json
{
    "id": "prompt",
    "type": "causal_lm",
    "config": {
        "model_name": "google/gemma-2-2b",
        "max_length": 128
    }
}
```

### `clap_text`
This loads the text encoder from a [CLAP](https://github.com/LAION-AI/CLAP) model, which can provide either a sequence of text features, or a single multimodal text/audio embedding.

The CLAP model must be provided with a local file path, set in the `clap_ckpt_path` property, along with the correct `audio_model_type` and `enable_fusion` properties for the provided model.

If the `use_text_features` property is set to `true`, the conditioner output will be a sequence of text features, instead of a single multimodal embedding. This allows for more fine-grained text information to be used by the model, at the cost of losing the ability to prompt with CLAP audio embeddings.

By default, if `use_text_features` is true, the last layer of the CLAP text encoder's features are returned. You can return the text features of earlier layers by specifying the index of the layer to return in the `feature_layer_ix` property. For example, you can return the text features of the next-to-last layer of the CLAP model by setting `feature_layer_ix` to `-2`.

If you set `finetune` to `true`, the CLAP model will be un-frozen and saved with the model checkpoint, allowing you to fine-tune the CLAP model.

CLAP text embeddings are compatible with global conditioning and cross attention conditioning. If `use_text_features` is set to `true`, the features are not compatible with global conditioning.

#### Example config
```json
{
    "id": "prompt",
    "type": "clap_text",
    "config": {
        "clap_ckpt_path": "/path/to/clap/model.ckpt",
        "audio_model_type": "HTSAT-base",
        "enable_fusion": true,
        "use_text_features": true,
        "feature_layer_ix": -2
    }
}
```

### `sat_clap_text`
This uses the Stable Audio Tools CLAP wrapper to encode text. Unlike `clap_text`, this conditioner uses the project's own CLAP implementation and is configured through the CLAP model config rather than a standalone checkpoint.

The `ckpt_path` property sets the path to a CLAP checkpoint to load.

The `use_text_features` property controls whether the output is a sequence of text features (`true`) or a single embedding (`false`).

The `feature_layer_ix` property sets which hidden layer's features to return (e.g. `-2` for the next-to-last layer).

If `use_model_pretransform` is set to `true`, the CLAP model will share the main model's pretransform encoder.

SAT CLAP text encodings are compatible with cross attention and global conditioning.

#### Example config
```json
{
    "id": "prompt",
    "type": "sat_clap_text",
    "config": {
        "ckpt_path": "/path/to/clap.ckpt",
        "use_text_features": true,
        "feature_layer_ix": -2
    }
}
```

### `phoneme`
This conditioner converts English text into phonemes using the `g2p_en` library and embeds them using a learned lookup table. This can be useful for conditioning on the phonetic content of text, e.g. for speech or singing synthesis.

The `max_length` property sets the maximum number of phonemes to embed (default 1024).

Phoneme encodings are compatible with cross attention conditioning.

#### Example config
```json
{
    "id": "phonemes",
    "type": "phoneme",
    "config": {
        "max_length": 512
    }
}
```

### `lut`
The TokenizerLUTConditioner embeds text using a learned lookup table built on a pretrained tokenizer's vocabulary. This is a lightweight alternative to frozen language model encoders like T5.

The `tokenizer_name` property specifies which HuggingFace tokenizer to use.

The `max_length` property sets the maximum token sequence length (default 1024).

If `use_abs_pos_emb` is set to `true`, absolute positional embeddings will be added to the token embeddings.

The `special_tokens` property is a list of additional special tokens to add to the tokenizer.

LUT encodings are compatible with cross attention conditioning.

#### Example config
```json
{
    "id": "prompt",
    "type": "lut",
    "config": {
        "tokenizer_name": "t5-base",
        "max_length": 256,
        "use_abs_pos_emb": true
    }
}
```

## Number encoders

### `int`
The IntConditioner takes in a list of integers in a given range, and returns a discrete learned embedding for each of those integers.

The `min_val` and `max_val` properties set the range of the embedding values. Input integers are clamped to this range.

This can be used for things like discrete timing embeddings, or learned class embeddings.

Int embeddings are compatible with global conditioning and cross attention conditioning.

#### Example config
```json
{
    "id": "seconds_start",
    "type": "int",
    "config": {
        "min_val": 0,
        "max_val": 512
    }
}
```

### `number`
The NumberConditioner takes in a list of floats in a given range, and returns a continuous Fourier embedding of the provided floats.

The `min_val` and `max_val` properties set the range of the float values. This is the range used to normalize the input float values.

The `fourier_features_type` property controls the type of Fourier features used: `"learned"` (default) or `"expo"`.

Number embeddings are compatible with global conditioning and cross attention conditioning.

#### Example config
```json
{
    "id": "seconds_total",
    "type": "number",
    "config": {
        "min_val": 0,
        "max_val": 512
    }
}
```

### `list`
The ListConditioner takes in a list of strings and returns a discrete learned embedding based on a fixed set of options. Strings not in the options list are mapped to a shared "unknown" embedding.

The `options` property is a list of valid string values.

List embeddings are compatible with global conditioning and cross attention conditioning.

#### Example config
```json
{
    "id": "genre",
    "type": "list",
    "config": {
        "options": ["rock", "jazz", "classical", "electronic"]
    }
}
```

## Audio conditioners

### `clap_audio`
This loads the audio encoder from a [CLAP](https://github.com/LAION-AI/CLAP) model to encode audio into a single multimodal embedding.

The CLAP model must be provided with a local file path via `clap_ckpt_path`, along with the correct `audio_model_type` and `enable_fusion` properties.

CLAP audio embeddings are compatible with global conditioning.

#### Example config
```json
{
    "id": "audio_prompt",
    "type": "clap_audio",
    "config": {
        "clap_ckpt_path": "/path/to/clap/model.ckpt",
        "audio_model_type": "HTSAT-base",
        "enable_fusion": true
    }
}
```

### `sat_clap_audio`
This uses the Stable Audio Tools CLAP wrapper to encode audio into a single embedding. Like `sat_clap_text`, it uses the project's own CLAP implementation.

The `ckpt_path` property sets the path to a CLAP checkpoint to load.

The `sample_rate` property must be specified for audio conditioners.

If `use_model_pretransform` is set to `true`, the CLAP model will share the main model's pretransform encoder.

SAT CLAP audio embeddings are compatible with global conditioning.

#### Example config
```json
{
    "id": "audio_prompt",
    "type": "sat_clap_audio",
    "config": {
        "ckpt_path": "/path/to/clap.ckpt",
        "sample_rate": 44100
    }
}
```

### `pretransform`
The PretransformConditioner encodes audio through a pretransform module (e.g. a VAE or DAC encoder) for conditioning. This allows you to condition on a latent representation of audio.

The `sample_rate` property must be specified.

If `use_model_pretransform` is `true`, the conditioner will share the main model's pretransform instead of creating its own. Otherwise, you must provide a `pretransform_config` to define the encoder.

The `pretransform_ckpt_path` property optionally specifies a checkpoint to load for the pretransform.

If `save_pretransform` is `true`, the pretransform weights will be saved as part of the model's state dict.

Pretransform encodings are compatible with input concatenation, local addition, and prepend conditioning.

#### Example config
```json
{
    "id": "audio_cond",
    "type": "pretransform",
    "config": {
        "sample_rate": 44100,
        "use_model_pretransform": true,
        "save_pretransform": false
    }
}
```

### `source_mix`
The SourceMixConditioner mixes projected audio embeddings from multiple source signals. Each source is encoded through a pretransform and then projected through a dedicated per-source head. The projections are summed to produce the final conditioning signal. This is useful for conditioning on multiple audio stems (e.g. vocals, drums, bass).

Key properties:
- `sample_rate`: Must be specified
- `source_keys`: A list of metadata keys corresponding to the potential audio sources
- `use_model_pretransform`: If `true`, shares the main model's pretransform
- `pretransform_config`: Config for a standalone pretransform (used if `use_model_pretransform` is `false`)
- `pretransform_ckpt_path`: Optional checkpoint for the pretransform
- `pre_encoded`: If `true`, inputs are already encoded latents (skip the pretransform)
- `allow_null_source`: If `true`, allows batch items with no sources by using a learned null embedding
- `source_length`: Required if `allow_null_source` is `true`; sets the length of the null source embedding
- `save_pretransform`: Whether to save pretransform weights in the state dict

Source mix outputs are compatible with input concatenation, local addition, and modular local conditioning.

#### Example config
```json
{
    "id": "source_mix",
    "type": "source_mix",
    "config": {
        "sample_rate": 44100,
        "source_keys": ["vocals", "drums", "bass", "other"],
        "use_model_pretransform": true,
        "allow_null_source": true,
        "source_length": 512
    }
}
```

## Gradio demo

The `run_control_gradio.py` script launches an interactive demo with audio conditioning controls. The UI auto-detects conditioners from the model config and provides audio input, conditioner selection, envelope tweaks, chord editing, and content editing.

```bash
python run_control_gradio.py --model-config config.json --ckpt-path checkpoint.ckpt --pretransform-ckpt-path pretransform.ckpt
```
