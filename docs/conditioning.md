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

## Prepend conditioning
Prepend conditioning involves prepending the conditioning tokens to the data tokens in the model, allowing for the information to be interpreted through the model's self-attention mechanism.

This kind of conditioning is currently only supported by Transformer-based models such as diffusion transformers.

Signals used for prepend conditioning should be of the shape `[batch, sequence, channels]`.

## Input concatenation
Input concatenation applies a spatial conditioning signal to the model that correlates in the sequence dimension with the model's input, and is of the same length. The conditioning signal will be concatenated with the model's input data along the channel dimension. This can be used for things like inpainting information, melody conditioning, or for creating a diffusion autoencoder.

Signals used for input concatenation conditioning should be of the shape `[batch, channels, sequence]` and must be the same length as the model's input.

# Conditioners and conditioning configs
`stable-audio-tools` uses Conditioner modules to translate human-readable metadata such as text prompts or a number of seconds into tensors that the model can take as input. 

Each conditioner has a corresponding `id` that it expects to find in the conditioning dictionary provided during training or inference. Each conditioner takes in the relevant conditioning data and returns a tuple containing the corresponding tensor and a mask.

The ConditionedDiffusionModelWrapper manages the translation between the user-provided metadata dictionary (e.g. `{"prompt": "a beautiful song", "seconds_start": 22, "seconds_total": 193}`) and the dictionary of different conditioning types that the model uses (e.g. `{"cross_attn_cond": ...}`).

To apply conditioning to a model, you must provide a `conditioning` configuration in the model's config. At the moment, we only support conditioning diffusion models though the `diffusion_cond` model type.

The `conditioning` configuration should contain a `configs` array, which allows you to define multiple conditioning signals. 

Each item in `configs` array should define the `id` for the corresponding metadata, the type of conditioner to be used, and the config for that conditioner.

The `cond_dim` property is used to enforce the same dimension on all conditioning inputs, however that can be overridden with an explicit `output_dim` property on any of the individual configs.

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
        }
    ],
    "cond_dim": 768
}
```

# Conditioners

## Text encoders

### `t5`
This uses a frozen [T5](https://huggingface.co/docs/transformers/model_doc/t5) text encoder from the `transformers` library to encode text prompts into a sequence of text features.

The `t5_model_name` property determines which T5 model is loaded from the `transformers` library.

The `max_length` property determines the maximum number of tokens that the text encoder will take in, as well as the sequence length of the output text features.

If you set `enable_grad` to `true`, the T5 model will be un-frozen and saved with the model checkpoint, allowing you to fine-tune the T5 model.

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

### `clap_text`
This loads the text encoder from a [CLAP](https://github.com/LAION-AI/CLAP) model, which can provide either a sequence of text features, or a single multimodal text/audio embedding.

The CLAP model must be provided with a local file path, set in the `clap_ckpt_path` property,along with the correct `audio_model_type` and `enable_fusion` properties for the provided model.

If the `use_text_features` property is set to `true`, the conditioner output will be a sequence of text features, instead of a single multimodal embedding. This allows for more fine-grained text information to be used by the model, at the cost of losing the ability to prompt with CLAP audio embeddings.

By default, if `use_text_features` is true, the last layer of the CLAP text encoder's features are returned. You can return the text features of earlier layers by specifying the index of the layer to return in the `feature_layer_ix` property. For example, you can return the text features of the next-to-last layer of the CLAP model by setting `feature_layer_ix` to `-2`.

If you set `enable_grad` to `true`, the CLAP model will be un-frozen and saved with the model checkpoint, allowing you to fine-tune the CLAP model.

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
The NumberConditioner takes in a a list of floats in a given range, and returns a continuous Fourier embedding of the provided floats.

The `min_val` and `max_val` properties set the range of the float values. This is the range used to normalize the input float values.

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