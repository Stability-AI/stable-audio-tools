# Pre Encoding

When training models on encoded latents from a frozen pre-trained autoencoder, the encoder is typically frozen. Because of that, it is common to pre-encode audio to latents and store them on disk instead of computing them on-the-fly during training. This can improve training throughput as well as free up GPU memory that would otherwise be used for encoding.

## Prerequisites

To pre-encode audio to latents, you'll need a dataset config file, an autoencoder model config file, and an **unwrapped** autoencoder checkpoint file.

**Note:** You can find a copy of the unwrapped VAE checkpoint (`vae_model.ckpt`) and config (`vae_config.json`) in the `stabilityai/stable-audio-open-1.0` Hugging Face [repo](https://huggingface.co/stabilityai/stable-audio-open-1.0). This is the same VAE used in  `stable-audio-open-small`.

## Run the Pre Encoding Script

To pre-encode latents from an autoencoder model, you can use `pre_encode.py`. This script will load a pre-trained autoencoder, encode the latents/tokens, and save them to disk in a format that can be easily loaded during training.

The `pre_encode.py` script accepts the following command line arguments:

- `--model-config`
  - Path to model config
- `--ckpt-path`
  - Path to **unwrapped** autoencoder model checkpoint
- `--model-half`
  - If true, uses half precision for model weights
  - Optional
- `--dataset-config`
  - Path to dataset config file
  - Required
- `--output-path`
  - Path to output folder
  - Required
- `--batch-size`
  - Batch size for processing
  - Optional, defaults to 1
- `--sample-size`
  - Number of audio samples to pad/crop to for pre-encoding
  - Optional, defaults to 1320960 (~30 seconds)
- `--is-discrete`
  - If true, treats the model as discrete, saving discrete tokens instead of continuous latents
  - Optional
- `--num-nodes`
  - Number of nodes to use for distributed processing, if available.
  - Optional, defaults to 1
- `--num-workers`
  - Number of dataloader workers
  - Optional, defaults to 4
- `--strategy`
  - PyTorch Lightning strategy
  - Optional, defaults to 'auto'
- `--limit-batches`
  - Limits the number of batches processed
  - Optional
- `--shuffle`
  - If true, shuffles the dataset
  - Optional

**Note:** When pre encoding, it's recommended to set `"drop_last": false` in your dataset config to ensure the last batch is processed even if it's not full.

For example, if you wanted to encode latents with padding up to 30 seconds long in half precision, you could run the following:

```bash
$ python3 ./pre_encode.py \
--model-config /path/to/model/config.json \
--ckpt-path /path/to/autoencoder/model.ckpt \
--model-half \
--dataset-config /path/to/dataset/config.json \
--output-path /path/to/output/dir \
--sample-size 1320960 \
```

When you run the above, the `--output-path` directory will contain numbered subdirectories for each GPU process used to encode the latents, and a `details.json` file that keeps track of settings used when the script was run.

Inside the numbered subdirectories, you will find the encoded latents as `.npy` files, along with associated `.json` metadata files.

```bash
/path/to/output/dir/
├── 0
│   ├── 0000000000000.json
│   ├── 0000000000000.npy
│   ├── 0000000000001.json
│   ├── 0000000000001.npy
│   ├── 0000000000002.json
│   ├── 0000000000002.npy
...
└── details.json
```

## Training on Pre Encoded Latents

Once you have saved your latents to disk, you can use them to train a model by providing a dataset config file to `train.py` that points to the pre-encoded latents, specifying `"dataset_type"` is `"pre_encoded"`. Under the hood, this will configure a `stable_audio_tools.data.dataset.PreEncodedDataset`. For more information on configuring pre encoded datasets, see the [Pre Encoded Datasets](datasets.md#pre-encoded-datasets) section of the datasets docs.

The dataset config file should look something like this:

```json
{
    "dataset_type": "pre_encoded",
    "datasets": [
        {
            "id": "my_audio",
            "path": "/path/to/output/dir"
        }
    ],
    "random_crop": false
}
```

In your diffusion model config, you'll also need to specify `pre_encoded: true` in the [`training` section](diffusion.md#training-configs) to tell the training wrapper to operate on pre encoded latents instead of audio.

```json
"training": {
    "pre_encoded": true,
    ...
}
```
