# Datasets

`stable-audio-tools` supports five dataset types for training:

- **`audio_dir`** — local directories of audio files
- **`pre_encoded`** — pre-encoded latent representations (`.npy` + `.json` pairs)
- **`s3` / `wds`** — WebDataset tar files from Amazon S3 or local storage

To specify the dataset used for training, provide a dataset config JSON file to `train.py` via the `--dataset_config` flag. An optional `--val_dataset_config` flag can be used for validation data.

# Dataset Config Structure

All dataset configs share the same top-level structure:

```json
{
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "my_dataset",
            "path": "/path/to/data/"
        }
    ],
    "random_crop": true
}
```

**Common top-level keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `dataset_type` | (required) | One of `audio_dir`, `pre_encoded`, `s3`, `wds` |
| `datasets` | (required) | Array of per-source dataset objects |
| `random_crop` | `true` | Crop from a random position in each file. If `false`, always crops from the beginning |
| `drop_last` | `true` | Drop the last incomplete batch |
| `custom_metadata_module` | — | Path to a custom metadata Python module (can also be set per-dataset entry) |

Additional type-specific keys are documented in each dataset type section below.

# Dataset Types

## audio_dir

Loads audio files from local directories. Recursively scans for compatible formats: FLAC, WAV, MP3, M4A, OGG, OPUS.

### Example config

```json
{
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "my_audio",
            "path": "/path/to/audio/dataset/"
        },
        {
            "id": "drums_only",
            "path": "/path/to/more/audio/",
            "keywords": ["drums", "percussion"],
            "weight": 2.0
        }
    ],
    "random_crop": true,
    "volume_norm": true,
    "volume_norm_param": [-16, 2]
}
```

### Per-dataset entry keys

| Key | Default | Description |
|-----|---------|-------------|
| `id` | (required) | Identifier for this dataset source |
| `path` | (required) | Path to directory containing audio files |
| `keywords` | — | List of keywords to filter filenames. Only files whose names contain at least one keyword are included (case-insensitive) |
| `filelist_path` | — | Path to a text file listing filenames (one per line). Overrides directory scanning |
| `custom_metadata_module` | — | Path to a custom metadata Python module for this dataset |
| `weight` | `1.0` | Sampling weight. Values other than 1.0 enable weighted random sampling |

### Top-level options

| Key | Default | Description |
|-----|---------|-------------|
| `random_crop` | `true` | Random crop position within each file |
| `volume_norm` | `false` | Enable LUFS-based volume normalization |
| `volume_norm_param` | `[-16, 2]` | `[target_lufs, random_gain_range_db]`. Audio is normalized to the target loudness with a random gain offset within the range |
| `strip_silence` | `false` | Strip trailing silence (below -60 dB) before cropping |
| `drop_last` | `true` | Drop incomplete final batch |

**Note:** Channel conversion (stereo/mono) is derived from the model config's `audio_channels` setting (2 = stereo, 1 = mono) and is not set in the dataset config. Phase flip augmentation (50% probability) is always applied.

## pre_encoded

Loads pre-encoded latent representations from `.npy` files with corresponding `.json` metadata files. Each sample consists of a pair: `sample_name.npy` (latent tensor of shape `[C, N]`) and `sample_name.json` (metadata dictionary).

The metadata JSON must include:
- `padding_mask` — list of 0s and 1s indicating valid frames
- `seconds_total` — total duration of the original audio in seconds

### Example config

```json
{
    "dataset_type": "pre_encoded",
    "datasets": [
        {
            "id": "encoded_v1",
            "path": "/path/to/latents/"
        }
    ],
    "latent_crop_length": 2048,
    "min_length_sec": 1.0,
    "max_length_sec": 120.0,
    "random_crop": false
}
```

### Per-dataset entry keys

| Key | Default | Description |
|-----|---------|-------------|
| `id` | (required) | Identifier for this dataset source |
| `path` | (required) | Directory containing `.npy` + `.json` pairs |
| `latent_extension` | `"npy"` | File extension for latent files |
| `filelist_path` | — | Path to explicit file list |
| `custom_metadata_module` | — | Path to custom metadata module |
| `weight` | `1.0` | Sampling weight |

### Top-level options

| Key | Default | Description |
|-----|---------|-------------|
| `latent_crop_length` | — | Maximum sequence length in latent frames. Longer samples are cropped; shorter are padded |
| `min_length_sec` | — | Reject samples shorter than this duration (seconds) |
| `max_length_sec` | — | Reject samples longer than this duration (seconds) |
| `random_crop` | `false` | When cropping long latents, use a random start offset instead of always starting at frame 0 |

### Silence padding

If a `silence.npy` file is placed in the dataset directory, it will be used to pad short samples with realistic silence latents instead of zeros. This file should be the autoencoder's encoding of a silent audio segment, with shape `[C, N]`. The silence latent is tiled as needed to fill the padding.

### Pre-tokenization

When the model has text conditioners (e.g., T5), text fields in the metadata are automatically tokenized in DataLoader workers rather than on the main training thread. This avoids CPU contention and requires no additional configuration — tokenizers are extracted from the model's conditioners automatically.

For each tokenized field, the raw text is preserved as `{key}_text` in the metadata (e.g., `prompt_text`) for use by text-based losses like CLAP.

## s3 / wds (WebDataset)

Streams data from WebDataset `.tar` files hosted on Amazon S3 (`"s3"`) or stored locally (`"wds"`). Both type names use the same implementation — `"s3"` is kept for backwards compatibility.

Each tar file should contain paired files that differ only in extension (e.g., `000001.flac` + `000001.json`).

### S3 example

```json
{
    "dataset_type": "s3",
    "datasets": [
        {
            "id": "s3-data",
            "s3_path": "s3://my-bucket/datasets/webdataset/audio/",
            "profile": "my-aws-profile"
        }
    ],
    "random_crop": true,
    "epoch_steps": 2000
}
```

### Local WebDataset example

```json
{
    "dataset_type": "wds",
    "datasets": [
        {
            "id": "local-wds",
            "path": "/path/to/tar/directory/"
        }
    ],
    "random_crop": true
}
```

### Per-dataset entry keys

| Key | Used By | Default | Description |
|-----|---------|---------|-------------|
| `id` | both | (required) | Identifier |
| `s3_path` | S3 | — | S3 bucket prefix containing tar files (searched recursively) |
| `path` | local wds | — | Local directory containing `.tar` files |
| `profile` | S3 | — | AWS CLI profile name for S3 access |
| `custom_metadata_module` | both | — | Path to custom metadata module |

### Top-level options

| Key | Default | Description |
|-----|---------|-------------|
| `random_crop` | `true` | Random crop position within each audio file |
| `volume_norm` | `false` | Enable LUFS volume normalization |
| `volume_norm_param` | `[-16, 2]` | `[target_lufs, random_gain_range_db]` |
| `strip_silence` | `false` | Strip trailing silence before cropping |
| `remove_silence` | `false` | Remove long internal silence segments and replace with short snippets (mono audio only) |
| `silence_threshold` | `[0.01, 0.5]` | `[energy_threshold, duration_threshold_sec]` for internal silence detection |
| `max_silence_duration` | `0.25` | Maximum allowed silence duration in seconds when removing silence |
| `epoch_steps` | `2000` | Number of steps per epoch (for resampled shards) |
| `resampled_shards` | `true` | Resample shards infinitely across epochs. If `false`, iterates through shards once per epoch |
| `pre_encoded` | `false` | Treat tar contents as `.npy` latents instead of audio files |
| `latent_crop_length` | — | Latent crop length (when `pre_encoded` is `true`) |
| `min_length_sec` | — | Reject samples shorter than this (when `pre_encoded` is `true`) |
| `max_length_sec` | — | Reject samples longer than this (when `pre_encoded` is `true`) |

**Note:** When `pre_encoded` is `true`, tar files should contain `.npy` + `.json` pairs. The JSON must include a `padding_mask` field.

**Note:** A `"text"` field in JSON metadata is automatically mapped to `"prompt"` for conditioning.

# Common Features

## Audio Processing Pipeline

For raw-audio dataset types (`audio_dir`, `s3`/`wds` without `pre_encoded`), audio samples go through the following processing steps in order:

1. **Load and resample** — audio is loaded and resampled to the model's target sample rate
2. **Volume normalization** — if enabled, audio is normalized to a target LUFS loudness with optional random gain augmentation
3. **Silence stripping** — if enabled, trailing silence below -60 dB is removed
4. **Internal silence removal** — (WebDataset only, mono only) long silent segments are replaced with short snippets
5. **Pad/crop** — audio is padded or cropped to `sample_size` with normalized timestamp tracking (`seconds_start`, `seconds_total`)
6. **Silence detection** — pure-silence samples are automatically skipped and a new sample is drawn
7. **Phase flip augmentation** — audio phase is randomly inverted with 50% probability
8. **Channel conversion** — audio is converted to stereo or mono based on the model config

The processing pipeline produces a `padding_mask` in the metadata, which indicates which frames contain valid audio versus padding.

## Weighted Sampling

Any local dataset type (`audio_dir`, `pre_encoded`) supports per-source sampling weights. Set the `weight` key on each dataset entry to control how frequently samples from that source are drawn.

When any weight differs from 1.0, a `WeightedRandomSampler` is created automatically, replacing the default shuffle. This is useful for balancing datasets of different sizes or oversampling important data.

```json
{
    "datasets": [
        { "id": "large_dataset", "path": "/path/a/", "weight": 1.0 },
        { "id": "small_important", "path": "/path/b/", "weight": 5.0 }
    ]
}
```

## File Discovery

Local dataset types support three modes of file discovery:

1. **Recursive directory scan** (default) — all files with supported audio extensions are found recursively under the given `path`
2. **Keyword filtering** — set `keywords` to a list of strings. Only files whose names contain at least one keyword are included
3. **Explicit filelist** — set `filelist_path` to a text file containing one relative filename per line. If not set, a `filelist.txt` file at the root of the dataset directory is used automatically if it exists

# Custom Metadata

Custom metadata modules let you control what conditioning information is passed to the model during training. Set `custom_metadata_module` either at the top level (applies to all dataset sources) or on individual dataset entries.

The module must be a Python file containing a `get_custom_metadata` function:

```python
def get_custom_metadata(info, audio):
    """
    Args:
        info: dict of metadata. Contents depend on dataset type:
            - audio_dir: path, relpath, timestamps, seconds_start, seconds_total,
              padding_mask, sample_rate, load_time
            - pre_encoded: all JSON metadata fields, plus latent_filename, padding_mask
            - s3/wds: all JSON metadata fields from the tar
        audio: the audio tensor (for audio_dir) or latent tensor (for pre_encoded)

    Returns:
        dict merged into the metadata passed to the model
    """
    return {"prompt": info.get("description", "")}
```

### Example config and module

```json
{
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "my_audio",
            "path": "/path/to/audio/dataset/",
            "custom_metadata_module": "/path/to/custom_metadata.py"
        }
    ],
    "random_crop": true
}
```

`custom_metadata.py`:
```python
def get_custom_metadata(info, audio):
    return {"prompt": info["relpath"]}
```

## Special Return Keys

The dictionary returned from `get_custom_metadata` can include special keys that modify the data loading behavior:

| Key | Type | Dataset Types | Effect |
|-----|------|---------------|--------|
| `__reject__` | `bool` | all | If `True`, skip this sample and draw another randomly |
| `__replace__` | `Tensor` or `None` | `pre_encoded` | Replace the latent tensor with a new one |
| `__audio__` | `dict[str, Tensor]` | `audio_dir` | Additional audio tensors to process. Each value goes through the same pad/crop/channel pipeline and is added to the metadata under its key |

Example using `__reject__` and `__audio__`:

```python
def get_custom_metadata(info, audio):
    # Skip very short samples
    if info.get("seconds_total", 0) < 1.0:
        return {"__reject__": True}

    return {
        "prompt": info.get("description", ""),
    }
```

For more information on how conditioning metadata is used by the model, see the [Conditioning documentation](./conditioning.md).

# Implementation Reference

| Module | Description |
|--------|-------------|
| `stable_audio_tools/data/dataset.py` | Dataset classes (`SampleDataset`, `PreEncodedDataset`, `WebDatasetDataLoader`) and the `create_dataloader_from_config` factory function |
| `stable_audio_tools/data/utils.py` | Audio transforms: `PadCrop_Normalized_T`, `VolumeNorm`, `Stereo`, `Mono`, `PhaseFlipper`, `strip_trailing_silence` |
| `stable_audio_tools/configs/dataset_configs/` | Example dataset config JSON files |
