# How to train your own finetune of Stable Audio Open 1.0 or Stable Audio Open Small
I did all of this on a ROG Z13 ACRNM with a mobile 4070 GPU with 8 GiB of VRAM, or an Eluktronics Mech-17 GP2 with a mobile 4090 GPU with 16 GiB of VRAM. I did everything with Python 3.11.9 and CUDA 12.9 on Windows 11 Pro. Ymmv.

## Raw files for your training dataset
* `/rawfiles` is what the example scripts and config files will use for finding raw audio files to pre-encode and use for training the model. You can change this and use another directory if you wish.
* Everything will automatically be converted to a 44100 Hz sampling rate using `torchaudio`. You can perform your own SRC (sample rate conversion) ahead of time if you wish.
* The SAO-small model can handle up to about 11.89 seconds of audio. If your files are longer than this, they will be implicitly truncated. If your files are shorter than this, they will be padded to fit.
* The full SAO-1.0 model can handle up to about 47.55 seconds of audio. If your files are longer than this, they will be implicitly truncated. If your files are shorter than this, they will be padded to fit.
* A good dataset will have at least ~ 5000 files. Less than this, and the model is likely to overfit even with a low number of training steps. More examples to show to the model = the better the results.
* A good dataset will have lots of different sounds. You can intentionally overfit by using only a few sounds, but this will make the model pretty much useless at doing anything other than spitting out the few sounds you fed it, with very little "variation" or "creativity".
* You can feed the model a ton of examples of something specific, like 808 bass one-shots, set all of the conditioning prompts for all of the files in your dataset to "808 bass one-shot" instead of using more complex logic in your custom metadata function(s) to create a distinct prompt for each audio file, and the model will learn many examples of what an "808 bass one-shot" is.
* It's better to feed the model "too many examples" and then refine things later, than it is to feed the model a tiny dataset and end up overfitting.

I strongly recommend using a lossless file format, such as `.WAV` or `.AIFF`, because lossy formats like `.MP3` change the audio in ways that can be disastrous for the audio quality.

Case in point:

<img width="376" height="690" alt="image" src="https://github.com/user-attachments/assets/ad7041cc-6732-4ce5-ae7a-7baf5b09d969" />

### Sample size, sample rate, latent length, audio file duration in seconds, etc.
* These models use a 44100 Hz `sample_rate`. Do not mess with this setting!
* `sample_size` is a confusing misnomer. It would be more accurate if it had been named `audio_input_length_in_samples` or `segment_size_in_samples` or something similar, because that's what it actually is. `sample_size` / 44100 = how long the audio inputs and outputs will be in seconds.
* "Latent" = what your audio gets turned into before the model trains on the data.
* `downsampling_ratio` = the ratio by which your raw audio inputs will be downsampled when they are converted into latents during pre-encoding. This value is 2048 for both SAO-1.0 and SAO-small. Do not mess with this setting!

#### Handy math examples
Latent length of 64 * downsampling ratio of 2048 = 131072 `sample_size`

131072 `sample_size` / `sample_rate` of 44100 = 2.97 seconds

"Segment size" = `sample_size` = latent length * `downsampling_ratio`.

Ergo the SAO-1.0 `model.ckpt` uses a `sample_size` of 2097152, since the model was trained using a latent length of 1024 and a `downsampling_ratio` of 2048.

Thus we arrive at 2097152 samples / 44100 samples per second = 47.55 seconds of audio.

Same values and math for SAO-small, but with a latent length of 256 instead of 1024, ergo 11.89 seconds of audio.

All pre-encoded latents derived from raw audio files will be silence-padded using a mask in order to fit the appropriate latent length for a given model during pre-encoding.

Ergo there is no point in using `latent_crop_length` when pre-encoding raw audio files which have a length in samples which is less than the model's native segment size, e.g., 2097152 with SAO-1.0 and 524288 with SAO-small.

`latent_crop_length` can be used to set the pre-encoded latent sizes to a consistent size. To give a practical example, you could pre-encode your data with a `sample_size` of  2097152 (SAO 1.0 length), then have two separate pre-encoded dataset configs with different `latent_crop_length` (1024 for SAO 1.0, 256 for SAO Small), both reading from the same pre-encoded directory. 

## Pre-encode the latents based on your raw audio files
Technically this is optional, but there is no reason not to pre-encode the latents.
* `pe_dataset_config.json`
  * "PE" stands for "pre-encoding". This file contains instructions for the `DataLoader` to read your raw audio files, such as `.WAV` or `.AIFF` files, which will be used to pre-encode the latents.
* `paths_md_pre_encode.py`
  * "PE" stands for "pre-encoding". This script provides the model with conditioning parameters during training. The only conditioning you need to handle is the `prompt`. The other one or two conditioning parameters, `seconds_start` and `seconds_total`, will be determined by other settings.
* `pre-encode.bat`
  * You can skip over this file and run the `pre_encode.py` command manually with your own settings if you wish.
  * If you run out of memory, lower the `batch_size`. If you still get OOM errors even with the minimum `batch_size` of 1, you probably need to buy a GPU with more VRAM, or you need to run this stuff on a remote hosting platform such as AWS EC2, RunPod, or Google Colab.
  * The example settings should work fine on a GPU with 8 GiB of VRAM. 

## Configure your dataset
* `dataset_config.json`
* `paths_md.py`

## Configure the model
* `/sao_small/base_model_config.json`
  * Use this model config for the SAO-small model.
* `model_config.json`
  * Use this model config for the full SAO-1.0 model.

## Train
`train.bat`

You can skip over this file and run the `training.py` command manually with your own settings if you wish.

If you run out of memory, lower the `batch_size`. If you still get OOM errors even with the minimum `batch_size` of 1, you probably need to buy a GPU with more VRAM, or you need to run this stuff on a remote hosting platform such as AWS EC2, RunPod, or Google Colab.

The example settings for SAO-small should work fine on a GPU with at least 8 GiB of VRAM.

If you want to train the full-sized SAO-1.0 model, you will need at least 24 GiB of VRAM.

### When is training done?
Whenever you feel like stopping.

Listen to the demos and decide when it sounds like the model has learned enough about your dataset, then kill the training process with Ctrl-C.

If you have a small number of files in the dataset, like only 100 .WAV files, then you will probably start to overfit after about 2000 steps. "Overfitting" means that the model is getting to a point where it will basically be hyper-optimized for recreating the exact audio you used for training whenever you generate new outputs during inference using the same or similer prompts that you used during training.

## WARNING: DO NOT MODIFY OR USE THESE FILES
Do not mess with these files:
* `/sao_small/model_config.json`
  * This is the config for the ARC post-trained `model.ckpt` of SAO-small, which you should not attempt to train.
* `/vae_model_config.json`
  * This is the config for the VAE model (the auto-encoder), which you should not mess with unless you know exactly what you are doing and why you are doing it.
 
# HELP!!!
* Static-y whine or drone = you probably used an unwrapped model instead of a wrapped one, or vice versa; or you used `--pretrained-ckpt-path` instead of `--ckpt-path`, or vice versa.
* If you need a pre-compiled wheel for `flash-attention`, I gotchu fam: https://github.com/sskalnik/flash_attn_wheels
* `RuntimeError: Given groups=1, weight of size [128, 2, 7], expected input[1, 64, 645] to have 2 channels, but got 64 channels instead` = you need to make sure `pre_encoded` is set to `True` in the model config JSON file you're using for training.
* `UserWarning: At least one mel filterbank has all zero values. The value for n_mels (128) may be set too high. Or, the value for n_freqs (513) may be set too low.` = You can ignore this.
