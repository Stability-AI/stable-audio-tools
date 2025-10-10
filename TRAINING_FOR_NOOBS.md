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

**Make sure that the training process has generated a checkpoint before you stop the training process!** You can always change the `--checkpoint-every` argument to a smaller value if you want to generate a checkpoint more quickly.

If you have a small number of files in the dataset, like only 100 .WAV files, then you will probably start to overfit after about 2000 steps. "Overfitting" means that the model is getting to a point where it will basically be hyper-optimized for recreating the exact audio you used for training whenever you generate new outputs during inference using the same or similer prompts that you used during training.

## WARNING: DO NOT MODIFY OR USE THESE FILES
Do not mess with these files:
* `/sao_small/model_config.json`
  * This is the config for the ARC post-trained `model.ckpt` of SAO-small, which you should not attempt to train.
* `/vae_model_config.json`
  * This is the config for the VAE model (the auto-encoder), which you should not mess with unless you know exactly what you are doing and why you are doing it.

# Terminology for noobs
* Epoch = one pass over all files in the dataset. If you have 1280 files in the dataset, it will take 1 Epoch to "show all 1280 files to the model".
* Batch = one chunk of the dataset. If your Batch Size is 32, it will take 1280 / 32 = 40 Steps to complete 1 Epoch.
* Step = one iteration of the training process, in which 1 Batch of latents (derived from your training dataset files) will be "shown to the model so it can learn from them". If you have 1280 files in your dataset, and you use a Batch Size of 8, it will take 1280 / 8 = 160 Steps to complete 1 Epoch.
* Gradient Accumulation = increases the effective Batch Size when your hardware can't handle a larger Batch Size. Effective Batch Size = Gradient Accumulation * Batch Size. Gradient Accumulation 4 * Batch Size 8 = Effective Batch Size of 32. Instead of actually "showing 32 latents or files to the model", you end up "showing 8 latents or files to the model" 4 times. This results in lower VRAM usage, but longer training times. It's usually better to just use the largest Batch Size you can without running out of VRAM, and not using Gradient Accumulation unless you have no other option.
* Learning Rate = how much the model learns from each Batch (in each Step), as a function of time. The Learning Rate could be constant, or it could change over time. This is usually expressed as a value between 0 and 1, with 0 meaning "learn nothing" and 1 meaning "study what you are exposed to in 100% depth, and let this experience influence you to the utmost".
  * Learning rate too low = takes longer to train, model seems to not have learned anything (underfits).
  * Learning rate too high = takes far less time to train than you probably expected, and the model probably overfits within 1 Epoch.
  * Small dataset = try a larger value for Learning Rate, such as 1e-2 (0.01). Not many examples to learn from, but you learn a lot from each example.
  * Large dataset = try a smaller value for Learning Rate, such as 5e-4 (0.0005). Learn just a bit from each example, but have a lot of examples.
* Weight decay = a regularization technique which limits how much each weight can be influenced by the training data. This helps mitigate the risk of overfitting. Much more info here: https://d2l.ai/chapter_linear-regression/weight-decay.html
* Learning Rate Optimizers = algorithms for optimizing the Learning Rate. Usually combined with a Learning Rate Scheduler. A typical choice is `AdamW`. `AdamW8bit` is a viable option for saving VRAM. Many other options exist, like Lion and Prodigy, but you should stick with `AdamW` or another `Adam` derivative unless you want to navigate unexplored territory and perform experiments.
* Learning Rate Schedulers = algorithms for changing the Learning Rate over time. https://machinelearningmastery.com/a-gentle-introduction-to-learning-rate-schedulers/ NOT TO BE CONFUSED WITH NOISE SCHEDULERS, AKA SAMPLERS! Stable Audio Open uses a custom `InverseLR` Scheduler. Another good option is `CosineAnnealing`.
* Noise Schedulers, aka Samplers = algorithms for adding or removing noise: https://huggingface.co/docs/diffusers/en/api/schedulers/overview https://civitai.com/articles/7484/understanding-stable-diffusion-samplers-beyond-image-comparisons

## What values should I use?
* You should train for at least 1 Epoch, or else the model won't "see" all of your dataset.
  * Too many Epochs (and similarly, too many total Steps) = the model is likely to overfit. Imagine someone going to "normal" school up to the 4th grade, and then being sent to a specialized school where they only learned about how to play modern jazz trumpet: they'd probably not be very good at many "normal" tasks, while excelling at modern jazz trumpet, and they'd be likely to interpret everything they experienced after graduation in the context of modern jazz trumpet.
  * Too few Epochs (and similarly, too few total Steps) = the model is likely to underfit. Imagine someone going to "normal" school up to the 4th grade, and then being sent to a specialized school where they only learned about how to play modern jazz trumpet, but then you pull them out of school after one week: they'd probably not suffer from "forgetting" everything from "normal" school, but they'd also have learned so little about modern jazz trumpet that they might not be much better than their peers who never studied modern jazz trumpet.
* You should use the largest Batch Size you can fit into VRAM, as a general rule.
  * Try to not use extremely small Batch Size values, such as 1, because the model is more likely to learn well from larger Batch Sizes. Consider a Batch Size of 8 if you're just starting out, then increase the Batch Size until you get an OOM (out of memory) error, then decrease the Batch Size until you no longer get OOM errors.
  * Do not use a Batch Size of 1 if at all possible! It is much better to have a minimum Batch Size of 2.
  * Try to use only Batch Size, and to not use Gradient Accumulation, whenever feasible.
* Some Optimizer + Scheduler combinations can figure out the appropriate Learning Rate for you. Even better: some Optimizer + Scheduler combinations can figure out the appropriate Learning Rate and the best way to adjust the Learning Rate over time, so you don't have a constant Learning Rate.

### I NEED SPECIFIC MAGICAL NUMBERS!!!
Training an AI/ML model is as much of an art as it is a science. Each scenario is unique. You will have to experiment in order to figure out whether training SAO-small on 500 drum one-shots for 2 Epochs with a Batch Size of 8 and a Learning Rate of 5e-3 (0.005) produces better results than training SAO-small on the same 500 drum one-shots for 20 Epochs with a Batch Size of 32 and a Learning Rate of 1e-5 (0.00001).

# HELP!!!
* Static-y whine or drone = you probably used an unwrapped model instead of a wrapped one, or vice versa; or you used `--pretrained-ckpt-path` instead of `--ckpt-path`, or vice versa.
* If you need a pre-compiled wheel for `flash-attention`, I gotchu fam: https://github.com/sskalnik/flash_attn_wheels
* `RuntimeError: Given groups=1, weight of size [128, 2, 7], expected input[1, 64, 645] to have 2 channels, but got 64 channels instead` = you need to make sure `pre_encoded` is set to `True` in the model config JSON file you're using for training.
* `UserWarning: At least one mel filterbank has all zero values. The value for n_mels (128) may be set too high. Or, the value for n_freqs (513) may be set too low.` = You can ignore this.
