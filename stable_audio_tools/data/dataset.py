import importlib
import numpy as np
import io
import json
import os
import posixpath
import random
import re
import subprocess
import time
import torch
import torchaudio
import webdataset as wds

from os import path
from torch import nn
from torchaudio import transforms as T
from typing import Optional, Callable, List

from .utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T, VolumeNorm

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")

# fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list,  # list of allowed file extensions,
    #max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any(
                        [keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words])
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    exts=['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus']
):
    "recursively get a list of audio filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames
        if keywords is not None:
            subfolders, files = keyword_scandir(path, exts, keywords)
        else:
            subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames

def get_latent_filenames(
    paths: list,  # directories in which to search
    extensions=['npy']
):
    "recursively get a list of pre-encoded filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames

        # Check for filelist.txt at the root of the directory
        filelist_path = path + "/filelist.txt"
        if os.path.exists(filelist_path):
            with open(filelist_path, "r") as f:
                files = f.readlines()
                files = [os.path.join(path, file.strip()) for file in files]
                filenames.extend(files)
            continue

        _, files = fast_scandir(path, extensions)
        filenames.extend(files)
    return filenames

class LocalDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn

class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs,
        sample_size=65536, 
        sample_rate=48000, 
        keywords=None, 
        random_crop=True,
        force_channels="stereo"
    ):
        super().__init__()
        self.filenames = []

        self.augs = torch.nn.Sequential(
            PhaseFlipper()
        )

        self.root_paths = []

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)

        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.sr = sample_rate

        self.custom_metadata_fns = {}

        for config in configs:
            self.root_paths.append(config.path)
            self.filenames.extend(get_audio_filenames(config.path, keywords))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = config.custom_metadata_fn

        print(f'Found {len(self.filenames)} files')

    def load_file(self, filename):
        ext = filename.split(".")[-1]

        audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        try:
            start_time = time.time()
            audio = self.load_file(audio_filename)

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)

            # Check for silence
            if is_silence(audio):
                return self[random.randrange(len(self))]

            # Run augmentations on this sample (including random crop)
            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)

            info = {}

            info["path"] = audio_filename

            for root_path in self.root_paths:
                if root_path in audio_filename:
                    info["relpath"] = path.relpath(audio_filename, root_path)

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = padding_mask
            info["sample_rate"] = self.sr

            end_time = time.time()

            info["load_time"] = end_time - start_time

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in audio_filename:
                    custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
                    custom_metadata = custom_metadata_fn(info, audio)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

                # Provide audio inputs as their own dictionary to be merged into info, each audio element will be normalized in the same way as the main audio
                if "__audio__" in info:
                    for audio_key, audio_value in info["__audio__"].items():
                        # Process the audio_value tensor, which should be a torch tensor
                        audio_value, _, _, _, _, _ = self.pad_crop(audio_value)
                        audio_value = audio_value.clamp(-1, 1)
                        if self.encoding is not None:
                            audio_value = self.encoding(audio_value)
                        info[audio_key] = audio_value
                
                    del info["__audio__"]

            return (audio, info)
        except Exception as e:
            print(f'Couldn\'t load file {audio_filename}: {e}')
            return self[random.randrange(len(self))]

class PreEncodedDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs: List[LocalDatasetConfig],
        latent_crop_length=None,
        min_length_sec=None,
        max_length_sec=None,
        random_crop=False,
        latent_extension='npy'
    ):
        super().__init__()
        self.filenames = []

        self.custom_metadata_fns = {}

        self.latent_extension = latent_extension

        for config in configs:
            self.filenames.extend(get_latent_filenames(config.path, [latent_extension]))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = config.custom_metadata_fn

        self.latent_crop_length = latent_crop_length
        self.random_crop = random_crop

        self.min_length_sec = min_length_sec
        self.max_length_sec = max_length_sec

        print(f'Found {len(self.filenames)} files')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        latent_filename = self.filenames[idx]
        try:
            latents = torch.from_numpy(np.load(latent_filename)) # [C, N]

            md_filename = latent_filename.replace(f".{self.latent_extension}", ".json")

            with open(md_filename, "r") as f:
                try:
                    info = json.load(f)
                except:
                    raise Exception(f"Couldn't load metadata file {md_filename}")

            info["latent_filename"] = latent_filename

            if self.latent_crop_length is not None:

                # Get the last index from the padding mask, the index of the last 1 in the sequence
                last_ix = len(info["padding_mask"]) - 1 - info["padding_mask"][::-1].index(1)

                if self.random_crop and last_ix > self.latent_crop_length:
                    start = random.randint(0, last_ix - self.latent_crop_length)
                else:
                    start = 0
                    
                latents = latents[:, start:start+self.latent_crop_length]

                info["padding_mask"] = info["padding_mask"][start:start+self.latent_crop_length]

                info["latent_crop_length"] = self.latent_crop_length
                info["latent_crop_start"] = start

            info["padding_mask"] = [torch.tensor(info["padding_mask"])]

            seconds_total = info["seconds_total"]

            if self.min_length_sec is not None and seconds_total < self.min_length_sec:
                return self[random.randrange(len(self))]

            if self.max_length_sec is not None and seconds_total > self.max_length_sec:
                return self[random.randrange(len(self))]

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in latent_filename:
                    custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
                    custom_metadata = custom_metadata_fn(info, None)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

                if "__replace__" in info and info["__replace__"] is not None:
                    # Replace the latents with the new latents if the custom metadata function returns a new set of latents
                    latents = info["__replace__"]

            info["audio"] = latents

            return (latents, info)
        except Exception as e:
            print(f'Couldn\'t load file {latent_filename}: {e}')
            return self[random.randrange(len(self))]

# S3 code and WDS preprocessing code based on implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

def get_s3_contents(dataset_path, s3_url_prefix=None, filter='', recursive=True, debug=False, profile=None):
    """
    Returns a list of full S3 paths to files in a given S3 bucket and directory path.
    """
    # Ensure dataset_path ends with a trailing slash
    if dataset_path != '' and not dataset_path.endswith('/'):
        dataset_path += '/'
    # Use posixpath to construct the S3 URL path
    bucket_path = posixpath.join(s3_url_prefix or '', dataset_path)
    # Construct the `aws s3 ls` command
    cmd = ['aws', 's3', 'ls', bucket_path]

    if profile is not None:
        cmd.extend(['--profile', profile])

    if recursive:
        # Add the --recursive flag if requested
        cmd.append('--recursive')
    
    # Run the `aws s3 ls` command and capture the output
    run_ls = subprocess.run(cmd, capture_output=True, check=True)
    # Split the output into lines and strip whitespace from each line
    contents = run_ls.stdout.decode('utf-8').split('\n')
    contents = [x.strip() for x in contents if x]
    # Remove the timestamp from lines that begin with a timestamp
    contents = [re.sub(r'^\S+\s+\S+\s+\d+\s+', '', x)
                if re.match(r'^\S+\s+\S+\s+\d+\s+', x) else x for x in contents]
    # Construct a full S3 path for each file in the contents list
    contents = [posixpath.join(s3_url_prefix or '', x)
                for x in contents if not x.endswith('/')]
    # Apply the filter, if specified
    if filter:
        contents = [x for x in contents if filter in x]
    # Remove redundant directory names in the S3 URL
    if recursive:
        # Get the main directory name from the S3 URL
        main_dir = "/".join(bucket_path.split('/')[3:])
        # Remove the redundant directory names from each file path
        contents = [x.replace(f'{main_dir}', '').replace(
            '//', '/') for x in contents]
    # Print debugging information, if requested
    if debug:
        print("contents = \n", contents)
    # Return the list of S3 paths to files
    return contents


def get_all_s3_urls(
    names=[],           # list of all valid [LAION AudioDataset] dataset names
    # list of subsets you want from those datasets, e.g. ['train','valid']
    subsets=[''],
    s3_url_prefix=None,  # prefix for those dataset names
    recursive=True,     # recursively list all tar files in all subdirs
    filter_str='tar',   # only grab files with this substring
    # print debugging info -- note: info displayed likely to change at dev's whims
    debug=False,
    profiles={},        # dictionary of profiles for each item in names, e.g. {'dataset1': 'profile1', 'dataset2': 'profile2'}
):
    "get urls of shards (tar files) for multiple datasets in one s3 bucket"
    urls = []
    for name in names:
        # If s3_url_prefix is not specified, assume the full S3 path is included in each element of the names list
        if s3_url_prefix is None:
            contents_str = name
        else:
            # Construct the S3 path using the s3_url_prefix and the current name value
            contents_str = posixpath.join(s3_url_prefix, name)
        if debug:
            print(f"get_all_s3_urls: {contents_str}:")
        for subset in subsets:
            subset_str = posixpath.join(contents_str, subset)
            if debug:
                print(f"subset_str = {subset_str}")
            # Get the list of tar files in the current subset directory
            profile = profiles.get(name, None)
            tar_list = get_s3_contents(
                subset_str, s3_url_prefix=None, recursive=recursive, filter=filter_str, debug=debug, profile=profile)
            for tar in tar_list:
                # Escape spaces and parentheses in the tar filename for use in the shell command
                tar = tar.replace(" ", "\ ").replace(
                    "(", "\(").replace(")", "\)")
                # Construct the S3 path to the current tar file
                s3_path = posixpath.join(name, subset, tar) + " -"
                # Construct the AWS CLI command to download the current tar file
                if s3_url_prefix is None:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {s3_path}"
                else:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {posixpath.join(s3_url_prefix, s3_path)}"
                if profiles.get(name):
                    request_str += f" --profile {profiles.get(name)}"
                if debug:
                    print("request_str = ", request_str)
                # Add the constructed URL to the list of URLs
                urls.append(request_str)
    return urls


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    print(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True

# get_dbmax and is_silence copied from https://github.com/drscotthawley/aeiou/blob/main/aeiou/core.py under Apache 2.0 License
# License can be found in LICENSES/LICENSE_AEIOU.txt
def get_dbmax(
    audio,       # torch tensor of (multichannel) audio
    ):
    "finds the loudest value in the entire clip and puts that into dB (full scale)"
    return 20*torch.log10(torch.flatten(audio.abs()).max()).cpu().numpy()

def is_silence(
    audio,       # torch tensor of (multichannel) audio
    thresh=-60,  # threshold in dB below which we declare to be silence
    ):
    "checks if entire clip is 'silence' below some dB threshold"
    dBmax = get_dbmax(audio)
    return dBmax < thresh

def is_valid_sample(sample):
    has_json = "json" in sample
    has_audio = "audio" in sample
    is_pre_encoded = sample.get("__pre_encoded__", False)
    is_silent = (not is_pre_encoded) and is_silence(sample["audio"])
    is_rejected = "__reject__" in sample["json"] and sample["json"]["__reject__"]

    return has_json and has_audio and not is_silent and not is_rejected


def remove_long_silence(audio, sample_rate, silence_threshold=[0.01, 0.5], max_silence_duration=0.25):
    """
    Removes silence longer than max_silence_duration and replaces it with a short silence.

    :param audio: torch tensor of shape [1, T]
    :param sample_rate: Sampling rate of the audio
    :param silence_threshold: List with [silence_energy_threshold, silence_duration_threshold] to consider a segment as silence
    :param max_silence_duration: Maximum allowed silence duration in seconds
    :return: Processed audio tensor
    """
    
    silence_energy_threshold, silence_duration_threshold = silence_threshold

    max_silence_samples = int(max_silence_duration * sample_rate)
    tiny_silence_samples = int(silence_duration_threshold * sample_rate)
    
    # Flatten the audio tensor
    audio = audio.flatten()
    
    # Detect silent segments
    silence_mask = torch.abs(audio) < silence_energy_threshold
    silence_mask_diff = torch.diff(silence_mask.int())
    
    # Find indices where silence starts and ends
    silence_starts = torch.where(silence_mask_diff == 1)[0] + 1
    silence_ends = torch.where(silence_mask_diff == -1)[0] + 1

    # Handle the case where the tensor starts or ends with silence
    if silence_mask[0]:
        silence_starts = torch.cat((torch.tensor([0], device=silence_starts.device), silence_starts))
    if silence_mask[-1]:
        silence_ends = torch.cat((silence_ends, torch.tensor([len(audio)], device=silence_ends.device)))

    processed_audio = []
    prev_end = 0
    for start, end in zip(silence_starts, silence_ends):
        # Add non-silence segment
        processed_audio.append(audio[prev_end:start])
        
        silence_segment = audio[start:end]
        if len(silence_segment) > max_silence_samples:
            # Replace long silence with a random segment of 0-0.5s silence
            if len(silence_segment) > tiny_silence_samples:
                start_idx = random.randint(0, len(silence_segment) - tiny_silence_samples)
                processed_audio.append(silence_segment[start_idx:start_idx + tiny_silence_samples])
            else:
                processed_audio.append(silence_segment[:tiny_silence_samples])
        else:
            # Keep the silence segment as is
            processed_audio.append(silence_segment)

        prev_end = end
    
    # Add the last non-silence segment if there is any
    if prev_end < len(audio):
        processed_audio.append(audio[prev_end:])
    
    # Concatenate all processed segments back into a single tensor
    processed_audio_tensor = torch.cat(processed_audio).unsqueeze(0)
    
    return processed_audio_tensor


def is_silence_audio(audio, silence_threshold=0.01, max_silence_ratio=0.3):
    # Calculate the ratio of silent frames in the audio sample
    silence_frames = torch.sum(audio.abs() < silence_threshold, dim=1)
    total_frames = audio.size(1)
    silence_ratio_per_channel = silence_frames / total_frames

    if torch.any(silence_ratio_per_channel > max_silence_ratio).item():
        # Save the tensor to an audio file
        output_path = f'rejected_audios/rejected_{silence_ratio_per_channel.item()}.wav'
        torchaudio.save(output_path, audio, 16000)
        print(f'Rejected: {silence_ratio_per_channel}')
    # Check if any channel exceeds the max silence ratio
    return torch.any(silence_ratio_per_channel > max_silence_ratio).item()

class S3DatasetConfig:
    def __init__(
        self,
        id: str,
        s3_path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        profile: Optional[str] = None,
    ):
        self.id = id
        self.path = s3_path
        self.custom_metadata_fn = custom_metadata_fn
        self.profile = profile
        self.urls = []

    def load_data_urls(self):
        self.urls = get_all_s3_urls(
            names=[self.path],
            s3_url_prefix=None,
            recursive=True,
            profiles={self.path: self.profile} if self.profile else {},
        )

        return self.urls

class LocalWebDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        profile: Optional[str] = None,
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.urls = []

    def load_data_urls(self):

        self.urls = fast_scandir(self.path, ["tar"])[1]

        return self.urls

def audio_decoder(key, value):
    # Get file extension from key
    ext = key.split(".")[-1]

    if ext in AUDIO_KEYS:
        return torchaudio.load(io.BytesIO(value))
    else:
        return None

def npy_decoder(key, value):
    # Get file extension from key
    ext = key.split(".")[-1]

    if ext == "npy":
        return np.lib.format.read_array(io.BytesIO(value))
    else:
        return None

def collation_fn(samples):
        batched = list(zip(*samples))
        result = []
        for b in batched:
            if isinstance(b[0], (int, float)):
                b = np.array(b)
            elif isinstance(b[0], torch.Tensor):
                b = torch.stack(b)
            elif isinstance(b[0], np.ndarray):
                b = np.array(b)
            else:
                b = b
            result.append(b)
        return result

class WebDatasetDataLoader():
    def __init__(
        self,
        datasets: List[S3DatasetConfig],
        batch_size,
        sample_size,
        sample_rate=48000,
        num_workers=8,
        epoch_steps=1000,
        random_crop=True,
        force_channels="stereo",
        augment_phase=True,
        remove_silence=True,
        silence_threshold=[0.01, 0.5],
        max_silence_duration=0.2,
        volume_norm=False,
        volume_norm_param=(-16, 2),
        pre_encoded=False,
        latent_crop_length=None,
        resampled_shards=True,
        **data_loader_kwargs
    ):

        self.datasets = datasets

        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.force_channels = force_channels
        self.augment_phase = augment_phase
        self.pre_encoded = pre_encoded
        self.latent_crop_length = latent_crop_length
        self.volume_norm = volume_norm
        self.volume_norm_param = volume_norm_param
        self.remove_silence = remove_silence
        self.silence_threshold = silence_threshold
        self.max_silence_duration = max_silence_duration

        urls = [dataset.load_data_urls() for dataset in datasets]

        # Flatten the list of lists of URLs
        urls = [url for dataset_urls in urls for url in dataset_urls]

        # Shuffle the urls
        random.shuffle(urls)

        self.dataset = wds.DataPipeline(
            wds.ResampledShards(urls) if resampled_shards else wds.SimpleShardList(urls),
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode(audio_decoder, handler=log_and_continue) if not self.pre_encoded else wds.decode(npy_decoder, handler=log_and_continue),
            wds.map(self.wds_preprocess, handler=log_and_continue),
            #wds.map(self.wds_preprocess),
            wds.select(is_valid_sample),
            wds.to_tuple("audio", "json", handler=log_and_continue),
            #wds.shuffle(bufsize=1000, initial=5000),
            wds.batched(batch_size, partial=False, collation_fn=collation_fn),
        )

        if resampled_shards:
            self.dataset = self.dataset.with_epoch(epoch_steps//num_workers if num_workers > 0 else epoch_steps)

        def worker_init_fn(worker_id):
            torch.multiprocessing.set_sharing_strategy('file_system')

        self.data_loader = wds.WebLoader(self.dataset, num_workers=num_workers, worker_init_fn=worker_init_fn, **data_loader_kwargs)

    def wds_preprocess(self, sample):

        if self.pre_encoded:
            audio = torch.from_numpy(sample["npy"])
            del sample["npy"]
            sample["__pre_encoded__"] = True

            padding_mask = sample["json"]["padding_mask"]
            if self.latent_crop_length is not None:

                # Get the last index from the padding mask, the index of the last 1 in the sequence
                last_ix = len(padding_mask) - 1 - padding_mask[::-1].index(1)

                if self.random_crop and last_ix > self.latent_crop_length:
                    start = random.randint(0, last_ix - self.latent_crop_length)
                else:
                    start = 0
                    
                audio = audio[:, start:start+self.latent_crop_length]

                padding_mask = padding_mask[start:start+self.latent_crop_length]

            sample["json"]["padding_mask"] = torch.tensor(padding_mask)
        else:
            found_key, rewrite_key = '', ''
            for k, v in sample.items():  # print the all entries in dict
                for akey in AUDIO_KEYS:
                    if k.endswith(akey):
                        # to rename long/weird key with its simpler counterpart
                        found_key, rewrite_key = k, akey
                        break
                if '' != found_key:
                    break
            if '' == found_key:  # got no audio!
                return None  # try returning None to tell WebDataset to skip this one

            audio, in_sr = sample[found_key]
            if in_sr != self.sample_rate:
                resample_tf = T.Resample(in_sr, self.sample_rate)
                audio = resample_tf(audio)

                    # Replace the long silence by the short for the mono audios
            if audio.shape[0] == 1 and self.remove_silence:
                audio = remove_long_silence(audio, self.sample_rate, self.silence_threshold, self.max_silence_duration)

            if self.sample_size is not None:
                # Pad/crop and get the relative timestamp
                pad_crop = PadCrop_Normalized_T(
                    self.sample_size, randomize=self.random_crop, sample_rate=self.sample_rate)
                audio, t_start, t_end, seconds_start, seconds_total, padding_mask = pad_crop(
                    audio)
                sample["json"]["seconds_start"] = seconds_start
                sample["json"]["seconds_total"] = seconds_total
                sample["json"]["padding_mask"] = padding_mask
            else:
                t_start, t_end = 0, 1

            # Check if audio is length zero, initialize to a single zero if so
            if audio.shape[-1] == 0:
                audio = torch.zeros(1, 1)

            # Make the audio stereo and augment by randomly inverting phase
            augs = torch.nn.Sequential(
                Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
                Mono() if self.force_channels == "mono" else torch.nn.Identity(),
                VolumeNorm(self.volume_norm_param, self.sample_rate) if self.volume_norm else torch.nn.Identity(),
                PhaseFlipper() if self.augment_phase else torch.nn.Identity()
            )

            audio = augs(audio)

            sample["json"]["timestamps"] = (t_start, t_end)

            if found_key != rewrite_key:   # rename long/weird key with its simpler counterpart
                del sample[found_key]

        if "text" in sample["json"]:
            sample["json"]["prompt"] = sample["json"]["text"]

        # Check for custom metadata functions
        for dataset in self.datasets:
            if dataset.custom_metadata_fn is None:
                continue
        
            if dataset.path in sample["__url__"]:
                custom_metadata = dataset.custom_metadata_fn(sample["json"], audio)
                sample["json"].update(custom_metadata)

        sample["audio"] = audio
        # Add audio to the metadata as well for conditioning
        sample["json"]["audio"] = audio
        
        return sample

def create_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate, audio_channels=2, num_workers=4, shuffle = True):

    dataset_type = dataset_config.get("dataset_type", None)

    assert dataset_type is not None, "Dataset type must be specified in dataset config"

    if audio_channels == 1:
        force_channels = "mono"
    else:
        force_channels = "stereo"

    if dataset_type == "audio_dir":

        audio_dir_configs = dataset_config.get("datasets", None)

        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        configs = []

        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"

            custom_metadata_fn = None
            custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                custom_metadata_fn = metadata_module.get_custom_metadata

            configs.append(
                LocalDatasetConfig(
                    id=audio_dir_config["id"],
                    path=audio_dir_path,
                    custom_metadata_fn=custom_metadata_fn
                )
            )

        train_set = SampleDataset(
            configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", True),
            force_channels=force_channels
        )

        return torch.utils.data.DataLoader(train_set, batch_size, shuffle=shuffle,
                                num_workers=num_workers, persistent_workers=True, pin_memory=True, drop_last=dataset_config.get("drop_last", True), collate_fn=collation_fn)

    elif dataset_type == "pre_encoded":

        pre_encoded_dir_configs = dataset_config.get("datasets", None)

        assert pre_encoded_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        latent_crop_length = dataset_config.get("latent_crop_length", None)
        min_length_sec = dataset_config.get("min_length_sec", None)
        max_length_sec = dataset_config.get("max_length_sec", None)
        random_crop = dataset_config.get("random_crop", False)

        configs = []

        for pre_encoded_dir_config in pre_encoded_dir_configs:
            pre_encoded_dir_path = pre_encoded_dir_config.get("path", None)
            assert pre_encoded_dir_path is not None, "Path must be set for local audio directory configuration"
            

            custom_metadata_fn = None
            custom_metadata_module_path = pre_encoded_dir_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                custom_metadata_fn = metadata_module.get_custom_metadata

            configs.append(
                LocalDatasetConfig(
                    id=pre_encoded_dir_config["id"],
                    path=pre_encoded_dir_path,
                    custom_metadata_fn=custom_metadata_fn
                )
            )

        latent_extension = dataset_config.get("latent_extension", 'npy')

        train_set = PreEncodedDataset(
            configs, 
            latent_crop_length=latent_crop_length, 
            min_length_sec=min_length_sec, 
            max_length_sec=max_length_sec, 
            random_crop=random_crop, 
            latent_extension=latent_extension
        )

        return torch.utils.data.DataLoader(train_set, batch_size, shuffle=shuffle,
                                num_workers=num_workers, persistent_workers=True, pin_memory=True, drop_last=dataset_config.get("drop_last", True), collate_fn=collation_fn)

    elif dataset_type in ["s3", "wds"]: # Support "s3" type for backwards compatibility
        wds_configs = []

        for wds_config in dataset_config["datasets"]:

            custom_metadata_fn = None
            custom_metadata_module_path = wds_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)                

                custom_metadata_fn = metadata_module.get_custom_metadata

            if "s3_path" in wds_config:

                wds_configs.append(
                    S3DatasetConfig(
                        id=wds_config["id"],
                        s3_path=wds_config["s3_path"],
                        custom_metadata_fn=custom_metadata_fn,
                        profile=wds_config.get("profile", None),
                    )
                )
            
            elif "path" in wds_config:
                    
                    wds_configs.append(
                        LocalWebDatasetConfig(
                            id=wds_config["id"],
                            path=wds_config["path"],
                            custom_metadata_fn=custom_metadata_fn
                        )
                    )

        return WebDatasetDataLoader(
            wds_configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            batch_size=batch_size,
            remove_silence=dataset_config.get("remove_silence", False),
            silence_threshold=dataset_config.get("silence_threshold", [0.01, 0.5]),
            max_silence_duration=dataset_config.get("max_silence_duration", 0.25),
            random_crop=dataset_config.get("random_crop", True),
            volume_norm=dataset_config.get("volume_norm", False),
            volume_norm_param=dataset_config.get("volume_norm_param", [-16, 2]),
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            force_channels=force_channels,
            epoch_steps=dataset_config.get("epoch_steps", 2000),
            pre_encoded=dataset_config.get("pre_encoded", False),
            latent_crop_length=dataset_config.get("latent_crop_length", None),
            resampled_shards=dataset_config.get("resampled_shards", True)
        ).data_loader
