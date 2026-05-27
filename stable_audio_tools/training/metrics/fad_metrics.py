import gc
import os
import requests
from tqdm import tqdm
import torch
import torchaudio
import numpy as np
import laion_clap
from packaging import version
import transformers
import glob
from scipy import linalg
import pandas as pd

import pytorch_lightning as pl
from stable_audio_tools.inference.sampling import sample_diffusion
import random

def calculate_embd_statistics(embd_lst):
    if isinstance(embd_lst, list):
        embd_lst = np.array(embd_lst)
    mu = np.mean(embd_lst, axis=0)
    sigma = np.cov(embd_lst, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    Adapted from: https://github.com/gudgud96/frechet-audio-distance/blob/main/frechet_audio_distance/fad.py

    Numpy implementation of the Frechet Distance.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Params:
    -- mu1: Embedding's mean statistics for generated samples.
    -- mu2: Embedding's mean statistics for reference samples.
    -- sigma1: Covariance matrix over embeddings for generated samples.
    -- sigma2: Covariance matrix over embeddings for reference samples.
    Returns:
    --  Fréchet Distance (or NaN if inputs are invalid).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # Check for NaN in inputs
    if not np.isfinite(mu1).all() or not np.isfinite(mu2).all():
        print('Warning: NaN/Inf in mean vectors, returning NaN for Frechet distance')
        return float('nan')
    if not np.isfinite(sigma1).all() or not np.isfinite(sigma2).all():
        print('Warning: NaN/Inf in covariance matrices, returning NaN for Frechet distance')
        return float('nan')

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
            'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        max_imag = np.max(np.abs(covmean.imag))
        if max_imag > 1e-3:
            print(f'Warning: Frechet distance calculation has significant imaginary component ({max_imag:.6f}), taking real part')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    # Ensure result is real (can be complex due to numerical issues)
    if np.iscomplexobj(fd):
        print(f'Warning: Frechet distance result is complex ({fd}), taking real part')
        fd = fd.real

    # Final NaN check
    if not np.isfinite(fd):
        print(f'Warning: Frechet distance result is not finite ({fd})')

    return float(fd)

# following documentation from https://github.com/LAION-AI/CLAP
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def int16_to_float32_torch(x):
    """Torch version of int16_to_float32 for GPU tensors."""
    return (x / 32767.0).float()

def float32_to_int16_torch(x):
    """Torch version of float32_to_int16 for GPU tensors."""
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).to(torch.int16)

def load_state_dict(checkpoint_path: str, map_location="cpu", skip_params=True):
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # removing position_ids to maintain compatibility with latest transformers update        
        if version.parse(transformers.__version__) >= version.parse("4.31.0") and "text_branch.embeddings.position_ids" in state_dict:
            del state_dict["text_branch.embeddings.position_ids"]

    return state_dict

def load_clap_model(clap_model='630k-audioset-fusion-best.pt', device='cuda'):
    """
    Load and return a CLAP model.

    Select one of the following clap models from https://github.com/LAION-AI/CLAP:
        - music_speech_audioset_epoch_15_esc_89.98.pt (used by musicgen)
        - music_audioset_epoch_15_esc_90.14.pt
        - music_speech_epoch_15_esc_89.25.pt
        - 630k-audioset-fusion-best.pt (our default, with "fusion" to handle longer inputs)
    """
    clap_configs = {
        'music_speech_audioset_epoch_15_esc_89.98.pt': {
            'url': 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt',
            'enable_fusion': False, 'amodel': 'HTSAT-base',
        },
        'music_audioset_epoch_15_esc_90.14.pt': {
            'url': 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt',
            'enable_fusion': False, 'amodel': 'HTSAT-base',
        },
        'music_speech_epoch_15_esc_89.25.pt': {
            'url': 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt',
            'enable_fusion': False, 'amodel': 'HTSAT-base',
        },
        '630k-audioset-fusion-best.pt': {
            'url': 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt',
            'enable_fusion': True, 'amodel': 'HTSAT-tiny',
        },
    }

    if clap_model not in clap_configs:
        raise ValueError(f'clap_model not implemented: {clap_model}')

    cfg = clap_configs[clap_model]
    url = cfg['url']
    clap_path = f'load/clap_score/{clap_model}'

    # File lock around CLAP_Module construction (downloads RoBERTa tokenizer
    # from HF Hub) and weights download — prevents multi-rank network races.
    from filelock import FileLock
    lock_path = os.path.join('load', 'clap_score', 'clap_init.lock')
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    with FileLock(lock_path):
        model = laion_clap.CLAP_Module(
            enable_fusion=cfg['enable_fusion'], amodel=cfg.get('amodel', 'HTSAT-tiny'), device=device,
        )

        if not os.path.exists(clap_path):
            print('Downloading ', clap_model, '...')
            temp_path = clap_path + '.tmp'
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(temp_path, 'wb') as file:
                with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
                    for data in response.iter_content(chunk_size=8192):
                        file.write(data)
                        progress_bar.update(len(data))
            # Atomic rename to avoid partial file reads
            os.rename(temp_path, clap_path)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # fixing CLAP-LION issue, see: https://github.com/LAION-AI/CLAP/issues/118
    pkg = load_state_dict(clap_path)
    pkg.pop('text_branch.embeddings.position_ids', None)
    model.model.load_state_dict(pkg)
    model.eval()
    
    return model


# ============================================================================
# PANNs CNN14 Model (from https://github.com/qiuqiangkong/audioset_tagging_cnn)
# Recommended for FAD per ICASSP 2024: "Adapting Frechet Audio Distance for Generative Music Evaluation"
# ============================================================================

def _init_layer(layer):
    torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)

def _init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class _ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        _init_layer(self.conv1)
        _init_layer(self.conv2)
        _init_bn(self.bn1)
        _init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = torch.relu_(self.bn1(self.conv1(x)))
        x = torch.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'avg':
            x = torch.nn.functional.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = torch.nn.functional.max_pool2d(x, kernel_size=pool_size)
        return x

class Cnn14(torch.nn.Module):
    """PANNs CNN14 model for audio embeddings."""
    def __init__(self, sample_rate=32000):
        super().__init__()
        self.bn0 = torch.nn.BatchNorm2d(64)
        self.conv_block1 = _ConvBlock(1, 64)
        self.conv_block2 = _ConvBlock(64, 128)
        self.conv_block3 = _ConvBlock(128, 256)
        self.conv_block4 = _ConvBlock(256, 512)
        self.conv_block5 = _ConvBlock(512, 1024)
        self.conv_block6 = _ConvBlock(1024, 2048)
        self.fc1 = torch.nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = torch.nn.Linear(2048, 527, bias=True)
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=320,
            n_mels=64, f_min=50, f_max=14000, power=2.0
        )
        _init_bn(self.bn0)
        _init_layer(self.fc1)
        _init_layer(self.fc_audioset)

    def forward(self, x):
        # x: (batch, samples)
        x = self.mel_extractor(x)  # (batch, 64, time)
        x = (x + 1e-10).log()
        x = x.unsqueeze(1).transpose(2, 3)  # (batch, 1, time, 64)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        x = torch.nn.functional.dropout(self.conv_block1(x, pool_size=(2, 2), pool_type='avg'), p=0.2, training=self.training)
        x = torch.nn.functional.dropout(self.conv_block2(x, pool_size=(2, 2), pool_type='avg'), p=0.2, training=self.training)
        x = torch.nn.functional.dropout(self.conv_block3(x, pool_size=(2, 2), pool_type='avg'), p=0.2, training=self.training)
        x = torch.nn.functional.dropout(self.conv_block4(x, pool_size=(2, 2), pool_type='avg'), p=0.2, training=self.training)
        x = torch.nn.functional.dropout(self.conv_block5(x, pool_size=(2, 2), pool_type='avg'), p=0.2, training=self.training)
        x = torch.nn.functional.dropout(self.conv_block6(x, pool_size=(1, 1), pool_type='avg'), p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        return torch.relu_(self.fc1(x))  # (batch, 2048)


PANNS_CONFIG = {'sample_rate': 32000, 'embedding_dim': 2048, 'window_seconds': 10.0,
                'url': 'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth',
                'path': 'load/panns/Cnn14_mAP=0.431.pth'}

VGGISH_CONFIG = {'sample_rate': 16000, 'embedding_dim': 128, 'window_seconds': 1.0}

# OpenL3 supports 48kHz which is ideal for high-quality audio evaluation
# Using 512-dim embeddings with music content type
OPENL3_CONFIG = {'sample_rate': 48000, 'embedding_dim': 512, 'window_seconds': 1.0,
                 'content_type': 'music', 'input_repr': 'mel256'}

# Latent embeddings use the pretransform (VAE) encoder
# Embedding dim = io_channels of the pretransform (typically 256)
# This computes Frechet Latent Distance (FLD) in the diffusion model's latent space
# window_seconds controls chunking (like other embedding types) for comparable FAD statistics
LATENT_CONFIG = {'window_seconds': 1.0}  # 1-second windows, matching VGGish/OpenL3


def load_panns_model(device='cuda'):
    """Load PANNs CNN14 model."""
    # Handle distributed downloading - only rank 0 downloads to avoid race conditions
    is_distributed = torch.distributed.is_initialized()
    is_rank_zero = not is_distributed or torch.distributed.get_rank() == 0

    if not os.path.exists(PANNS_CONFIG['path']):
        if is_rank_zero:
            print('Downloading PANNs CNN14 checkpoint...')
            os.makedirs(os.path.dirname(PANNS_CONFIG['path']), exist_ok=True)
            temp_path = PANNS_CONFIG['path'] + '.tmp'
            response = requests.get(PANNS_CONFIG['url'], stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(temp_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for data in response.iter_content(chunk_size=8192):
                        f.write(data)
                        pbar.update(len(data))
            # Atomic rename to avoid partial file reads
            os.rename(temp_path, PANNS_CONFIG['path'])

        if is_distributed:
            torch.distributed.barrier()

    model = Cnn14(sample_rate=PANNS_CONFIG['sample_rate'])
    checkpoint = torch.load(PANNS_CONFIG['path'], map_location='cpu', weights_only=False)
    # Use strict=False because checkpoint has custom mel extractor keys (spectrogram_extractor, logmel_extractor)
    # but we use torchaudio's MelSpectrogram instead
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    return model


def load_vggish_model(device='cuda'):
    """Load VGGish model via torchaudio or torchvggish. Returns (model, input_processor, backend_type)."""
    try:
        from torchaudio.prototype.pipelines import VGGISH
        model = VGGISH.get_model().to(device)
        model.eval()
        # VGGish requires mel spectrogram input - get the input processor
        input_processor = VGGISH.get_input_processor()
        return model, input_processor, 'torchaudio'
    except (ImportError, AttributeError):
        pass
    try:
        import torchvggish
        model = torchvggish.vggish()
        model.to(device)
        model.eval()
        return model, None, 'torchvggish'
    except ImportError:
        raise ImportError("VGGish requires torchaudio nightly or torchvggish. Install: pip install torchvggish")


def load_openl3_model(device='cuda', content_type='music', input_repr='mel256', embedding_size=512):
    """
    Load OpenL3 model for audio embeddings.

    Args:
        device: Device to load model on
        content_type: 'music' or 'env' (environment sounds)
        input_repr: 'mel128' or 'mel256'
        embedding_size: 512 or 6144

    Returns:
        (model, sample_rate) tuple
    """
    try:
        import openl3
        # openl3 uses TensorFlow by default, but we want PyTorch
        raise ImportError("Using torchopenl3 instead of openl3")
    except ImportError:
        pass

    try:
        import torchopenl3
        model = torchopenl3.core.load_audio_embedding_model(
            content_type=content_type,
            input_repr=input_repr,
            embedding_size=embedding_size
        )
        model = model.to(device)
        model.eval()
        return model, OPENL3_CONFIG['sample_rate']
    except ImportError:
        raise ImportError("OpenL3 requires torchopenl3. Install: pip install torchopenl3")


def extract_embeddings(
    id2audio,
    embedding_types=('clap',),
    sample_rate=44100,
    show_progress=True,
    device='cuda',
    # CLAP-specific args
    id2text=None,
    clap_model='630k-audioset-fusion-best.pt'
):
    """
    Extract audio embeddings using any combination of CLAP, PANNs, and VGGish.

    Args:
        id2audio: dict mapping id to audio tensor on GPU (shape: C, T)
        embedding_types: tuple/list of types to extract, e.g. ('clap', 'panns', 'vggish')
        sample_rate: Input audio sample rate
        show_progress: Whether to show progress bar
        device: Device for computation
        id2text: (CLAP only) dict mapping id to text prompt
        clap_model: (CLAP only) CLAP model name

    Returns:
        dict mapping embedding_type to dict with keys:
            'score_sum': Sum of scores (CLAP only, else None)
            'score_count': Count of scores (CLAP only, else None)
            'embeddings': numpy array (M, embedding_dim)
    """
    embedding_types = [t.lower() for t in embedding_types]
    results = {}

    # Load all requested models
    models = {}
    configs = {}
    resamplers = {}

    for emb_type in embedding_types:
        if emb_type == 'clap':
            models['clap'] = load_clap_model(clap_model, device=device)
            configs['clap'] = {'sr': 48000, 'window_sec': 10.0, 'emb_dim': 512}
            # Get text embeddings for CLAP score
            text_emb = {}
            if id2text is not None:
                batch_size = 64
                for i in range(0, len(id2text), batch_size):
                    batch_ids = list(id2text.keys())[i:i+batch_size]
                    batch_texts = [id2text[bid] for bid in batch_ids]
                    with torch.no_grad():
                        embeddings = models['clap'].get_text_embedding(batch_texts, use_tensor=True)
                    for bid, emb in zip(batch_ids, embeddings):
                        text_emb[bid] = emb
                configs['clap']['text_emb'] = text_emb
        elif emb_type == 'panns':
            models['panns'] = load_panns_model(device)
            configs['panns'] = {'sr': PANNS_CONFIG['sample_rate'], 'window_sec': PANNS_CONFIG['window_seconds'], 'emb_dim': PANNS_CONFIG['embedding_dim']}
        elif emb_type == 'vggish':
            model, input_processor, backend = load_vggish_model(device)
            models['vggish'] = model
            configs['vggish'] = {'sr': VGGISH_CONFIG['sample_rate'], 'window_sec': VGGISH_CONFIG['window_seconds'], 'emb_dim': VGGISH_CONFIG['embedding_dim'], 'backend': backend, 'input_processor': input_processor}
        elif emb_type == 'openl3':
            model, sr = load_openl3_model(
                device=device,
                content_type=OPENL3_CONFIG['content_type'],
                input_repr=OPENL3_CONFIG['input_repr'],
                embedding_size=OPENL3_CONFIG['embedding_dim']
            )
            models['openl3'] = model
            configs['openl3'] = {'sr': sr, 'window_sec': OPENL3_CONFIG['window_seconds'], 'emb_dim': OPENL3_CONFIG['embedding_dim']}
        else:
            raise ValueError(f"Unknown embedding type: {emb_type}. Use 'clap', 'panns', 'vggish', or 'openl3'")

        # Create resampler if needed
        target_sr = configs[emb_type]['sr']
        if sample_rate != target_sr:
            resamplers[emb_type] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr).to(device)

    # Initialize results
    for emb_type in embedding_types:
        results[emb_type] = {
            'score_sum': 0.0 if emb_type == 'clap' else None,
            'score_count': 0 if emb_type == 'clap' else None,
            'embeddings': []
        }

    iterator = tqdm(id2audio.keys(), desc=f"[metrics] extracting embeddings") if show_progress else id2audio.keys()

    for id in iterator:
        with torch.no_grad():
            audio_orig = id2audio[id]

            # NaN check
            if torch.isnan(audio_orig).any() or torch.isinf(audio_orig).any():
                print(f'Warning: NaN/Inf in audio for id {id}, skipping')
                continue

            # Make mono (shared preprocessing)
            if audio_orig.dim() == 2 and audio_orig.shape[0] > 1:
                audio_mono = audio_orig.mean(dim=0, keepdim=True)
            elif audio_orig.dim() == 1:
                audio_mono = audio_orig.unsqueeze(0)
            else:
                audio_mono = audio_orig

            # Peak normalize
            peak = torch.abs(audio_mono).max()
            if peak <= 0:
                print(f'Warning: Audio for id {id} is all zeros, skipping')
                continue
            audio_norm = audio_mono / peak * (10 ** (-1.0 / 20))

            # Extract embeddings for each model
            for emb_type in embedding_types:
                cfg = configs[emb_type]
                model = models[emb_type]
                window_size = int(cfg['window_sec'] * cfg['sr'])

                # Resample if needed
                if emb_type in resamplers:
                    audio = resamplers[emb_type](audio_norm)
                else:
                    audio = audio_norm

                if emb_type == 'clap':
                    # CLAP uses (1, T) and int16 quantization
                    audio = int16_to_float32_torch(float32_to_int16_torch(audio))

                    # Full-audio embedding for CLAP score
                    text_emb = cfg.get('text_emb', {})
                    if text_emb and id in text_emb:
                        audio_emb_full = model.get_audio_embedding_from_data(x=audio, use_tensor=True)
                        if not (torch.isnan(audio_emb_full).any() or torch.isnan(text_emb[id]).any()):
                            cosine_sim = torch.nn.functional.cosine_similarity(
                                audio_emb_full, text_emb[id].unsqueeze(0), dim=1, eps=1e-8
                            )[0]
                            if torch.isfinite(cosine_sim):
                                results[emb_type]['score_sum'] += cosine_sim.item()
                                results[emb_type]['score_count'] += 1

                    # Chunk for FAD embeddings
                    if audio.size(1) >= window_size:
                        patches = audio.unfold(1, window_size, window_size).squeeze(0)
                    else:
                        patches = audio
                    audio_emb = model.get_audio_embedding_from_data(x=patches, use_tensor=True)

                elif emb_type == 'openl3':
                    # OpenL3 uses torchopenl3's get_audio_embedding
                    import torchopenl3
                    audio = audio.squeeze(0)  # (T,)
                    # torchopenl3 expects (batch, samples) or (samples,) at target sample rate
                    # hop_size controls the embedding rate (in seconds)
                    # sampler="julius" uses pure PyTorch, avoiding resampy dependency
                    # (though we already resample to 48kHz so no resampling happens)
                    emb, ts = torchopenl3.get_audio_embedding(
                        audio.unsqueeze(0),  # (1, T)
                        sr=cfg['sr'],
                        model=model,
                        center=True,
                        hop_size=cfg['window_sec'],  # one embedding per window
                        batch_size=32,
                        sampler="julius",
                        verbose=False
                    )
                    # emb shape: (1, num_frames, embedding_dim)
                    audio_emb = emb.squeeze(0)  # (num_frames, embedding_dim)

                else:  # PANNs or VGGish
                    audio = audio.squeeze(0)  # (T,)

                    # Chunk
                    if audio.size(0) >= window_size:
                        num_windows = audio.size(0) // window_size
                        chunks = audio[:num_windows * window_size].view(num_windows, window_size)
                    else:
                        chunks = torch.nn.functional.pad(audio, (0, window_size - audio.size(0))).unsqueeze(0)

                    if emb_type == 'panns':
                        audio_emb = model(chunks)
                    else:  # vggish
                        backend = cfg.get('backend', 'torchaudio')
                        if backend == 'torchaudio':
                            # torchaudio VGGish requires mel spectrogram input via input_processor
                            input_processor = cfg.get('input_processor')
                            if input_processor is not None:
                                # input_processor expects 1D waveform at 16kHz, returns (n_examples, 1, n_frames, 64)
                                # Process each chunk and collect embeddings
                                chunk_embeddings = []
                                for chunk in chunks:
                                    mel = input_processor(chunk)  # (n_examples, 1, n_frames, 64)
                                    if mel.numel() > 0:
                                        mel = mel.to(device)
                                        emb = model(mel)  # (n_examples, 128)
                                        chunk_embeddings.append(emb)
                                if chunk_embeddings:
                                    audio_emb = torch.cat(chunk_embeddings, dim=0)
                                else:
                                    audio_emb = torch.zeros(0, cfg['emb_dim'], device=device)
                            else:
                                audio_emb = model(chunks)
                        else:
                            audio_emb = torch.from_numpy(model.forward(chunks.cpu().numpy())).to(device)

                # Filter NaN embeddings
                emb_np = audio_emb.cpu().numpy()
                valid_mask = np.isfinite(emb_np).all(axis=1)
                if not valid_mask.all():
                    emb_np = emb_np[valid_mask]
                if len(emb_np) > 0:
                    results[emb_type]['embeddings'].append(emb_np)

    # Concatenate embeddings
    for emb_type in embedding_types:
        emb_list = results[emb_type]['embeddings']
        emb_dim = configs[emb_type]['emb_dim']
        if len(emb_list) > 0:
            results[emb_type]['embeddings'] = np.concatenate(emb_list, axis=0)
        else:
            results[emb_type]['embeddings'] = np.array([]).reshape(0, emb_dim)

    # Cleanup models
    models.clear()
    resamplers.clear()
    gc.collect()
    torch.cuda.empty_cache()

    return results


def compute_metrics_from_embeddings(
    eval_embeddings,
    path_ref_audios,
    audios_extension,
    embedding_type='clap',
    clap_model='630k-audioset-fusion-best.pt',
    score_sum=None,
    score_count=None,
    show_progress=False,
    device='cuda'
):
    """
    Compute FAD (and optionally CLAP score) from pre-extracted embeddings.

    Args:
        eval_embeddings: numpy array of embeddings (M, embedding_dim)
        path_ref_audios: path to reference audio files
        audios_extension: file extension (e.g., '.wav', '.mp3')
        embedding_type: 'clap', 'panns', or 'vggish'
        clap_model: (CLAP only) model name for cache key
        score_sum: (CLAP only) sum of cosine similarities
        score_count: (CLAP only) count of similarities
        show_progress: Whether to show progress bars
        device: Device to use for computation (default: 'cuda')

    Returns:
        score: CLAP score (or None for non-CLAP)
        fd: Frechet Distance
    """
    embedding_type = embedding_type.lower()

    # Compute score (CLAP only)
    score = None
    if embedding_type == 'clap' and score_sum is not None and score_count is not None:
        if score_count > 0:
            score = score_sum / score_count
            if not np.isfinite(score):
                print(f'Warning: CLAP score is not finite ({score}), setting to NaN')
                score = float('nan')
        else:
            print('Warning: No valid samples for CLAP score computation')
            score = float('nan')

    # Check embeddings
    if len(eval_embeddings) < 2:
        print(f'Warning: Not enough {embedding_type} embeddings for FD ({len(eval_embeddings)})')
        return score, float('nan')

    mu_eval, sigma_eval = calculate_embd_statistics(eval_embeddings)

    # Reference embeddings cache
    cache_dir = 'load/fad_metrics'
    sanitized_path = path_ref_audios.strip('/').replace('/', '-')
    model_key = clap_model if embedding_type == 'clap' else embedding_type
    cache_path = os.path.join(cache_dir, f"{sanitized_path}__{model_key}.npz")

    if os.path.exists(cache_path):
        loaded = np.load(cache_path)
        mu_ref, sigma_ref = loaded['mu_ref'], loaded['sigma_ref']
    else:
        # Compute reference embeddings by reusing extract_embeddings()
        import librosa
        import pyloudnorm as pyln

        if not os.path.isdir(path_ref_audios):
            raise ValueError(f"path_ref_audios does not exist: {path_ref_audios}")
        wav_files = glob.glob(os.path.join(path_ref_audios, '*' + audios_extension))
        if len(wav_files) == 0:
            raise ValueError(f'No files with extension {audios_extension} in {path_ref_audios}')

        # Determine target sample rate for this embedding type
        if embedding_type == 'clap':
            target_sr = 48000
        elif embedding_type == 'panns':
            target_sr = PANNS_CONFIG['sample_rate']
        elif embedding_type == 'vggish':
            target_sr = VGGISH_CONFIG['sample_rate']
        elif embedding_type == 'openl3':
            target_sr = OPENL3_CONFIG['sample_rate']
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        # Load all reference audio files into dict format expected by extract_embeddings
        id2audio = {}
        iterator = tqdm(wav_files, desc=f"[metrics] loading {embedding_type} REFERENCE audio") if show_progress else wav_files
        for p in iterator:
            audio, _ = librosa.load(p, sr=target_sr, mono=True)
            audio = pyln.normalize.peak(audio, -1.0)
            # Convert to tensor format (C, T) expected by extract_embeddings
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)  # (1, T)
            id2audio[p] = audio_tensor

        # Use unified extract_embeddings function
        with torch.no_grad():
            results = extract_embeddings(
                id2audio=id2audio,
                embedding_types=(embedding_type,),
                sample_rate=target_sr,
                show_progress=show_progress,
                device=device,
                clap_model=clap_model
            )

        ref_embeddings = results[embedding_type]['embeddings']
        del id2audio
        torch.cuda.empty_cache()

        if len(ref_embeddings) < 2:
            raise ValueError(f"Not enough reference embeddings for {embedding_type}: {len(ref_embeddings)}")

        mu_ref, sigma_ref = calculate_embd_statistics(ref_embeddings)

        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_path, mu_ref=mu_ref, sigma_ref=sigma_ref)

    fd = calculate_frechet_distance(mu_eval, sigma_eval, mu_ref, sigma_ref)
    return score, fd


def compute_latent_fad(
    eval_embeddings,
    path_ref_audios,
    audios_extension,
    pretransform,
    sample_rate,
    device='cuda',
    show_progress=False
):
    """
    Compute Frechet Latent Distance (FLD) - FAD in the VAE latent space.

    Args:
        eval_embeddings: numpy array of latent embeddings (N, latent_dim), mean-pooled over sequence
        path_ref_audios: path to reference audio files
        audios_extension: file extension (e.g., '.wav', '.mp3')
        pretransform: Pretransform model (VAE encoder) for encoding reference audio
        sample_rate: Sample rate for loading reference audio
        device: Device for computation

    Returns:
        fd: Frechet Latent Distance
    """
    if len(eval_embeddings) < 2:
        print(f'Warning: Not enough latent embeddings for FLD ({len(eval_embeddings)})')
        return float('nan')

    mu_eval, sigma_eval = calculate_embd_statistics(eval_embeddings)

    # Reference embeddings cache
    # Include a hash of pretransform parameters to ensure cache validity across different models
    import hashlib
    param_hash = hashlib.md5()
    for name, param in sorted(pretransform.state_dict().items()):
        param_hash.update(name.encode())
        param_hash.update(param.cpu().numpy().tobytes())
    pretransform_hash = param_hash.hexdigest()[:12]  # Use first 12 chars for brevity

    cache_dir = 'load/fad_metrics'
    sanitized_path = path_ref_audios.strip('/').replace('/', '-')
    latent_dim = eval_embeddings.shape[1]
    window_sec = LATENT_CONFIG['window_seconds']
    cache_path = os.path.join(cache_dir, f"{sanitized_path}__latent_{latent_dim}_{pretransform_hash}_w{window_sec}.npz")

    if os.path.exists(cache_path):
        loaded = np.load(cache_path)
        mu_ref, sigma_ref = loaded['mu_ref'], loaded['sigma_ref']
    else:
        # Compute reference latent embeddings
        import librosa
        import pyloudnorm as pyln

        if not os.path.isdir(path_ref_audios):
            raise ValueError(f"path_ref_audios does not exist: {path_ref_audios}")
        wav_files = glob.glob(os.path.join(path_ref_audios, '*' + audios_extension))
        if len(wav_files) == 0:
            raise ValueError(f'No files with extension {audios_extension} in {path_ref_audios}')

        ref_embeddings = []
        pretransform_dtype = next(pretransform.parameters()).dtype
        downsampling_ratio = pretransform.downsampling_ratio

        # Window settings (same as eval embeddings)
        latent_window_sec = LATENT_CONFIG['window_seconds']
        latent_frames_per_sec = sample_rate / downsampling_ratio
        window_frames = int(latent_window_sec * latent_frames_per_sec)

        iterator = tqdm(wav_files, desc="[metrics] encoding REFERENCE audio to latents") if show_progress else wav_files
        for p in iterator:
            audio, _ = librosa.load(p, sr=sample_rate, mono=False)
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)  # Convert mono to stereo
            audio = pyln.normalize.peak(audio, -1.0)
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)  # (1, C, T)
            audio_tensor = audio_tensor.to(pretransform_dtype)

            with torch.no_grad():
                latents = pretransform.encode(audio_tensor)  # (1, latent_dim, seq_len)
                seq_len = latents.shape[-1]

                if seq_len >= window_frames:
                    # Chunk into windows and mean-pool within each
                    chunks = latents.unfold(-1, window_frames, window_frames)  # (1, latent_dim, num_windows, window_frames)
                    chunk_embs = chunks.mean(dim=-1).squeeze(0).T  # (num_windows, latent_dim)
                    ref_embeddings.extend(chunk_embs.cpu().numpy())
                else:
                    # Audio too short, use global mean pool
                    latent_emb = latents.mean(dim=-1).squeeze(0)  # (latent_dim,)
                    ref_embeddings.append(latent_emb.cpu().numpy())

        ref_embeddings = np.array(ref_embeddings)
        torch.cuda.empty_cache()

        if len(ref_embeddings) < 2:
            raise ValueError(f"Not enough reference latent embeddings: {len(ref_embeddings)}")

        mu_ref, sigma_ref = calculate_embd_statistics(ref_embeddings)

        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_path, mu_ref=mu_ref, sigma_ref=sigma_ref)

    fd = calculate_frechet_distance(mu_eval, sigma_eval, mu_ref, sigma_ref)
    return fd


class DiffusionMetricsCallbackDistributed(pl.Callback):
    def __init__(
        self,
        metrics_every=1000,
        cfg_scale=7,
        sampling_steps=50,
        num_generations=250,
        min_length=False,
        max_length=False,
        prompts_type='song_describer-nosinging',
        ref_audios_path=False,
        ref_audios_ext=False,  # '.mp3' or '.wav'
        clap_model='music_speech_audioset_epoch_15_esc_89.98.pt',
        embedding_types=('clap',),  # Any combo of 'clap', 'panns', 'vggish', 'openl3', 'latent'
        show_progress=False,  # Show progress bars (only on rank 0)
        skip_first=True,  # Skip metrics on step 0
        decode_batch_size=4,  # Max samples to decode at once through pretransform
    ):
        super().__init__()
        self.metrics_every = metrics_every
        self.skip_first = skip_first
        self.decode_batch_size = decode_batch_size
        self.cfg_scale = cfg_scale
        self.sampling_steps = sampling_steps
        self.num_generations = num_generations
        self.show_progress = show_progress
        assert min_length is not None and max_length is not None, "Must specify min_length and max_length"
        self.min_length = min_length
        self.max_length = max_length

        cache_dir = 'load/clap_metrics'
        self.prompts_type = prompts_type+'.csv'
        url = 'https://raw.githubusercontent.com/Stability-AI/stable-audio-metrics/main/load/'+self.prompts_type
        self.prompts_path = os.path.join(cache_dir, self.prompts_type)
        if not os.path.exists(self.prompts_path):
            print(f"Downloading {self.prompts_path}...")
            os.makedirs(os.path.dirname(self.prompts_path), exist_ok=True)
            response = requests.get(url)
            response.raise_for_status()
            with open(self.prompts_path, 'wb') as f:
                f.write(response.content)        
        self.csv_data = pd.read_csv(self.prompts_path)
        self.total_data = len(self.csv_data)

        assert ref_audios_path is not None, "Must specify ref_audios_path"
        self.ref_audios_path = ref_audios_path
        assert ref_audios_ext is not None, "Must specify ref_audios_ext"
        self.ref_audios_ext = ref_audios_ext
        self.clap_model = clap_model

        self.embedding_types = tuple(t.lower() for t in embedding_types)

        self.seed = 11

    @torch.no_grad()
    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if trainer.global_step % self.metrics_every != 0:
            return
        if self.skip_first and trainer.global_step == 0:
            return

        # Get sample_rate and samples_latent from the training wrapper
        sample_rate = module.sample_rate
        if module.sample_size is not None:
            downsampling_ratio = module.diffusion.pretransform.downsampling_ratio if module.diffusion.pretransform is not None else 1
            samples_latent = module.sample_size // downsampling_ratio
        else:
            raise ValueError("sample_size not set on training wrapper. Cannot compute samples_latent for metrics.")

        # get distributed info
        world_size = trainer.world_size
        rank = trainer.global_rank
        is_rank_zero = (rank == 0)

        batch_size = batch[0].shape[0]

        if self.num_generations <= self.total_data:
            # use fixed seed so all GPUs get the same sample
            data_to_process = self.csv_data.sample(n=self.num_generations, random_state=self.seed)
        else:
            # Re-use prompts to reach num_generations (different noise → independent samples for FAD)
            num_full_repeats = self.num_generations // self.total_data
            remainder = self.num_generations % self.total_data
            parts = [self.csv_data] * num_full_repeats
            if remainder > 0:
                parts.append(self.csv_data.sample(n=remainder, random_state=self.seed))
            data_to_process = pd.concat(parts)

        # reset index to get unique integer IDs (important when prompts are re-used)
        data_to_process = data_to_process.reset_index(drop=True)
        # each GPU takes every world_size-th sample starting from its rank
        local_data = data_to_process.iloc[rank::world_size]

        # Local storage for this GPU's results
        local_id2audio = {}
        local_id2text = {}
        local_latent_embeddings = []  # For Frechet Latent Distance

        # Separate latent from audio-based embedding types
        audio_embedding_types = tuple(t for t in self.embedding_types if t != 'latent')
        compute_latent_emb = 'latent' in self.embedding_types

        show_progress = self.show_progress and is_rank_zero
        iterator = tqdm(range(0, len(local_data), batch_size), desc=f"[metrics] generating audio") if show_progress else range(0, len(local_data), batch_size)

        for i in iterator:
            batch_df = local_data.iloc[i : i + batch_size]
            num_gen_batch = len(batch_df)
            
            if self.prompts_type.startswith('song_describer'):
                base_ids = batch_df['caption_id'].tolist()
            elif self.prompts_type.startswith('musiccaps'):
                base_ids = batch_df['ytid'].tolist()
            elif self.prompts_type.startswith('audiocaps'):
                base_ids = batch_df['audiocap_id'].tolist()
            else:
                raise NotImplementedError(f'Unknown prompts type: {self.prompts_type}')
            # Append row index for uniqueness when prompts are re-used
            ids_list = [f"{base_id}_{idx}" for base_id, idx in zip(base_ids, batch_df.index)]
            
            texts_list = batch_df['caption'].tolist()
            batch_id2text = dict(zip(ids_list, texts_list))
            # Use torch.randint for reproducibility in distributed training (Python's random is not synced across ranks)
            batch_id2length = {id: torch.randint(self.min_length, self.max_length + 1, (1,)).item() for id in ids_list}

            # preparing prompts
            formatted_prompts = [{"prompt": batch_id2text[id], "seconds_total": batch_id2length[id]} for id in ids_list]
            conditioning = module.diffusion.conditioner(formatted_prompts, module.device)
            
            # preparing inpainting masks
            mask = torch.zeros((num_gen_batch, 1, samples_latent), device=module.device)
            conditioning['inpaint_mask'] = [mask]
            masked_input = torch.zeros((num_gen_batch, module.diffusion.io_channels, samples_latent), device=module.device)
            conditioning['inpaint_masked_input'] = [masked_input]
            # note: for inpainting we provide empty masks since we generate audio from scratch
            # note: for non-inpainting models these conditioning keys will be ignored   
            
            # get conditioning tensors
            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

            # sample noise
            noise = torch.randn([num_gen_batch, module.diffusion.io_channels, samples_latent]).to(module.device)
            model_dtype = next(module.diffusion.parameters()).dtype
            noise = noise.to(model_dtype)

            # generate latents using unified sampling function (decode separately per-sample to avoid OOM)
            diffusion_model = module.diffusion_ema.ema_model if module.diffusion_ema is not None else module.diffusion.model
            with torch.amp.autocast("cuda"):
                sampled_latents = sample_diffusion(
                    model=diffusion_model,
                    noise=noise,
                    cond_inputs=cond_inputs,
                    diffusion_objective=module.diffusion_objective,
                    steps=self.sampling_steps,
                    cfg_scale=self.cfg_scale,
                    # Varlen support
                    conditioning=formatted_prompts,
                    sample_rate=sample_rate,
                    pretransform=module.diffusion.pretransform,
                    mask_padding_attention=module.diffusion.mask_padding_attention,
                    use_effective_length_for_schedule=module.diffusion.use_effective_length_for_schedule,
                    headroom_seconds=5.0,
                    dist_shift=module.diffusion.sampling_dist_shift,
                    batch_cfg=True,
                    decode=False,  # decode per-sample below to avoid OOM
                    disable_tqdm=not show_progress
                )

            # Free generation intermediates before decoding
            del noise, conditioning, cond_inputs
            torch.cuda.empty_cache()

            # Extract latent embeddings before decoding (avoids re-encoding)
            if compute_latent_emb:
                latent_window_sec = LATENT_CONFIG['window_seconds']
                latent_frames_per_sec = sample_rate / downsampling_ratio
                window_frames = int(latent_window_sec * latent_frames_per_sec)
                seq_len = sampled_latents.shape[-1]

                if seq_len >= window_frames:
                    chunks = sampled_latents.unfold(-1, window_frames, window_frames)
                    chunk_embs = chunks.mean(dim=-1)
                    chunk_embs = chunk_embs.permute(0, 2, 1).reshape(-1, sampled_latents.shape[1])
                    local_latent_embeddings.extend(chunk_embs.cpu().numpy())
                else:
                    latent_embs = sampled_latents.mean(dim=-1)
                    local_latent_embeddings.extend(latent_embs.cpu().numpy())

            # Decode in sub-batches to avoid OOM from full-batch VAE decode
            pretransform = module.diffusion.pretransform
            batch_id2audio = {}
            if pretransform is not None:
                pretransform_dtype = next(pretransform.parameters()).dtype
                for sub_start in range(0, num_gen_batch, self.decode_batch_size):
                    sub_end = min(sub_start + self.decode_batch_size, num_gen_batch)
                    with torch.amp.autocast("cuda"):
                        audio_sub = pretransform.decode(sampled_latents[sub_start:sub_end].to(pretransform_dtype))
                    audio_sub = audio_sub.to(torch.float32).detach()
                    for j in range(sub_start, sub_end):
                        audio_j = audio_sub[j - sub_start]
                        audio_j = audio_j / (audio_j.abs().max() + 1e-8)
                        id = ids_list[j]
                        length_sec = batch_id2length[id] + 1
                        num_samples = int(length_sec * sample_rate)
                        batch_id2audio[id] = audio_j[:, :num_samples].clone()
                    del audio_sub
            else:
                for j, id in enumerate(ids_list):
                    audio_j = sampled_latents[j].to(torch.float32)
                    audio_j = audio_j / (audio_j.abs().max() + 1e-8)
                    length_sec = batch_id2length[id] + 1
                    num_samples = int(length_sec * sample_rate)
                    batch_id2audio[id] = audio_j[:, :num_samples].detach()

            local_id2audio.update(batch_id2audio)
            local_id2text.update(batch_id2text)

            # explicitly free batch tensors
            del sampled_latents, batch_id2audio
            torch.cuda.empty_cache()
        
        # Extract embeddings for audio-based model types locally on each GPU
        if audio_embedding_types:
            local_results = extract_embeddings(
                id2audio=local_id2audio,
                embedding_types=audio_embedding_types,
                sample_rate=sample_rate,
                show_progress=show_progress,
                id2text=local_id2text,
                clap_model=self.clap_model
            )
        else:
            local_results = {}

        # Add latent embeddings to results
        if compute_latent_emb:
            local_results['latent'] = {
                'embeddings': np.array(local_latent_embeddings) if local_latent_embeddings else np.array([]).reshape(0, module.diffusion.io_channels)
            }

        # Free GPU memory from stored audio tensors
        del local_id2audio
        torch.cuda.empty_cache()

        # ============ Optimized distributed gather ============
        # Strategy:
        # 1. Use tensor-based all_reduce for CLAP score scalars (avoids pickle, ~16 bytes)
        # 2. Use gather_object to rank 0 only (not all_gather - saves memory on other ranks)
        # 3. Only send embeddings arrays, not score_sum/score_count dicts

        if torch.distributed.is_initialized():
            # 1. Reduce CLAP score scalars separately (much more efficient than pickling)
            if 'clap' in local_results and local_results['clap'].get('score_sum') is not None:
                clap_score_sum = torch.tensor(local_results['clap']['score_sum'], device=module.device, dtype=torch.float64)
                clap_score_count = torch.tensor(local_results['clap']['score_count'], device=module.device, dtype=torch.int64)
                torch.distributed.all_reduce(clap_score_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(clap_score_count, op=torch.distributed.ReduceOp.SUM)
                total_clap_score_sum = clap_score_sum.item()
                total_clap_score_count = clap_score_count.item()
            else:
                total_clap_score_sum = 0.0
                total_clap_score_count = 0

            # 2. Prepare embeddings-only dict for gather (exclude score_sum/score_count to reduce pickle overhead)
            embeddings_only = {}
            for emb_type in self.embedding_types:
                if emb_type in local_results:
                    embeddings_only[emb_type] = local_results[emb_type]['embeddings']

            # 3. Use gather_object to rank 0 only (not all_gather - saves memory on other ranks)
            if is_rank_zero:
                gathered_embeddings = [None] * world_size
            else:
                gathered_embeddings = None
            torch.distributed.gather_object(embeddings_only, gathered_embeddings, dst=0)
        else:
            # Single GPU case
            gathered_embeddings = [{emb_type: local_results[emb_type]['embeddings'] for emb_type in self.embedding_types if emb_type in local_results}]
            total_clap_score_sum = local_results.get('clap', {}).get('score_sum', 0.0) or 0.0
            total_clap_score_count = local_results.get('clap', {}).get('score_count', 0) or 0

        # Only rank 0 computes final metrics
        if is_rank_zero:
            metrics_to_log = {}

            # Aggregate and compute metrics for each embedding type
            for emb_type in self.embedding_types:
                all_embeddings = []

                for gpu_embeddings in gathered_embeddings:
                    if emb_type in gpu_embeddings and len(gpu_embeddings[emb_type]) > 0:
                        all_embeddings.append(gpu_embeddings[emb_type])

                emb_dim = {'clap': 512, 'panns': 2048, 'vggish': 128, 'openl3': 512, 'latent': module.diffusion.io_channels}.get(emb_type, 512)
                all_embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([]).reshape(0, emb_dim)

                print(f"[{emb_type}] Total embeddings: {len(all_embeddings)}")

                if emb_type == 'latent':
                    # Frechet Latent Distance - compute directly using pretransform
                    fd = compute_latent_fad(
                        eval_embeddings=all_embeddings,
                        path_ref_audios=self.ref_audios_path,
                        audios_extension=self.ref_audios_ext,
                        pretransform=module.diffusion.pretransform,
                        sample_rate=sample_rate,
                        device=module.device,
                        show_progress=show_progress
                    )
                    metrics_to_log["metrics/latent-fad"] = fd
                else:
                    score, fd = compute_metrics_from_embeddings(
                        eval_embeddings=all_embeddings,
                        path_ref_audios=self.ref_audios_path,
                        audios_extension=self.ref_audios_ext,
                        embedding_type=emb_type,
                        clap_model=self.clap_model,
                        score_sum=total_clap_score_sum if emb_type == 'clap' else None,
                        score_count=total_clap_score_count if emb_type == 'clap' else None,
                        show_progress=show_progress,
                        device=module.device
                    )

                    if emb_type == 'clap':
                        metrics_to_log["metrics/clap-score"] = score
                        metrics_to_log["metrics/clap-fad"] = fd
                    else:
                        metrics_to_log[f"metrics/{emb_type}-fad"] = fd

            trainer.logger.experiment.log_metrics(metrics_to_log, step=trainer.global_step)

        # Final cleanup — free all intermediate data before training resumes
        del local_results, local_latent_embeddings, gathered_embeddings
        gc.collect()
        torch.cuda.empty_cache()


class AutoencoderMetricsCallback(pl.Callback):
    """FAD metrics for autoencoder training.

    Compares reconstructed reference audio vs original reference audio
    using Frechet Audio Distance. Uses the autoencoder to encode+decode
    reference files, then computes FAD between reconstructed and original
    embedding distributions.
    """
    def __init__(
        self,
        metrics_every=1000,
        ref_audios_path=False,
        ref_audios_ext='.wav',
        embedding_types=('clap',),
        clap_model='music_speech_audioset_epoch_15_esc_89.98.pt',
        show_progress=False,
        skip_first=True,
        max_samples=0,  # 0 = use all reference files
    ):
        super().__init__()
        self.metrics_every = metrics_every
        self.skip_first = skip_first
        self.show_progress = show_progress
        self.embedding_types = tuple(t.lower() for t in embedding_types)
        self.clap_model = clap_model

        assert ref_audios_path, "Must specify ref_audios_path"
        self.ref_audios_path = ref_audios_path
        self.ref_audios_ext = ref_audios_ext
        self.max_samples = max_samples

        self.seed = 42

    @torch.no_grad()
    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if trainer.global_step % self.metrics_every != 0:
            return
        if self.skip_first and trainer.global_step == 0:
            return

        # Switch to eval mode so bottleneck noise / running norms behave correctly
        was_training = module.training
        module.eval()

        try:
            import librosa
            import pyloudnorm as pyln

            sample_rate = module.sample_rate
            world_size = trainer.world_size
            rank = trainer.global_rank
            is_rank_zero = (rank == 0)
            show_progress = self.show_progress and is_rank_zero

            # List reference audio files
            wav_files = sorted(glob.glob(os.path.join(self.ref_audios_path, '*' + self.ref_audios_ext)))
            if len(wav_files) == 0:
                raise ValueError(f'No files with extension {self.ref_audios_ext} in {self.ref_audios_path}')

            # Optionally subsample for faster evaluation
            if self.max_samples > 0 and len(wav_files) > self.max_samples:
                rng = random.Random(self.seed)
                wav_files = rng.sample(wav_files, self.max_samples)
                wav_files.sort()

            # Distribute files across GPUs
            local_files = wav_files[rank::world_size]

            # Use EMA model if available
            autoencoder = module.autoencoder_ema.ema_model if module.autoencoder_ema is not None else module.autoencoder
            autoencoder_dtype = next(autoencoder.parameters()).dtype
            force_input_mono = module.force_input_mono
            in_channels = module.autoencoder.in_channels

            local_id2reconstructed = {}

            batch_size = batch[0].shape[0]
            iterator = tqdm(range(0, len(local_files), batch_size), desc="[ae-metrics] reconstructing") if show_progress else range(0, len(local_files), batch_size)

            for i in iterator:
                batch_files = local_files[i:i + batch_size]
                batch_audios = []
                batch_ids = []
                batch_lengths = []

                for filepath in batch_files:
                    audio, _ = librosa.load(filepath, sr=sample_rate, mono=False)
                    if audio.ndim == 1:
                        if in_channels == 2:
                            audio = np.stack([audio, audio], axis=0)
                        else:
                            audio = audio[np.newaxis, :]
                    elif audio.ndim == 2 and audio.shape[0] > in_channels:
                        audio = audio[:in_channels]
                    audio = pyln.normalize.peak(audio, -1.0)
                    audio_tensor = torch.from_numpy(audio).float()
                    batch_audios.append(audio_tensor)
                    batch_ids.append(filepath)
                    batch_lengths.append(audio_tensor.shape[-1])

                # Pad to same length within sub-batch
                max_len = max(a.shape[-1] for a in batch_audios)
                for j in range(len(batch_audios)):
                    if batch_audios[j].shape[-1] < max_len:
                        batch_audios[j] = torch.nn.functional.pad(batch_audios[j], (0, max_len - batch_audios[j].shape[-1]))

                batch_tensor = torch.stack(batch_audios, dim=0).to(module.device)

                encoder_input = batch_tensor
                if force_input_mono and encoder_input.shape[1] > 1:
                    encoder_input = encoder_input.mean(dim=1, keepdim=True)

                with torch.amp.autocast("cuda"):
                    latents = autoencoder.encode(encoder_input.to(autoencoder_dtype))
                    reconstructed = autoencoder.decode(latents)

                reconstructed = reconstructed.to(torch.float32).detach()
                min_len = min(reconstructed.shape[-1], batch_tensor.shape[-1])

                for j, fid in enumerate(batch_ids):
                    sample_len = min(min_len, batch_lengths[j])
                    recon_j = reconstructed[j, :, :sample_len]
                    recon_j = recon_j / (recon_j.abs().max() + 1e-8)
                    local_id2reconstructed[fid] = recon_j

                del batch_tensor, encoder_input, latents, reconstructed

            # Extract embeddings from reconstructed audio
            audio_embedding_types = tuple(t for t in self.embedding_types if t != 'latent')

            if audio_embedding_types:
                local_results = extract_embeddings(
                    id2audio=local_id2reconstructed,
                    embedding_types=audio_embedding_types,
                    sample_rate=sample_rate,
                    show_progress=show_progress,
                    clap_model=self.clap_model
                )
            else:
                local_results = {}

            del local_id2reconstructed
            torch.cuda.empty_cache()

            # Distributed gather
            if torch.distributed.is_initialized():
                embeddings_only = {}
                for emb_type in audio_embedding_types:
                    if emb_type in local_results:
                        embeddings_only[emb_type] = local_results[emb_type]['embeddings']

                if is_rank_zero:
                    gathered_embeddings = [None] * world_size
                else:
                    gathered_embeddings = None
                torch.distributed.gather_object(embeddings_only, gathered_embeddings, dst=0)
            else:
                gathered_embeddings = [{emb_type: local_results[emb_type]['embeddings'] for emb_type in audio_embedding_types if emb_type in local_results}]

            # Only rank 0 computes final metrics
            if is_rank_zero:
                metrics_to_log = {}

                for emb_type in audio_embedding_types:
                    all_embeddings = []
                    for gpu_embeddings in gathered_embeddings:
                        if emb_type in gpu_embeddings and len(gpu_embeddings[emb_type]) > 0:
                            all_embeddings.append(gpu_embeddings[emb_type])

                    emb_dim = {'clap': 512, 'panns': 2048, 'vggish': 128, 'openl3': 512}.get(emb_type, 512)
                    all_embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([]).reshape(0, emb_dim)

                    print(f"[ae-metrics/{emb_type}] Reconstructed embeddings: {len(all_embeddings)}")

                    # compute_metrics_from_embeddings handles reference embedding caching
                    score, fd = compute_metrics_from_embeddings(
                        eval_embeddings=all_embeddings,
                        path_ref_audios=self.ref_audios_path,
                        audios_extension=self.ref_audios_ext,
                        embedding_type=emb_type,
                        clap_model=self.clap_model,
                        show_progress=show_progress,
                        device=module.device
                    )

                    metrics_to_log[f"metrics/ae-{emb_type}-fad"] = fd

                trainer.logger.experiment.log_metrics(metrics_to_log, step=trainer.global_step)

            # Final cleanup
            del local_results, gathered_embeddings
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            module.train(was_training)