import torch
import typing as tp
from audiocraft.models import MusicGen, CompressionModel, LMModel
import audiocraft.quantization as qt
from .autoencoders import AudioAutoencoder
from .bottleneck import DACRVQBottleneck, DACRVQVAEBottleneck

from audiocraft.modules.codebooks_patterns import (
    DelayedPatternProvider,
    MusicLMPattern,
    ParallelPatternProvider,
    UnrolledPatternProvider,
    VALLEPattern,
)

from audiocraft.modules.conditioners import (
    ConditionFuser,
    ConditioningProvider,
    T5Conditioner,
)

def create_musicgen_from_config(config):
    model_config = config.get('model', None)
    assert model_config is not None, 'model config must be specified in config'

    if model_config.get("pretrained", False):
        model = MusicGen.get_pretrained(model_config["pretrained"], device="cpu")

        if model_config.get("reinit_lm", False):
            model.lm._init_weights("gaussian", "current", True)
    
        return model
    
    # Create MusicGen model from scratch
    compression_config = model_config.get('compression', None)
    assert compression_config is not None, 'compression config must be specified in model config'

    compression_type = compression_config.get('type', None)
    assert compression_type is not None, 'type must be specified in compression config'

    if compression_type == 'pretrained':
        compression_model = CompressionModel.get_pretrained(compression_config["config"]["name"])
    elif compression_type == "dac_rvq_ae":
        from .autoencoders import create_autoencoder_from_config
        autoencoder = create_autoencoder_from_config({"model": compression_config["config"], "sample_rate": config["sample_rate"]})
        autoencoder.load_state_dict(torch.load(compression_config["ckpt_path"], map_location="cpu")["state_dict"])
        compression_model = DACRVQCompressionModel(autoencoder)
    
    lm_config = model_config.get('lm', None)
    assert lm_config is not None, 'lm config must be specified in model config'

    codebook_pattern = lm_config.pop("codebook_pattern", "delay")

    pattern_providers = {
        'parallel': ParallelPatternProvider,
        'delay': DelayedPatternProvider,
        'unroll': UnrolledPatternProvider,
        'valle': VALLEPattern,
        'musiclm': MusicLMPattern,
    }

    pattern_provider = pattern_providers[codebook_pattern](n_q=compression_model.num_codebooks)

    conditioning_config = model_config.get("conditioning", {})

    condition_output_dim = conditioning_config.get("output_dim", 768)

    condition_provider = ConditioningProvider(
        conditioners = {
            "description": T5Conditioner(
                name="t5-base",
                output_dim=condition_output_dim,
                word_dropout=0.3,
                normalize_text=False,
                finetune=False,
                device="cpu"
            )
        }
    )

    condition_fuser = ConditionFuser(fuse2cond={
        "cross": ["description"],
        "prepend": [],
        "sum": []
        })

    lm = LMModel(
        pattern_provider = pattern_provider,
        condition_provider = condition_provider,
        fuser = condition_fuser,
        n_q = compression_model.num_codebooks,
        card = compression_model.cardinality,
        **lm_config
    )


    model = MusicGen(
        name = model_config.get("name", "musicgen-scratch"),
        compression_model = compression_model,
        lm = lm,
        max_duration=30
    )

    return model

class DACRVQCompressionModel(CompressionModel):
    def __init__(self, autoencoder: AudioAutoencoder):
        super().__init__()
        self.model = autoencoder.eval()

        assert isinstance(self.model.bottleneck, DACRVQBottleneck) or isinstance(self.model.bottleneck, DACRVQVAEBottleneck), "Autoencoder must have a DACRVQBottleneck or DACRVQVAEBottleneck"

        self.n_quantizers = self.model.bottleneck.num_quantizers

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        raise NotImplementedError("Forward and training with DAC RVQ not supported")

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        _, info = self.model.encode(x, return_info=True, n_quantizers=self.n_quantizers)
        codes = info["codes"]
        return codes, None

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        assert scale is None
        z_q = self.decode_latent(codes)
        return self.model.decode(z_q)

    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.model.bottleneck.quantizer.from_codes(codes)[0]

    @property
    def channels(self) -> int:
        return self.model.io_channels

    @property
    def frame_rate(self) -> float:
        return self.model.sample_rate / self.model.downsampling_ratio

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @property
    def cardinality(self) -> int:
        return self.model.bottleneck.quantizer.codebook_size

    @property
    def num_codebooks(self) -> int:
        return self.n_quantizers

    @property
    def total_codebooks(self) -> int:
        self.model.bottleneck.num_quantizers

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer.
        """
        assert n >= 1
        assert n <= self.total_codebooks
        self.n_quantizers = n