import torch
import torchaudio
import wandb
from einops import rearrange
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from ema_pytorch import EMA
import auraloss
import pytorch_lightning as pl
from ..models.autoencoders import AudioAutoencoder
from ..models.discriminators import EncodecDiscriminator, OobleckDiscriminator, DACGANLoss
from ..models.bottleneck import VAEBottleneck, RVQBottleneck, DACRVQBottleneck, DACRVQVAEBottleneck, RVQVAEBottleneck, WassersteinBottleneck
from .utils import create_optimizer_from_config, create_scheduler_from_config


from pytorch_lightning.utilities.rank_zero import rank_zero_only
from aeiou.viz import pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image

class AutoencoderTrainingWrapper(pl.LightningModule):
    def __init__(
            self, 
            autoencoder: AudioAutoencoder,
            lr: float = 1e-4,
            warmup_steps: int = 0,
            encoder_freeze_on_warmup: bool = False,
            sample_rate=48000,
            loss_config: dict = None,
            optimizer_configs: dict = None,
            use_ema: bool = True,
            ema_copy = None,
            force_input_mono = False,
            latent_mask_ratio = 0.0,
            teacher_model: AudioAutoencoder = None
    ):
        super().__init__()

        self.automatic_optimization = False

        self.autoencoder = autoencoder

        self.warmed_up = False
        self.warmup_steps = warmup_steps
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup
        self.lr = lr

        self.force_input_mono = force_input_mono

        self.teacher_model = teacher_model

        if optimizer_configs is None:
            optimizer_configs ={
                "autoencoder": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": lr,
                            "betas": (.8, .99)
                        }
                    }
                },
                "discriminator": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": lr,
                            "betas": (.8, .99)
                        }
                    }
                }

            } 
            
        self.optimizer_configs = optimizer_configs

        if loss_config is None:
            scales = [2048, 1024, 512, 256, 128, 64, 32]
            hop_sizes = []
            win_lengths = []
            overlap = 0.75
            for s in scales:
                hop_sizes.append(int(s * (1 - overlap)))
                win_lengths.append(s)
        
            loss_config = {
                "discriminator": {
                    "type": "encodec",
                    "config": {
                        "n_ffts": scales,
                        "hop_lengths": hop_sizes,
                        "win_lengths": win_lengths,
                        "filters": 32
                    },
                    "weights": {
                        "adversarial": 0.1,
                        "feature_matching": 5.0,
                    }
                },
                "spectral": {
                    "type": "mrstft",
                    "config": {
                        "fft_sizes": scales,
                        "hop_sizes": hop_sizes,
                        "win_lengths": win_lengths,
                        "perceptual_weighting": True
                    },
                    "weights": {
                        "mrstft": 1.0,
                    }
                },
                "time": {
                    "type": "l1",
                    "config": {},
                    "weights": {
                        "l1": 0.0,
                    }
                }
            }
        
        self.loss_config = loss_config
       
        # Spectral reconstruction loss

        stft_loss_args = loss_config['spectral']['config']

        if self.autoencoder.out_channels == 2:
            self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
        else:
            self.sdstft = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)

        # Discriminator

        if loss_config['discriminator']['type'] == 'oobleck':
            self.discriminator = OobleckDiscriminator(**loss_config['discriminator']['config'])
        elif loss_config['discriminator']['type'] == 'encodec':
            self.discriminator = EncodecDiscriminator(in_channels=self.autoencoder.out_channels, **loss_config['discriminator']['config'])
        elif loss_config['discriminator']['type'] == 'dac':
            self.discriminator = DACGANLoss(channels=self.autoencoder.out_channels, sample_rate=sample_rate, **loss_config['discriminator']['config'])

        self.autoencoder_ema = None
        
        self.use_ema = use_ema

        if self.use_ema:
            self.autoencoder_ema = EMA(
                self.autoencoder,
                ema_model=ema_copy,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1
            )

        self.latent_mask_ratio = latent_mask_ratio

    def configure_optimizers(self):

        opt_gen = create_optimizer_from_config(self.optimizer_configs['autoencoder']['optimizer'], self.autoencoder.parameters())
        opt_disc = create_optimizer_from_config(self.optimizer_configs['discriminator']['optimizer'], self.discriminator.parameters())

        if "scheduler" in self.optimizer_configs['autoencoder'] and "scheduler" in self.optimizer_configs['discriminator']:
            sched_gen = create_scheduler_from_config(self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
            sched_disc = create_scheduler_from_config(self.optimizer_configs['discriminator']['scheduler'], opt_disc)
            return [opt_gen, opt_disc], [sched_gen, sched_disc]

        return [opt_gen, opt_disc]
  
    def training_step(self, batch, batch_idx):
        reals, _ = batch

        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if self.global_step >= self.warmup_steps:
            self.warmed_up = True

        encoder_input = reals

        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        if self.warmed_up and self.encoder_freeze_on_warmup:
            with torch.no_grad():
                latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)
        else:
            latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)

        # Encode with teacher model for distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_latents = self.teacher_model.encode(encoder_input, return_info=False)

        # Optionally mask out some latents for noise resistance
        if self.latent_mask_ratio > 0.0:
            mask = torch.rand_like(latents) < self.latent_mask_ratio
            latents = torch.where(mask, torch.zeros_like(latents), latents)

        decoded = self.autoencoder.decode(latents)

        # Distillation with spectral losses
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_decoded = self.teacher_model.decode(teacher_latents)
                own_latents_teacher_decoded = self.teacher_model.decode(latents) #Distilled model's latents decoded by teacher
                teacher_latents_own_decoded = self.autoencoder.decode(teacher_latents) #Teacher's latents decoded by distilled model


            mrstft_loss = self.sdstft(reals, decoded) # Reconstruction loss
            mrstft_loss += self.sdstft(reals, own_latents_teacher_decoded) # Distilled model's encoder is compatible with teacher's decoder
            mrstft_loss += self.sdstft(decoded, teacher_decoded) # Distilled model's decoder is compatible with teacher's decoder
            mrstft_loss += self.sdstft(reals, teacher_latents_own_decoded) # Teacher's encoder is compatible with distilled model's decoder
            mrstft_loss /= 4

            latent_loss = F.l1_loss(latents, teacher_latents)

        else:
            mrstft_loss = self.sdstft(reals, decoded)

            latent_loss = torch.tensor(0.).to(reals)

        # Time-domain reconstruction loss
        l1_time_loss = F.l1_loss(reals, decoded)

        if self.warmed_up:
            loss_dis, loss_adv, feature_matching_distance = self.discriminator.loss(reals, decoded)
        else:
            loss_dis = torch.tensor(0.).to(reals)
            loss_adv = torch.tensor(0.).to(reals)
            feature_matching_distance = torch.tensor(0.).to(reals)

        opt_gen, opt_disc = self.optimizers()

        lr_schedulers = self.lr_schedulers()

        sched_gen = None
        sched_disc = None

        if lr_schedulers is not None:
            sched_gen, sched_disc = lr_schedulers

        # Train the discriminator
        if self.global_step % 2 and self.warmed_up:
            loss = loss_dis

            log_dict = {
                'train/discriminator_loss': loss_dis.detach()  ,
                'train/disc_lr': opt_disc.param_groups[0]['lr']
            }

            opt_disc.zero_grad()
            self.manual_backward(loss_dis)
            opt_disc.step()

            if sched_disc is not None:
                # sched step every step
                sched_disc.step()

        # Train the generator 
        else:

            loss_adv = loss_adv * self.loss_config['discriminator']['weights']['adversarial']

            feature_matching_distance = feature_matching_distance * self.loss_config['discriminator']['weights']['feature_matching']

            mrstft_loss = mrstft_loss * self.loss_config['spectral']['weights']['mrstft']

            l1_time_loss = l1_time_loss * self.loss_config['time']['weights']['l1']

            # Combine spectral loss, KL loss, time-domain loss, and adversarial loss
            loss = mrstft_loss + loss_adv + feature_matching_distance + l1_time_loss + latent_loss

            if isinstance(self.autoencoder.bottleneck, VAEBottleneck) or isinstance(self.autoencoder.bottleneck, DACRVQVAEBottleneck) or isinstance(self.autoencoder.bottleneck, RVQVAEBottleneck):
                kl = encoder_info['kl']

                try:
                    kl_weight = self.loss_config['bottleneck']['weights']['kl']
                except:
                    kl_weight = 1e-6

                kl_loss = kl_weight * kl 
                loss = loss + kl_loss

            if isinstance(self.autoencoder.bottleneck, WassersteinBottleneck):
                mmd_loss = encoder_info['mmd']            

                try:
                    mmd_weight = self.loss_config['bottleneck']['weights']['mmd']
                except:
                    mmd_weight = 100

                loss = loss + mmd_weight * mmd_loss

            if isinstance(self.autoencoder.bottleneck, RVQBottleneck) or isinstance(self.autoencoder.bottleneck, RVQVAEBottleneck):
                quantizer_loss = encoder_info['quantizer_loss']
                loss = loss + quantizer_loss
                
            if isinstance(self.autoencoder.bottleneck, DACRVQBottleneck) or isinstance(self.autoencoder.bottleneck, DACRVQVAEBottleneck):
                codebook_loss = encoder_info["vq/codebook_loss"]
                commitment_loss = 0.25 * encoder_info["vq/commitment_loss"]
                loss = loss + codebook_loss + commitment_loss

            if self.use_ema:
                self.autoencoder_ema.update()

            opt_gen.zero_grad()
            self.manual_backward(loss)
            opt_gen.step()

            if sched_gen is not None:
                # scheduler step every step
                sched_gen.step()

            log_dict = {
                'train/loss': loss.detach(),
                'train/mrstft_loss': mrstft_loss.detach(),   
                'train/l1_time_loss': l1_time_loss.detach(),
                'train/loss_adv': loss_adv.detach(),
                'train/feature_matching': feature_matching_distance.detach(),
                'train/latent_std': latents.std().detach(),
                'train/gen_lr': opt_gen.param_groups[0]['lr']
            }

            if self.teacher_model is not None:
                log_dict['train/latent_loss'] = latent_loss.detach()

            if isinstance(self.autoencoder.bottleneck, VAEBottleneck) or isinstance(self.autoencoder.bottleneck, DACRVQVAEBottleneck) or isinstance(self.autoencoder.bottleneck, RVQVAEBottleneck):
                log_dict['train/kl_loss'] = kl_loss.detach()
            
            if isinstance(self.autoencoder.bottleneck, RVQBottleneck):
                log_dict['train/quantizer_loss'] = quantizer_loss.detach()
            
            if isinstance(self.autoencoder.bottleneck, DACRVQBottleneck) or isinstance(self.autoencoder.bottleneck, DACRVQVAEBottleneck):
                log_dict['train/codebook_loss'] = codebook_loss.detach()
                log_dict['train/commitment_loss'] = commitment_loss.detach()
                
            if isinstance(self.autoencoder.bottleneck, WassersteinBottleneck):
                log_dict['train/mmd_loss'] = mmd_loss.detach()
            
        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss
    
    def export_model(self, path):
        if self.autoencoder_ema is not None:
            export_state_dict = {"state_dict": self.autoencoder_ema.ema_model.state_dict()}
        else:
            export_state_dict = {"state_dict": self.autoencoder.state_dict()}
            
        torch.save(export_state_dict, path)
        

class AutoencoderDemoCallback(pl.Callback):
    def __init__(
        self, 
        demo_dl, 
        demo_every=2000,
        sample_size=65536,
        sample_rate=48000
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_samples = sample_size
        self.demo_dl = iter(demo_dl)
        self.sample_rate = sample_rate
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx): 
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return
        
        self.last_demo_step = trainer.global_step

        module.eval()

        try:
            demo_reals, _ = next(self.demo_dl)

            # Remove extra dimension added by WebDataset
            if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
                demo_reals = demo_reals[0]

            encoder_input = demo_reals
            
            encoder_input = encoder_input.to(module.device)

            if module.force_input_mono:
                encoder_input = encoder_input.mean(dim=1, keepdim=True)

            demo_reals = demo_reals.to(module.device)

            with torch.no_grad():
                if module.use_ema:

                    latents = module.autoencoder_ema.ema_model.encode(encoder_input)

                    fakes = module.autoencoder_ema.ema_model.decode(latents)
                else:
                    latents = module.autoencoder.encode(encoder_input)

                    fakes = module.autoencoder.decode(latents)

            #Interleave reals and fakes
            reals_fakes = rearrange([demo_reals, fakes], 'i b d n -> (b i) d n')

            # Put the demos together
            reals_fakes = rearrange(reals_fakes, 'b d n -> d (b n)')

            log_dict = {}
            
            filename = f'recon_{trainer.global_step:08}.wav'
            reals_fakes = reals_fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, reals_fakes, self.sample_rate)

            log_dict[f'recon'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Reconstructed')
            
            log_dict[f'embeddings_3dpca'] = pca_point_cloud(latents)
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(latents))

            log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(reals_fakes))

            trainer.logger.experiment.log(log_dict)
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            raise e
        finally:
            module.train()