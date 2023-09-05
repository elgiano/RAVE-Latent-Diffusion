import torch
from torch import optim
import pytorch_lightning as pl

from audio_diffusion_pytorch import UNetV0, VDiffusion, VSampler
from audio_diffusion_pytorch.utils import groupby

from typing import Callable
from torch import Tensor


class LightningDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        net_t: Callable,
        diffusion_t: Callable = VDiffusion,
        sampler_t: Callable = VSampler,
        loss_fn: Callable = torch.nn.functional.mse_loss,
        dim: int = 1,
        finetune: bool = False,
        scheduler_steps: int = 100,
        num_latents: int = 2048,
        **kwargs,
    ):
        super().__init__()
        diffusion_kwargs, kwargs = groupby("diffusion_", kwargs)
        sampler_kwargs, kwargs = groupby("sampler_", kwargs)

        self.net = net_t(dim=dim, **kwargs)
        self.diffusion = diffusion_t(net=self.net, loss_fn=loss_fn, **diffusion_kwargs)
        self.sampler = sampler_t(net=self.net, **sampler_kwargs)

        self.latent_length = num_latents
        self.finetune = finetune
        self.scheduler_steps = scheduler_steps
        self.conditioning = 'embedding_features' in kwargs
        # self.example_input_array = torch.zeros((1, kwargs['in_channels'], num_latents))
        self.save_hyperparameters()

    def forward(self, *args, **kwargs) -> Tensor:
        return self.diffusion(*args, **kwargs)

    def configure_optimizers(self):
        if self.finetune:
            o = optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-5)
            s = optim.lr_scheduler.StepLR(o, gamma=0.99,
                                          step_size=self.scheduler_steps)
        else:
            o = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
            s = optim.lr_scheduler.CosineAnnealingLR(o, eta_min=1e-6,
                                                     T_max=self.scheduler_steps)
        return {'optimizer': o, 'lr_scheduler': s}

    @torch.no_grad()
    def sample(self, *args, **kwargs) -> Tensor:
        return self.sampler(*args, **kwargs)

    def training_step(self, train_batch, batch_idx):
        if self.conditioning:
            x, c = train_batch
            loss = self.forward(x, embedding=c)
        else:
            loss = self.forward(train_batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        if self.conditioning:
            x, c = val_batch
            loss = self.forward(x, embedding=c)
        else:
            loss = self.forward(val_batch)
        self.log("val_loss", loss)
        return loss


def RAVELDModel(latent_dims, latent_length):
    return LightningDiffusionModel(
        net_t=UNetV0,
        in_channels=latent_dims,
        channels=[256, 256, 256, 256, 512, 512, 512, 768, 768],
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
        attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
        attention_heads=12,
        attention_features=64,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
        num_latents=latent_length
    )


def RAVELDConditioningModel(latent_dims, latent_length):
    return LightningDiffusionModel(
        net_t=UNetV0,
        in_channels=latent_dims,
        channels=[256, 256, 256, 256, 512, 512, 512, 768, 768],
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
        attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
        cross_attentions=[0, 0, 0, 1, 1, 1, 1, 1, 1],
        attention_heads=12,
        attention_features=64,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
        num_latents=latent_length,
        embedding_features=latent_length,
        embedding_max_length=latent_length,
        use_embedding_cfg=True,
    )
