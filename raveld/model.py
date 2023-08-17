import torch
from torch import optim
import pytorch_lightning as pl

from audio_diffusion_pytorch import VDiffusion, VSampler
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

        self.finetune = finetune
        self.scheduler_steps = scheduler_steps
        self.conditioning = kwargs['embedding_features'] is not None
        # self.example_input_array = torch.zeros((1, kwargs['in_channels'], num_latents))

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

