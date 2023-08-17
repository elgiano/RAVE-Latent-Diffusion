import torch
from model import LightningDiffusionModel
import nn_tilde


class ScriptedRAVELD(nn_tilde.Module):

    def __init__(self,
                 pretrained: LightningDiffusionModel) -> None:
        super().__init__()

        self.net = pretrained.net
        self.diffusion = pretrained.diffusion
        self.sampler = pretrained.sampler

        self.latent_dims = pretrained.in_channels
        self.min_latent_length = pretrained.num_latents

        self.register_method(
            "forward",
            in_channels=1,
            in_ratio=ratio_encode,
            out_channels=self.latent_dims,
            out_ratio=ratio_encode,
            input_labels=['(signal) Temperature'],
            output_labels=[
                f'(signal) Generated latent {channel}'
                for channel in channels
            ],
        )

    def forward(self, temperature):

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        rave_dims = self.latent_dims
        z_length = self.min_latent_length

        noise = torch.randn(1, rave_dims, z_length)
        noise = noise * temperature

        diffusion_steps = 100

        diff = self.sample(noise, num_steps=diffusion_steps, show_progress=False)
        diff = (diff - diff.mean()) / diff.std()

        return diff

