import os
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from a_unet import Module
from typing import Type, Callable, Optional


def RAVEConditioningPlugin(
    net_t: Type[nn.Module]
) -> Callable[..., nn.Module]:
    """Adds RAVE conditioning"""
    # embedder = embedder if exists(embedder) else T5Embedder()
    # msg = "RAVEConditioningPlugin embedder requires embedding_features attribute"
    # assert hasattr(embedder, "embedding_features"), msg
    # features: int = embedder.embedding_features  # type: ignore

    def Net(embedding_features: int, **kwargs) -> nn.Module:
        # msg = f"TextConditioningPlugin requires embedding_features={features}"
        # assert embedding_features == features, msg
        net = net_t(embedding_features=embedding_features, **kwargs)  # type: ignore

        def forward(
            x: Tensor, *args, conditioning: Tensor, embedding: Optional[Tensor] = None, **kwargs
        ):
            if embedding is not None:
                conditioning = torch.cat([conditioning, embedding])

            return net(x, embedding=conditioning, *args, **kwargs)

        return Module([net], forward)  # type: ignore

    return Net


class RAVEConditioningDataset(Dataset):
    def __init__(self, latent_files, conditioning_files):
        self.latent_data = []
        self.conditioning = bool(conditioning_files)
        self.conditioning_data = []

        for latent_path in latent_files:
            z = np.load(latent_path)
            z = torch.from_numpy(z).float().squeeze()
            self.latent_data.append(z)

        for latent_path in conditioning_files:
            z = np.load(latent_path)
            z = torch.from_numpy(z).float().squeeze()
            self.conditioning_data.append(z)

        self.latent_dims = self.latent_data[0].shape[0]
        self.cond_latent_dims = self.conditioning_data[0].shape[0]
        self.num_latents = self.latent_data[0].shape[-1]

    def __len__(self):
        return len(self.latent_data)

    def __getitem__(self, index):
        if self.conditioning:
            return self.latent_data[index], self.conditioning_data[index]
        else:
            return self.latent_data[index]


def RAVEConditioningModel(in_channels, embedding_features):
    return DiffusionModel(
        net_t=RAVEConditioningPlugin(UNetV0),
        in_channels=in_channels,
        channels=[256, 256, 256, 256, 512, 512, 512, 768, 768],
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
        attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
        cross_attentions=[0, 0, 0, 1, 1, 1, 1, 1, 1],
        attention_heads=12,
        attention_features=64,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
        embedding_features=embedding_features,
    )


def shuffle_unison(a, b):
    msg = "train and condition datasets must be the same size"
    assert len(a) == len(b), msg
    p = np.random.permutation(len(a))
    return list(np.array(a)[p]), list(np.array(b)[p])


def load_cond_datasets(latent_folder, cond_folder, split_ratio):
    assert os.path.isdir(latent_folder), f"latent folder '{latent_folder}' not found"
    latent_files = [os.path.join(latent_folder, f)
                    for f in os.listdir(latent_folder) if f.endswith(".npy")]
    assert len(latent_files) > 0, f"no latent files found in '{latent_folder}'"

    assert os.path.isdir(cond_folder), f"cond folder '{cond_folder}' not found"
    cond_files = [os.path.join(cond_folder, f)
                  for f in os.listdir(cond_folder) if f.endswith(".npy")]
    assert len(cond_files) > 0, f"no cond latent files found in '{cond_folder}'"

    latent_files, cond_files = shuffle_unison(latent_files, cond_files)

    num_train = int(len(latent_files) * split_ratio)
    train_dataset = RAVEConditioningDataset(latent_files[:num_train], cond_files[:num_train])
    val_dataset = RAVEConditioningDataset(latent_files[num_train:], cond_files[num_train:])

    return train_dataset, val_dataset, num_train, len(latent_files) - num_train
