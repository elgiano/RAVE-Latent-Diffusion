import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad

import numpy as np
import os
import random


class RaveDataset(Dataset):
    def __init__(self, latent_files, latent_length):
        self.num_files = len(latent_files)
        self.latent_length = latent_length
        self.latent_data = []

        for latent_path in latent_files:
            z = np.load(latent_path)
            z = torch.from_numpy(z).float()
            z = pad(z, (0, -z.size(-1) % latent_length)).split(latent_length, -1)
            self.latent_data += z

        self.latent_dims = self.latent_data[0].shape[0]
        self.num_latents = self.latent_data[0].shape[-1]

    def __len__(self):
        return len(self.latent_data)

    def __getitem__(self, index):
        return self.latent_data[index]


def load_dataset(latent_folder, latent_length, split_ratio):
    latent_files = [os.path.join(latent_folder, f)
                    for f in os.listdir(latent_folder) if f.endswith(".npy")]

    assert len(latent_files) > 0, f"no latent files found in '{latent_folder}'"

    random.shuffle(latent_files)

    num_train = int(len(latent_files) * split_ratio)
    train_dataset = RaveDataset(latent_files[:num_train], latent_length)
    val_dataset = RaveDataset(latent_files[num_train:], latent_length)

    return train_dataset, val_dataset


class RAVEConditioningDataset(Dataset):
    def __init__(self, latent_files, conditioning_files, latent_length):
        self.num_files = len(latent_files)
        self.latent_length = latent_length
        self.latent_data = []
        self.conditioning_data = []

        for latent_path in latent_files:
            z = np.load(latent_path)
            z = torch.from_numpy(z).float()
            z = pad(z, (0, -z.size(-1) % latent_length)).split(latent_length, -1)
            self.latent_data += z

        for latent_path in conditioning_files:
            z = np.load(latent_path)
            z = torch.from_numpy(z).float()
            z = pad(z, (0, -z.size(-1) % latent_length)).split(latent_length, -1)
            self.conditioning_data += z

        self.latent_dims = self.latent_data[0].shape[0]
        self.cond_latent_dims = self.conditioning_data[0].shape[0]
        self.num_latents = self.latent_data[0].shape[-1]

    def __len__(self):
        return len(self.latent_data)

    def __getitem__(self, index):
        return self.latent_data[index], self.conditioning_data[index]


def shuffle_unison(a, b):
    msg = "train and condition datasets must be the same size"
    assert len(a) == len(b), msg
    p = np.random.permutation(len(a))
    return list(np.array(a)[p]), list(np.array(b)[p])


def load_cond_datasets(latent_folder, cond_folder, latent_length, split_ratio):
    latent_files = [os.path.join(latent_folder, f)
                    for f in os.listdir(latent_folder) if f.endswith(".npy")]
    assert len(latent_files) > 0, f"no latent files found in '{latent_folder}'"

    cond_files = [os.path.join(cond_folder, f)
                  for f in os.listdir(cond_folder) if f.endswith(".npy")]
    assert len(cond_files) > 0, f"no cond latent files found in '{cond_folder}'"

    latent_files, cond_files = shuffle_unison(latent_files, cond_files)

    num_train = int(len(latent_files) * split_ratio)
    train_dataset = RAVEConditioningDataset(latent_files[:num_train], cond_files[:num_train], latent_length)
    val_dataset = RAVEConditioningDataset(latent_files[num_train:], cond_files[num_train:], latent_length)

    return train_dataset, val_dataset


class RAVESelfConditioningDataset(RAVEConditioningDataset):
    def __init__(self, latent_files, latent_length):
        self.num_files = len(latent_files)
        self.latent_length = latent_length
        self.latent_data = []

        for latent_path in latent_files:
            z = np.load(latent_path)
            z = torch.from_numpy(z).float()
            z = pad(z, (0, -z.size(-1) % latent_length)).split(latent_length, -1)
            self.latent_data += z

        self.latent_dims = self.cond_latent_dims = self.latent_data[0].shape[0]
        self.num_latents = self.latent_data[0].shape[-1]

    def __len__(self):
        return len(self.latent_data)

    def __getitem__(self, index):
        return self.latent_data[index], self.latent_data[index-1]

def load_self_cond_datasets(latent_folder, latent_length, split_ratio):
    latent_files = [os.path.join(latent_folder, f)
                    for f in os.listdir(latent_folder) if f.endswith(".npy")]
    assert len(latent_files) > 0, f"no latent files found in '{latent_folder}'"

    random.shuffle(latent_files)

    num_train = int(len(latent_files) * split_ratio)
    train_dataset = RAVESelfConditioningDataset(latent_files[:num_train], latent_length)
    val_dataset = RAVESelfConditioningDataset(latent_files[num_train:], latent_length)

    return train_dataset, val_dataset
