import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.functional import pad
import numpy as np
from os import path, listdir
import random


class RaveDataset(torch.utils.data.Dataset):
    def __init__(self, latent_data):
        self.latent_data = latent_data

    def __len__(self):
        return len(self.latent_data)

    def __getitem__(self, index):
        return self.latent_data[index]


def read_latents(latent_path, latent_length):
    z = np.load(latent_path)
    z = torch.from_numpy(z).float()
    z = pad(z, (0, -z.size(-1) % latent_length)).split(latent_length, -1)
    return list(z)


def read_latent_folder(latent_folder, latent_length):
    latent_files = [path.join(latent_folder, f)
                    for f in listdir(latent_folder) if f.endswith(".npy")]
    assert len(latent_files) > 0, f"no latent files found in '{latent_folder}'"
    # sort by ctime to align latent and conditioning files
    latent_files.sort(key=lambda x: path.getctime(x))
    latent_data = [read_latents(p, latent_length) for p in latent_files]
    latent_dims = set(z[0].shape[0] for z in latent_data)
    msg = f"latent files in '{latent_folder}' have different latent dims: {latent_dims}"
    assert len(latent_dims) == 1, msg
    return latent_data


def split_train_val(latent_data, split_ratio, split_files):
    train_data, val_data = ([], [])
    if split_files:
        random.shuffle(latent_data)
        num_train = int(len(latent_data) * split_ratio)
        for z in latent_data[:num_train]:
            train_data += z
        for z in latent_data[num_train:]:
            val_data += z
    else:
        for z in latent_data:
            random.shuffle(z)
            num_train = int(len(z) * split_ratio)
            train_data += z[:num_train]
            val_data += z[num_train:]
    return train_data, val_data


class RAVEDataModule(pl.LightningDataModule):
    def __init__(self, seq_len, latent_folder,
                 cond_folder=None, cond_recursive=False,
                 split_ratio=0.8, split_files=False,
                 batch_size=32, num_workers=4):
        self.seq_len = seq_len
        self.batch_size, self.num_workers = batch_size, num_workers
        self.split_ratio, self.split_files = split_ratio, split_files
        self.latent_folder = latent_folder
        self.conditioning = cond_folder is not None or cond_recursive
        self.cond_folder = cond_folder
        self.cond_recursive = cond_recursive

    def setup(self, stage):
        latent_data = read_latent_folder(self.latent_folder, self.seq_len)
        print(f"Read {len(latent_data)} latent files in {self.latent_folder}")
        self.latent_dims = latent_data[0][0].shape[0]
        if self.conditioning:
            if self.cond_recursive:
                # cond_data is shifted latents
                cond_data = [(torch.zeros_like(z[0]),) + z[:-1] for z in latent_data]
            elif self.cond_folder:
                # read conditioning data
                cond_data = read_latent_folder(self.cond_folder, self.seq_len)
                print(f"Read {len(cond_data)} conditioning latent files in {self.cond_folder}")
            # zip (x,c) for each window for each file
            latent_data = [list(zip(x, c)) for x, c in zip(latent_data, cond_data)]

        # split train-val
        train_data, val_data = split_train_val(latent_data, self.split_ratio, self.split_files)
        self.train_dataset = RaveDataset(tuple(train_data))
        self.val_dataset = RaveDataset(tuple(val_data))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


