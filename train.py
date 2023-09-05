#### RAVE-Latent Diffusion
#### https://github.com/moiseshorta/RAVE-Latent-Diffusion
####
#### Author: Moises Horta Valenzuela / @hexorcismos
#### Year: 2023

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import GPUtil as gpu

import argparse
import datetime
import numpy as np
import random

from raveld.model import LightningDiffusionModel, RAVELDModel, RAVELDConditioningModel
from raveld.utils import read_latent_folder


class RaveDataset(torch.utils.data.Dataset):
    def __init__(self, latent_data):
        self.latent_data = latent_data

    def __len__(self):
        return len(self.latent_data)

    def __getitem__(self, index):
        return self.latent_data[index]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with a new dataset.")
    parser.add_argument("--name", type=str, default=f"run_{datetime.date.today()}",
                        help="Name of your training run.")
    parser.add_argument("--latent_folder", type=str, default="./latents/",
                        help="Path to the directory containing the latent files.")
    parser.add_argument("--latent_length", type=int, default=2048,
                        choices=[2**n for n in range(4, 15)],
                        help="Length of latent sequences to train on (perceptual field).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Resume training from checkpoint.")
    parser.add_argument("--save_out_path", type=str, default="./runs/",
                        help="Path to the directory where the model checkpoints will be saved.")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Ratio for splitting the dataset into training and validation sets.")
    parser.add_argument("--split_files", action="store_true",
                        help="Split files for validation. Otherwise, split latents within files (default)")
    parser.add_argument("--max_epochs", type=int, default=25000,
                        help="Maximum epochs to train model.")
    parser.add_argument("--scheduler_steps", type=int, default=100,
                        help="Diffusion steps for scheduler.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training.")
    parser.add_argument("--accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Interval (number of epochs) at which to save the model.")
    parser.add_argument("--finetune", type=bool, default=False,
                        help="Finetune model.")
    parser.add_argument("--cond_latent_folder", type=str,
                        help="Path to the directory containing the latent files for conditioning.")
    parser.add_argument("--self_conditioning", action='store_true',
                        help="Set conditioning on previous latents from same dataset")
    parser.add_argument("--gpu", type=int, nargs="+", default=None,
                        help="GPU to use")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of workers to spawn for dataset loading")
    return parser.parse_args()


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


def set_seed(seed=664):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse command-line arguments
    args = parse_args()
    latent_folder = args.latent_folder
    latent_length = args.latent_length
    checkpoint_path = args.checkpoint_path
    save_out_path = args.save_out_path
    split_ratio = args.split_ratio
    split_files = args.split_files
    batch_size = args.batch_size
    num_workers = args.workers

    set_seed(664)

    cond_folder = args.cond_latent_folder
    self_cond = args.self_conditioning
    conditioning = self_cond or cond_folder

    # load datasets
    latent_data = read_latent_folder(latent_folder, latent_length)
    print(f"Read {len(latent_data)} latent files in {latent_folder}")
    latent_dims = latent_data[0][0].shape[0]

    if conditioning:
        if cond_folder:
            # read conditioning data
            cond_data = read_latent_folder(cond_folder, latent_length)
            print(f"Read {len(cond_data)} conditioning latent files in {cond_folder}")
        elif args.self_conditioning:
            # cond_data is shifted latents
            cond_data = [(torch.zeros_like(z[0]),) + z[:-1] for z in latent_data]
        # zip (x,c) for each window for each file
        latent_data = [list(zip(x, c)) for x, c in zip(latent_data, cond_data)]

    # split train-val
    train_data, val_data = split_train_val(latent_data, split_ratio, split_files)
    train_dataset = RaveDataset(tuple(train_data))
    val_dataset = RaveDataset(tuple(val_data))

    train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

    print(f"Training: {len(train_dataset)} sequences")
    print(f"Validation: {len(val_dataset)} sequences")

    # make model

    if checkpoint_path is not None:
        print(f"Resuming training from: {checkpoint_path}\n")
        try:
            model = LightningDiffusionModel.load_from_checkpoint(checkpoint_path)
        except TypeError:
            cls = RAVELDConditioningModel if conditioning else RAVELDModel
            model = cls(latent_dims, latent_length)
            state_dict = torch.load(checkpoint_path)["state_dict"]
            model.load_state_dict(state_dict)

        # is checkpoint compatible with dataset?
        msg = f"checkpoint latent_dims ({model.in_channels}) doesn't match dataset ({latent_dims})"
        assert model.in_channels == latent_dims, msg
        msg = f"checkpoint latent_length ({model.latent_length}) doesn't match dataset ({latent_length})"
        assert model.latent_length == latent_length, msg
        if conditioning:
            msg = f"checkpoint embedding_features ({model.embedding_features}) doesn't match dataset ({latent_length})"
            assert model.embedding_features == latent_length, msg
        else:
            msg = "checkpoint had conditioning, but no training conditioning is provided"
            assert model.embedding_features is None, msg

    elif conditioning:
        model = RAVELDConditioningModel(latent_dims, latent_length)
    else:
        model = RAVELDModel(latent_dims, latent_length)

    # config training

    accelerator = None
    devices = None
    if args.gpu == [-1]:
        pass
    elif torch.cuda.is_available():
        accelerator = "cuda"
        devices = args.gpu or gpu.getAvailable()
        print("Selected GPU:", devices)

    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor="val_loss",
                                     save_last=True,
                                     save_top_k=1,
                                     every_n_epochs=1,
                                     auto_insert_metric_name=True,
                                     filename='best-{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}'
                                     ),
        pl.callbacks.ModelCheckpoint(every_n_epochs=args.save_interval,
                                     filename='{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}'
                                     ),
    ]

    logger = pl.loggers.TensorBoardLogger(save_out_path, name=args.name)

    # train

    trainer = pl.Trainer(
        resume_from_checkpoint=checkpoint_path,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulation_steps,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks
    )

    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    main()
