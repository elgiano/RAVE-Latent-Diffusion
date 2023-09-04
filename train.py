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
from raveld.dataset import load_dataset, load_cond_datasets, load_self_cond_datasets

current_date = datetime.date.today()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with a new dataset.")
    parser.add_argument("--name", type=str, default=f"run_{current_date}",
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
    batch_size = args.batch_size
    num_workers = args.workers

    set_seed(664)

    conditioning = args.self_conditioning or args.cond_latent_folder

    # load datasets

    if not conditioning:
        res = load_dataset(latent_folder, latent_length, split_ratio)
        train_dataset, val_dataset = res
    else:
        if args.self_conditioning:
            res = load_self_cond_datasets(latent_folder,
                                          latent_length, split_ratio)
        else:
            res = load_cond_datasets(latent_folder, args.cond_latent_folder,
                                     latent_length, split_ratio)
        train_dataset, val_dataset = res

    train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

    print(f"Training: {train_dataset.num_files} files, {len(train_dataset)} sequences")
    print(f"Validation: {val_dataset.num_files} files, {len(val_dataset)} sequences")

    # make model

    if checkpoint_path is not None:
        print(f"Resuming training from: {checkpoint_path}\n")
        model = LightningDiffusionModel.load_from_checkpoint(checkpoint_path)
    elif conditioning:
        model = RAVELDConditioningModel(train_dataset.latent_dims, train_dataset.latent_length)
    else:
        model = RAVELDModel(train_dataset.latent_dims, train_dataset.latent_length)

    # print("Model Architecture:")
    # print(model)

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
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_last=True,
            every_n_epochs=args.save_interval
        ),
    ]

    logger = pl.loggers.TensorBoardLogger(save_out_path, name=args.name)

    # train

    trainer = pl.Trainer(
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
