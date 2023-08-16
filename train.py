#### RAVE-Latent Diffusion
#### https://github.com/moiseshorta/RAVE-Latent-Diffusion
####
#### Author: Moises Horta Valenzuela / @hexorcismos
#### Year: 2023

import argparse
import torch.multiprocessing as mp
import torch
import os
import datetime
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

from rave_conditioning import RAVEConditioningModel, load_cond_datasets

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")
else:
    device = torch.device("cpu")
current_date = datetime.date.today()


class RaveDataset(Dataset):
    def __init__(self, latent_files):
        self.latent_data = []

        for latent_path in latent_files:
            z = np.load(latent_path)
            z = torch.from_numpy(z).float().squeeze()
            self.latent_data.append(z)

        self.latent_dims = self.latent_data[0].shape[0]
        self.num_latents = self.latent_data[0].shape[-1]

    def __len__(self):
        return len(self.latent_data)

    def __getitem__(self, index):
        return self.latent_data[index]


def load_dataset(latent_folder, split_ratio):
    assert os.path.isdir(latent_folder), f"latent folder '{latent_folder}' not found"

    latent_files = [os.path.join(latent_folder, f)
                    for f in os.listdir(latent_folder) if f.endswith(".npy")]

    assert len(latent_files) > 0, f"no latent files found in '{latent_folder}'"

    random.shuffle(latent_files)

    num_train = int(len(latent_files) * split_ratio)
    train_dataset = RaveDataset(latent_files[:num_train])
    val_dataset = RaveDataset(latent_files[num_train:])

    return train_dataset, val_dataset, num_train, len(latent_files) - num_train


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with a new dataset.")
    parser.add_argument("--name", type=str, default=f"run_{current_date}",
                        help="Name of your training run.")
    parser.add_argument("--latent_folder", type=str, default="./latents/",
                        help="Path to the directory containing the latent files.")
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
    parser.add_argument("--cond_latent_folder", type=str, default="./latents/",
                        help="Path to the directory containing the latent files for conditioning.")
    return parser.parse_args()


def set_seed(seed=664):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
        else:
            print("The checkpoint file does not contain the required keys. Training will start from scratch.")
            start_epoch = 0
    else:
        start_epoch = 0

    return start_epoch


def main():
    # Parse command-line arguments
    args = parse_args()
    latent_folder = args.latent_folder
    checkpoint_path = args.checkpoint_path
    save_out_path = args.save_out_path
    split_ratio = args.split_ratio
    batch_size = args.batch_size
    save_interval = args.save_interval
    batch_size = args.batch_size

    global best_loss
    global best_epoch
    best_epoch = None
    best_loss = float('inf')

    os.makedirs(args.save_out_path, exist_ok=True)

    set_seed(664)

    if not args.cond_latent_folder:
        res = load_dataset(latent_folder, split_ratio)
        train_dataset, val_dataset, num_train_files, num_val_files = res

        model = DiffusionModel(
            net_t=UNetV0,
            in_channels=train_dataset.latent_dims,
            channels=[256, 256, 256, 256, 512, 512, 512, 768, 768],
            factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
            items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
            attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
            attention_heads=12,
            attention_features=64,
            diffusion_t=VDiffusion,
            sampler_t=VSampler,
        )
    else:
        res = load_cond_datasets(latent_folder, args.cond_latent_folder, split_ratio)
        train_dataset, val_dataset, num_train_files, num_val_files = res
        model = RAVEConditioningModel(in_channels=train_dataset.latent_dims,
                                      embedding_features=train_dataset.cond_latent_dims)

    model = model.to(device)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=8, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=8, pin_memory=True)

    print("Model Architecture:")
    print(model)
    print("\nModel Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    print(f"Number of trainable parameters: {trainable_params}\n")

    print("Training:", num_train_files, "files")
    print("Validation:", num_val_files, "files")

    if checkpoint_path is not None:
        print(f"Resuming training from: {checkpoint_path}\n")

    if not args.finetune:
        ##### TRAIN FROM SCRATCH
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_steps, gamma=0.99)
        start_epoch = resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler)
    else:
        #### FINETUNE
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5) # Change the learning rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.scheduler_steps, eta_min=1e-6) # Replace the StepLR scheduler with the CosineAnnealingLR scheduler
        start_epoch = resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler)

    accumulation_steps = args.accumulation_steps

    for i in range(start_epoch, args.max_epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_data_loader):
            if args.conditioning is None:
                batch_rave_tensor = batch.to(device)
                loss = model(batch_rave_tensor)
            else:
                x, c = batch
                x_rave_tensor = x.to(device)
                cond_rave_tensor = c.to(device)
                loss = model(x_rave_tensor, conditioning=cond_rave_tensor)

            train_loss += loss.item()

            if (step + 1) % accumulation_steps == 0:
                loss = loss / accumulation_steps
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= len(train_data_loader)
        print(f"Epoch {i+1}, train loss: {train_loss}")

        with torch.no_grad():
            model.eval()

            val_loss = 0
            for batch in val_data_loader:
                if args.conditioning is None:
                    batch_rave_tensor = batch.to(device)
                    loss = model(batch_rave_tensor)
                else:
                    x, c = batch
                    x_rave_tensor = x.to(device)
                    cond_rave_tensor = c.to(device)
                    loss = model(x_rave_tensor, conditioning=cond_rave_tensor)

                val_loss += loss.item()

            val_loss /= len(val_data_loader)
            print(f"Epoch {i+1}, validation loss: {val_loss}")

            # Save the best model
            if val_loss < best_loss:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': i
                }
                new_checkpoint_path = f"{save_out_path}/{args.name}_best_epoch{i}_loss_{val_loss}.pt"
                torch.save(checkpoint, new_checkpoint_path)
                print(f"Saved new best model with validation loss {val_loss}")

                # If a previous best model exists, remove it
                if best_epoch is not None:
                    old_checkpoint_path = f"{save_out_path}/{args.name}_best_epoch{best_epoch}_loss_{best_loss}.pt"
                    if os.path.exists(old_checkpoint_path):
                        os.remove(old_checkpoint_path)
                best_epoch = i
                best_loss = val_loss

            # Save a checkpoint every n epochs
            if i % save_interval == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': i
                }
                torch.save(checkpoint, f"{save_out_path}/{args.name}_epoch{i}.pt")

            scheduler.step()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
