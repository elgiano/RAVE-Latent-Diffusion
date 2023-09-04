import numpy as np
import torch
from torch.nn.functional import pad
from os import path, listdir


def zeropad_to_multiple(x, base):
    return pad(x, (0, -x.size(-1) % base))


def split_in_chunks(x, chunk_size):
    zeropad_to_multiple(x, chunk_size).split(chunk_size, -1)


def read_latents(latent_path, latent_length):
    z = np.load(latent_path)
    z = torch.from_numpy(z).float()
    z = pad(z, (0, -z.size(-1) % latent_length)).split(latent_length, -1)
    return list(z)


def read_latent_folder(latent_folder, latent_length):
    latent_files = [path.join(latent_folder, f)
                    for f in listdir(latent_folder) if f.endswith(".npy")]
    assert len(latent_files) > 0, f"no latent files found in '{latent_folder}'"
    latent_data = [read_latents(p, latent_length) for p in latent_files]
    latent_dims = set(z[0].shape[0] for z in latent_data)
    msg = f"latent files in '{latent_folder}' have different latent dims: {latent_dims}"
    assert len(latent_dims) == 1, msg
    return latent_data


