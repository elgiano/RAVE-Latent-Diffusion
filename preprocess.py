# RAVE-Latent Diffusion
# Fork by: Gianluca Elia / elgiano
# Original author: Moises Horta Valenzuela / @hexorcismos
# https://github.com/elgiano/RAVE-Latent-Diffusion
# Year: 2023
#
# Preprocessing encodes each audiofile in a npy file with RAVE latents

import argparse
import torch
import os
import numpy as np
from pathlib import Path
import librosa
from tqdm import tqdm
# for hashing paths
import hashlib


if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")
else:
    device = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_folder', type=str, default='/path/to/audio_folder',
                        help='Path to the folder containing audio files.')
    parser.add_argument('--rave_model', type=str, default='/path/to/rave_model',
                        help='Path to the exported (.ts) RAVE model.')
    parser.add_argument('--latent_folder', type=str, default='latents',
                        help='Path to the folder where RAVE latent files will be saved.')
    parser.add_argument('--max_chunk_size', type=int, default=1000,
                        help='Maximum number of latents to be encoded by RAVE at once')
    parser.add_argument('--sample_rate', type=int, default=48000, choices=[44100, 48000],
                        help='Set sample rate for the audio files, if not readable from RAVE model.')
    parser.add_argument('--normalize_latents', action="store_true",
                        help='Normalize latents (default: False): z = (z - z.mean) / z.std')
    parser.add_argument('--extensions', type=str, nargs="+",
                        default=['wav', 'opus', 'mp3', 'aac', 'flac'],
                        help='Extensions to search for in audio_folder')
    return parser.parse_args()


def encode_latents(rave, audio, max_chunk_size, normalize_latents=False):
    rave_block_size = rave.decode_params[1].item()
    with torch.no_grad():
        x = torch.from_numpy(audio)
        x = x.split(rave_block_size * max_chunk_size)

        latents = None
        for chunk in tqdm(x, desc="Encoding file with RAVE", leave=False):
            chunk = chunk.reshape(1, 1, -1)
            z = rave.encode(chunk.to(device))[0]
            if device.type != 'cpu':
                z = z.cpu()
            # z = torch.nn.functional.pad(z, (0, latent_length - z.shape[2]))
            z = z.detach().numpy()
            if latents is None:
                latents = z
            else:
                latents = np.concatenate((latents, z), -1)

    # Why should we normalize latents?
    if normalize_latents:
        latents = (latents - latents.mean()) / latents.std()
    return latents


def main():
    args = parse_args()

    os.makedirs(args.latent_folder, exist_ok=True)

    rave = torch.jit.load(args.rave_model).to(device)

    sample_rate = args.sample_rate
    if hasattr(rave, 'sr'):
        print(f"RAVE model's sample rate: {rave.sr}")
        sample_rate = rave.sr

    audio_folder = Path(args.audio_folder)
    audio_files = [f for ext in args.extensions for f in audio_folder.rglob(f'*.{ext}')]

    pbar = tqdm(audio_files)
    for audio_file in pbar:
        relpath = os.path.relpath(audio_file, args.audio_folder)
        pbar.set_description(relpath)

        audio, _ = librosa.load(os.path.abspath(audio_file),
                                sr=sample_rate, mono=True)

        latents = encode_latents(rave, audio,
                                 args.max_chunk_size, args.normalize_latents)

        output_dir = os.path.join(args.latent_folder, relpath)
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{os.path.splitext(os.path.basename(audio_file))[0]}.npy"
        np.save(os.path.join(output_dir, output_file), latents)

    print('Done encoding RAVE latents')
    print('Path to latents:', args.latent_folder)


if __name__ == "__main__":
    main()
