#### RAVE-Latent Diffusion
#### https://github.com/moiseshorta/RAVE-Latent-Diffusion
####
#### Author: Moises Horta Valenzuela / @hexorcismos
#### Year: 2023

import librosa
import argparse
import os
import torch
import numpy as np
import random
import soundfile as sf
from raveld.model import LightningDiffusionModel, RAVELDConditioningModel, RAVELDModel
from raveld.utils import zeropad_to_multiple

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def get_latent_dim(rave):
    return rave.decode_params[0].item()

# Parse the input arguments for the script.
def parse_args():
    parser = argparse.ArgumentParser(description="Generate RAVE latents using diffusion model.")
    parser.add_argument("--model_path", type=str, required=True, default=None, help="Path to the pretrained diffusion model checkpoint.")
    parser.add_argument("--rave_model", type=str, required=True, default=None, help="Path to the pretrained RAVE model (.ts).")
    parser.add_argument("--sample_rate", type=int, default=None, choices=[44100, 48000], help="Sample rate for generated audio. Should match samplerate of RAVE model.")
    parser.add_argument("--diffusion_steps", type=int, default=100, help="Number of steps for denoising diffusion.")
    parser.add_argument("--seed", type=int, default=random.randint(0,2**31-1), help="Random seed for generation.")
    parser.add_argument("--latent_length", type=int, default=2048, help="latent_length the model was trained with")
    parser.add_argument("--length_mult", type=int, default=1, help="Multiply the duration of output by default model window.")
    parser.add_argument("--output_path", type=str, default="./", help="Path to the output audio file.")
    parser.add_argument("--num", type=int, default=1, help="Number of audio to generate.")
    parser.add_argument("--name", type=str, default="out", help="Name of audio to generate.")
    parser.add_argument("--lerp", type=bool, default=False, help="Interpolate between two seeds.")
    parser.add_argument("--lerp_factor", type=float, default=1.0, help="Interpolating factor between two seeds.")
    parser.add_argument("--seed_a", type=int, default=random.randint(0,2**31-1), help="Starting seed for interpolation.")
    parser.add_argument("--seed_b", type=int, default=random.randint(0,2**31-1), help="Ending seed for interpolation.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature of the random noise before diffusion.")
    parser.add_argument("--conditioning", type=str, help="Audio or .pt latents file for conditioning")
    parser.add_argument("--rave_conditioning", type=str, help="RAVE model to encode the conditioning file. Defaults to --rave_model")
    parser.add_argument("--regenerate", type=str, help="Audio file to regenerate")
    return parser.parse_args()


def slerp(val, low, high):
    omega = torch.acos((low/torch.norm(low, dim=2, keepdim=True) * high/torch.norm(high, dim=2, keepdim=True)).sum(dim=2, keepdim=True).clamp(-1, 1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so) * low + (torch.sin(val*omega)/so) * high
    return res


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Generate the audio using the provided models and settings.
def generate_audio(model, rave, args, seed):
    with torch.no_grad():
        set_seed(seed)
        rave_dims = get_latent_dim(rave)
        z_length = model.latent_length * args.length_mult

        noise = torch.randn(1, rave_dims, z_length).to(device)
        noise = noise * args.temperature

        rave_model_name = os.path.basename(args.rave_model).split(".")[0]
        diffusion_model_name = os.path.basename(args.model_path)

        print(f"Generating {z_length} latent codes with Diffusion model:", diffusion_model_name)
        print("Decoding using RAVE Model:", rave_model_name)
        print("Seed:", seed)

        model.eval()

        ### GENERATING WITH .PT FILE
        diff = model.sample(noise, num_steps=args.diffusion_steps, show_progress=True)
        # diff = model(noise)
        # noise = diff

        diff = (diff - diff.mean()) / diff.std()

        rave = rave.cpu()
        diff = diff.cpu()
        print("Decoding using RAVE Model...")
        y = decode_latents(diff, rave)

        path = f'{args.output_path}/rave-latent_diffusion_seed{seed}_{args.name}_{rave_model_name}.wav'
        print(f"Writing {path}")
        sf.write(path, y, args.sample_rate)

# Generate audio by slerping between two diffusion generated RAVE latents.
def interpolate_seeds(model, rave, args, seed):
    with torch.no_grad():
        set_seed(seed)

        z_length = model.latent_length * args.length_mult

        rave_dims = get_latent_dim(rave)

        torch.manual_seed(args.seed_a)
        noise1 = torch.randn(1, rave_dims, z_length).to(device) * args.temperature
        torch.manual_seed(args.seed_b)
        noise2 = torch.randn(1, rave_dims, z_length).to(device) * args.temperature

        rave_model_name = os.path.basename(args.rave_model).split(".")[0]
        diffusion_model_name = os.path.basename(args.model_path)

        print(f"Generating {z_length} latent codes with Diffusion model:", os.path.basename(args.model_path))
        print("Decoding using RAVE Model:", os.path.basename(args.rave_model))
        print("Interpolating with factor", args.lerp_factor)
        print("Seed A:", args.seed_a)
        print("Seed B:", args.seed_b)

        model.eval()

        diff1 = model.sample(noise1, num_steps=args.diffusion_steps, show_progress=True)
        diff2 = model.sample(noise2, num_steps=args.diffusion_steps, show_progress=True)
        diff = slerp(torch.linspace(0., args.lerp_factor, z_length).to(device), diff1, diff2)

        diff = (diff - diff.mean()) / diff.std()

        rave = rave.cpu()
        diff = diff.cpu()
        print("Decoding using RAVE Model...")
        y = decode_latents(diff, rave)
        path = f'{args.output_path}/rave-latent_diffusion_seed{seed}_{args.name}_{rave_model_name}_slerp.wav'
        print(f"Writing {path}")
        sf.write(path, y, args.sample_rate)


def encode_audiofile(encoder, audiofile_path):
    with torch.no_grad():
        cond_y, _ = librosa.load(audiofile_path, sr=encoder.sr, mono=True)
        cond_y = torch.tensor(cond_y.reshape((1, 1, -1))).to(device)
        cond_latents = encoder.encode(cond_y)

    return cond_latents


def regenerate_audio(model, rave, latents, args, seed):
    with torch.no_grad():
        set_seed(seed)

        noise = torch.rand_like(latents) * args.temperature
        latents = latents * (1 - args.temperature)
        noise = noise * args.temperature + latents * (1 - args.temperature)
        noise = noise.to(device)

        rave_model_name = os.path.basename(args.rave_model).split(".")[0]
        diffusion_model_name = os.path.basename(args.model_path)

        print(f"Regenerating {noise.shape} latent codes with Diffusion model:", diffusion_model_name)
        print("Decoding using RAVE Model:", rave_model_name)
        print("Seed:", seed)

        model.eval()

        ### GENERATING WITH .PT FILE
        diff = model.sample(noise, num_steps=args.diffusion_steps, show_progress=True)
        # diff = model(noise)
        # noise = diff

        diff = (diff - diff.mean()) / diff.std()

        rave = rave.cpu()
        diff = diff.cpu()
        print("Decoding using RAVE Model...")
        y = decode_latents(diff, rave)
        fname = f"rave-latent-diffusion_seed{seed}_{args.name}_{os.path.basename(args.regenerate)}.wav"
        path = os.path.join(args.output_path, fname)
        print(f"Writing {path}")
        sf.write(path, y, args.sample_rate)


def generate_audio_conditioning(model, rave, cond_latents, args, seed):
    with torch.no_grad():
        set_seed(seed)

        # rave_dims = get_latent_dim(rave)
        noise = torch.randn_like(cond_latents)
        noise = noise * args.temperature
        z_length = noise.size(-1)

        rave_model_name = os.path.basename(args.rave_model).split(".")[0]
        diffusion_model_name = os.path.basename(args.model_path)

        print(f"Generating {z_length} latent codes with Diffusion model:", diffusion_model_name)
        print("Decoding using RAVE Model:", rave_model_name)
        print("Seed:", seed)

        model.eval()

        ### GENERATING WITH .PT FILE
        diff = model.sample(noise, conditioning=cond_latents,
                            num_steps=args.diffusion_steps, show_progress=True)
        # diff = model(noise)
        # noise = diff

        diff = (diff - diff.mean()) / diff.std()

        rave = rave.cpu()
        diff = diff.cpu()
        print("Decoding using RAVE Model...")
        y = decode_latents(diff, rave)

        path = f'{args.output_path}/rave-latent_diffusion_seed{seed}_{args.name}_{rave_model_name}.wav'
        print(f"Writing {path}")
        sf.write(path, y, args.sample_rate)


def decode_latents(latents, rave):
    y = rave.decode(latents)
    y = y.reshape(-1).detach().numpy()

    if rave.stereo:
        y_l = y[:len(y)//2]
        y_r = y[len(y)//2:]

        y = np.stack((y_l, y_r), axis=-1)

    return y


def load_model(rave, args, device):
    try:
        model = LightningDiffusionModel.load_from_checkpoint(args.model_path)
    except KeyError:
        print("Can't load lightning model, trying to load model_state_dict")
        ckpt = torch.load(args.model_path, map_location=device)
        latent_dims = get_latent_dim(rave)
        if args.conditioning:
            model = RAVELDConditioningModel(latent_dims, args.latent_length)
        else:
            model = RAVELDModel(latent_dims, args.latent_length)

        model.load_state_dict(ckpt["model_state_dict"])

    return model.to(device)


def load_latents_or_encode(path, cond_encoder, device):
    if os.path.splitext(path)[1] == ".pt":
        print(f"Loading conditioning latents: {path}")
        return torch.load(path, map_location=device)
    else:
        if cond_encoder is str:
            cond_encoder = torch.jit.load(path).to(device)
        print(f"Encoding conditioning audiofile: {path}")
        return encode_audiofile(cond_encoder, path)


def reshape_latents_for_model(z, model):
    return zeropad_to_multiple(
        z, model.latent_length
    ).reshape((-1, z.size(1), model.latent_length))


# Main function sets up the models and generates the audio.
def main():
    args = parse_args()

    rave = torch.jit.load(args.rave_model).to(device)

    if not args.sample_rate:
        msg = "RAVE model doesn't store its sample rate. --sample_rate is required."
        assert hasattr(rave, "sr"), msg
        args.sample_rate = rave.sr

    model = load_model(rave, args, device)

    if args.conditioning is not None:
        cond_latents = load_latents_or_encode(args.conditioning,
                                              args.rave_conditioning or rave,
                                              device)

        cond_latents = reshape_latents_for_model(cond_latents, model)

        for i in range(args.num):
            seed = args.seed + i
            generate_audio_conditioning(model, rave, cond_latents, args, seed)
    elif args.regenerate:
        latents = load_latents_or_encode(args.regenerate, rave, device)
        latents = reshape_latents_for_model(latents, model)
        for i in range(args.num):
            seed = args.seed + i
            regenerate_audio(model, rave, latents, args, seed)
    elif args.lerp:
        for i in range(args.num):
            seed = args.seed + i
            interpolate_seeds(model, rave, args, seed)
    else:
        for i in range(args.num):
            seed = args.seed + i
            generate_audio(model, rave, args, seed)

if __name__ == "__main__":
    main()
