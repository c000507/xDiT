import argparse
import os

import torch
from diffusers import AutoencoderKL
from transformers import T5EncoderModel

from xfuser import xFuserArgs, xFuserPixArtAlphaPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import get_world_group


def parse_args():
    parser = FlexibleArgumentParser(description="Generate an image with custom component paths")
    xFuserArgs.add_cli_args(parser)

    parser.add_argument(
        "--vae-path",
        "--vae_path",
        dest="vae_path",
        type=str,
        default=None,
        help="Optional path to a standalone VAE checkpoint. Defaults to the base model's VAE subfolder.",
    )
    parser.add_argument(
        "--text-encoder-path",
        "--text_encoder_path",
        dest="text_encoder_path",
        type=str,
        default=None,
        help="Optional path to a standalone text encoder checkpoint. Defaults to the base model's text_encoder subfolder.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/custom_components_result.png",
        help="Where to save the generated image.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()

    # Stick with float16 for broad GPU support
    engine_config.runtime_config.dtype = torch.float16

    def load_state_dict(path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(path)

        return torch.load(path, map_location="cpu")

    # Resolve component locations
    model_root = engine_config.model_config.model
    vae_location = args.vae_path or model_root
    text_encoder_location = args.text_encoder_path or model_root

    def has_available_submodule_config(root, subfolder):
        """Return True if the given root can supply a diffusers submodule config."""

        if os.path.isfile(root):
            # A single checkpoint file cannot host nested configs
            return False

        if os.path.isdir(root):
            return os.path.isfile(os.path.join(root, subfolder, "config.json"))

        # Assume remote repos can provide the needed config files
        return True

    base_has_vae = has_available_submodule_config(model_root, "vae")
    base_has_text_encoder = has_available_submodule_config(model_root, "text_encoder")

    if args.vae_path and os.path.isfile(args.vae_path):
        # Direct weight file without a config; load using a base config from the primary model when available
        try:
            vae = AutoencoderKL.from_single_file(args.vae_path, torch_dtype=torch.float16)
        except Exception as exc:
            if base_has_vae:
                vae = AutoencoderKL.from_pretrained(model_root, subfolder="vae", torch_dtype=torch.float16)
                vae.load_state_dict(load_state_dict(args.vae_path))
            else:
                raise RuntimeError(
                    "Failed to load VAE weights as a single file and no diffusers-formatted VAE config "
                    "was found under the base model path. Please provide a diffusers-formatted VAE directory "
                    "via --vae-path."
                ) from exc
    elif args.vae_path:
        vae = AutoencoderKL.from_pretrained(vae_location, subfolder=None, torch_dtype=torch.float16)
    else:
        if base_has_vae:
            vae = AutoencoderKL.from_pretrained(model_root, subfolder="vae", torch_dtype=torch.float16)
        else:
            raise RuntimeError(
                "No VAE path provided and the base model location does not contain a diffusers-formatted VAE. "
                "Supply --vae-path with a diffusers VAE directory or single-file checkpoint."
            )

    if args.text_encoder_path and os.path.isfile(args.text_encoder_path):
        if not base_has_text_encoder:
            raise RuntimeError(
                "A standalone text encoder weight file was provided but the base model path does not "
                "expose a diffusers-formatted text_encoder config. Please supply a directory path for "
                "--text-encoder-path instead."
            )

        text_encoder = T5EncoderModel.from_pretrained(
            model_root, subfolder="text_encoder", torch_dtype=torch.float16
        )
        text_encoder.load_state_dict(load_state_dict(args.text_encoder_path))
        text_encoder.to(dtype=torch.float16)
    elif args.text_encoder_path:
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_location, subfolder=None, torch_dtype=torch.float16)
    else:
        if base_has_text_encoder:
            text_encoder = T5EncoderModel.from_pretrained(
                model_root, subfolder="text_encoder", torch_dtype=torch.float16
            )
        else:
            raise RuntimeError(
                "No text encoder path provided and the base model location does not contain a diffusers-formatted "
                "text_encoder directory. Supply --text-encoder-path with compatible weights."
            )

    pipe = xFuserPixArtAlphaPipeline.from_pretrained(
        engine_config.model_config.model,
        engine_config=engine_config,
        vae=vae,
        text_encoder=text_encoder,
        torch_dtype=torch.float16,
    )

    local_rank = get_world_group().local_rank
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    pipe.prepare_run(input_config)

    generator = torch.Generator(device=f"cuda:{local_rank}").manual_seed(input_config.seed)
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        guidance_scale=input_config.guidance_scale,
        generator=generator,
        max_sequence_length=input_config.max_sequence_length,
    )

    if pipe.is_dp_last_group():
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        output.images[0].save(args.output_path)
        print(f"Image saved to {args.output_path}")


if __name__ == "__main__":
    main()
