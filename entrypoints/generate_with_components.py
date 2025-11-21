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
        type=str,
        default=None,
        help="Optional path to a standalone VAE checkpoint. Defaults to the base model's VAE subfolder.",
    )
    parser.add_argument(
        "--text-encoder-path",
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

    # Resolve component locations
    vae_location = args.vae_path or engine_config.model_config.model
    text_encoder_location = args.text_encoder_path or engine_config.model_config.model

    vae = AutoencoderKL.from_pretrained(
        vae_location,
        subfolder=None if args.vae_path else "vae",
        torch_dtype=torch.float16,
    )
    text_encoder = T5EncoderModel.from_pretrained(
        text_encoder_location,
        subfolder=None if args.text_encoder_path else "text_encoder",
        torch_dtype=torch.float16,
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
