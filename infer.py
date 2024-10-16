import os
import argparse

from PIL import Image
import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from diffusers.utils import export_to_gif

from diffusers_sv3d import SV3DUNetSpatioTemporalConditionModel, StableVideo3DDiffusionPipeline

SV3D_DIFFUSERS = "chenguolin/sv3d-diffusers"

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "~/.cache/huggingface"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="out/", type=str, help="Output filepath")
    parser.add_argument("--image_path", default="assets/bubble_mart_blue.png", type=str, help="Image filepath")
    parser.add_argument("--elevation", default=10, type=float, help="Camera elevation of the input image")
    parser.add_argument("--half_precision", action="store_true", help="Use fp16 half precision")
    parser.add_argument("--seed", default=-1, type=int, help="Random seed")
    args = parser.parse_args()

    unet = SV3DUNetSpatioTemporalConditionModel.from_pretrained(SV3D_DIFFUSERS, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(SV3D_DIFFUSERS, subfolder="vae")
    scheduler = EulerDiscreteScheduler.from_pretrained(SV3D_DIFFUSERS, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(SV3D_DIFFUSERS, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(SV3D_DIFFUSERS, subfolder="feature_extractor")

    pipeline = StableVideo3DDiffusionPipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor, 
        unet=unet, vae=vae,
        scheduler=scheduler,
    )

    num_frames, sv3d_res = 21, 576
    elevations_deg = [args.elevation] * num_frames
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    azimuths_rad[:-1].sort()
    pipeline = pipeline.to("cuda")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16 if args.half_precision else torch.float32, enabled=True):
            image = Image.open(args.image_path)
            image.load()  # required for `.split()`
            if len(image.split()) == 4:  # RGBA
                input_image = Image.new("RGB", image.size, (255, 255, 255))  # pure white bg
                input_image.paste(image, mask=image.split()[3])  # 3rd is the alpha channel
            else:
                input_image = image
            video_frames = pipeline(
                input_image.resize((sv3d_res, sv3d_res)),
                height=sv3d_res,
                width=sv3d_res,
                num_frames=num_frames,
                decode_chunk_size=8,  # smaller to save memory
                polars_rad=polars_rad,
                azimuths_rad=azimuths_rad,
                generator=torch.manual_seed(args.seed) if args.seed >= 0 else None,
            ).frames[0]

    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.basename(args.image_path).split(".")[0]
    export_to_gif(video_frames, f"{args.output_dir}/{name}.gif", fps=7)


if __name__ == "__main__":
    main()
