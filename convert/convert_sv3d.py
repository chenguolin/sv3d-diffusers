import os
import argparse

from huggingface_hub import hf_hub_download
import safetensors.torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import (
    AutoencoderKL,
    # AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
)

from convert.convert_svd_to_diffusers import (
    convert_ldm_unet_checkpoint,
    # convert_ldm_vae_checkpoint,
    create_unet_diffusers_config,
)
from diffusers_sv3d import SV3DUNetSpatioTemporalConditionModel, StableVideo3DDiffusionPipeline

SVD_V1_CKPT = "stabilityai/stable-video-diffusion-img2vid-xt"
SD_V15_CKPT = "chenguolin/stable-diffusion-v1-5"
HF_HOME = "~/.cache/huggingface"
HF_TOKEN = ""
HF_USERNAME = ""

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_USERNAME"] = HF_USERNAME


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_ckpt_path", default=os.path.expanduser(f"{HF_HOME}/hub/models--stabilityai--sv3d/snapshots/31213729b4314a44b574ce7cc2d0c28356f097ed/sv3d_p.safetensors"), type=str,  help="Path to the checkpoint to convert.")
    parser.add_argument("--hf_token", default=HF_TOKEN, type=str, help="your HuggingFace token")
    parser.add_argument("--config_path", default="convert/sv3d_p.yaml", type=str, help="Config filepath.")
    parser.add_argument("--repo_name", default="sv3d-diffusers", type=str)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.original_ckpt_path):
        token = HF_TOKEN  # open(os.path.expanduser("~/.cache/huggingface/token"), "r").read()
        hf_hub_download("stabilityai/sv3d", filename="sv3d_p.safetensors", token=token)
    original_ckpt = safetensors.torch.load_file(args.original_ckpt_path, device="cpu")

    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config_path)

    unet_config = create_unet_diffusers_config(config, image_size=576)

    ori_config = unet_config.copy()
    unet_config.pop("attention_head_dim")
    unet_config.pop("use_linear_projection")
    unet_config.pop("class_embed_type")
    unet_config.pop("addition_embed_type")
    unet = SV3DUNetSpatioTemporalConditionModel(**unet_config)
    unet_state_dict = convert_ldm_unet_checkpoint(original_ckpt, ori_config)
    unet.load_state_dict(unet_state_dict, strict=True)

    # unet.save_pretrained("out/sv3d-diffusers", push_to_hub=True)

    vae = AutoencoderKL.from_pretrained(SD_V15_CKPT, subfolder="vae")
    scheduler = EulerDiscreteScheduler.from_pretrained(SVD_V1_CKPT, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(SVD_V1_CKPT, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(SVD_V1_CKPT, subfolder="feature_extractor")

    pipeline = StableVideo3DDiffusionPipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor, 
        unet=unet, vae=vae,
        scheduler=scheduler,
    )

    if args.push_to_hub:
        pipeline.push_to_hub(args.repo_name)
