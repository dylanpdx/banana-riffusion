import torch
from riffusion_pipeline import RiffusionPipeline
from huggingface_hub import hf_hub_download
def download_model():
    model = RiffusionPipeline.from_pretrained(
        "riffusion/riffusion-model-v1",
        revision="main",
        torch_dtype=torch.float16,
        # Disable the NSFW filter, causes incorrect false positives
        safety_checker=lambda images, **kwargs: (images, False),
    )
    unet_file = hf_hub_download(
        "riffusion/riffusion-model-v1", filename="unet_traced.pt", subfolder="unet_traced"
    )

if __name__ == "__main__":
    download_model()