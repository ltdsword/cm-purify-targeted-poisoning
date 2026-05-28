import torch
from diffusers import DDPMScheduler, UNet2DModel
import os

os.makedirs('local_ddpm_model', exist_ok=True)
model_id = "google/ddpm-cifar10-32"
print(f"Downloading {model_id} to local_ddpm_model...")
unet = UNet2DModel.from_pretrained(model_id)
unet.save_pretrained('local_ddpm_model/unet')

scheduler = DDPMScheduler.from_pretrained(model_id)
scheduler.save_pretrained('local_ddpm_model/scheduler')
print("Saved successfully!")
