# --- Usage Example ---
# Assuming `batch_tensors` is a batch of CIFAR-10 images loaded via a DataLoader
# purified_tensors = purify_gdmp_batch(batch_tensors, unet, scheduler, M=4, Tc=36)

import torch
import torch.nn.functional as F
import os
import logging
from PIL import Image
from torchvision import transforms
from diffusers import DDPMScheduler, UNet2DModel

device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["HF_HUB_OFFLINE"] = "0"

# 1. Load Pre-trained CIFAR-10 DDPM
model_id = "google/ddpm-cifar10-32"
unet = UNet2DModel.from_pretrained(model_id).to(device)
scheduler = DDPMScheduler.from_pretrained(model_id)

# 2. Setup Image Transforms (Rules for converting pictures to numbers and back)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) 
])

inverse_transform = transforms.Compose([
    transforms.Normalize([-1.0], [2.0]), 
    transforms.ToPILImage()
])

# 3. GDMP Algorithm Implementation (The Purification Process)
def purify_gdmp_batch(x_adv, unet, scheduler, M=4, Tc=36, gamma=8/255, a=1.0, verbose=True):
    batch_size = x_adv.shape[0]
    
    # Make a working copy of the images
    x_current = x_adv.clone()
    
    # Set the total rulebook steps to 1000
    scheduler.set_timesteps(1000)
    
    # Get standard math numbers from the scheduler
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    betas = scheduler.betas.to(device)
    
    for iteration in range(M):
        if verbose:
            logging.info(f"--- Starting Purification Iteration {iteration + 1}/{M} ---")
            
        # STEP A: Add random noise up to step Tc
        noise = torch.randn_like(x_current)
        timesteps = torch.full((batch_size,), Tc, device=device, dtype=torch.long)
        x_t = scheduler.add_noise(x_current, noise, timesteps)
        
        # STEP B: The Guided Reverse Process (Denoise step-by-step back to 0)
        for t_val in range(Tc, -1, -1):
            t_tensor = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            
            # Math setup for the "Guidance"
            alpha_bar_t = alphas_cumprod[t_val]
            s_t = (3 * torch.sqrt(1 - alpha_bar_t)) / (gamma * torch.sqrt(alpha_bar_t)) * a
            sigma_sq = betas[t_val]
            
            # Diffuse the original poisoned image to the exact same step t
            noise_adv = torch.randn_like(x_adv)
            x_adv_t = scheduler.add_noise(x_adv, noise_adv, t_tensor)
            
            # Turn on PyTorch's math tracker for the image
            x_t = x_t.detach().requires_grad_(True)
            
            # Calculate how far our current image is from the original (Loss)
            loss = F.mse_loss(x_t, x_adv_t)
            
            # Calculate the direction (gradient) to pull the image back to normal
            grad_x_t = torch.autograd.grad(loss, x_t)[0]
            
            # Print the loss so we can watch it drop!
            if verbose and (t_val % 5 == 0 or t_val == Tc or t_val == 0):
                logging.info(f"Step {t_val:02d} | Guidance Loss (MSE): {loss.item():.6f}")
            
            # Standard AI Denoising
            with torch.no_grad():
                # The AI guesses what the noise looks like
                noise_pred = unet(x_t, t_tensor).sample
                
                # The rulebook removes a tiny bit of that noise
                step_output = scheduler.step(noise_pred, t_val, x_t)
                x_prev = step_output.prev_sample
                
                # Apply our specific "Guidance Shift" to save the image contents
                x_prev = x_prev - (s_t * sigma_sq * grad_x_t)
            
            # Move to the next step down
            x_t = x_prev
            
        # The output of this iteration becomes the starting point for the next
        x_current = x_t.detach()
        
    return x_current

def run_purification_pipeline(input_dir="poisons/poisoned_images", output_dir="gdmp/purified_images", Tc=36, M=4):
    """
    Reads poisoned images, runs GDMP purification, and saves them to identical filenames.
    """
    logging.info(f"Starting purification batch. Reading from {input_dir}, saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return
        
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png') or f.endswith('.jpg')]
    logging.info(f"Found {len(image_files)} poisoned images to purify.")
    
    if len(image_files) == 0:
        logging.warning("No images found to process. Exiting.")
        return

    # Process in batches to not blow up GPU memory
    # Reduced from 100 to 64 to ensure safety under max 1 GPU / 32GB server RAM limits.
    # CIFAR10 is 32x32, so 64 is extremely safe for a loaded Diffusers UNet2DModel.
    batch_size = 64
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size} (size: {len(batch_files)})")
        
        batch_tensors = []
        for img_name in batch_files:
            img_path = os.path.join(input_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            batch_tensors.append(img_tensor)
            
        batch_tensors = torch.stack(batch_tensors).to(device)
        
        # Purify
        # Default settings: M = 4 and Tc = 36
        purified_tensors = purify_gdmp_batch(batch_tensors, unet, scheduler, M, Tc, verbose=False)
        
        # Save back
        for j, img_name in enumerate(batch_files):
            out_img = inverse_transform(purified_tensors[j].cpu())
            out_path = os.path.join(output_dir, img_name)
            out_img.save(out_path)
            
    logging.info("Purification pipeline complete!")

if __name__ == "__main__":
    run_purification_pipeline()