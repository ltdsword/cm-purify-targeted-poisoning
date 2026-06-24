"""DDPM schedule helpers for pixel-space consistency distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


# Purpose: Import torch only in code paths that need tensor math.
# Input: no arguments.
# Output: the imported torch module, or a clear ImportError.
def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Torch is required for DDPM schedule tensor operations."
        ) from exc
    return torch


# Purpose: Create a standard DDPM beta schedule.
# Input: number of timesteps, beta endpoints, and schedule name.
# Output: float32 torch tensor of betas with shape [num_train_timesteps].
def make_beta_schedule(
    num_train_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    schedule: str = "linear",
):
    torch = _require_torch()
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    if schedule == "scaled_linear":
        return torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
    raise ValueError(f"Unsupported beta schedule: {schedule}")


# Purpose: Try to read a diffusers scheduler config and reproduce its beta schedule.
# Input: model id/path that contains a scheduler config, or a direct scheduler directory.
# Output: float32 torch tensor of betas.
def load_diffusers_beta_schedule(model_or_scheduler_path: str):
    try:
        from diffusers import DDPMScheduler
    except ImportError as exc:
        raise ImportError("diffusers is required for --schedule-source diffusers") from exc

    path = Path(model_or_scheduler_path)
    if path.is_dir() and (path / "scheduler").is_dir():
        scheduler = DDPMScheduler.from_pretrained(str(path), subfolder="scheduler")
    else:
        scheduler = DDPMScheduler.from_pretrained(model_or_scheduler_path)
    torch = _require_torch()
    return torch.as_tensor(scheduler.betas, dtype=torch.float32)


# Purpose: Convert betas into alpha products and square-root schedules.
# Input: beta tensor with shape [T].
# Output: tuple of alphas_cumprod, sqrt(alpha_bar), and sqrt(1-alpha_bar).
def compute_alpha_sigma(betas):
    torch = _require_torch()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alpha_schedule = torch.sqrt(alphas_cumprod)
    sigma_schedule = torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=0.0))
    return alphas_cumprod, alpha_schedule, sigma_schedule


# Purpose: Gather a one-dimensional schedule into a broadcastable tensor.
# Input: schedule tensor [T], integer timesteps [B], and target tensor shape.
# Output: tensor shaped [B, 1, 1, ...] for broadcasting.
def extract(schedule, timesteps, target_shape):
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must be rank 1, got shape {tuple(timesteps.shape)}")
    values = schedule.to(timesteps.device).gather(0, timesteps)
    return values.reshape(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))


# Purpose: Apply the DDPM forward process with caller-supplied noise.
# Input: clean image tensor, timesteps, noise tensor, and alpha/sigma schedules.
# Output: noised image tensor at the requested timesteps.
def q_sample(x_start, timesteps, noise, alpha_schedule, sigma_schedule):
    if noise.shape != x_start.shape:
        raise ValueError(f"noise shape {tuple(noise.shape)} must match x_start {tuple(x_start.shape)}")
    alpha_t = extract(alpha_schedule, timesteps, x_start.shape)
    sigma_t = extract(sigma_schedule, timesteps, x_start.shape)
    return alpha_t * x_start + sigma_t * noise


# Purpose: Recover predicted clean image from an epsilon prediction.
# Input: noised tensor, predicted epsilon, timesteps, and alpha/sigma schedules.
# Output: predicted x_0 tensor.
def predict_x0_from_eps(x_t, eps, timesteps, alpha_schedule, sigma_schedule):
    alpha_t = extract(alpha_schedule, timesteps, x_t.shape)
    sigma_t = extract(sigma_schedule, timesteps, x_t.shape)
    return (x_t - sigma_t * eps) / alpha_t.clamp_min(1e-8)


# Purpose: Recover predicted epsilon from a clean-image prediction.
# Input: noised tensor, predicted x_0, timesteps, and alpha/sigma schedules.
# Output: predicted epsilon tensor.
def predict_eps_from_x0(x_t, pred_x0, timesteps, alpha_schedule, sigma_schedule):
    alpha_t = extract(alpha_schedule, timesteps, x_t.shape)
    sigma_t = extract(sigma_schedule, timesteps, x_t.shape)
    return (x_t - alpha_t * pred_x0) / sigma_t.clamp_min(1e-8)


# Purpose: Compute Consistency Model boundary scaling coefficients.
# Input: timesteps, target tensor rank, sigma_data, and timestep scaling.
# Output: c_skip and c_out tensors broadcastable to image tensors.
def boundary_scalings(
    timesteps,
    target_ndim: int,
    sigma_data: float = 0.5,
    timestep_scaling: float = 10.0,
) -> Tuple[object, object]:
    scaled_t = timesteps.float() * timestep_scaling
    c_skip = sigma_data**2 / (scaled_t**2 + sigma_data**2)
    c_out = scaled_t / torch_sqrt(scaled_t**2 + sigma_data**2)
    shape = (timesteps.shape[0],) + (1,) * (target_ndim - 1)
    return c_skip.reshape(shape), c_out.reshape(shape)


# Purpose: Take a torch square root while keeping torch import local to this module.
# Input: torch tensor.
# Output: square-root tensor.
def torch_sqrt(tensor):
    torch = _require_torch()
    return torch.sqrt(tensor)


# Purpose: Sample DDIM timestep indices and their start/end training timesteps.
# Input: batch size, solver timesteps, fraction of low-noise timesteps, and target device.
# Output: tuple of solver indices, start timesteps, and previous timesteps.
def sample_ddim_training_timesteps(
    batch_size: int,
    solver_timesteps,
    timestep_fraction: float,
    device,
):
    torch = _require_torch()
    if not 0.0 < timestep_fraction <= 1.0:
        raise ValueError(f"timestep_fraction must be in (0, 1], got {timestep_fraction}")
    max_index = max(1, int(len(solver_timesteps) * timestep_fraction))
    index = torch.randint(0, max_index, (batch_size,), device=device)
    start_timesteps = solver_timesteps.to(device).gather(0, index)
    previous_index = torch.clamp(index - 1, min=0)
    end_timesteps = solver_timesteps.to(device).gather(0, previous_index)
    end_timesteps = torch.where(index == 0, torch.zeros_like(end_timesteps), end_timesteps)
    return index, start_timesteps.long(), end_timesteps.long()


# Purpose: Convert a tensor in [-1, 1] into [0, 1] for image saving or classifier input.
# Input: image tensor in the purifier training range.
# Output: clamped tensor in [0, 1].
def minus_one_to_one_to_zero_one(tensor):
    return ((tensor + 1.0) * 0.5).clamp(0.0, 1.0)


# Purpose: Convert a tensor in [0, 1] into [-1, 1] for purifier model input.
# Input: image tensor in standard image range.
# Output: tensor in purifier training range.
def zero_one_to_minus_one_to_one(tensor):
    return tensor * 2.0 - 1.0

