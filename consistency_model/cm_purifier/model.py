"""Model definitions for the pixel-space consistency purifier."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedules import boundary_scalings, predict_x0_from_eps


ModelBackbone = Literal["diffusers", "tiny"]
CMOutputMode = Literal["pred_x0", "full_boundary", "no_skip_boundary"]


# Purpose: Build sinusoidal timestep embeddings for the tiny denoiser.
# Input: integer timesteps with shape [B] and embedding dimension.
# Output: float tensor with shape [B, dim].
def sinusoidal_timestep_embedding(timesteps, dim: int):
    half = dim // 2
    frequencies = torch.exp(
        -torch.arange(half, device=timesteps.device, dtype=torch.float32)
        * torch.log(torch.tensor(10000.0, device=timesteps.device))
        / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TinyDenoiser(nn.Module):
    # Purpose: Initialize a compact denoising network for smoke tests and debugging.
    # Input: image channels, hidden channels, and time embedding dimension.
    # Output: torch module that predicts epsilon-like noise.
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64, time_dim: int = 128) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_mid_1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_mid_2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

    # Purpose: Predict diffusion noise for a noised image batch.
    # Input: image tensor [B, C, H, W] and integer timesteps [B].
    # Output: noise prediction tensor with the same shape as input image tensor.
    def forward(self, x, timesteps):
        time_embedding = sinusoidal_timestep_embedding(timesteps, self.time_dim)
        time_features = self.time_mlp(time_embedding).view(x.shape[0], -1, 1, 1)
        hidden = F.silu(self.conv_in(x) + time_features)
        hidden = F.silu(self.conv_mid_1(hidden))
        hidden = F.silu(self.conv_mid_2(hidden))
        return self.conv_out(hidden)


# Purpose: Create a small random denoiser for local smoke checks.
# Input: hidden channel count.
# Output: TinyDenoiser instance.
def create_tiny_denoiser(hidden_channels: int = 64) -> TinyDenoiser:
    return TinyDenoiser(hidden_channels=hidden_channels)


# Purpose: Load a diffusers UNet2DModel from a model id, root path, or unet directory.
# Input: model identifier/path.
# Output: UNet2DModel instance.
def load_diffusers_unet(model_name_or_path: str):
    try:
        from diffusers import UNet2DModel
    except ImportError as exc:
        raise ImportError("diffusers is required when --backbone diffusers is used.") from exc

    path = Path(model_name_or_path)
    if path.is_dir() and (path / "unet").is_dir():
        return UNet2DModel.from_pretrained(str(path / "unet"))
    return UNet2DModel.from_pretrained(model_name_or_path)


# Purpose: Create the requested denoising backbone.
# Input: backbone name, model id/path, and tiny hidden channel count.
# Output: torch module that maps (x_t, t) to predicted epsilon.
def load_denoiser(
    backbone: ModelBackbone,
    model_name_or_path: str,
    tiny_hidden_channels: int = 64,
):
    if backbone == "diffusers":
        return load_diffusers_unet(model_name_or_path)
    if backbone == "tiny":
        return create_tiny_denoiser(hidden_channels=tiny_hidden_channels)
    raise ValueError(f"Unsupported backbone: {backbone}")


# Purpose: Extract the tensor sample from either a plain torch module or a diffusers output object.
# Input: raw model output from a denoiser.
# Output: tensor prediction.
def get_model_sample(model_output):
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "sample"):
        return model_output.sample
    if isinstance(model_output, (tuple, list)) and model_output:
        return model_output[0]
    raise TypeError(f"Cannot extract sample tensor from output type {type(model_output)!r}")


class CMBoundaryWrapper(nn.Module):
    # Purpose: Initialize a wrapper that converts epsilon predictions into CM clean-image outputs.
    # Input: denoiser module, output mode, and boundary scaling constants.
    # Output: torch module that returns purified clean-image predictions.
    def __init__(
        self,
        denoiser: nn.Module,
        output_mode: CMOutputMode = "pred_x0",
        sigma_data: float = 0.5,
        timestep_scaling: float = 10.0,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.output_mode = output_mode
        self.sigma_data = sigma_data
        self.timestep_scaling = timestep_scaling

    # Purpose: Predict epsilon from the wrapped denoising backbone.
    # Input: noised image tensor and integer timesteps.
    # Output: predicted epsilon tensor.
    def predict_eps(self, x_t, timesteps):
        model_output = self.denoiser(x_t, timesteps)
        return get_model_sample(model_output)

    # Purpose: Predict a clean image under the selected consistency output parameterization.
    # Input: noised image, timesteps, alpha schedule, and sigma schedule.
    # Output: clean-image prediction tensor in the training value range.
    def forward(self, x_t, timesteps, alpha_schedule, sigma_schedule):
        eps_pred = self.predict_eps(x_t, timesteps)
        pred_x0 = predict_x0_from_eps(x_t, eps_pred, timesteps, alpha_schedule, sigma_schedule)
        if self.output_mode == "pred_x0":
            return pred_x0
        c_skip, c_out = boundary_scalings(
            timesteps,
            target_ndim=x_t.ndim,
            sigma_data=self.sigma_data,
            timestep_scaling=self.timestep_scaling,
        )
        if self.output_mode == "full_boundary":
            return c_skip * x_t + c_out * pred_x0
        if self.output_mode == "no_skip_boundary":
            return c_out * pred_x0
        raise ValueError(f"Unsupported output_mode: {self.output_mode}")


# Purpose: Build a complete CM purifier model from a denoising backbone.
# Input: backbone selection, model path, tiny size, and CM output mode.
# Output: CMBoundaryWrapper instance.
def build_cm_model(
    backbone: ModelBackbone,
    model_name_or_path: str,
    tiny_hidden_channels: int = 64,
    output_mode: CMOutputMode = "pred_x0",
) -> CMBoundaryWrapper:
    denoiser = load_denoiser(backbone, model_name_or_path, tiny_hidden_channels)
    return CMBoundaryWrapper(denoiser=denoiser, output_mode=output_mode)

