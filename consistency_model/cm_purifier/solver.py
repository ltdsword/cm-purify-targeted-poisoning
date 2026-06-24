"""Small DDIM solver used for one-step consistency distillation targets."""

from __future__ import annotations

import numpy as np

from .schedules import extract


class DDIMSolver:
    # Purpose: Build DDIM solver timesteps and previous alpha products.
    # Input: alpha cumulative products, number of DDPM timesteps, and number of DDIM timesteps.
    # Output: solver instance with tensors used by ddim_step.
    def __init__(self, alphas_cumprod, num_train_timesteps: int = 1000, num_ddim_timesteps: int = 50) -> None:
        torch = self._require_torch()
        step_ratio = num_train_timesteps // num_ddim_timesteps
        if step_ratio <= 0:
            raise ValueError("num_ddim_timesteps must be <= num_train_timesteps")
        ddim_timesteps_np = (np.arange(1, num_ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        alphas_np = alphas_cumprod.detach().cpu().numpy()
        self.ddim_timesteps = torch.from_numpy(ddim_timesteps_np).long()
        self.ddim_alphas_cumprod = torch.from_numpy(alphas_np[ddim_timesteps_np]).float()
        previous = np.asarray([alphas_np[0]] + alphas_np[ddim_timesteps_np[:-1]].tolist(), dtype=np.float32)
        self.ddim_alphas_cumprod_prev = torch.from_numpy(previous).float()

    # Purpose: Import torch only when solver tensor methods are used.
    # Input: no arguments beyond the solver instance.
    # Output: imported torch module.
    def _require_torch(self):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Torch is required for DDIMSolver.") from exc
        return torch

    # Purpose: Move solver buffers to the requested device.
    # Input: torch device or device string.
    # Output: this solver instance after moving buffers.
    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alphas_cumprod = self.ddim_alphas_cumprod.to(device)
        self.ddim_alphas_cumprod_prev = self.ddim_alphas_cumprod_prev.to(device)
        return self

    # Purpose: Compute one deterministic DDIM step toward a shallower timestep.
    # Input: predicted x0, predicted noise, and DDIM timestep indices.
    # Output: x_prev tensor on the same probability-flow trajectory.
    def step(self, pred_x0, pred_noise, timestep_index):
        alpha_prev = extract(self.ddim_alphas_cumprod_prev, timestep_index, pred_x0.shape)
        direction = (1.0 - alpha_prev).sqrt() * pred_noise
        return alpha_prev.sqrt() * pred_x0 + direction

