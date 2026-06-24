"""Checkpoint save/load helpers for trained CM purifier models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .ema import ema_state_dict, load_ema_state_dict, sync_ema
from .model import build_cm_model
from .schedules import compute_alpha_sigma, make_beta_schedule


# Purpose: Save all training state needed for inference or resume.
# Input: output path, student model, EMA teacher, optimizer, args, global step, and DDPM betas.
# Output: none; a .pth/.pt checkpoint file is written to disk.
def save_training_checkpoint(
    path: str | Path,
    model,
    teacher,
    optimizer,
    args,
    global_step: int,
    betas,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "cm_purifier_checkpoint_v1",
        "model": model.state_dict(),
        "ema": ema_state_dict(teacher),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "global_step": global_step,
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
        "betas": betas.detach().cpu(),
    }
    torch.save(checkpoint, path)


# Purpose: Read a serialized purifier checkpoint dictionary from disk.
# Input: checkpoint path and target device.
# Output: checkpoint dictionary loaded by torch.
def load_checkpoint_dict(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


# Purpose: Restore a training checkpoint into an existing student, EMA teacher, and optimizer.
# Input: checkpoint path, model, teacher, optional optimizer, and target device.
# Output: restored global step.
def load_training_checkpoint(
    checkpoint_path: Optional[str | Path],
    model,
    teacher,
    optimizer,
    device: torch.device,
) -> int:
    if checkpoint_path is None:
        return 0
    checkpoint = load_checkpoint_dict(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"])
    if "ema" in checkpoint and checkpoint["ema"] is not None:
        load_ema_state_dict(teacher, checkpoint["ema"])
    else:
        sync_ema(teacher, model)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("global_step", 0))


# Purpose: Build alpha/sigma schedules stored inside a checkpoint.
# Input: checkpoint dictionary, training args dictionary, and target device.
# Output: alpha and sigma schedule tensors.
def build_schedules_from_checkpoint(checkpoint, train_args, device: torch.device):
    if "betas" in checkpoint and checkpoint["betas"] is not None:
        betas = checkpoint["betas"].to(device).float()
    else:
        betas = make_beta_schedule(num_train_timesteps=int(train_args.get("num_train_timesteps", 1000))).to(device)
    _, alpha_schedule, sigma_schedule = compute_alpha_sigma(betas)
    return alpha_schedule, sigma_schedule


# Purpose: Reconstruct a purifier model from a saved checkpoint file.
# Input: checkpoint path, target device, and whether to load student instead of EMA weights.
# Output: model, alpha schedule, sigma schedule, and saved training args.
def load_purifier_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    use_student: bool = False,
):
    checkpoint = load_checkpoint_dict(checkpoint_path, device)
    train_args = checkpoint.get("args", {})
    model = build_cm_model(
        backbone=train_args.get("backbone", "diffusers"),
        model_name_or_path=train_args.get("teacher_model", "google/ddpm-cifar10-32"),
        tiny_hidden_channels=int(train_args.get("tiny_hidden_channels", 64)),
        output_mode=train_args.get("cm_output_mode", "pred_x0"),
    ).to(device)
    state_key = "model" if use_student or "ema" not in checkpoint else "ema"
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    alpha_schedule, sigma_schedule = build_schedules_from_checkpoint(checkpoint, train_args, device)
    return model, alpha_schedule, sigma_schedule, train_args

