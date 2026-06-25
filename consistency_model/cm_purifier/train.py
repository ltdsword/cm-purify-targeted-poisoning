"""Train Algorithm 2: poison-aware pixel-space consistency distillation."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import ATTACK_TO_ID
from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .dataset import PoisonPairDataset
from .ema import create_ema_model, update_ema
from .losses import LossWeights, compute_loss_dict
from .model import build_cm_model
from .schedules import (
    compute_alpha_sigma,
    load_diffusers_beta_schedule,
    make_beta_schedule,
    predict_x0_from_eps,
    q_sample,
    sample_ddim_training_timesteps,
)
from .solver import DDIMSolver


LOGGER = logging.getLogger("cm_purifier.train")


# Purpose: Build the command-line parser for Algorithm 2 training.
# Input: no arguments.
# Output: argparse parser with all training options.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a pixel-space CM purifier with poison-aware distillation.")
    parser.add_argument("--pair-dir", type=str, default="dataset_generation/datasets/train")
    parser.add_argument("--out", type=str, default="checkpoints/cm_purifier.pth")
    parser.add_argument("--teacher-model", type=str, default="google/ddpm-cifar10-32")
    parser.add_argument("--backbone", choices=["diffusers", "tiny"], default="diffusers")
    parser.add_argument("--tiny-hidden-channels", type=int, default=64)
    parser.add_argument("--cm-output-mode", choices=["pred_x0", "full_boundary", "no_skip_boundary"], default="full_boundary")
    parser.add_argument("--schedule-source", choices=["linear", "diffusers"], default="diffusers")
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--num-ddim-timesteps", type=int, default=50)
    parser.add_argument("--timestep-fraction", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--gamma-clean", type=float, default=0.0)
    parser.add_argument("--gamma-wb", type=float, default=1.0)
    parser.add_argument("--gamma-bp", type=float, default=1.0)
    parser.add_argument("--lambda-distill", type=float, default=1.0)
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-id", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=0.0)
    parser.add_argument("--distill-loss-type", choices=["l1", "l2", "huber"], default="l2")
    parser.add_argument("--classifier-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    return parser


# Purpose: Configure human-readable training logs for console and Slurm log files.
# Input: no arguments.
# Output: configured logger instance.
def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return LOGGER


# Purpose: Emit a visible section header in training logs.
# Input: section title string.
# Output: none; formatted title is written to the logger.
def log_section(title: str) -> None:
    LOGGER.info("=" * 72)
    LOGGER.info(title)
    LOGGER.info("=" * 72)


# Purpose: Convert a dictionary into stable pretty JSON for logs.
# Input: dictionary-like object.
# Output: indented JSON string.
def to_pretty_json(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


# Purpose: Format elapsed or remaining seconds as a compact HH:MM:SS string.
# Input: duration in seconds.
# Output: human-readable time string.
def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# Purpose: Read the current learning rate from an optimizer.
# Input: torch optimizer.
# Output: learning rate from the first parameter group.
def get_learning_rate(optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


# Purpose: Format CUDA memory usage for readable progress logs.
# Input: torch device used for training.
# Output: memory string, or CPU marker when CUDA is not used.
def format_gpu_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    allocated_gb = torch.cuda.memory_allocated(device) / 1e9
    reserved_gb = torch.cuda.memory_reserved(device) / 1e9
    return f"{allocated_gb:.1f}/{reserved_gb:.1f} GB"


# Purpose: Estimate remaining training time from average step duration.
# Input: start time, current step, starting step, and max training steps.
# Output: ETA in seconds.
def estimate_eta_seconds(start_time: float, global_step: int, start_step: int, max_steps: int) -> float:
    completed_steps = max(global_step - start_step, 1)
    elapsed = time.monotonic() - start_time
    seconds_per_step = elapsed / completed_steps
    remaining_steps = max(max_steps - global_step, 0)
    return remaining_steps * seconds_per_step


# Purpose: Format scalar training metrics into a poison-pipeline-style progress line.
# Input: global step, max steps, metrics, optimizer, start step, start time, and device.
# Output: compact readable string for training logs.
def format_metrics(
    global_step: int,
    max_steps: int,
    metrics: Dict[str, float],
    optimizer,
    start_step: int,
    start_time: float,
    device: torch.device,
) -> str:
    percent = 100.0 * global_step / max(max_steps, 1)
    elapsed_seconds = time.monotonic() - start_time
    eta_seconds = estimate_eta_seconds(start_time, global_step, start_step, max_steps)
    return (
        f"Step: {global_step}/{max_steps} | "
        f"{percent:.2f}% | "
        f"lr: {get_learning_rate(optimizer):.6f} | "
        f"Training loss is {metrics.get('loss', 0.0):.4f} | "
        f"distill: {metrics.get('loss_distill', 0.0):.4f} | "
        f"rec: {metrics.get('loss_reconstruction', 0.0):.4f} | "
        f"id: {metrics.get('loss_identity', 0.0):.4f} | "
        f"cls: {metrics.get('loss_classifier', 0.0):.4f} | "
        f"elapsed: {format_duration(elapsed_seconds)} | "
        f"eta: {format_duration(eta_seconds)} | "
        f"gpu: {format_gpu_memory(device)}"
    )


# Purpose: Log the most important run settings before training starts.
# Input: parsed args, resolved device, dataset summary, and output path.
# Output: none; readable configuration is written to logs.
def log_run_configuration(args, device: torch.device, dataset_summary: Dict[str, object]) -> None:
    selected_args = {
        "pair_dir": args.pair_dir,
        "out": args.out,
        "teacher_model": args.teacher_model,
        "backbone": args.backbone,
        "cm_output_mode": args.cm_output_mode,
        "schedule_source": args.schedule_source,
        "num_train_timesteps": args.num_train_timesteps,
        "num_ddim_timesteps": args.num_ddim_timesteps,
        "timestep_fraction": args.timestep_fraction,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "ema_decay": args.ema_decay,
        "gamma_clean": args.gamma_clean,
        "gamma_wb": args.gamma_wb,
        "gamma_bp": args.gamma_bp,
        "lambda_distill": args.lambda_distill,
        "lambda_rec": args.lambda_rec,
        "lambda_id": args.lambda_id,
        "lambda_cls": args.lambda_cls,
        "device": str(device),
    }
    log_section("CM purifier training configuration")
    LOGGER.info("Selected args:\n%s", to_pretty_json(selected_args))
    LOGGER.info("Dataset summary:\n%s", to_pretty_json(dataset_summary))


# Purpose: Log model and schedule details after construction.
# Input: model, betas tensor, solver, and resolved device.
# Output: none; summary is written to logs.
def log_model_summary(model, betas, solver, device: torch.device) -> None:
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    log_section("Model and schedule")
    LOGGER.info("Model class: %s", model.__class__.__name__)
    LOGGER.info("Trainable parameters: %s", f"{trainable_parameters:,}")
    LOGGER.info("Total parameters: %s", f"{total_parameters:,}")
    LOGGER.info("DDPM timesteps: %d", len(betas))
    LOGGER.info("DDIM timesteps: %d", len(solver.ddim_timesteps))
    LOGGER.info("First/last DDIM timestep: %d / %d", int(solver.ddim_timesteps[0]), int(solver.ddim_timesteps[-1]))
    LOGGER.info("Training device: %s", device)
    if device.type == "cuda":
        LOGGER.info("CUDA device name: %s", torch.cuda.get_device_name(device))
        LOGGER.info("CUDA memory allocated before training: %.2f GB", torch.cuda.memory_allocated(device) / 1e9)


# Purpose: Seed Python, numpy, and torch for reproducible training.
# Input: integer seed.
# Output: none; random number generators are updated in place.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Purpose: Resolve the requested torch device.
# Input: device argument string.
# Output: torch.device object.
def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# Purpose: Load a frozen classifier for optional label preservation loss.
# Input: optional classifier checkpoint path and device.
# Output: eval-mode classifier module, or None when disabled.
def load_optional_classifier(classifier_path: Optional[str], device: torch.device):
    if classifier_path is None:
        return None
    checkpoint = torch.load(classifier_path, map_location=device)
    if isinstance(checkpoint, torch.nn.Module):
        classifier = checkpoint
    else:
        try:
            classifier = torch.jit.load(classifier_path, map_location=device)
        except Exception as exc:
            raise ValueError(
                "classifier_path must point to a torch Module or TorchScript model; "
                "a raw state_dict is ambiguous without architecture code."
            ) from exc
    classifier.to(device)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    return classifier


# Purpose: Build the DDPM alpha/sigma schedules requested by training arguments.
# Input: parsed arguments and target device.
# Output: betas, alpha cumulative products, alpha schedule, and sigma schedule.
def build_schedules(args, device: torch.device):
    if args.schedule_source == "diffusers":
        betas = load_diffusers_beta_schedule(args.teacher_model)
    else:
        betas = make_beta_schedule(num_train_timesteps=args.num_train_timesteps)
    betas = betas.to(device)
    alphas_cumprod, alpha_schedule, sigma_schedule = compute_alpha_sigma(betas)
    return betas, alphas_cumprod, alpha_schedule, sigma_schedule


# Purpose: Build per-sample poison residual strengths from attack ids.
# Input: attack id tensor and gamma arguments.
# Output: broadcastable gamma tensor with shape [B, 1, 1, 1].
def build_gamma_tensor(attack_ids, args):
    gamma_lookup = torch.zeros(3, device=attack_ids.device, dtype=torch.float32)
    gamma_lookup[ATTACK_TO_ID["clean"]] = args.gamma_clean
    gamma_lookup[ATTACK_TO_ID["wb"]] = args.gamma_wb
    gamma_lookup[ATTACK_TO_ID["bp"]] = args.gamma_bp
    return gamma_lookup.gather(0, attack_ids).view(-1, 1, 1, 1)


# Purpose: Move tensor fields in a dataloader batch to the target device.
# Input: batch dictionary and torch device.
# Output: batch dictionary with tensor values moved.
def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


# Purpose: Run one Algorithm 2 optimization step.
# Input: batch, models, schedules, solver, classifier, optimizer, loss weights, args, and device.
# Output: dictionary of scalar loss metrics.
def train_one_step(
    batch,
    model,
    teacher,
    alpha_schedule,
    sigma_schedule,
    solver,
    classifier,
    optimizer,
    loss_weights: LossWeights,
    args,
    device,
):
    batch = move_batch_to_device(batch, device)
    clean = batch["clean"].float()
    poison = batch["poison"].float()
    labels = batch["label"].long()
    attack_ids = batch["attack_id"].long()
    is_clean = batch["is_clean"].bool()

    delta = poison - clean
    gamma = build_gamma_tensor(attack_ids, args)
    poison_aware_noise = torch.randn_like(clean) + gamma * delta

    solver_index, start_timesteps, end_timesteps = sample_ddim_training_timesteps(
        batch_size=clean.shape[0],
        solver_timesteps=solver.ddim_timesteps,
        timestep_fraction=args.timestep_fraction,
        device=device,
    )
    x_t_star = q_sample(clean, start_timesteps, poison_aware_noise, alpha_schedule, sigma_schedule)
    student_prediction = model(x_t_star, start_timesteps, alpha_schedule, sigma_schedule)

    with torch.no_grad():
        oracle_x0 = predict_x0_from_eps(
            x_t_star,
            poison_aware_noise,
            start_timesteps,
            alpha_schedule,
            sigma_schedule,
        )
        x_prev = solver.step(oracle_x0, poison_aware_noise, solver_index)
        teacher_target = teacher(x_prev, end_timesteps, alpha_schedule, sigma_schedule)

    loss_dict = compute_loss_dict(
        student_prediction=student_prediction,
        teacher_target=teacher_target,
        clean_target=clean,
        labels=labels,
        clean_mask=is_clean,
        classifier=classifier,
        weights=loss_weights,
        distill_loss_type=args.distill_loss_type,
    )
    optimizer.zero_grad(set_to_none=True)
    loss_dict["loss"].backward()
    if args.max_grad_norm is not None and args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    update_ema(teacher, model, args.ema_decay)
    return {key: float(value.detach().cpu()) for key, value in loss_dict.items()}


# Purpose: Execute the full Algorithm 2 training loop.
# Input: parsed command-line arguments.
# Output: final checkpoint path after training completes.
def main(args=None):
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args(args)
    log_section("STARTING CM PURIFIER TRAINING")
    set_seed(args.seed)
    device = resolve_device(args.device)
    LOGGER.info("Seed: %d", args.seed)
    LOGGER.info("Resolved device: %s", device)

    log_section("1. VALIDATING DATASET...")
    dataset = PoisonPairDataset(args.pair_dir, image_size=args.image_size, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if len(dataloader) == 0:
        raise ValueError("Training dataloader is empty; lower batch size or check pair-dir.")
    dataset_summary = dataset.summary()
    log_run_configuration(args, device, dataset_summary)

    log_section("2. BUILDING MODEL AND DDPM SCHEDULE...")
    betas, alphas_cumprod, alpha_schedule, sigma_schedule = build_schedules(args, device)
    solver = DDIMSolver(
        alphas_cumprod=alphas_cumprod,
        num_train_timesteps=len(betas),
        num_ddim_timesteps=args.num_ddim_timesteps,
    ).to(device)

    model = build_cm_model(
        backbone=args.backbone,
        model_name_or_path=args.teacher_model,
        tiny_hidden_channels=args.tiny_hidden_channels,
        output_mode=args.cm_output_mode,
    ).to(device)
    teacher = create_ema_model(model).to(device)
    log_model_summary(model, betas, solver, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    global_step = load_training_checkpoint(args.resume, model, teacher, optimizer, device)
    if args.resume is not None:
        LOGGER.info("Resumed from %s at global step %d", args.resume, global_step)
    classifier = load_optional_classifier(args.classifier_path, device)
    if args.lambda_cls > 0.0 and classifier is None:
        raise ValueError("--lambda-cls > 0 requires --classifier-path")
    classifier_for_loss = classifier if args.lambda_cls > 0.0 else None
    loss_weights = LossWeights(
        distill=args.lambda_distill,
        reconstruction=args.lambda_rec,
        identity=args.lambda_id,
        classifier=args.lambda_cls,
    )

    log_section("3. TRAINING CM PURIFIER...")
    LOGGER.info("Start step: %d", global_step)
    LOGGER.info("Batches per epoch-style pass: %d", len(dataloader))
    training_start_time = time.monotonic()
    start_step = global_step
    model.train()
    teacher.eval()
    while global_step < args.max_steps:
        for batch in dataloader:
            metrics = train_one_step(
                batch=batch,
                model=model,
                teacher=teacher,
                alpha_schedule=alpha_schedule,
                sigma_schedule=sigma_schedule,
                solver=solver,
                classifier=classifier_for_loss,
                optimizer=optimizer,
                loss_weights=loss_weights,
                args=args,
                device=device,
            )
            global_step += 1
            if global_step % args.log_steps == 0 or global_step == 1:
                LOGGER.info(
                    format_metrics(
                        global_step=global_step,
                        max_steps=args.max_steps,
                        metrics=metrics,
                        optimizer=optimizer,
                        start_step=start_step,
                        start_time=training_start_time,
                        device=device,
                    )
                )
            if global_step % args.save_steps == 0:
                save_training_checkpoint(args.out, model, teacher, optimizer, args, global_step, betas)
                LOGGER.info("Saved checkpoint to %s at step %d", args.out, global_step)
            if global_step >= args.max_steps:
                break

    log_section("4. SAVING FINAL CHECKPOINT...")
    save_training_checkpoint(args.out, model, teacher, optimizer, args, global_step, betas)
    log_section("Training complete")
    LOGGER.info("Finished training at step %d", global_step)
    LOGGER.info("Final checkpoint saved to %s", args.out)
    return args.out


if __name__ == "__main__":
    main()
