"""Shared benchmark utilities for logging, images, and result files."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# Purpose: Configure timestamped stdout logging for Slurm logs.
# Input: logger name.
# Output: configured logger instance.
def setup_logging(name: str = "benchmark") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(name)


# Purpose: Emit a visible section banner in benchmark logs.
# Input: logger and section title.
# Output: none; logs are written to stdout.
def log_section(logger: logging.Logger, title: str) -> None:
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)


# Purpose: Format elapsed or remaining seconds as HH:MM:SS.
# Input: duration in seconds.
# Output: human-readable duration.
def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# Purpose: Format CUDA memory use for progress logs.
# Input: torch device.
# Output: memory string, or "cpu" for CPU runs.
def format_gpu_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    allocated_gb = torch.cuda.memory_allocated(device) / 1e9
    reserved_gb = torch.cuda.memory_reserved(device) / 1e9
    return f"{allocated_gb:.1f}/{reserved_gb:.1f} GB"


# Purpose: Return whether a path points to a supported image.
# Input: filesystem path.
# Output: True for supported image files, otherwise False.
def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


# Purpose: Remove and recreate a directory for fresh case artifacts.
# Input: directory path.
# Output: empty directory exists.
def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# Purpose: Copy a directory if it exists, replacing the destination.
# Input: source and destination directories.
# Output: destination directory copied, or skipped when source is absent.
def copy_dir_if_exists(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


# Purpose: Load an image path as RGB PIL.
# Input: filesystem path.
# Output: PIL RGB image.
def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


# Purpose: Save a PIL image to a path, creating parent directories first.
# Input: PIL image and destination path.
# Output: image written to disk.
def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


# Purpose: Convert an image path to a CIFAR-normalized tensor.
# Input: image path and optional mean/std.
# Output: CHW float tensor normalized by CIFAR statistics.
def load_cifar_normalized_tensor(
    path: Path,
    mean: Sequence[float] = CIFAR_MEAN,
    std: Sequence[float] = CIFAR_STD,
) -> torch.Tensor:
    image = load_rgb(path)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


# Purpose: Convert a PIL image to a CIFAR-normalized tensor.
# Input: PIL image and optional mean/std.
# Output: CHW float tensor normalized by CIFAR statistics.
def pil_to_cifar_normalized_tensor(
    image: Image.Image,
    mean: Sequence[float] = CIFAR_MEAN,
    std: Sequence[float] = CIFAR_STD,
) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


# Purpose: Convert a CIFAR-normalized tensor back into a PIL image.
# Input: CHW tensor normalized by CIFAR statistics.
# Output: PIL RGB image.
def cifar_normalized_tensor_to_pil(
    tensor: torch.Tensor,
    mean: Sequence[float] = CIFAR_MEAN,
    std: Sequence[float] = CIFAR_STD,
) -> Image.Image:
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    image = (tensor.detach() * std_tensor + mean_tensor).clamp(0, 1).cpu()
    array = image.permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


# Purpose: Recursively list images under a directory in stable order.
# Input: root directory.
# Output: sorted list of image paths.
def list_images(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*") if is_image(path))


# Purpose: Serialize a Python object as pretty JSON.
# Input: output path and JSON-compatible object.
# Output: JSON file written to disk.
def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# Purpose: Append one JSON object per line.
# Input: output path and JSON-compatible object.
# Output: JSONL row appended to disk.
def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


# Purpose: Write the benchmark CSV table with stable columns.
# Input: output path and row dictionaries.
# Output: CSV file written to disk.
def write_results_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    fieldnames = [
        "Case",
        "Target",
        "Attack",
        "Clean Accuracy (Poison)",
        "Target Acc (Poison)",
        "Clean Acc (Purified)",
        "Target Acc (Purified)",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


# Purpose: Build a compact progress line for long image processing phases.
# Input: current count, total count, start time, and device.
# Output: readable progress string.
def format_progress(processed: int, total: int, started_at: float, device: torch.device) -> str:
    elapsed = time.monotonic() - started_at
    rate = processed / max(elapsed, 1e-8)
    remaining = (total - processed) / max(rate, 1e-8)
    percent = 100.0 * processed / max(total, 1)
    return (
        f"Images: {processed}/{total} | {percent:.2f}% | {rate:.2f} img/s | "
        f"elapsed: {format_duration(elapsed)} | eta: {format_duration(remaining)} | "
        f"gpu: {format_gpu_memory(device)}"
    )
