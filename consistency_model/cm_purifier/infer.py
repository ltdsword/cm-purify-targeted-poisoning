"""One-step dataset sanitization with a trained CM purifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from .dataset import load_image_tensor
from .checkpoint import load_purifier_from_checkpoint
from .schedules import minus_one_to_one_to_zero_one, q_sample


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# Purpose: Build the command-line parser for one-step purification.
# Input: no arguments.
# Output: argparse parser with inference options.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Purify an image directory with a trained CM purifier.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--t-star", type=float, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-student", action="store_true", help="Use student weights instead of EMA weights.")
    parser.add_argument("--recursive", action="store_true", help="Read images recursively from input directory.")
    return parser


# Purpose: Resolve the requested torch device.
# Input: device argument string.
# Output: torch.device object.
def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# Purpose: List input images from a file or directory.
# Input: input path and recursive flag.
# Output: sorted list of image paths.
def list_images(input_path: Path, recursive: bool = False) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    pattern = "**/*" if recursive else "*"
    images = [
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    return sorted(images)


# Purpose: Convert a possibly fractional t-star into a valid integer timestep.
# Input: t-star argument and number of DDPM timesteps.
# Output: integer timestep in [0, num_train_timesteps - 1].
def resolve_t_star(t_star: float, num_train_timesteps: int) -> int:
    if 0.0 < t_star <= 1.0:
        resolved = round(t_star * (num_train_timesteps - 1))
    else:
        resolved = round(t_star)
    return int(max(0, min(num_train_timesteps - 1, resolved)))


# Purpose: Save a single CHW tensor in [0, 1] as an RGB image.
# Input: tensor and destination path.
# Output: none; image is written to disk.
def save_image_tensor(tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


# Purpose: Purify one batch of image tensors with one neural function evaluation.
# Input: model, batch tensor, t-star integer, alpha schedule, sigma schedule, and device.
# Output: purified batch tensor in [0, 1].
def purify_batch(model, batch, t_star: int, alpha_schedule, sigma_schedule, device: torch.device):
    batch = batch.to(device)
    timesteps = torch.full((batch.shape[0],), t_star, dtype=torch.long, device=device)
    noise = torch.randn_like(batch)
    x_t = q_sample(batch, timesteps, noise, alpha_schedule, sigma_schedule)
    with torch.no_grad():
        purified = model(x_t, timesteps, alpha_schedule, sigma_schedule)
    return minus_one_to_one_to_zero_one(purified)


# Purpose: Run one-step purification over all images in the input path.
# Input: parsed command-line arguments.
# Output: number of purified images.
def main(args=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(args)
    device = resolve_device(args.device)
    input_path = Path(args.input)
    output_path = Path(args.output)
    image_paths = list_images(input_path, recursive=args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {input_path}")

    model, alpha_schedule, sigma_schedule, train_args = load_purifier_from_checkpoint(
        Path(args.checkpoint),
        device=device,
        use_student=args.use_student,
    )
    image_size = int(train_args.get("image_size", 32))
    t_star = resolve_t_star(args.t_star, len(alpha_schedule))

    for start in range(0, len(image_paths), args.batch_size):
        paths = image_paths[start : start + args.batch_size]
        batch = torch.stack([load_image_tensor(path, image_size=image_size) for path in paths], dim=0)
        purified = purify_batch(model, batch, t_star, alpha_schedule, sigma_schedule, device)
        for tensor, source_path in zip(purified, paths):
            relative = source_path.name if input_path.is_file() else source_path.relative_to(input_path)
            save_image_tensor(tensor, output_path / relative)

    print(f"purified {len(image_paths)} images to {output_path} at t_star={t_star}")
    return len(image_paths)


if __name__ == "__main__":
    main()
