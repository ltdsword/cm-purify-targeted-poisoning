"""Reusable Algorithm 3 purifier module.

This file is intentionally small and callable: it exposes the paper-style
function f(image) -> purified image while sharing the same CM checkpoint loader
used by the training package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image

from consistency_model.cm_purifier.checkpoint import load_purifier_from_checkpoint
from consistency_model.cm_purifier.dataset import load_image_tensor
from consistency_model.cm_purifier.infer import resolve_t_star, save_image_tensor
from consistency_model.cm_purifier.schedules import minus_one_to_one_to_zero_one, q_sample


# Purpose: Resolve a user-facing device string into a torch device.
# Input: device string such as "auto", "cuda", or "cpu".
# Output: concrete torch.device object.
def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# Purpose: Seed torch RNGs used for the DDPM noise draw during purification.
# Input: optional integer seed.
# Output: none; global torch RNG state is updated when seed is provided.
def seed_torch(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Purpose: Convert a PIL image to a CHW tensor in the requested purifier range.
# Input: PIL image, image size, and value range.
# Output: CHW float tensor in [0, 1] or [-1, 1].
def pil_to_tensor(image: Image.Image, image_size: int, value_range: str = "minus_one_to_one") -> torch.Tensor:
    image = image.convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if value_range == "minus_one_to_one":
        return tensor * 2.0 - 1.0
    if value_range == "zero_to_one":
        return tensor
    raise ValueError(f"Unknown value_range: {value_range}")


# Purpose: Convert a CHW tensor in [0, 1] into a PIL RGB image.
# Input: CHW tensor with values in [0, 1].
# Output: PIL RGB image.
def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


@dataclass
class CMPurifier:
    """Callable CM purifier for one-step Algorithm 3 inference."""

    model: torch.nn.Module
    alpha_schedule: torch.Tensor
    sigma_schedule: torch.Tensor
    train_args: dict
    image_size: int
    t_star: int
    device: torch.device

    # Purpose: Load a trained CM purifier from a .pth checkpoint.
    # Input: checkpoint path, t-star, device, optional seed, and student/EMA selector.
    # Output: initialized CMPurifier instance ready for image or batch purification.
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        t_star: float = 200,
        device: str | torch.device = "auto",
        seed: int | None = None,
        use_student: bool = False,
    ) -> "CMPurifier":
        device = resolve_device(device)
        seed_torch(seed)
        model, alpha_schedule, sigma_schedule, train_args = load_purifier_from_checkpoint(
            checkpoint_path,
            device=device,
            use_student=use_student,
        )
        image_size = int(train_args.get("image_size", 32))
        resolved_t_star = resolve_t_star(t_star, len(alpha_schedule))
        return cls(
            model=model,
            alpha_schedule=alpha_schedule,
            sigma_schedule=sigma_schedule,
            train_args=train_args,
            image_size=image_size,
            t_star=resolved_t_star,
            device=device,
        )

    # Purpose: Convert a tensor batch into the purifier's expected [-1, 1] range.
    # Input: CHW or BCHW tensor and declared input range.
    # Output: BCHW tensor in [-1, 1] and whether the original input was single-image.
    def _prepare_batch(self, images: torch.Tensor, input_range: str) -> tuple[torch.Tensor, bool]:
        single_image = images.dim() == 3
        if single_image:
            images = images.unsqueeze(0)
        if images.dim() != 4:
            raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(images.shape)}")
        images = images.float()
        if input_range == "zero_to_one":
            images = images * 2.0 - 1.0
        elif input_range != "minus_one_to_one":
            raise ValueError(f"Unknown input_range: {input_range}")
        return images, single_image

    # Purpose: Run the CM purifier on a tensor image or batch.
    # Input: CHW/BCHW tensor, input range, and desired output range.
    # Output: purified tensor in the requested output range, preserving CHW vs BCHW shape.
    def purify_tensor(
        self,
        images: torch.Tensor,
        input_range: str = "minus_one_to_one",
        output_range: str = "zero_to_one",
    ) -> torch.Tensor:
        batch, single_image = self._prepare_batch(images, input_range=input_range)
        batch = batch.to(self.device)
        timesteps = torch.full((batch.shape[0],), self.t_star, dtype=torch.long, device=self.device)
        noise = torch.randn(batch.shape, dtype=batch.dtype, device=self.device)
        x_t = q_sample(batch, timesteps, noise, self.alpha_schedule, self.sigma_schedule)
        with torch.no_grad():
            purified = self.model(x_t, timesteps, self.alpha_schedule, self.sigma_schedule)
        if output_range == "zero_to_one":
            purified = minus_one_to_one_to_zero_one(purified)
        elif output_range != "minus_one_to_one":
            raise ValueError(f"Unknown output_range: {output_range}")
        purified = purified.detach().cpu()
        return purified[0] if single_image else purified

    # Purpose: Purify a single in-memory PIL image.
    # Input: PIL image.
    # Output: purified PIL RGB image.
    def purify_image(self, image: Image.Image) -> Image.Image:
        tensor = pil_to_tensor(image, image_size=self.image_size, value_range="minus_one_to_one")
        purified = self.purify_tensor(tensor, input_range="minus_one_to_one", output_range="zero_to_one")
        return tensor_to_pil(purified)

    # Purpose: Purify a single image file and optionally save it.
    # Input: source image path and optional output image path.
    # Output: purified tensor in [0, 1].
    def purify_path(self, source_path: str | Path, output_path: str | Path | None = None) -> torch.Tensor:
        source_path = Path(source_path)
        tensor = load_image_tensor(source_path, image_size=self.image_size)
        purified = self.purify_tensor(tensor, input_range="minus_one_to_one", output_range="zero_to_one")
        if output_path is not None:
            save_image_tensor(purified, Path(output_path))
        return purified

    # Purpose: Purify many image paths in batches while preserving caller-chosen destinations.
    # Input: source paths, output paths, and batch size.
    # Output: number of images purified and saved.
    def purify_paths(
        self,
        source_paths: Sequence[str | Path],
        output_paths: Sequence[str | Path],
        batch_size: int = 64,
    ) -> int:
        if len(source_paths) != len(output_paths):
            raise ValueError("source_paths and output_paths must have the same length")
        source_paths = [Path(path) for path in source_paths]
        output_paths = [Path(path) for path in output_paths]
        processed = 0
        for start in range(0, len(source_paths), batch_size):
            batch_sources = source_paths[start : start + batch_size]
            batch_outputs = output_paths[start : start + batch_size]
            batch = torch.stack(
                [load_image_tensor(path, image_size=self.image_size) for path in batch_sources],
                dim=0,
            )
            purified = self.purify_tensor(batch, input_range="minus_one_to_one", output_range="zero_to_one")
            for tensor, output_path in zip(purified, batch_outputs):
                save_image_tensor(tensor, output_path)
            processed += len(batch_sources)
        return processed


# Purpose: Convenience wrapper matching the paper notation f(image).
# Input: PIL image, checkpoint path, t-star, device, and optional seed.
# Output: purified PIL image.
def purify_image(
    image: Image.Image,
    checkpoint_path: str | Path,
    t_star: float = 200,
    device: str | torch.device = "auto",
    seed: int | None = None,
) -> Image.Image:
    purifier = CMPurifier.from_checkpoint(checkpoint_path, t_star=t_star, device=device, seed=seed)
    return purifier.purify_image(image)


# Purpose: Purify an iterable of source paths into an iterable of output paths.
# Input: image paths, output paths, checkpoint path, t-star, device, seed, and batch size.
# Output: number of purified images.
def purify_paths(
    source_paths: Iterable[str | Path],
    output_paths: Iterable[str | Path],
    checkpoint_path: str | Path,
    t_star: float = 200,
    device: str | torch.device = "auto",
    seed: int | None = None,
    batch_size: int = 64,
) -> int:
    purifier = CMPurifier.from_checkpoint(checkpoint_path, t_star=t_star, device=device, seed=seed)
    return purifier.purify_paths(list(source_paths), list(output_paths), batch_size=batch_size)
