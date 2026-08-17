"""Configurable, resumable Narcissus trigger generation.

The upstream ``narcissus_function.py`` is notebook-oriented and hard-codes a
machine path and CUDA device.  This module preserves its three optimization
stages while providing deterministic inputs, atomic checkpoints, and a
pixel-space trigger artifact suitable for the project pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .models.resnet import ResNet18_201


@dataclass(frozen=True)
class TriggerGenerationConfig:
    target_class: int
    kind: str
    seed: int
    cifar_root: str
    pood_root: str
    output_path: str
    checkpoint_dir: str
    synthesis_indices: tuple[int, ...]
    epsilon: float = 8.0 / 255.0
    surrogate_epochs: int = 200
    warmup_epochs: int = 5
    trigger_rounds: int = 1000
    batch_size: int = 350
    num_workers: int = 8
    surrogate_lr: float = 0.1
    warmup_lr: float = 0.1
    trigger_lr: float = 0.01
    checkpoint_interval: int = 10
    device: str = "cuda"

    def validate(self) -> None:
        if self.kind not in {"train", "eval"}:
            raise ValueError(f"kind must be train or eval, got {self.kind!r}")
        if not 0 <= self.target_class <= 9:
            raise ValueError(f"target_class must be in [0, 9], got {self.target_class}")
        if self.epsilon <= 0 or self.epsilon > 1:
            raise ValueError(f"Invalid epsilon: {self.epsilon}")
        if not self.synthesis_indices:
            raise ValueError("At least one target-class synthesis index is required")
        if min(self.synthesis_indices) < 0 or max(self.synthesis_indices) >= 50000:
            raise ValueError("CIFAR-10 synthesis index is out of range")
        for value, name in [
            (self.surrogate_epochs, "surrogate_epochs"),
            (self.warmup_epochs, "warmup_epochs"),
            (self.trigger_rounds, "trigger_rounds"),
            (self.batch_size, "batch_size"),
            (self.checkpoint_interval, "checkpoint_interval"),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive")


class ConcoctDataset(Dataset):
    """POOD images retain labels 0..199; target-class images use label 200."""

    def __init__(self, target_dataset: Dataset, pood_dataset: Dataset) -> None:
        self.target_dataset = target_dataset
        self.pood_dataset = pood_dataset

    def __len__(self) -> int:
        return len(self.pood_dataset) + len(self.target_dataset)

    def __getitem__(self, index: int):
        if index < len(self.pood_dataset):
            return self.pood_dataset[index]
        image, _ = self.target_dataset[index - len(self.pood_dataset)]
        return image, 200


def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _config_payload(config: TriggerGenerationConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["synthesis_indices_sha256"] = hashlib.sha256(
        np.asarray(config.synthesis_indices, dtype=np.int64).tobytes()
    ).hexdigest()
    payload.pop("synthesis_indices")
    payload["format"] = "cm_narcissus_trigger_v1"
    return payload


def config_fingerprint(config: TriggerGenerationConfig) -> str:
    return hashlib.sha256(_canonical_json(_config_payload(config)).encode("utf-8")).hexdigest()


def trigger_sha256(trigger: torch.Tensor) -> str:
    array = trigger.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_trigger(trigger: torch.Tensor, epsilon: float, tolerance: float = 1e-6) -> None:
    if tuple(trigger.shape) == (1, 3, 32, 32):
        trigger = trigger[0]
    if tuple(trigger.shape) != (3, 32, 32):
        raise ValueError(f"Expected trigger shape (3, 32, 32), got {tuple(trigger.shape)}")
    if not torch.isfinite(trigger).all():
        raise ValueError("Trigger contains NaN or Inf")
    maximum = float(trigger.abs().max().item())
    if maximum > epsilon + tolerance:
        raise ValueError(f"Trigger L-inf {maximum:.8f} exceeds epsilon {epsilon:.8f}")


def apply_trigger_array(image: np.ndarray, trigger: torch.Tensor, scale: float = 1.0) -> np.ndarray:
    """Apply a pixel-space CHW trigger to an HWC uint8/float image."""
    if not math.isfinite(float(scale)) or float(scale) < 0:
        raise ValueError(f"Trigger scale must be finite and non-negative, got {scale}")
    validate_trigger(trigger, epsilon=float(trigger.abs().max().item()) + 1e-8)
    if tuple(trigger.shape) == (1, 3, 32, 32):
        trigger = trigger[0]
    source = np.asarray(image)
    if source.shape != (32, 32, 3):
        raise ValueError(f"Expected image shape (32, 32, 3), got {source.shape}")
    was_uint8 = source.dtype == np.uint8
    pixels = source.astype(np.float32) / 255.0 if was_uint8 else source.astype(np.float32)
    delta = trigger.detach().cpu().permute(1, 2, 0).numpy() * float(scale)
    patched = np.clip(pixels + delta, 0.0, 1.0)
    if was_uint8:
        return np.clip(np.rint(patched * 255.0), 0, 255).astype(np.uint8)
    return patched.astype(np.float32)


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset: Dataset, config: TriggerGenerationConfig, epoch_seed: int, shuffle: bool = True) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(epoch_seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = (epoch_seed + worker_id + 1) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=str(config.device).startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _load_checkpoint(path: Path, config: TriggerGenerationConfig) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    expected = config_fingerprint(config)
    if state.get("config_fingerprint") != expected:
        raise RuntimeError(
            f"Checkpoint configuration mismatch at {path}. Move the old checkpoint aside "
            "or rerun with the exact original configuration."
        )
    return state


def _save_stage_checkpoint(
    path: Path,
    config: TriggerGenerationConfig,
    stage: str,
    next_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any = None,
    perturbation: torch.Tensor | None = None,
    perturbation_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "format": "cm_narcissus_checkpoint_v1",
        "config": _config_payload(config),
        "config_fingerprint": config_fingerprint(config),
        "stage": stage,
        "next_step": int(next_step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": _capture_rng_state(),
        "selected_indices": list(config.synthesis_indices),
    }
    if perturbation is not None:
        payload["perturbation"] = perturbation.detach().cpu()
    if perturbation_optimizer is not None:
        payload["perturbation_optimizer"] = perturbation_optimizer.state_dict()
    _atomic_torch_save(payload, path)


def _build_datasets(config: TriggerGenerationConfig):
    pood_root = Path(config.pood_root)
    if not pood_root.is_dir():
        raise FileNotFoundError(
            f"Tiny ImageNet POOD directory is missing: {pood_root}. "
            "Set --pood-root or NARCISSUS_POOD_ROOT to its ImageFolder train directory."
        )
    target_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    pood_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    cifar_train = datasets.CIFAR10(root=config.cifar_root, train=True, download=False, transform=target_transform)
    bad = [index for index in config.synthesis_indices if int(cifar_train.targets[index]) != config.target_class]
    if bad:
        raise ValueError(f"Synthesis indices do not all belong to target class {config.target_class}: {bad[:3]}")
    target_subset = Subset(cifar_train, list(config.synthesis_indices))
    pood_dataset = datasets.ImageFolder(root=str(pood_root), transform=pood_transform)
    if len(pood_dataset.classes) != 200:
        raise ValueError(f"Expected 200 Tiny ImageNet classes under {pood_root}, found {len(pood_dataset.classes)}")
    return target_subset, ConcoctDataset(target_subset, pood_dataset)


def load_trigger_artifact(path: str | Path) -> tuple[torch.Tensor, Dict[str, Any]]:
    try:
        artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(Path(path), map_location="cpu")
    if isinstance(artifact, torch.Tensor):
        return artifact.squeeze(0).float(), {}
    trigger = artifact["trigger"].squeeze(0).float()
    return trigger, dict(artifact.get("metadata", {}))


def generate_trigger(config: TriggerGenerationConfig) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Generate or resume one class-oriented trigger and return its final artifact."""
    config.validate()
    output_path = Path(config.output_path)
    metadata_path = output_path.with_suffix(".json")
    checkpoint_path = Path(config.checkpoint_dir) / "state.pt"
    if output_path.is_file() and metadata_path.is_file():
        trigger, metadata = load_trigger_artifact(output_path)
        validate_trigger(trigger, config.epsilon)
        if metadata.get("config_fingerprint") != config_fingerprint(config):
            raise RuntimeError(f"Completed trigger configuration mismatch: {output_path}")
        if metadata.get("sha256") != trigger_sha256(trigger):
            raise RuntimeError(f"Completed trigger hash mismatch: {output_path}")
        return trigger, metadata

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Narcissus trigger generation requested CUDA, but CUDA is unavailable")
    _seed_everything(config.seed)
    target_subset, surrogate_dataset = _build_datasets(config)
    criterion = torch.nn.CrossEntropyLoss()
    model = ResNet18_201().to(device)
    checkpoint = _load_checkpoint(checkpoint_path, config)
    stage = checkpoint.get("stage") if checkpoint else "surrogate"
    next_step = int(checkpoint.get("next_step", 0)) if checkpoint else 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        _restore_rng_state(checkpoint["rng_state"])

    if stage == "surrogate":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.surrogate_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.surrogate_epochs)
        if checkpoint and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
        for epoch in range(next_step, config.surrogate_epochs):
            model.train()
            for images, labels in _loader(surrogate_dataset, config, config.seed + epoch):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            if (epoch + 1) % config.checkpoint_interval == 0 or epoch + 1 == config.surrogate_epochs:
                _save_stage_checkpoint(checkpoint_path, config, "surrogate", epoch + 1, model, optimizer, scheduler)
        optimizer = torch.optim.RAdam(model.parameters(), lr=config.warmup_lr)
        _save_stage_checkpoint(checkpoint_path, config, "warmup", 0, model, optimizer)
        checkpoint = _load_checkpoint(checkpoint_path, config)
        stage, next_step = "warmup", 0

    if stage == "warmup":
        optimizer = torch.optim.RAdam(model.parameters(), lr=config.warmup_lr)
        if checkpoint and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        for epoch in range(next_step, config.warmup_epochs):
            model.train()
            for images, labels in _loader(target_subset, config, config.seed + 10000 + epoch):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
            _save_stage_checkpoint(checkpoint_path, config, "warmup", epoch + 1, model, optimizer)
        perturbation = torch.zeros((1, 3, 32, 32), device=device, requires_grad=True)
        perturbation_optimizer = torch.optim.RAdam([perturbation], lr=config.trigger_lr)
        _save_stage_checkpoint(
            checkpoint_path,
            config,
            "trigger",
            0,
            model,
            None,
            perturbation=perturbation,
            perturbation_optimizer=perturbation_optimizer,
        )
        checkpoint = _load_checkpoint(checkpoint_path, config)
        stage, next_step = "trigger", 0

    if stage != "trigger":
        raise RuntimeError(f"Unsupported Narcissus checkpoint stage: {stage}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    perturbation = checkpoint["perturbation"].to(device).detach().requires_grad_(True)
    perturbation_optimizer = torch.optim.RAdam([perturbation], lr=config.trigger_lr)
    if checkpoint.get("perturbation_optimizer") is not None:
        perturbation_optimizer.load_state_dict(checkpoint["perturbation_optimizer"])
    next_step = int(checkpoint.get("next_step", 0))
    normalized_bound = 2.0 * config.epsilon
    completed_trigger_rounds = next_step
    for round_index in range(next_step, config.trigger_rounds):
        for images, labels in _loader(target_subset, config, config.seed + 20000 + round_index):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            patched = torch.clamp(images + torch.clamp(perturbation, -normalized_bound, normalized_bound), -1.0, 1.0)
            perturbation_optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(patched), labels)
            loss.backward()
            perturbation_optimizer.step()
        completed_trigger_rounds = round_index + 1
        gradient_sum = float(perturbation.grad.detach().abs().sum().item()) if perturbation.grad is not None else 0.0
        should_stop = gradient_sum == 0.0
        if should_stop or (round_index + 1) % config.checkpoint_interval == 0 or round_index + 1 == config.trigger_rounds:
            _save_stage_checkpoint(
                checkpoint_path,
                config,
                "trigger",
                round_index + 1,
                model,
                None,
                perturbation=perturbation,
                perturbation_optimizer=perturbation_optimizer,
            )
        if should_stop:
            break

    pixel_trigger = torch.clamp(perturbation.detach().cpu()[0] / 2.0, -config.epsilon, config.epsilon).float()
    validate_trigger(pixel_trigger, config.epsilon)
    metadata = {
        **_config_payload(config),
        "config_fingerprint": config_fingerprint(config),
        "trigger_id": f"ns_{config.kind}_trigger_c{config.target_class:02d}_seed_{config.seed}",
        "sha256": trigger_sha256(pixel_trigger),
        "shape": list(pixel_trigger.shape),
        "linf": float(pixel_trigger.abs().max().item()),
        "space": "pixel_zero_to_one",
        "trigger_scale": 1.0,
        "completed_trigger_rounds": completed_trigger_rounds,
        "complete": True,
    }
    _atomic_torch_save({"trigger": pixel_trigger, "metadata": metadata}, output_path)
    _atomic_json_save(metadata, metadata_path)
    return pixel_trigger, metadata
