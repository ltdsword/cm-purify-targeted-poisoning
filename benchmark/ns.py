"""Narcissus clean-label backdoor victim training and evaluation.

The victim is a CIFAR ResNet-18 trained from scratch on either the poisoned or
CM-purified 50,000-image training folder. Test queries are never purified.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset_generation.Narcissus.integration import apply_trigger_array, load_trigger_artifact
from dataset_generation.Narcissus.models.resnet import ResNet18

from .common import CIFAR_MEAN, CIFAR_STD
from .materialize import MaterializedCase


@dataclass(frozen=True)
class NSVictimConfig:
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    milestones: Tuple[int, ...] = (100, 150)
    num_workers: int = 8
    seed: int = 62000
    checkpoint_interval: int = 1
    triggered_test_limit: int | None = None
    resume: bool = True

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("NS victim epochs and batch size must be positive")
        if self.num_workers < 0 or self.checkpoint_interval <= 0:
            raise ValueError("NS num_workers must be non-negative and checkpoint interval positive")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _fingerprint(config: NSVictimConfig, condition: str, case_name: str) -> str:
    config_payload = asdict(config)
    config_payload.pop("triggered_test_limit", None)
    config_payload.pop("resume", None)
    payload = {**config_payload, "condition": condition, "case": case_name, "model": "cifar_resnet18"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _atomic_save(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def folder_records(train_dir: Path) -> List[Tuple[Path, int, int]]:
    records: List[Tuple[Path, int, int]] = []
    label_dirs = sorted((path for path in train_dir.iterdir() if path.is_dir()), key=lambda path: int(path.name))
    for label_dir in label_dirs:
        label = int(label_dir.name)
        for image_path in sorted(label_dir.glob("*.png"), key=lambda path: int(path.stem)):
            records.append((image_path, label, int(image_path.stem)))
    records.sort(key=lambda record: record[2])
    if len(records) != 50000:
        raise ValueError(f"Expected 50,000 NS victim training images in {train_dir}, found {len(records)}")
    return records


class FolderTrainingDataset(Dataset):
    def __init__(self, records: Sequence[Tuple[Path, int, int]], transform) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path, label, _ = self.records[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


class TriggeredCIFAR10(Dataset):
    def __init__(self, base_dataset, indices: Sequence[int], trigger: torch.Tensor, target_class: int, transform) -> None:
        self.base_dataset = base_dataset
        self.indices = [int(value) for value in indices]
        self.trigger = trigger
        self.target_class = int(target_class)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        test_index = self.indices[position]
        image, label = self.base_dataset[test_index]
        if int(label) == self.target_class:
            raise ValueError("Triggered NS evaluation must exclude target-class test images")
        patched = apply_trigger_array(np.asarray(image.convert("RGB"), dtype=np.uint8), self.trigger, scale=1.0)
        return self.transform(Image.fromarray(patched, mode="RGB")), int(label)


def _worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_seed,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def _load_checkpoint(path: Path, fingerprint: str, device: torch.device) -> Dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("config_fingerprint") != fingerprint:
        raise ValueError(f"Incompatible NS victim checkpoint: {path}")
    return checkpoint


def train_victim(
    materialized: MaterializedCase,
    train_dir: Path,
    condition: str,
    config: NSVictimConfig,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[torch.nn.Module, Dict[str, object]]:
    """Train or resume one deterministic NS victim condition."""
    config.validate()
    _seed_everything(config.seed)
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    dataset = FolderTrainingDataset(folder_records(train_dir), transform)
    model = ResNet18().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(config.milestones), gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    checkpoint_path = materialized.case_output_dir / "victim_checkpoints" / condition / "state.pt"
    fingerprint = _fingerprint(config, condition, materialized.case.name)
    checkpoint = _load_checkpoint(checkpoint_path, fingerprint, device) if config.resume else None
    next_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        next_epoch = int(checkpoint["next_epoch"])
        _restore_rng_state(checkpoint["rng_state"])
        logger.info("Resuming NS %s victim at epoch %d/%d", condition, next_epoch, config.epochs)

    started_at = time.monotonic()
    for epoch in range(next_epoch, config.epochs):
        loader = _make_loader(dataset, config.batch_size, config.num_workers, True, config.seed + epoch)
        model.train()
        correct = total = 0
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)
        scheduler.step()
        logger.info(
            "NS %s victim epoch %d/%d | loss=%.4f | train_acc=%.2f%% | elapsed=%.1fs",
            condition, epoch + 1, config.epochs, total_loss / max(total, 1),
            100.0 * correct / max(total, 1), time.monotonic() - started_at,
        )
        if (epoch + 1) % config.checkpoint_interval == 0 or epoch + 1 == config.epochs:
            _atomic_save(
                {
                    "format": "cm_ns_victim_checkpoint_v1",
                    "config_fingerprint": fingerprint,
                    "condition": condition,
                    "next_epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng_state": _rng_state(),
                }, checkpoint_path,
            )
    return model, {"checkpoint": str(checkpoint_path), "completed_epochs": config.epochs, "config": asdict(config)}


@torch.no_grad()
def _accuracy(model, loader: DataLoader, device: torch.device, target_class: int | None = None) -> tuple[float, int]:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        if target_class is not None:
            mask = labels == target_class
            predictions, labels = predictions[mask], labels[mask]
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    return 100.0 * correct / max(total, 1), total


@torch.no_grad()
def _attack_success(model, loader: DataLoader, device: torch.device, target_class: int) -> tuple[float, int]:
    model.eval()
    successes = total = 0
    for images, _ in loader:
        predictions = model(images.to(device, non_blocking=True)).argmax(dim=1)
        successes += int((predictions == target_class).sum().item())
        total += int(predictions.numel())
    return 100.0 * successes / max(total, 1), total


def _read_query_indices(path: Path, limit: int | None) -> List[int]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    indices = [int(record["test_index"]) for record in records]
    return indices if limit is None else indices[:limit]


def evaluate_model(
    model: torch.nn.Module,
    materialized: MaterializedCase,
    cifar_root: Path,
    config: NSVictimConfig,
    device: torch.device,
) -> Dict[str, object]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    clean_test = torchvision.datasets.CIFAR10(root=str(cifar_root), train=False, download=False, transform=transform)
    raw_test = torchvision.datasets.CIFAR10(root=str(cifar_root), train=False, download=False)
    query_indices = _read_query_indices(materialized.case.case_dir / "triggered_test_manifest.jsonl", config.triggered_test_limit)
    trigger, trigger_metadata = load_trigger_artifact(materialized.case.case_dir / "trigger.pt")
    triggered_test = TriggeredCIFAR10(raw_test, query_indices, trigger, materialized.case.class_idx, transform)
    clean_loader = _make_loader(clean_test, config.batch_size, config.num_workers, False, config.seed)
    triggered_loader = _make_loader(triggered_test, config.batch_size, config.num_workers, False, config.seed)
    natural_accuracy, clean_count = _accuracy(model, clean_loader, device)
    target_accuracy, target_count = _accuracy(model, clean_loader, device, materialized.case.class_idx)
    asr, triggered_count = _attack_success(model, triggered_loader, device, materialized.case.class_idx)
    return {
        "clean_acc": natural_accuracy,
        "natural_accuracy": natural_accuracy,
        "target_acc": asr,
        "attack_success": asr,
        "asr": asr,
        "target_class_acc": target_accuracy,
        "clean_test_count": clean_count,
        "target_class_test_count": target_count,
        "triggered_test_count": triggered_count,
        "trigger_sha256": trigger_metadata["sha256"],
        "test_queries_purified": False,
        "trigger_scale": 1.0,
    }


def evaluate_ns_case(
    materialized: MaterializedCase,
    cifar_root: Path,
    config: NSVictimConfig,
    device: str,
    logger: logging.Logger,
) -> Dict[str, object]:
    resolved_device = torch.device(device)
    results: Dict[str, object] = {}
    training: Dict[str, object] = {}
    for condition, train_dir in (("poison", materialized.poisoned_train_dir), ("purified", materialized.purified_train_dir)):
        model, training_metadata = train_victim(materialized, train_dir, condition, config, resolved_device, logger)
        results[condition] = evaluate_model(model, materialized, cifar_root, config, resolved_device)
        training[condition] = training_metadata
    results["training"] = training
    return results
