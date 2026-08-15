"""Materialize poisoned and purified train sets for benchmark cases."""

from __future__ import annotations

import logging
import json
import pickle
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from purify.purifier import CMPurifier

from .cases import BenchmarkCase, bp_poison_name, ns_poison_name, wb_poison_name
from .common import copy_dir_if_exists, format_progress, list_images, load_rgb, reset_dir, save_image, write_json


@dataclass(frozen=True)
class MaterializedCase:
    case: BenchmarkCase
    case_output_dir: Path
    poisoned_train_dir: Path
    purified_train_dir: Path
    purify_dir: Path
    target_dir: Path
    poison_relpaths: Dict[str, Path]
    bp_flat_base_indices: Optional[List[int]]
    train_image_count: int
    poison_image_count: int


@dataclass(frozen=True)
class BPSplitRecord:
    flat_index: int
    class_relative_index: int
    image: Image.Image
    label: int


@dataclass(frozen=True)
class PurificationStats:
    image_count: int
    elapsed_seconds: float
    seconds_per_image: float
    images_per_second: float
    mean_batch_seconds_per_image: float
    median_batch_seconds_per_image: float
    batch_size: int
    device: str
    t_star: int
    inference_seed: int
    checkpoint_sha256: str
    reused: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "image_count": self.image_count,
            "elapsed_seconds": self.elapsed_seconds,
            "seconds_per_image": self.seconds_per_image,
            "images_per_second": self.images_per_second,
            "mean_batch_seconds_per_image": self.mean_batch_seconds_per_image,
            "median_batch_seconds_per_image": self.median_batch_seconds_per_image,
            "batch_size": self.batch_size,
            "device": self.device,
            "t_star": self.t_star,
            "inference_seed": self.inference_seed,
            "checkpoint_sha256": self.checkpoint_sha256,
            "reused": self.reused,
        }


def load_completed_materialized_case(
    case: BenchmarkCase,
    output_root: Path,
) -> tuple[MaterializedCase, PurificationStats] | None:
    """Load a completed materialization/purification without changing artifacts."""
    case_output_dir = output_root / case.name
    summary_path = case_output_dir / "summary.json"
    if not summary_path.is_file():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    timing_payload = payload.get("purification_timing")
    poisoned_train_dir = case_output_dir / "poisoned_train"
    purified_train_dir = case_output_dir / "purified_train"
    if not isinstance(timing_payload, dict) or not poisoned_train_dir.is_dir() or not purified_train_dir.is_dir():
        return None
    expected_count = int(payload.get("train_image_count", 0))
    poisoned_count = sum(1 for _ in poisoned_train_dir.rglob("*.png"))
    purified_count = sum(1 for _ in purified_train_dir.rglob("*.png"))
    if expected_count <= 0 or poisoned_count != expected_count or purified_count != expected_count:
        return None
    timing_payload = dict(timing_payload)
    timing_payload["reused"] = True
    payload["purification_timing"] = timing_payload
    write_json(summary_path, payload)
    stats = PurificationStats(**timing_payload)
    bp_flat_base_indices = payload.get("bp_flat_base_indices")
    poison_relpaths: Dict[str, Path] = {}
    if case.attack == "WB":
        poison_relpaths = {
            wb_poison_name(case.class_idx, int(index)): Path(str(case.class_idx)) / f"{int(index)}.png"
            for index in case.setup["base indices"]
        }
    elif case.attack == "BP" and bp_flat_base_indices is not None:
        group_idx = int(case.group_idx) if case.group_idx is not None else 0
        poison_relpaths = {
            bp_poison_name(case.class_idx, group_idx, int(relative_index)): Path(str(case.class_idx)) / f"{int(flat_index)}.png"
            for relative_index, flat_index in zip(case.setup["base indices"], bp_flat_base_indices)
        }
    elif case.attack == "NS":
        poison_relpaths = {
            ns_poison_name(case.class_idx, int(index)): Path(str(case.class_idx)) / f"{int(index)}.png"
            for index in case.setup["base indices"]
        }
    materialized = MaterializedCase(
        case=case,
        case_output_dir=case_output_dir,
        poisoned_train_dir=poisoned_train_dir,
        purified_train_dir=purified_train_dir,
        purify_dir=case_output_dir / "purify",
        target_dir=case_output_dir / "target",
        poison_relpaths=poison_relpaths,
        bp_flat_base_indices=bp_flat_base_indices,
        train_image_count=int(payload["train_image_count"]),
        poison_image_count=int(payload["poison_image_count"]),
    )
    return materialized, stats


# Purpose: Import torchvision only when benchmark materialization needs CIFAR.
# Input: no arguments.
# Output: imported torchvision module.
def require_torchvision():
    try:
        import torchvision
    except ImportError as exc:
        raise ImportError("torchvision is required for benchmark dataset materialization") from exc
    return torchvision


# Purpose: Read the Bullseye compact split from pickle or torch serialization.
# Input: split path.
# Output: loaded split dictionary.
def load_bp_split(path: str | Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing BullseyePoison split: {path}")
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        import torch

        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


# Purpose: Convert one split subset into flat records plus class-relative indices.
# Input: loaded split object and subset name.
# Output: list of BPSplitRecord objects.
def build_bp_split_records(split, subset: str = "others") -> List[BPSplitRecord]:
    subset_obj = split[subset]
    if isinstance(subset_obj, dict) and "data" in subset_obj and "targets" in subset_obj:
        data = subset_obj["data"]
        targets = subset_obj["targets"]
        iterable = list(zip(data, targets))
    else:
        iterable = list(subset_obj)

    class_counts: Dict[int, int] = {}
    records: List[BPSplitRecord] = []
    for flat_index, (image_obj, label_obj) in enumerate(iterable):
        label = int(label_obj)
        class_relative_index = class_counts.get(label, 0)
        class_counts[label] = class_relative_index + 1
        if isinstance(image_obj, Image.Image):
            image = image_obj.convert("RGB")
        else:
            image = Image.fromarray(np.asarray(image_obj).astype(np.uint8)).convert("RGB")
        records.append(
            BPSplitRecord(
                flat_index=flat_index,
                class_relative_index=class_relative_index,
                image=image,
                label=label,
            )
        )
    return records


# Purpose: Build a lookup from BP class-relative index to flat split index.
# Input: BP split records and class label.
# Output: dictionary mapping class-relative index to flat split index.
def bp_class_relative_to_flat(records: List[BPSplitRecord], class_idx: int) -> Dict[int, int]:
    return {
        record.class_relative_index: record.flat_index
        for record in records
        if record.label == class_idx
    }


# Purpose: Prepare a clean case output directory with standard subdirectories.
# Input: case, benchmark output root, and overwrite flag.
# Output: MaterializedCase paths except counts/lookups.
def prepare_case_output(case: BenchmarkCase, output_root: Path, overwrite: bool) -> Path:
    case_output_dir = output_root / case.name
    if overwrite:
        reset_dir(case_output_dir)
    else:
        case_output_dir.mkdir(parents=True, exist_ok=True)
    return case_output_dir


# Purpose: Materialize a WB tampered full CIFAR-10 train folder.
# Input: case metadata, output root, CIFAR root, overwrite flag, and logger.
# Output: MaterializedCase with poison filename to train-relative path mapping.
def materialize_wb_case(
    case: BenchmarkCase,
    output_root: Path,
    cifar_root: Path,
    overwrite: bool,
    logger: logging.Logger,
) -> MaterializedCase:
    torchvision = require_torchvision()
    case_output_dir = prepare_case_output(case, output_root, overwrite=overwrite)
    poisoned_train_dir = case_output_dir / "poisoned_train"
    purified_train_dir = case_output_dir / "purified_train"
    purify_dir = case_output_dir / "purify"
    target_dir = case_output_dir / "target"
    reset_dir(poisoned_train_dir)
    reset_dir(purified_train_dir)
    reset_dir(purify_dir)
    copy_dir_if_exists(case.target_dir, target_dir)

    train_set = torchvision.datasets.CIFAR10(root=str(cifar_root), train=True, download=False)
    base_indices = [int(index) for index in case.setup["base indices"]]
    poison_by_index = {
        base_index: case.poison_dir / wb_poison_name(case.class_idx, base_index)
        for base_index in base_indices
    }
    missing = [path for path in poison_by_index.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing WB poison images for {case.name}: {missing[:3]}")

    poison_relpaths: Dict[str, Path] = {}
    started_at = time.monotonic()
    for index in range(len(train_set)):
        image, label = train_set[index]
        source_path = poison_by_index.get(index)
        if source_path is not None:
            image = load_rgb(source_path)
        relpath = Path(str(label)) / f"{index}.png"
        save_image(image, poisoned_train_dir / relpath)
        if source_path is not None:
            poison_relpaths[source_path.name] = relpath
        if (index + 1) % 10000 == 0:
            logger.info("Materialized WB poisoned train | %s", format_progress(index + 1, len(train_set), started_at, torch_device_cpu()))

    return MaterializedCase(
        case=case,
        case_output_dir=case_output_dir,
        poisoned_train_dir=poisoned_train_dir,
        purified_train_dir=purified_train_dir,
        purify_dir=purify_dir,
        target_dir=target_dir,
        poison_relpaths=poison_relpaths,
        bp_flat_base_indices=None,
        train_image_count=len(train_set),
        poison_image_count=len(base_indices),
    )


# Purpose: Return a CPU torch.device without importing torch at module import time.
# Input: no arguments.
# Output: torch CPU device.
def torch_device_cpu():
    import torch

    return torch.device("cpu")


# Purpose: Materialize a BP tampered full compact-split train folder.
# Input: case metadata, output root, BP split path, overwrite flag, and logger.
# Output: MaterializedCase with BP flat base indices for evaluator use.
def materialize_bp_case(
    case: BenchmarkCase,
    output_root: Path,
    bp_split_path: Path,
    overwrite: bool,
    logger: logging.Logger,
) -> MaterializedCase:
    case_output_dir = prepare_case_output(case, output_root, overwrite=overwrite)
    poisoned_train_dir = case_output_dir / "poisoned_train"
    purified_train_dir = case_output_dir / "purified_train"
    purify_dir = case_output_dir / "purify"
    target_dir = case_output_dir / "target"
    reset_dir(poisoned_train_dir)
    reset_dir(purified_train_dir)
    reset_dir(purify_dir)
    copy_dir_if_exists(case.target_dir, target_dir)

    split = load_bp_split(bp_split_path)
    records = build_bp_split_records(split, subset="others")
    relative_to_flat = bp_class_relative_to_flat(records, class_idx=case.class_idx)
    group_idx = int(case.group_idx) if case.group_idx is not None else 0
    class_relative_indices = [int(index) for index in case.setup["base indices"]]
    flat_base_indices = [relative_to_flat[index] for index in class_relative_indices]
    poison_by_flat_index = {
        flat_index: case.poison_dir / bp_poison_name(case.class_idx, group_idx, class_relative_index)
        for flat_index, class_relative_index in zip(flat_base_indices, class_relative_indices)
    }
    missing = [path for path in poison_by_flat_index.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing BP poison images for {case.name}: {missing[:3]}")

    poison_relpaths: Dict[str, Path] = {}
    started_at = time.monotonic()
    for record in records:
        image = record.image
        source_path = poison_by_flat_index.get(record.flat_index)
        if source_path is not None:
            image = load_rgb(source_path)
        relpath = Path(str(record.label)) / f"{record.flat_index}.png"
        save_image(image, poisoned_train_dir / relpath)
        if source_path is not None:
            poison_relpaths[source_path.name] = relpath
        if (record.flat_index + 1) % 10000 == 0:
            logger.info("Materialized BP poisoned train | %s", format_progress(record.flat_index + 1, len(records), started_at, torch_device_cpu()))

    return MaterializedCase(
        case=case,
        case_output_dir=case_output_dir,
        poisoned_train_dir=poisoned_train_dir,
        purified_train_dir=purified_train_dir,
        purify_dir=purify_dir,
        target_dir=target_dir,
        poison_relpaths=poison_relpaths,
        bp_flat_base_indices=flat_base_indices,
        train_image_count=len(records),
        poison_image_count=len(class_relative_indices),
    )


def materialize_ns_case(
    case: BenchmarkCase,
    output_root: Path,
    cifar_root: Path,
    overwrite: bool,
    logger: logging.Logger,
) -> MaterializedCase:
    """Materialize one 50,000-image Narcissus attacked training set."""
    torchvision = require_torchvision()
    case_output_dir = prepare_case_output(case, output_root, overwrite=overwrite)
    poisoned_train_dir = case_output_dir / "poisoned_train"
    purified_train_dir = case_output_dir / "purified_train"
    purify_dir = case_output_dir / "purify"
    target_dir = case_output_dir / "target"
    reset_dir(poisoned_train_dir)
    reset_dir(purified_train_dir)
    reset_dir(purify_dir)
    train_set = torchvision.datasets.CIFAR10(root=str(cifar_root), train=True, download=False)
    base_indices = [int(index) for index in case.setup["base indices"]]
    if len(base_indices) != int(case.setup["num_poison_train"]):
        raise ValueError(f"NS poison count disagrees with metadata for {case.name}")
    poison_by_index = {
        base_index: case.poison_dir / ns_poison_name(case.class_idx, base_index)
        for base_index in base_indices
    }
    missing = [path for path in poison_by_index.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing NS poison images for {case.name}: {missing[:3]}")
    poison_relpaths: Dict[str, Path] = {}
    started_at = time.monotonic()
    for index in range(len(train_set)):
        image, label = train_set[index]
        source_path = poison_by_index.get(index)
        if source_path is not None:
            if int(label) != case.class_idx:
                raise ValueError(f"NS poison index {index} is not target class {case.class_idx}")
            image = load_rgb(source_path)
        relpath = Path(str(label)) / f"{index}.png"
        save_image(image, poisoned_train_dir / relpath)
        if source_path is not None:
            poison_relpaths[source_path.name] = relpath
        if (index + 1) % 10000 == 0:
            logger.info(
                "Materialized NS poisoned train | %s",
                format_progress(index + 1, len(train_set), started_at, torch_device_cpu()),
            )
    return MaterializedCase(
        case=case,
        case_output_dir=case_output_dir,
        poisoned_train_dir=poisoned_train_dir,
        purified_train_dir=purified_train_dir,
        purify_dir=purify_dir,
        target_dir=target_dir,
        poison_relpaths=poison_relpaths,
        bp_flat_base_indices=None,
        train_image_count=len(train_set),
        poison_image_count=len(base_indices),
    )


# Purpose: Purify all images in a materialized train folder and write the inspection subset.
# Input: MaterializedCase, CMPurifier, batch size, log interval, and logger.
# Output: synchronized per-case purification timing statistics.
def purify_materialized_case(
    materialized: MaterializedCase,
    purifier: CMPurifier,
    batch_size: int,
    log_steps: int,
    logger: logging.Logger,
    checkpoint_sha256: str = "",
) -> PurificationStats:
    source_paths = list_images(materialized.poisoned_train_dir)
    if not source_paths:
        raise FileNotFoundError(f"No images found in {materialized.poisoned_train_dir}")
    output_paths = [
        materialized.purified_train_dir / source_path.relative_to(materialized.poisoned_train_dir)
        for source_path in source_paths
    ]

    import torch

    if purifier.device.type == "cuda":
        torch.cuda.synchronize(purifier.device)
    started_at = time.monotonic()
    processed = 0
    batch_seconds_per_image: List[float] = []
    for start in range(0, len(source_paths), batch_size):
        batch_sources = source_paths[start : start + batch_size]
        batch_outputs = output_paths[start : start + batch_size]
        batch_started_at = time.monotonic()
        purifier.purify_paths(batch_sources, batch_outputs, batch_size=len(batch_sources))
        if purifier.device.type == "cuda":
            torch.cuda.synchronize(purifier.device)
        batch_elapsed = time.monotonic() - batch_started_at
        batch_seconds_per_image.append(batch_elapsed / max(len(batch_sources), 1))
        processed += len(batch_sources)
        if processed == len(source_paths) or processed % max(log_steps, 1) == 0:
            logger.info("Purified full train | %s", format_progress(processed, len(source_paths), started_at, purifier.device))

    elapsed_seconds = time.monotonic() - started_at
    for poison_name, relpath in materialized.poison_relpaths.items():
        source = materialized.purified_train_dir / relpath
        destination = materialized.purify_dir / poison_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    schedule = purifier.schedule_statistics()
    stats = PurificationStats(
        image_count=processed,
        elapsed_seconds=elapsed_seconds,
        seconds_per_image=elapsed_seconds / max(processed, 1),
        images_per_second=processed / max(elapsed_seconds, 1e-8),
        mean_batch_seconds_per_image=statistics.fmean(batch_seconds_per_image),
        median_batch_seconds_per_image=statistics.median(batch_seconds_per_image),
        batch_size=batch_size,
        device=str(purifier.device),
        t_star=int(schedule["t_star"]),
        inference_seed=int(schedule["seed"]),
        checkpoint_sha256=checkpoint_sha256,
    )
    write_json(
        materialized.case_output_dir / "summary.json",
        {
            "case": materialized.case.name,
            "attack": materialized.case.attack,
            "target_class": int(materialized.case.setup["target class"]),
            "target_index": int(materialized.case.setup["target index"]),
            "base_class": int(materialized.case.setup["base class"]),
            "poisoned_train_dir": str(materialized.poisoned_train_dir),
            "purified_train_dir": str(materialized.purified_train_dir),
            "purify_dir": str(materialized.purify_dir),
            "target_dir": str(materialized.target_dir),
            "train_image_count": materialized.train_image_count,
            "poison_image_count": materialized.poison_image_count,
            "bp_flat_base_indices": materialized.bp_flat_base_indices,
            "purification_timing": stats.to_dict(),
        },
    )
    logger.info(
        "Purification timing for %s | %.3f s | %.6f s/image | %.2f images/s",
        materialized.case.name,
        stats.elapsed_seconds,
        stats.seconds_per_image,
        stats.images_per_second,
    )
    return stats
