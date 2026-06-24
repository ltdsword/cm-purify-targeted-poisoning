"""Dataset utilities for clean/poison CIFAR-10 purifier pairs."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

from . import ATTACK_TO_ID


_LABEL_PATTERN = re.compile(r"(?:^|_)c(?P<label>[0-9]+)(?:_|\.)")


@dataclass(frozen=True)
class PairRecord:
    filename: str
    clean_path: Path
    poison_path: Path
    attack_type: str
    attack_id: int
    label: int


# Purpose: Import torch only when a caller actually needs tensor output.
# Input: no arguments.
# Output: the imported torch module, or a clear ImportError with install guidance.
def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Torch is required for tensor dataset items. Install the project ML "
            "dependencies before training or inference."
        ) from exc
    return torch


# Purpose: Infer the purifier attack type from a generated pair filename.
# Input: a PNG filename such as wb_c2_24919.png, bp_c0_g0_1500.png, or clean_c1_1.png.
# Output: one of clean, wb, or bp.
def parse_attack_type(filename: str) -> str:
    prefix = filename.split("_", 1)[0].lower()
    if prefix not in ATTACK_TO_ID:
        raise ValueError(f"Unsupported attack prefix in filename: {filename}")
    return prefix


# Purpose: Infer the CIFAR-10 class label from a generated pair filename.
# Input: a PNG filename containing a c{label} token.
# Output: the integer class label.
def parse_label(filename: str) -> int:
    match = _LABEL_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Could not parse class label from filename: {filename}")
    label = int(match.group("label"))
    if label < 0 or label > 9:
        raise ValueError(f"Parsed label is outside CIFAR-10 range in {filename}: {label}")
    return label


# Purpose: Load an RGB image from disk as a float numpy array.
# Input: image path and expected spatial size.
# Output: float32 array in HWC layout with values in [0, 1].
def load_image_array(path: Path, image_size: int = 32) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


# Purpose: Convert an image file into a normalized torch tensor.
# Input: image path, expected image size, and target value range.
# Output: CHW tensor with values in either [0, 1] or [-1, 1].
def load_image_tensor(path: Path, image_size: int = 32, value_range: str = "minus_one_to_one"):
    torch = _require_torch()
    array = load_image_array(path, image_size=image_size)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if value_range == "minus_one_to_one":
        tensor = tensor * 2.0 - 1.0
    elif value_range != "zero_to_one":
        raise ValueError(f"Unknown value_range: {value_range}")
    return tensor


# Purpose: Build records by matching clean and poison images by filename.
# Input: pair directory with clean/ and poisons/ subdirectories.
# Output: sorted list of PairRecord objects.
def build_pair_records(pair_dir: Path) -> List[PairRecord]:
    clean_dir = pair_dir / "clean"
    poison_dir = pair_dir / "poisons"
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Missing clean image directory: {clean_dir}")
    if not poison_dir.is_dir():
        raise FileNotFoundError(f"Missing poison image directory: {poison_dir}")

    clean_files = {path.name: path for path in clean_dir.glob("*.png")}
    poison_files = {path.name: path for path in poison_dir.glob("*.png")}
    missing_poison = sorted(set(clean_files) - set(poison_files))
    missing_clean = sorted(set(poison_files) - set(clean_files))
    if missing_poison or missing_clean:
        raise ValueError(
            "Clean/poison PNG filenames do not match. "
            f"Missing poison={len(missing_poison)}, missing clean={len(missing_clean)}"
        )

    records: List[PairRecord] = []
    for filename in sorted(clean_files):
        attack_type = parse_attack_type(filename)
        records.append(
            PairRecord(
                filename=filename,
                clean_path=clean_files[filename],
                poison_path=poison_files[filename],
                attack_type=attack_type,
                attack_id=ATTACK_TO_ID[attack_type],
                label=parse_label(filename),
            )
        )
    return records


# Purpose: Count class labels and attack types for quick dataset validation.
# Input: iterable of PairRecord objects.
# Output: dictionary containing total, attack counts, and label counts.
def summarize_records(records: Iterable[PairRecord]) -> Dict[str, object]:
    records = list(records)
    attack_counts = Counter(record.attack_type for record in records)
    label_counts = Counter(record.label for record in records)
    return {
        "total": len(records),
        "attacks": dict(sorted(attack_counts.items())),
        "labels": dict(sorted(label_counts.items())),
    }


class PoisonPairDataset:
    # Purpose: Initialize the clean/poison paired dataset.
    # Input: pair directory, image size, optional max sample count, and tensor value range.
    # Output: dataset object ready for torch DataLoader.
    def __init__(
        self,
        pair_dir: str | Path,
        image_size: int = 32,
        max_samples: Optional[int] = None,
        value_range: str = "minus_one_to_one",
    ) -> None:
        self.pair_dir = Path(pair_dir)
        self.image_size = image_size
        self.value_range = value_range
        records = build_pair_records(self.pair_dir)
        if max_samples is not None:
            records = records[:max_samples]
        self.records = records

    # Purpose: Return the number of paired examples.
    # Input: no arguments beyond the dataset instance.
    # Output: integer dataset length.
    def __len__(self) -> int:
        return len(self.records)

    # Purpose: Load one clean/poison pair and metadata for training.
    # Input: integer sample index.
    # Output: dictionary with clean tensor, poison tensor, label, attack id, clean mask, and filename.
    def __getitem__(self, index: int) -> Dict[str, object]:
        torch = _require_torch()
        record = self.records[index]
        clean = load_image_tensor(record.clean_path, self.image_size, self.value_range)
        poison = load_image_tensor(record.poison_path, self.image_size, self.value_range)
        return {
            "clean": clean,
            "poison": poison,
            "label": torch.tensor(record.label, dtype=torch.long),
            "attack_id": torch.tensor(record.attack_id, dtype=torch.long),
            "is_clean": torch.tensor(record.attack_type == "clean", dtype=torch.bool),
            "filename": record.filename,
        }

    # Purpose: Provide a dependency-light summary for smoke tests and review.
    # Input: no arguments beyond the dataset instance.
    # Output: dictionary with total, attack counts, and label counts.
    def summary(self) -> Dict[str, object]:
        return summarize_records(self.records)


# Purpose: Inspect a pair directory without importing torch.
# Input: pair directory path.
# Output: dictionary with dataset counts and path information.
def inspect_pair_directory(pair_dir: str | Path) -> Dict[str, object]:
    pair_dir = Path(pair_dir)
    records = build_pair_records(pair_dir)
    summary = summarize_records(records)
    summary["pair_dir"] = str(pair_dir)
    summary["clean_dir"] = str(pair_dir / "clean")
    summary["poison_dir"] = str(pair_dir / "poisons")
    return summary
