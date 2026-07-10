"""BullseyePoison benchmark adapter.

BP intentionally does not share the WB retraining path. It uses the original
BullseyePoison transfer-learning evaluator style: pretrained victim network,
linear-head retraining by default, compact CIFAR split, and target attack check.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from .common import CIFAR_MEAN, CIFAR_STD, load_cifar_normalized_tensor
from .materialize import MaterializedCase


@contextlib.contextmanager
def pushd(path: Path):
    # Purpose: Temporarily change working directory for legacy BP relative paths.
    # Input: target directory.
    # Output: context manager restoring the previous working directory.
    import os

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class FolderBackedBPDataset(torch.utils.data.Dataset):
    # Purpose: Mimic BullseyePoison PoisonedDataset while reading from materialized folders.
    # Input: folder records, transform, poison tuples, base flat indices, sampling settings.
    # Output: dataset yielding normalized image tensors and integer labels.
    def __init__(
        self,
        records: Sequence[Tuple[int, Path, int]],
        transform,
        poison_tuple_list: Sequence[Tuple[torch.Tensor, int]],
        poison_indices: Sequence[int],
        num_per_label: int = 50,
        class_labels: Sequence[int] = tuple(range(10)),
        subset_group: int = 0,
    ) -> None:
        self.records = list(records)
        self.transform = transform
        self.poison_tuple_list = list(poison_tuple_list)
        self.poison_indices = set(int(index) for index in poison_indices)
        self.class_labels = set(int(label) for label in class_labels)
        self.valid_positions = self._build_valid_positions(num_per_label, subset_group)

    # Purpose: Select clean examples exactly like BP's PoisonedDataset.
    # Input: per-class count and subset group.
    # Output: list of positions into self.records.
    def _build_valid_positions(self, num_per_label: int, subset_group: int) -> List[int]:
        num_per_label_dict: Dict[int, int] = {}
        idx_cursors = {label: 0 for label in self.class_labels}
        for poison_index in self.poison_indices:
            for flat_index, _, label in self.records:
                if flat_index == poison_index and label in self.class_labels:
                    num_per_label_dict[label] = num_per_label_dict.get(label, 0) + 1
                    break

        if num_per_label <= 0:
            return [position for position, _ in enumerate(self.records)]

        valid_positions: List[int] = []
        start_idx = subset_group * num_per_label
        end_idx = (subset_group + 1) * num_per_label
        for position, (flat_index, _, label) in enumerate(self.records):
            if label not in self.class_labels:
                continue
            idx_cursors[label] += 1
            if flat_index in self.poison_indices:
                continue
            num_per_label_dict.setdefault(label, 0)
            if num_per_label_dict[label] < num_per_label and start_idx < idx_cursors[label] <= end_idx:
                valid_positions.append(position)
                num_per_label_dict[label] += 1
        return valid_positions

    # Purpose: Return dataset length including explicit poison tuples.
    # Input: no arguments beyond self.
    # Output: integer item count.
    def __len__(self) -> int:
        return len(self.poison_tuple_list) + len(self.valid_positions)

    # Purpose: Load one poison or clean/purified sample.
    # Input: integer item index.
    # Output: image tensor and integer label.
    def __getitem__(self, index: int):
        if index < len(self.poison_tuple_list):
            return self.poison_tuple_list[index]
        position = self.valid_positions[index - len(self.poison_tuple_list)]
        _, path, label = self.records[position]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


# Purpose: Add the project root and BP root to sys.path for legacy imports.
# Input: repository root.
# Output: none; sys.path is updated.
def add_bp_import_paths(repo_dir: Path) -> None:
    for path in [repo_dir, repo_dir / "dataset_generation" / "BullseyePoison"]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


# Purpose: Build folder records from a materialized BP train directory.
# Input: train directory with <label>/<flat_idx>.png files.
# Output: list of flat-index/path/label records.
def build_folder_records(train_dir: Path) -> List[Tuple[int, Path, int]]:
    records: List[Tuple[int, Path, int]] = []
    for label_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
        label = int(label_dir.name)
        for path in sorted(label_dir.glob("*.png")):
            records.append((int(path.stem), path, label))
    return sorted(records, key=lambda item: item[0])


# Purpose: Build BP poison tuples from materialized train paths.
# Input: materialized case and train directory.
# Output: list of normalized poison tensors with base-class labels.
def build_poison_tuples(materialized: MaterializedCase, train_dir: Path) -> List[Tuple[torch.Tensor, int]]:
    tuples: List[Tuple[torch.Tensor, int]] = []
    base_class = int(materialized.case.setup["base class"])
    for poison_name in sorted(materialized.poison_relpaths):
        relpath = materialized.poison_relpaths[poison_name]
        tensor = load_cifar_normalized_tensor(train_dir / relpath)
        tuples.append((tensor, base_class))
    return tuples


# Purpose: Load the one target image saved with the case as a normalized batch tensor.
# Input: materialized case.
# Output: target tensor with shape [1, C, H, W].
def load_target_tensor(materialized: MaterializedCase) -> torch.Tensor:
    target_images = sorted(materialized.target_dir.glob("*.png"))
    if not target_images:
        raise FileNotFoundError(f"No target PNG found in {materialized.target_dir}")
    return load_cifar_normalized_tensor(target_images[0]).unsqueeze(0)


# Purpose: Build a Namespace matching BP's transfer evaluator defaults.
# Input: benchmark paths and optional hyperparameter overrides.
# Output: argparse.Namespace consumed by BP training helpers.
def build_bp_args(
    train_data_path: Path,
    dset_path: Path,
    model_resume_path: str,
    device: str,
    retrain_epochs: int,
    retrain_bsize: int,
    target_label: int,
    poison_label: int,
    subset_group: int = 0,
) -> Namespace:
    return Namespace(
        end2end=False,
        retrain_opt="adam",
        retrain_lr=0.1,
        retrain_wd=0.0,
        retrain_momentum=0.9,
        lr_decay_epoch=[30, 45],
        retrain_epochs=retrain_epochs,
        retrain_bsize=retrain_bsize,
        num_per_class=50,
        subset_group=subset_group,
        train_data_path=str(train_data_path),
        dset_path=str(dset_path),
        model_resume_path=model_resume_path,
        target_label=target_label,
        poison_label=poison_label,
        device=device,
    )


# Purpose: Run one BP retraining/evaluation pass on poisoned or purified data.
# Input: materialized case, train folder, repo paths, BP settings, and logger.
# Output: dictionary with clean accuracy, target attack accuracy, and prediction details.
def run_bp_pass(
    materialized: MaterializedCase,
    train_dir: Path,
    repo_dir: Path,
    bp_root: Path,
    cifar_root: Path,
    bp_split_path: Path,
    device: str,
    victim_net: str,
    checkpoint_name: str,
    retrain_epochs: int,
    retrain_bsize: int,
    logger: logging.Logger,
) -> Dict[str, object]:
    if device != "cuda":
        raise RuntimeError("BP transfer evaluation requires CUDA because the original trainer hard-codes cuda.")
    add_bp_import_paths(repo_dir)
    from dataset_generation.BullseyePoison.eval_poisons_transfer import train_network_with_poison
    from dataset_generation.BullseyePoison.utils import load_pretrained_net

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    records = build_folder_records(train_dir)
    poison_tuples = build_poison_tuples(materialized, train_dir)
    base_indices = materialized.bp_flat_base_indices
    if base_indices is None:
        raise ValueError(f"BP flat base indices are missing for {materialized.case.name}")
    poisoned_dataset = FolderBackedBPDataset(
        records=records,
        transform=transform,
        poison_tuple_list=poison_tuples,
        poison_indices=base_indices,
        num_per_label=50,
        subset_group=0,
    )
    testset = torchvision.datasets.CIFAR10(root=str(cifar_root), train=False, download=False, transform=transform)
    target_tensor = load_target_tensor(materialized)
    args = build_bp_args(
        train_data_path=bp_split_path,
        dset_path=cifar_root,
        model_resume_path="model-chks",
        device=device,
        retrain_epochs=retrain_epochs,
        retrain_bsize=retrain_bsize,
        target_label=int(materialized.case.setup["target class"]),
        poison_label=int(materialized.case.setup["base class"]),
    )

    logger.info("Loading BP victim %s from %s", victim_net, checkpoint_name)
    with pushd(bp_root):
        net = load_pretrained_net(
            victim_net,
            checkpoint_name,
            model_chk_path=args.model_resume_path,
            device=args.device,
        )
        result = train_network_with_poison(
            net,
            target_tensor,
            [],
            poison_tuples,
            poisoned_dataset,
            base_indices,
            args,
            testset,
        )
    target_acc = 100.0 if int(result["prediction"]) == int(materialized.case.setup["base class"]) else 0.0
    return {
        "clean_acc": float(result["clean acc"]),
        "target_acc": target_acc,
        "prediction": int(result["prediction"]),
        "malicious_score": float(result["malicious score"]),
        "poison_predictions": result["poisons predictions"],
    }


# Purpose: Run BP poisoned and purified benchmark passes for one case.
# Input: materialized case, repo paths, device/settings, and logger.
# Output: combined benchmark metric dictionary.
def evaluate_bp_case(
    materialized: MaterializedCase,
    repo_dir: Path,
    cifar_root: Path,
    bp_split_path: Path,
    device: str,
    victim_net: str,
    checkpoint_name: str,
    retrain_epochs: int,
    retrain_bsize: int,
    logger: logging.Logger,
) -> Dict[str, object]:
    bp_root = repo_dir / "dataset_generation" / "BullseyePoison"
    poison_result = run_bp_pass(
        materialized=materialized,
        train_dir=materialized.poisoned_train_dir,
        repo_dir=repo_dir,
        bp_root=bp_root,
        cifar_root=cifar_root,
        bp_split_path=bp_split_path,
        device=device,
        victim_net=victim_net,
        checkpoint_name=checkpoint_name,
        retrain_epochs=retrain_epochs,
        retrain_bsize=retrain_bsize,
        logger=logger,
    )
    purified_result = run_bp_pass(
        materialized=materialized,
        train_dir=materialized.purified_train_dir,
        repo_dir=repo_dir,
        bp_root=bp_root,
        cifar_root=cifar_root,
        bp_split_path=bp_split_path,
        device=device,
        victim_net=victim_net,
        checkpoint_name=checkpoint_name,
        retrain_epochs=retrain_epochs,
        retrain_bsize=retrain_bsize,
        logger=logger,
    )
    return {
        "poison": poison_result,
        "purified": purified_result,
    }
