"""Witches' Brew benchmark adapter.

WB intentionally uses the original poisoning-gradient-matching Forest stack:
Kettle for benchmark metadata, ResNet18 Victim training, and target checks.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from .common import format_progress, load_cifar_normalized_tensor
from .materialize import MaterializedCase


# Purpose: Add the Forest package path for the WB implementation.
# Input: repository root.
# Output: none; sys.path is updated.
def add_forest_import_path(repo_dir: Path) -> None:
    forest_root = repo_dir / "dataset_generation" / "poisoning-gradient-matching"
    forest_root_str = str(forest_root)
    if forest_root_str not in sys.path:
        sys.path.insert(0, forest_root_str)


# Purpose: Build the exact Forest argument object used for WB validation.
# Input: repo paths, benchmark index, optional epoch override, and dryrun flag.
# Output: parsed Forest args object.
def build_wb_args(
    repo_dir: Path,
    cifar_root: Path,
    case_name: str,
    benchmark_index: int,
    epochs: int | None,
    dryrun: bool,
):
    add_forest_import_path(repo_dir)
    import forest

    argv = [
        "--name",
        f"benchmark_{case_name}",
        "--benchmark",
        str(repo_dir / "dataset_generation" / "configs" / "wb_benchmark_setups.pickle"),
        "--benchmark_idx",
        str(benchmark_index),
        "--vruns",
        "1",
        "--eps",
        "8",
        "--ensemble",
        "1",
        "--net",
        "ResNet18",
        "--data_path",
        str(cifar_root),
        "--table_path",
        str(repo_dir / "benchmark" / "tables"),
    ]
    if epochs is not None:
        argv.extend(["--epochs", str(epochs)])
    if dryrun:
        argv.append("--dryrun")
    return forest.options().parse_args(argv)


# Purpose: Extract the last metric from a Forest stats dictionary as percent.
# Input: stats dictionary and metric key.
# Output: float percentage or NaN when missing.
def last_percent(stats: Dict[str, Sequence[float]], key: str) -> float:
    values = stats.get(key, [])
    if not values:
        return float("nan")
    value = float(values[-1])
    return value * 100.0 if value <= 1.0 else value


# Purpose: Build a normalized delta tensor from a materialized train folder.
# Input: Kettle, train folder, CIFAR indices to replace, and logger.
# Output: delta tensor and lookup mapping from CIFAR index to delta row.
def build_delta_from_folder(
    kettle,
    train_dir: Path,
    indices: Iterable[int],
    logger: logging.Logger,
    description: str,
) -> tuple[torch.Tensor, Dict[int, int]]:
    indices = [int(index) for index in indices]
    deltas: List[torch.Tensor] = []
    lookup: Dict[int, int] = {}
    started_at = time.monotonic()
    for row, index in enumerate(indices):
        clean, label, returned_index = kettle.trainset[index]
        if int(returned_index) != index:
            raise ValueError(f"Kettle trainset returned index {returned_index}, expected {index}")
        altered_path = train_dir / str(label) / f"{index}.png"
        if not altered_path.is_file():
            raise FileNotFoundError(f"Missing materialized WB image: {altered_path}")
        altered = load_cifar_normalized_tensor(altered_path)
        deltas.append(altered - clean.cpu())
        lookup[index] = row
        if (row + 1) % 10000 == 0 or (row + 1) == len(indices):
            logger.info("%s delta build | %s", description, format_progress(row + 1, len(indices), started_at, torch.device("cpu")))
    if not deltas:
        raise ValueError("Cannot build an empty WB delta tensor")
    return torch.stack(deltas, dim=0), lookup


# Purpose: Run one WB validation pass on either poisoned or purified data.
# Input: materialized case, train folder, delta index policy, repo paths, settings, and logger.
# Output: dictionary with clean and target attack accuracy.
def run_wb_pass(
    materialized: MaterializedCase,
    train_dir: Path,
    use_full_train_delta: bool,
    repo_dir: Path,
    cifar_root: Path,
    epochs: int | None,
    dryrun: bool,
    logger: logging.Logger,
) -> Dict[str, object]:
    add_forest_import_path(repo_dir)
    import forest

    args = build_wb_args(
        repo_dir=repo_dir,
        cifar_root=cifar_root,
        case_name=materialized.case.name,
        benchmark_index=materialized.case.setup_index,
        epochs=epochs,
        dryrun=dryrun,
    )
    setup = forest.utils.system_startup(args)
    model = forest.Victim(args, setup=setup)
    kettle = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    if use_full_train_delta:
        indices = range(len(kettle.trainset))
        description = "WB purified full-train"
    else:
        indices = [int(index) for index in materialized.case.setup["base indices"]]
        description = "WB poisoned base-only"
    poison_delta, lookup = build_delta_from_folder(
        kettle=kettle,
        train_dir=train_dir,
        indices=indices,
        logger=logger,
        description=description,
    )
    kettle.poison_lookup = lookup
    stats = model.validate(kettle, poison_delta)
    return {
        "clean_acc": last_percent(stats, "valid_accs"),
        "target_acc": last_percent(stats, "target_accs"),
        "target_clean_acc": last_percent(stats, "target_accs_clean"),
        "raw_stats": {key: list(value) for key, value in stats.items()},
    }


# Purpose: Run WB poisoned and purified benchmark passes for one case.
# Input: materialized case, repo paths, WB settings, and logger.
# Output: combined benchmark metric dictionary.
def evaluate_wb_case(
    materialized: MaterializedCase,
    repo_dir: Path,
    cifar_root: Path,
    epochs: int | None,
    dryrun: bool,
    logger: logging.Logger,
) -> Dict[str, object]:
    poison_result = run_wb_pass(
        materialized=materialized,
        train_dir=materialized.poisoned_train_dir,
        use_full_train_delta=False,
        repo_dir=repo_dir,
        cifar_root=cifar_root,
        epochs=epochs,
        dryrun=dryrun,
        logger=logger,
    )
    purified_result = run_wb_pass(
        materialized=materialized,
        train_dir=materialized.purified_train_dir,
        use_full_train_delta=True,
        repo_dir=repo_dir,
        cifar_root=cifar_root,
        epochs=epochs,
        dryrun=dryrun,
        logger=logger,
    )
    return {
        "poison": poison_result,
        "purified": purified_result,
    }
