"""Build CM-training pairs and held-out Narcissus evaluation cases."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torchvision
from PIL import Image

from .integration import (
    TriggerGenerationConfig,
    apply_trigger_array,
    generate_trigger,
    load_trigger_artifact,
    trigger_sha256,
    validate_trigger,
)


CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
DEFAULT_SPLIT_SEED = 121
CM_PAIR_SELECTION_SEED = 11000
TRAIN_TRIGGER_BASE_SEED = 12000
EVAL_TRIGGER_BASE_SEED = 22000
EVAL_POISON_BASE_SEED = 32000
SMOKE_SELECTION_BASE_SEED = 42000
CM_INFERENCE_SEED = 52000
VICTIM_TRAINING_SEED = 62000


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
    os.replace(temporary, path)


def shuffled_indices_by_class(targets: Sequence[int], seed: int = DEFAULT_SPLIT_SEED) -> Dict[int, np.ndarray]:
    rng = np.random.RandomState(seed)
    target_array = np.asarray(targets)
    result: Dict[int, np.ndarray] = {}
    for class_idx in range(10):
        indices = np.where(target_array == class_idx)[0]
        if len(indices) != 5000:
            raise ValueError(f"Expected 5,000 CIFAR-10 train images for class {class_idx}, found {len(indices)}")
        rng.shuffle(indices)
        result[class_idx] = indices
    return result


def build_ns_setup_records(shuffled_indices: Mapping[int, Sequence[int]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for target_class in range(10):
        cm_sources: Dict[str, List[int]] = {}
        for source_class in range(10):
            source_indices = list(map(int, shuffled_indices[source_class]))
            start = 3520 + target_class * 100
            cm_sources[str(source_class)] = source_indices[start : start + 100]
        target_indices = list(map(int, shuffled_indices[target_class]))
        eval_rng = np.random.RandomState(EVAL_POISON_BASE_SEED + target_class)
        eval_poison_indices = [
            int(value) for value in eval_rng.permutation(target_indices[1000:1500]).tolist()
        ]
        records.append(
            {
                "case_id": f"NS_c{target_class}",
                "attack": "narcissus",
                "target_class": target_class,
                "target_class_name": CIFAR10_CLASS_NAMES[target_class],
                "cm_pair_source_indices": cm_sources,
                "eval_poison_indices": eval_poison_indices,
                "train_trigger_synthesis_indices": target_indices[:2500],
                "eval_trigger_synthesis_indices": target_indices[2500:],
                "shared_with_wb_eval": True,
                "seeds": {
                    "cm_pair_selection": CM_PAIR_SELECTION_SEED,
                    "train_trigger": TRAIN_TRIGGER_BASE_SEED + target_class,
                    "eval_trigger": EVAL_TRIGGER_BASE_SEED + target_class,
                    "eval_poison_selection": EVAL_POISON_BASE_SEED + target_class,
                    "cm_inference": CM_INFERENCE_SEED,
                    "victim_training": VICTIM_TRAINING_SEED,
                },
            }
        )
    return records


def create_ns_setup_file(train_targets: Sequence[int], output_path: str | Path) -> List[Dict[str, Any]]:
    shuffled = shuffled_indices_by_class(train_targets)
    records = build_ns_setup_records(shuffled)
    _atomic_json(
        Path(output_path),
        {
            "format": "cm_narcissus_setups_v1",
            "split_seed": DEFAULT_SPLIT_SEED,
            "cm_train_position_range": [3520, 4520],
            "eval_position_range": [1000, 1500],
            "eval_indices_shared_with": "WB evaluation bases",
            "records": records,
        },
    )
    return records


def load_ns_setup_file(path: str | Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "cm_narcissus_setups_v1":
        raise ValueError(f"Unsupported Narcissus setup format in {path}")
    return list(payload["records"])


def ns_pair_name(source_class: int, target_class: int, clean_index: int) -> str:
    return f"ns_c{source_class}_t{target_class}_{clean_index}.png"


def ns_eval_name(target_class: int, clean_index: int) -> str:
    return f"ns_c{target_class}_{clean_index}.png"


def trigger_artifact_path(
    data_root: str | Path,
    kind: str,
    target_class: int,
    profile: str = "final",
) -> Path:
    profile_parts = [] if profile == "final" else [profile]
    return Path(data_root) / "narcissus" / Path(*profile_parts) / "triggers" / kind / f"class_{target_class:02d}.pt"


def generate_ns_triggers(
    setup_path: str | Path,
    data_root: str | Path,
    narcissus_root: str | Path,
    cifar_root: str | Path,
    pood_root: str | Path,
    classes: Sequence[int],
    kinds: Sequence[str],
    device: str = "cuda",
    surrogate_epochs: int = 200,
    warmup_epochs: int = 5,
    trigger_rounds: int = 1000,
    batch_size: int = 350,
    num_workers: int = 8,
    checkpoint_interval: int = 10,
    epsilon: float = 8.0 / 255.0,
    profile: str = "final",
) -> None:
    setups = {int(record["target_class"]): record for record in load_ns_setup_file(setup_path)}
    for kind in kinds:
        if kind not in {"train", "eval"}:
            raise ValueError(f"Unsupported trigger kind: {kind}")
        for target_class in classes:
            setup = setups[target_class]
            seed = int(setup["seeds"][f"{kind}_trigger"])
            synthesis_indices = tuple(int(value) for value in setup[f"{kind}_trigger_synthesis_indices"])
            output_path = trigger_artifact_path(data_root, kind, target_class, profile=profile)
            checkpoint_parts = [] if profile == "final" else [profile]
            checkpoint_dir = Path(narcissus_root) / "checkpoints" / Path(*checkpoint_parts) / kind / f"class_{target_class:02d}"
            config = TriggerGenerationConfig(
                target_class=target_class,
                kind=kind,
                seed=seed,
                cifar_root=str(cifar_root),
                pood_root=str(pood_root),
                output_path=str(output_path),
                checkpoint_dir=str(checkpoint_dir),
                synthesis_indices=synthesis_indices,
                epsilon=epsilon,
                surrogate_epochs=surrogate_epochs,
                warmup_epochs=warmup_epochs,
                trigger_rounds=trigger_rounds,
                batch_size=batch_size,
                num_workers=num_workers,
                checkpoint_interval=checkpoint_interval,
                device=device,
            )
            trigger, metadata = generate_trigger(config)
            print(
                f"Narcissus {kind} trigger c{target_class} ready: "
                f"sha256={metadata['sha256']} linf={float(trigger.abs().max()):.8f}"
            )
            other_kind = "eval" if kind == "train" else "train"
            other_path = trigger_artifact_path(data_root, other_kind, target_class, profile=profile)
            if other_path.is_file():
                other_trigger, _ = load_trigger_artifact(other_path)
                if trigger_sha256(trigger) == trigger_sha256(other_trigger):
                    raise AssertionError(
                        f"Narcissus train/eval trigger hashes are identical for class {target_class}"
                    )


def _load_cifar(data_root: str | Path):
    train_set = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=False)
    test_set = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=False)
    return train_set, test_set


def export_ns_cm_bank(
    setup_path: str | Path,
    data_root: str | Path,
    train_clean_dir: str | Path,
    train_poison_dir: str | Path,
    classes: Sequence[int] = tuple(range(10)),
    epsilon: float = 8.0 / 255.0,
    profile: str = "final",
) -> Path:
    setups = {int(record["target_class"]): record for record in load_ns_setup_file(setup_path)}
    train_set, _ = _load_cifar(data_root)
    clean_dir, poison_dir = Path(train_clean_dir), Path(train_poison_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    poison_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: List[Dict[str, Any]] = []
    for target_class in classes:
        setup = setups[target_class]
        trigger_path = trigger_artifact_path(data_root, "train", target_class, profile=profile)
        trigger, trigger_metadata = load_trigger_artifact(trigger_path)
        validate_trigger(trigger, epsilon)
        for source_class in range(10):
            source_indices = [int(value) for value in setup["cm_pair_source_indices"][str(source_class)]]
            if len(source_indices) != 100:
                raise ValueError(f"Expected 100 NS CM sources for target={target_class}, source={source_class}")
            for clean_index in source_indices:
                image, label = train_set[clean_index]
                if int(label) != source_class:
                    raise ValueError(f"NS source label mismatch at CIFAR index {clean_index}")
                filename = ns_pair_name(source_class, target_class, clean_index)
                clean_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
                poison_array = apply_trigger_array(clean_array, trigger, scale=1.0)
                Image.fromarray(clean_array, mode="RGB").save(clean_dir / filename)
                Image.fromarray(poison_array, mode="RGB").save(poison_dir / filename)
                manifest_records.append(
                    {
                        "pair_id": Path(filename).stem,
                        "attack": "narcissus",
                        "purpose": "cm_train",
                        "clean_split": "train",
                        "clean_index": clean_index,
                        "source_class": source_class,
                        "source_class_name": CIFAR10_CLASS_NAMES[source_class],
                        "trigger_target_class": target_class,
                        "trigger_target_name": CIFAR10_CLASS_NAMES[target_class],
                        "trigger_id": trigger_metadata["trigger_id"],
                        "trigger_seed": int(trigger_metadata["seed"]),
                        "selection_seed": int(setup["seeds"]["cm_pair_selection"]),
                        "trigger_sha256": trigger_sha256(trigger),
                        "epsilon": epsilon,
                        "trigger_scale": 1.0,
                        "clean_path": str(clean_dir / filename),
                        "poison_path": str(poison_dir / filename),
                        "label": source_class,
                    }
                )
    expected = 1000 * len(classes)
    if len(manifest_records) != expected:
        raise AssertionError(f"Expected {expected} exported NS CM pairs, found {len(manifest_records)}")
    manifest_path = clean_dir.parent / "narcissus_manifest.jsonl"
    existing_records: List[Dict[str, Any]] = []
    if manifest_path.is_file():
        existing_records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    selected_classes = {int(value) for value in classes}
    existing_records = [
        record
        for record in existing_records
        if int(record.get("trigger_target_class", -1)) not in selected_classes
    ]
    merged_records = existing_records + manifest_records
    merged_records.sort(key=lambda record: (int(record["trigger_target_class"]), str(record["pair_id"])))
    _atomic_jsonl(manifest_path, merged_records)
    return manifest_path


def export_ns_eval_cases(
    setup_path: str | Path,
    data_root: str | Path,
    test_root: str | Path,
    classes: Sequence[int] = tuple(range(10)),
    profile: str = "final",
    epsilon: float = 8.0 / 255.0,
) -> None:
    setups = {int(record["target_class"]): record for record in load_ns_setup_file(setup_path)}
    train_set, test_set = _load_cifar(data_root)
    poison_count = 500 if profile == "final" else 50
    triggered_count = None if profile == "final" else 500
    for target_class in classes:
        setup = setups[target_class]
        case_dir = Path(test_root) / f"NS_c{target_class}"
        clean_dir, poison_dir = case_dir / "clean", case_dir / "poisons"
        clean_dir.mkdir(parents=True, exist_ok=True)
        poison_dir.mkdir(parents=True, exist_ok=True)
        trigger_path = trigger_artifact_path(data_root, "eval", target_class, profile=profile)
        trigger, trigger_metadata = load_trigger_artifact(trigger_path)
        validate_trigger(trigger, epsilon)
        selected = [int(value) for value in setup["eval_poison_indices"][:poison_count]]
        train_records: List[Dict[str, Any]] = []
        for clean_index in selected:
            image, label = train_set[clean_index]
            if int(label) != target_class:
                raise ValueError(f"NS evaluation base {clean_index} is not class {target_class}")
            filename = ns_eval_name(target_class, clean_index)
            clean_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            poison_array = apply_trigger_array(clean_array, trigger, scale=1.0)
            Image.fromarray(clean_array, mode="RGB").save(clean_dir / filename)
            Image.fromarray(poison_array, mode="RGB").save(poison_dir / filename)
            train_records.append(
                {
                    "train_index": clean_index,
                    "label": target_class,
                    "target_class": target_class,
                    "clean_path": str(clean_dir / filename),
                    "poison_path": str(poison_dir / filename),
                    "trigger_sha256": trigger_sha256(trigger),
                    "poison_selection_seed": int(setup["seeds"]["eval_poison_selection"]),
                    "epsilon": epsilon,
                    "trigger_scale": 1.0,
                }
            )
        candidate_queries = [index for index, label in enumerate(test_set.targets) if int(label) != target_class]
        if triggered_count is not None:
            rng = np.random.RandomState(SMOKE_SELECTION_BASE_SEED + target_class)
            candidate_queries = sorted(rng.choice(candidate_queries, size=triggered_count, replace=False).tolist())
        query_records = [
            {
                "test_index": int(index),
                "original_label": int(test_set.targets[index]),
                "attack_target": target_class,
                "trigger_scale": 1.0,
                "trigger_sha256": trigger_sha256(trigger),
            }
            for index in candidate_queries
        ]
        if profile == "final" and len(query_records) != 9000:
            raise AssertionError(f"Expected 9,000 triggered NS queries for c{target_class}")
        shutil.copy2(trigger_path, case_dir / "trigger.pt")
        trigger_json = trigger_path.with_suffix(".json")
        if trigger_json.is_file():
            shutil.copy2(trigger_json, case_dir / "trigger.json")
        _atomic_json(case_dir / "poison_indices.json", selected)
        _atomic_jsonl(case_dir / "train_manifest.jsonl", train_records)
        _atomic_jsonl(case_dir / "triggered_test_manifest.jsonl", query_records)
        _atomic_json(
            case_dir / "metadata.json",
            {
                "format": "cm_narcissus_eval_case_v1",
                "case_id": f"NS_c{target_class}",
                "attack": "narcissus",
                "profile": profile,
                "dataset": "cifar10",
                "target_class": target_class,
                "target_class_name": CIFAR10_CLASS_NAMES[target_class],
                "epsilon": epsilon,
                "trigger_scale_train": 1.0,
                "trigger_scale_test": 1.0,
                "num_train_total": 50000,
                "num_poison_train": len(selected),
                "poison_rate_total": len(selected) / 50000.0,
                "poison_rate_target_class": len(selected) / 5000.0,
                "num_clean_test": 10000,
                "num_triggered_test": len(query_records),
                "trigger_id": trigger_metadata["trigger_id"],
                "trigger_seed": int(trigger_metadata["seed"]),
                "trigger_sha256": trigger_sha256(trigger),
                "poison_selection_seed": int(setup["seeds"]["eval_poison_selection"]),
                "cm_inference_seed": CM_INFERENCE_SEED,
                "victim_training_seed": VICTIM_TRAINING_SEED,
                "shared_base_indices_with_wb_eval": True,
                "poison_indices": selected,
                "semantic_label": target_class,
                "paths": {
                    "case_dir": str(case_dir),
                    "clean_dir": str(clean_dir),
                    "poison_dir": str(poison_dir),
                    "trigger": str(case_dir / "trigger.pt"),
                    "poison_indices": str(case_dir / "poison_indices.json"),
                    "train_manifest": str(case_dir / "train_manifest.jsonl"),
                    "triggered_test_manifest": str(case_dir / "triggered_test_manifest.jsonl"),
                },
                "seeds": dict(setup["seeds"]),
                "allocation": {
                    "cm_train_position_range": [3520, 4520],
                    "evaluation_position_range": [1000, 1500],
                    "evaluation_bases_shared_with": "WB evaluation cases",
                },
            },
        )


def validate_ns_artifacts(
    setup_path: str | Path,
    data_root: str | Path,
    train_clean_dir: str | Path,
    train_poison_dir: str | Path,
    test_root: str | Path,
    classes: Sequence[int] = tuple(range(10)),
    profile: str = "final",
    epsilon: float = 8.0 / 255.0,
) -> Dict[str, Any]:
    setups = {int(record["target_class"]): record for record in load_ns_setup_file(setup_path)}
    expected_pairs = 1000 * len(classes)
    clean_names = {path.name for path in Path(train_clean_dir).glob("ns_*.png")}
    poison_names = {path.name for path in Path(train_poison_dir).glob("ns_*.png")}
    if clean_names != poison_names:
        raise AssertionError("Narcissus CM clean/poison filenames do not match")
    selected_names = {name for name in clean_names if int(name.split("_t", 1)[1].split("_", 1)[0]) in classes}
    if len(selected_names) != expected_pairs:
        raise AssertionError(f"Expected {expected_pairs} NS pairs for selected classes, found {len(selected_names)}")
    expected_names = {
        ns_pair_name(source_class, target_class, int(clean_index))
        for target_class in classes
        for source_class in range(10)
        for clean_index in setups[target_class]["cm_pair_source_indices"][str(source_class)]
    }
    if selected_names != expected_names:
        raise AssertionError("Narcissus CM filenames do not match the locked allocation")
    cases: Dict[str, Any] = {}
    for target_class in classes:
        train_trigger, _ = load_trigger_artifact(trigger_artifact_path(data_root, "train", target_class, profile=profile))
        eval_trigger, _ = load_trigger_artifact(trigger_artifact_path(data_root, "eval", target_class, profile=profile))
        validate_trigger(train_trigger, epsilon)
        validate_trigger(eval_trigger, epsilon)
        if trigger_sha256(train_trigger) == trigger_sha256(eval_trigger):
            raise AssertionError(f"Training and evaluation triggers are identical for class {target_class}")
        setup = setups[target_class]
        cm_indices = {
            int(index)
            for values in setup["cm_pair_source_indices"].values()
            for index in values
        }
        eval_indices = {int(index) for index in setup["eval_poison_indices"]}
        if cm_indices & eval_indices:
            raise AssertionError(f"NS CM/evaluation index leakage for class {target_class}")
        case_dir = Path(test_root) / f"NS_c{target_class}"
        metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
        expected_poisons = 500 if profile == "final" else 50
        expected_queries = 9000 if profile == "final" else 500
        if metadata["num_poison_train"] != expected_poisons or metadata["num_triggered_test"] != expected_queries:
            raise AssertionError(f"Unexpected NS case counts for c{target_class}: {metadata}")
        if metadata["trigger_sha256"] != trigger_sha256(eval_trigger):
            raise AssertionError(f"NS case trigger hash mismatch for c{target_class}")
        poison_indices = json.loads((case_dir / "poison_indices.json").read_text(encoding="utf-8"))
        if len(poison_indices) != expected_poisons or len(set(poison_indices)) != expected_poisons:
            raise AssertionError(f"NS poison indices are not unique/complete for c{target_class}")
        expected_eval_names = {ns_eval_name(target_class, int(index)) for index in poison_indices}
        clean_eval_names = {path.name for path in (case_dir / "clean").glob("*.png")}
        poison_eval_names = {path.name for path in (case_dir / "poisons").glob("*.png")}
        if clean_eval_names != expected_eval_names or poison_eval_names != expected_eval_names:
            raise AssertionError(f"NS clean/poison evaluation files do not match indices for c{target_class}")
        train_records = [
            json.loads(line)
            for line in (case_dir / "train_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(train_records) != expected_poisons or any(
            int(record["label"]) != target_class or int(record["target_class"]) != target_class
            for record in train_records
        ):
            raise AssertionError(f"NS train manifest labels/count are invalid for c{target_class}")
        query_records = [
            json.loads(line)
            for line in (case_dir / "triggered_test_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        query_indices = [int(record["test_index"]) for record in query_records]
        if len(query_records) != expected_queries or len(set(query_indices)) != expected_queries:
            raise AssertionError(f"NS triggered query manifest is not unique/complete for c{target_class}")
        if any(
            int(record["original_label"]) == target_class
            or int(record["attack_target"]) != target_class
            or record["trigger_sha256"] != metadata["trigger_sha256"]
            for record in query_records
        ):
            raise AssertionError(f"NS triggered query metadata is invalid for c{target_class}")
        cases[f"NS_c{target_class}"] = metadata
    return {"pairs": expected_pairs, "cases": cases, "profile": profile}
