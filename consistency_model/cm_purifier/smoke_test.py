"""Smoke checks for the pixel-space CM purifier package."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .dataset import inspect_pair_directory


# Purpose: Build the command-line parser for local smoke checks.
# Input: no arguments.
# Output: argparse parser with smoke-test options.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lightweight checks for cm_purifier.")
    parser.add_argument("--pair-dir", type=str, default="dataset_generation/datasets/train")
    parser.add_argument("--expected-total", type=int, default=30000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--tmp-checkpoint", type=str, default="/tmp/cm_purifier_smoke.pth")
    parser.add_argument("--tmp-image-dir", type=str, default="/tmp/cm_purifier_smoke_images")
    parser.add_argument("--tmp-output-dir", type=str, default="/tmp/cm_purifier_smoke_outputs")
    return parser


# Purpose: Check whether torch is importable in the current Python environment.
# Input: no arguments.
# Output: True when torch import succeeds, otherwise False.
def torch_is_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


# Purpose: Validate pair directory shape and class/attack counts.
# Input: pair directory path and expected total pair count.
# Output: summary dictionary.
def run_dataset_checks(pair_dir: str, expected_total: int):
    summary = inspect_pair_directory(pair_dir)
    if summary["total"] != expected_total:
        raise AssertionError(f"Expected {expected_total} pairs, found {summary['total']}")
    expected_attacks = {"bp": 10000, "clean": 10000, "wb": 10000}
    if expected_total == 30000 and summary["attacks"] != expected_attacks:
        raise AssertionError(f"Unexpected attack counts: {summary['attacks']}")
    if expected_total == 30000:
        expected_labels = {label: 3000 for label in range(10)}
        if summary["labels"] != expected_labels:
            raise AssertionError(f"Unexpected label counts: {summary['labels']}")
    return summary


# Purpose: Run a tiny two-step training job when torch is installed.
# Input: pair directory, number of batches/steps, and temporary checkpoint path.
# Output: checkpoint path string.
def run_tiny_training_smoke(pair_dir: str, max_batches: int, tmp_checkpoint: str) -> str:
    from .train import main as train_main

    train_main(
        [
            "--pair-dir",
            pair_dir,
            "--out",
            tmp_checkpoint,
            "--backbone",
            "tiny",
            "--schedule-source",
            "linear",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--max-samples",
            "16",
            "--max-steps",
            str(max_batches),
            "--save-steps",
            str(max_batches),
            "--log-steps",
            "1",
            "--device",
            "cpu",
        ]
    )
    if not Path(tmp_checkpoint).is_file():
        raise AssertionError(f"Smoke checkpoint was not created: {tmp_checkpoint}")
    return tmp_checkpoint


# Purpose: Prepare two real poison images for checkpoint load/inference smoke testing.
# Input: pair directory and temporary image directory.
# Output: list of copied image paths.
def prepare_two_image_smoke_inputs(pair_dir: str, tmp_image_dir: str):
    source_dir = Path(pair_dir) / "poisons"
    image_dir = Path(tmp_image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    for old_path in image_dir.glob("*"):
        if old_path.is_file():
            old_path.unlink()
    source_paths = sorted(source_dir.glob("*.png"))[:2]
    if len(source_paths) != 2:
        raise AssertionError(f"Expected at least two PNG images in {source_dir}")
    copied_paths = []
    for source_path in source_paths:
        target_path = image_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied_paths.append(target_path)
    return copied_paths


# Purpose: Load a saved checkpoint and purify exactly two smoke-test images.
# Input: checkpoint path, pair directory, temporary input directory, and temporary output directory.
# Output: dictionary describing the loaded checkpoint and generated outputs.
def run_two_image_checkpoint_smoke(
    checkpoint_path: str,
    pair_dir: str,
    tmp_image_dir: str,
    tmp_output_dir: str,
):
    import torch
    from PIL import Image

    from .checkpoint import load_purifier_from_checkpoint
    from .dataset import load_image_tensor
    from .infer import purify_batch, resolve_t_star, save_image_tensor

    input_paths = prepare_two_image_smoke_inputs(pair_dir, tmp_image_dir)
    output_dir = Path(tmp_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.glob("*"):
        if old_path.is_file():
            old_path.unlink()

    device = torch.device("cpu")
    model, alpha_schedule, sigma_schedule, train_args = load_purifier_from_checkpoint(checkpoint_path, device)
    image_size = int(train_args.get("image_size", 32))
    t_star = resolve_t_star(20, len(alpha_schedule))
    batch = torch.stack([load_image_tensor(path, image_size=image_size) for path in input_paths], dim=0)
    purified = purify_batch(model, batch, t_star, alpha_schedule, sigma_schedule, device)

    output_paths = []
    for tensor, input_path in zip(purified, input_paths):
        output_path = output_dir / input_path.name
        save_image_tensor(tensor, output_path)
        output_paths.append(output_path)

    for output_path in output_paths:
        with Image.open(output_path) as image:
            if image.mode != "RGB" or image.size != (image_size, image_size):
                raise AssertionError(f"Unexpected output image format for {output_path}: {image.mode} {image.size}")

    return {
        "checkpoint": str(checkpoint_path),
        "input_count": len(input_paths),
        "output_count": len(output_paths),
        "outputs": [str(path) for path in output_paths],
    }


# Purpose: Run all smoke checks available in the current environment.
# Input: optional command-line arguments.
# Output: zero on success.
def main(args=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(args)
    summary = run_dataset_checks(args.pair_dir, args.expected_total)
    print(json.dumps({"dataset_check": summary}, indent=2))
    if args.skip_training:
        print("training smoke skipped by --skip-training")
        return 0
    if not torch_is_available():
        print("training smoke skipped because torch is not installed in this Python environment")
        return 0
    if args.checkpoint is not None:
        checkpoint_smoke = run_two_image_checkpoint_smoke(
            args.checkpoint,
            args.pair_dir,
            args.tmp_image_dir,
            args.tmp_output_dir,
        )
        print(json.dumps({"checkpoint_two_image_smoke": checkpoint_smoke}, indent=2))
        return 0
    checkpoint = run_tiny_training_smoke(args.pair_dir, args.max_batches, args.tmp_checkpoint)
    checkpoint_smoke = run_two_image_checkpoint_smoke(
        checkpoint,
        args.pair_dir,
        args.tmp_image_dir,
        args.tmp_output_dir,
    )
    print(json.dumps({"tiny_training_checkpoint": checkpoint, "two_image_checkpoint_smoke": checkpoint_smoke}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
