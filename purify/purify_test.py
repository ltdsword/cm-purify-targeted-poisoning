"""Algorithm 3 inference-time sanitization for held-out poison test cases."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

from consistency_model.cm_purifier.checkpoint import load_purifier_from_checkpoint
from consistency_model.cm_purifier.dataset import load_image_tensor
from consistency_model.cm_purifier.infer import resolve_t_star, save_image_tensor
from consistency_model.cm_purifier.schedules import minus_one_to_one_to_zero_one, q_sample

from .dataset import TestCase, build_purify_records, discover_test_cases, summarize_cases


LOGGER = logging.getLogger("purify.algorithm3")


# Purpose: Build CLI parser for dataset-level Algorithm 3 purification.
# Input: no arguments.
# Output: argparse.ArgumentParser with all purification options.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Purify held-out poison cases with a trained CM purifier.")
    parser.add_argument("--checkpoint", type=str, default="consistency_model/checkpoints/cm_purifier.pth")
    parser.add_argument("--input", type=str, default="dataset_generation/datasets/test")
    parser.add_argument("--output", type=str, default="purify/outputs/test_purified")
    parser.add_argument("--t-star", type=float, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--copy-reference-dirs", action="store_true")
    parser.add_argument("--use-student", action="store_true")
    return parser


# Purpose: Configure readable logs for Slurm output files.
# Input: no arguments.
# Output: configured logger.
def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return LOGGER


# Purpose: Emit a visible section header in purification logs.
# Input: section title.
# Output: none; title is written through the logger.
def log_section(title: str) -> None:
    LOGGER.info("=" * 72)
    LOGGER.info(title)
    LOGGER.info("=" * 72)


# Purpose: Resolve requested torch device.
# Input: device argument string.
# Output: torch.device object.
def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# Purpose: Seed torch CPU and CUDA RNGs for reproducible inference noise.
# Input: integer seed.
# Output: none; RNG state is updated.
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Purpose: Format elapsed or remaining seconds as HH:MM:SS.
# Input: duration in seconds.
# Output: human-readable time string.
def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# Purpose: Format CUDA memory usage for progress logs.
# Input: torch device.
# Output: GPU memory string or CPU marker.
def format_gpu_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    allocated_gb = torch.cuda.memory_allocated(device) / 1e9
    reserved_gb = torch.cuda.memory_reserved(device) / 1e9
    return f"{allocated_gb:.1f}/{reserved_gb:.1f} GB"


# Purpose: Estimate remaining purification time from current throughput.
# Input: start time, processed image count, and total image count.
# Output: ETA in seconds.
def estimate_eta_seconds(start_time: float, processed: int, total: int) -> float:
    processed = max(processed, 1)
    elapsed = time.monotonic() - start_time
    images_per_second = processed / max(elapsed, 1e-8)
    return max(total - processed, 0) / max(images_per_second, 1e-8)


# Purpose: Format one progress line for purification logs.
# Input: processed count, total count, start time, and device.
# Output: readable progress string.
def format_progress(processed: int, total: int, start_time: float, device: torch.device) -> str:
    elapsed = time.monotonic() - start_time
    percent = 100.0 * processed / max(total, 1)
    images_per_second = processed / max(elapsed, 1e-8)
    eta = estimate_eta_seconds(start_time, processed, total)
    return (
        f"Images: {processed}/{total} | "
        f"{percent:.2f}% | "
        f"{images_per_second:.2f} img/s | "
        f"elapsed: {format_duration(elapsed)} | "
        f"eta: {format_duration(eta)} | "
        f"gpu: {format_gpu_memory(device)}"
    )


# Purpose: Copy clean and target folders to the output tree for evaluation context.
# Input: test cases and output root.
# Output: none; reference directories are copied if present.
def copy_reference_dirs(cases: List[TestCase], output_root: Path) -> None:
    for case in cases:
        for folder_name, source_dir in [("clean", case.clean_dir), ("target", case.target_dir)]:
            if not source_dir.is_dir():
                continue
            destination = output_root / case.name / folder_name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source_dir, destination)


# Purpose: Purify one tensor batch using Algorithm 3.
# Input: model, batch tensor, t-star, schedules, and device.
# Output: purified image tensor batch in [0, 1].
def purify_batch(model, batch, t_star: int, alpha_schedule, sigma_schedule, device: torch.device):
    batch = batch.to(device)
    timesteps = torch.full((batch.shape[0],), t_star, dtype=torch.long, device=device)
    noise = torch.randn(batch.shape, dtype=batch.dtype, device=device)
    x_t = q_sample(batch, timesteps, noise, alpha_schedule, sigma_schedule)
    with torch.no_grad():
        purified = model(x_t, timesteps, alpha_schedule, sigma_schedule)
    return minus_one_to_one_to_zero_one(purified)


# Purpose: Write a JSON summary file for the purification run.
# Input: output root and summary dictionary.
# Output: path to the written summary file.
def write_summary(output_root: Path, summary: Dict[str, object]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary_path


# Purpose: Emit readable per-case start and finish messages as records progress.
# Input: current case state, batch records, and cumulative per-case counts.
# Output: updated current case name.
def update_case_logging(current_case, batch_records, case_done_counts: Dict[str, int]):
    for record in batch_records:
        if record.case_name != current_case:
            if current_case is not None:
                LOGGER.info("Finished case: %s | images purified: %d", current_case, case_done_counts[current_case])
            current_case = record.case_name
            LOGGER.info("Starting case: %s", current_case)
        case_done_counts[record.case_name] = case_done_counts.get(record.case_name, 0) + 1
    return current_case


# Purpose: Run Algorithm 3 purification over the held-out test dataset.
# Input: optional CLI argument list.
# Output: path to the output directory.
def main(args=None):
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args(args)
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    input_root = Path(args.input)
    output_root = Path(args.output)
    device = resolve_device(args.device)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint file: {checkpoint_path}. "
            "Submit purify/run_purify_test.sh so Slurm can train it first, "
            "or run consistency_model/run_cm_purifier_training.sh before direct Python inference."
        )

    log_section("1. DISCOVERING TEST POISON CASES...")
    cases = discover_test_cases(input_root)
    records = build_purify_records(cases)
    if args.max_images is not None:
        records = records[: args.max_images]
    case_summary = summarize_cases(cases)
    LOGGER.info("Checkpoint: %s", checkpoint_path)
    LOGGER.info("Input root: %s", input_root)
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Discovered %d cases and %d poison images", case_summary["total_cases"], len(records))
    LOGGER.info("t_star: %s | batch size: %d | seed: %d", args.t_star, args.batch_size, args.seed)

    log_section("2. LOADING TRAINED CM PURIFIER...")
    model, alpha_schedule, sigma_schedule, train_args = load_purifier_from_checkpoint(
        checkpoint_path,
        device=device,
        use_student=args.use_student,
    )
    image_size = int(train_args.get("image_size", 32))
    t_star = resolve_t_star(args.t_star, len(alpha_schedule))
    LOGGER.info("Resolved device: %s", device)
    LOGGER.info("Resolved integer t_star: %d", t_star)
    LOGGER.info("Checkpoint training args image_size: %d", image_size)

    log_section("3. PURIFYING POISON IMAGES...")
    output_root.mkdir(parents=True, exist_ok=True)
    processed = 0
    current_case = None
    case_done_counts: Dict[str, int] = {}
    started_at = time.monotonic()

    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        current_case = update_case_logging(current_case, batch_records, case_done_counts)

        batch = torch.stack(
            [load_image_tensor(record.source_path, image_size=image_size) for record in batch_records],
            dim=0,
        )
        purified = purify_batch(model, batch, t_star, alpha_schedule, sigma_schedule, device)
        for tensor, record in zip(purified, batch_records):
            save_image_tensor(tensor, output_root / record.relative_output_path)

        processed += len(batch_records)
        if processed == len(records) or processed % max(args.log_steps, 1) == 0:
            LOGGER.info(format_progress(processed, len(records), started_at, device))

    if current_case is not None:
        LOGGER.info("Finished case: %s | images purified: %d", current_case, case_done_counts[current_case])

    if args.copy_reference_dirs:
        log_section("4. COPYING CLEAN/TARGET REFERENCE DIRECTORIES...")
        copy_reference_dirs(cases, output_root)
    else:
        log_section("4. SKIPPING CLEAN/TARGET REFERENCE COPY...")

    log_section("5. WRITING SUMMARY...")
    elapsed = time.monotonic() - started_at
    summary = {
        "checkpoint": str(checkpoint_path),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "t_star": t_star,
        "requested_t_star": args.t_star,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "copy_reference_dirs": bool(args.copy_reference_dirs),
        "cases": case_summary,
        "purified_images": processed,
        "elapsed_seconds": elapsed,
        "images_per_second": processed / max(elapsed, 1e-8),
    }
    summary_path = write_summary(output_root, summary)
    LOGGER.info("Summary written to %s", summary_path)
    LOGGER.info("Done. Purified images are in %s", output_root)
    return output_root


if __name__ == "__main__":
    main()
