"""Run CM purification benchmark over held-out WB, BP, and NS test cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch

from purify.purifier import CMPurifier, resolve_device

from . import DEFAULT_CHECKPOINT_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_TEST_DIR
from .bp import evaluate_bp_case
from .cases import BenchmarkCase, discover_benchmark_cases, summarize_cases
from .common import append_jsonl, log_section, setup_logging, write_json, write_results_csv
from .materialize import materialize_bp_case, materialize_wb_case, purify_materialized_case
from .materialize import PurificationStats, load_completed_materialized_case, materialize_ns_case
from .ns import NSVictimConfig, evaluate_ns_case
from .wb import evaluate_wb_case


LOGGER = setup_logging("benchmark.run")


# Purpose: Build CLI parser for Slurm benchmark orchestration.
# Input: no arguments.
# Output: argparse.ArgumentParser with benchmark options.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark CM purification on WB/BP/NS held-out poisoning cases.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--test-dir", type=str, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--attack-filter", type=str, default="all",
        choices=["all", "WB", "BP", "NS", "wb", "bp", "ns"],
    )
    parser.add_argument("--case-filter", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--t-star", type=float, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=52000)
    parser.add_argument("--log-steps", type=int, default=1024)
    parser.add_argument("--skip-purify", action="store_true")
    parser.add_argument("--skip-retrain", action="store_true")
    parser.add_argument("--overwrite-artifacts", action="store_true")
    parser.add_argument("--wb-epochs", type=int, default=None)
    parser.add_argument("--wb-dryrun", action="store_true")
    parser.add_argument("--bp-victim-net", type=str, default="ResNet18")
    parser.add_argument("--bp-checkpoint-name", type=str, default="ckpt-%s-4800.t7")
    parser.add_argument("--bp-retrain-epochs", type=int, default=60)
    parser.add_argument("--bp-retrain-bsize", type=int, default=64)
    parser.add_argument("--ns-profile", choices=["smoke", "final"], default="final")
    parser.add_argument("--ns-victim-epochs", type=int, default=None)
    parser.add_argument("--ns-victim-batch-size", type=int, default=128)
    parser.add_argument("--ns-victim-workers", type=int, default=8)
    parser.add_argument("--ns-victim-seed", type=int, default=62000)
    parser.add_argument("--ns-victim-checkpoint-interval", type=int, default=1)
    parser.add_argument("--ns-no-victim-resume", action="store_true")
    parser.add_argument("--ns-triggered-test-limit", type=int, default=None)
    return parser


# Purpose: Resolve the repository root from this file location.
# Input: no arguments.
# Output: absolute repository root path.
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Purpose: Build a timestamped run id when the user did not provide one.
# Input: no arguments.
# Output: stable run id string.
def default_run_id() -> str:
    slurm_id = os.environ.get("SLURM_JOB_ID")
    if slurm_id:
        return f"slurm_{slurm_id}"
    return time.strftime("local_%Y%m%d_%H%M%S")


# Purpose: Validate required local files before a long benchmark starts.
# Input: repository root and parsed args.
# Output: dictionary of resolved important paths.
def resolve_paths(repo_dir: Path, args) -> Dict[str, Path]:
    paths = {
        "checkpoint": (repo_dir / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint),
        "test_dir": (repo_dir / args.test_dir).resolve() if not Path(args.test_dir).is_absolute() else Path(args.test_dir),
        "output_dir": (repo_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir),
        "cifar_root": repo_dir / "dataset_generation" / "datasets",
        "wb_config": repo_dir / "dataset_generation" / "configs" / "wb_benchmark_setups.pickle",
        "bp_config": repo_dir / "dataset_generation" / "configs" / "bp_benchmark_setups.pickle",
        "bp_split": repo_dir / "dataset_generation" / "BullseyePoison" / "datasets" / "CIFAR10_TRAIN_Split.pth",
    }
    if not paths["checkpoint"].is_file() and not args.skip_purify:
        raise FileNotFoundError(
            f"Missing CM purifier checkpoint: {paths['checkpoint']}. "
            "Submit consistency_model/run_cm_purifier_training.sh first."
        )
    requested_attack = args.attack_filter.upper()
    required = ["test_dir", "cifar_root"]
    if requested_attack in {"ALL", "WB"}:
        required.append("wb_config")
    if requested_attack in {"ALL", "BP"}:
        required.extend(["bp_config", "bp_split"])
    for key in required:
        path = paths[key]
        if key.endswith("dir") or key == "cifar_root":
            if not path.is_dir():
                raise FileNotFoundError(f"Missing required directory {key}: {path}")
        elif not path.is_file():
            raise FileNotFoundError(f"Missing required file {key}: {path}")
    return paths


# Purpose: Build the user-facing target descriptor for a case.
# Input: BenchmarkCase object.
# Output: compact target string for CSV.
def target_descriptor(case: BenchmarkCase) -> str:
    if case.attack == "NS":
        return f"target_class_c{int(case.setup['target class'])}"
    return f"target_c{int(case.setup['target class'])}_{int(case.setup['target index'])}"


# Purpose: Convert nested evaluator output into the required CSV row.
# Input: BenchmarkCase and optional evaluation result.
# Output: row dictionary matching the benchmark CSV schema.
def build_result_row(
    case: BenchmarkCase,
    result: Dict[str, object] | None,
    timing: PurificationStats | None = None,
) -> Dict[str, object]:
    if result is None:
        row = {
            "Case": case.name,
            "Target": target_descriptor(case),
            "Attack": case.attack,
            "Clean Accuracy (Poison)": "",
            "Target Acc (Poison)": "",
            "Clean Acc (Purified)": "",
            "Target Acc (Purified)": "",
        }
        poison = purified = {}
    else:
        poison = result["poison"]
        purified = result["purified"]
        row = {
            "Case": case.name,
            "Target": target_descriptor(case),
            "Attack": case.attack,
            "Clean Accuracy (Poison)": f"{float(poison['clean_acc']):.4f}",
            "Target Acc (Poison)": f"{float(poison['target_acc']):.4f}",
            "Clean Acc (Purified)": f"{float(purified['clean_acc']):.4f}",
            "Target Acc (Purified)": f"{float(purified['target_acc']):.4f}",
        }
    if result is not None:
        row.update(
            {
                "Natural Accuracy (Poison)": f"{float(poison.get('natural_accuracy', poison['clean_acc'])):.4f}",
                "Attack Success/ASR (Poison)": f"{float(poison.get('attack_success', poison['target_acc'])):.4f}",
                "Target-Class Accuracy (Poison)": _optional_metric(poison, "target_class_acc"),
                "Natural Accuracy (Purified)": f"{float(purified.get('natural_accuracy', purified['clean_acc'])):.4f}",
                "Attack Success/ASR (Purified)": f"{float(purified.get('attack_success', purified['target_acc'])):.4f}",
                "Target-Class Accuracy (Purified)": _optional_metric(purified, "target_class_acc"),
            }
        )
    if timing is not None:
        stats = timing.to_dict()
        row.update(
            {
                "Purification Images": stats["image_count"],
                "Purification Seconds": f"{float(stats['elapsed_seconds']):.6f}",
                "Purification Seconds/Image": f"{float(stats['seconds_per_image']):.9f}",
                "Purification Images/Second": f"{float(stats['images_per_second']):.6f}",
                "Purification Mean Batch Seconds/Image": f"{float(stats['mean_batch_seconds_per_image']):.9f}",
                "Purification Median Batch Seconds/Image": f"{float(stats['median_batch_seconds_per_image']):.9f}",
                "Purification Batch Size": stats["batch_size"],
                "Purification Device": stats["device"],
                "Purification Timestep": stats["t_star"],
                "Purification Seed": stats["inference_seed"],
                "Purifier Checkpoint SHA256": stats["checkpoint_sha256"],
                "Purification Reused": stats["reused"],
            }
        )
    return row


def _optional_metric(metrics: Dict[str, object], key: str) -> str:
    value = metrics.get(key)
    return "" if value is None else f"{float(value):.4f}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_ns_aggregate(run_root: Path, case_results: List[Dict[str, object]]) -> None:
    """Write report-ready aggregate metrics for completed Narcissus cases."""
    jsonl_path = run_root / "benchmark_results.jsonl"
    if jsonl_path.is_file():
        latest_by_case: Dict[str, Dict[str, object]] = {}
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                latest_by_case[str(item["case"])] = item
        case_results = list(latest_by_case.values())
    ns_results = [item for item in case_results if item["attack"] == "NS" and item["result"] is not None]
    if not ns_results:
        return
    poison_asr = [float(item["result"]["poison"]["asr"]) for item in ns_results]
    defended_asr = [float(item["result"]["purified"]["asr"]) for item in ns_results]
    poison_natural = [float(item["result"]["poison"]["natural_accuracy"]) for item in ns_results]
    defended_natural = [float(item["result"]["purified"]["natural_accuracy"]) for item in ns_results]
    count = len(ns_results)
    payload = {
        "case_count": count,
        "reportable_final": count == 10 and all(
            int(item["result"]["poison"]["triggered_test_count"]) == 9000 for item in ns_results
        ),
        "average_asr_poison": sum(poison_asr) / count,
        "maximum_asr_poison": max(poison_asr),
        "average_asr_purified": sum(defended_asr) / count,
        "maximum_asr_purified": max(defended_asr),
        "average_natural_accuracy_poison": sum(poison_natural) / count,
        "average_natural_accuracy_purified": sum(defended_natural) / count,
        "average_asr_reduction": sum(a - b for a, b in zip(poison_asr, defended_asr)) / count,
        "average_natural_accuracy_change": sum(b - a for a, b in zip(poison_natural, defended_natural)) / count,
        "cases": [item["case"] for item in ns_results],
    }
    write_json(run_root / "narcissus_summary.json", payload)


# Purpose: Materialize the tampered train set for a WB or BP case.
# Input: case, paths, output root, overwrite flag, and logger.
# Output: MaterializedCase object.
def materialize_case(case: BenchmarkCase, paths: Dict[str, Path], run_root: Path, overwrite: bool):
    if case.attack == "WB":
        return materialize_wb_case(
            case=case,
            output_root=run_root,
            cifar_root=paths["cifar_root"],
            overwrite=overwrite,
            logger=LOGGER,
        )
    if case.attack == "BP":
        return materialize_bp_case(
        case=case,
        output_root=run_root,
        bp_split_path=paths["bp_split"],
        overwrite=overwrite,
        logger=LOGGER,
        )
    return materialize_ns_case(
        case=case,
        output_root=run_root,
        cifar_root=paths["cifar_root"],
        overwrite=overwrite,
        logger=LOGGER,
    )


# Purpose: Run the correct retrain/eval adapter for a case.
# Input: case artifacts, repo paths, parsed args, and device string.
# Output: evaluator result dictionary.
def evaluate_case(materialized, repo_dir: Path, paths: Dict[str, Path], args, device: str) -> Dict[str, object]:
    if materialized.case.attack == "WB":
        return evaluate_wb_case(
            materialized=materialized,
            repo_dir=repo_dir,
            cifar_root=paths["cifar_root"],
            epochs=args.wb_epochs,
            dryrun=args.wb_dryrun,
            logger=LOGGER,
        )
    if materialized.case.attack == "BP":
        return evaluate_bp_case(
        materialized=materialized,
        repo_dir=repo_dir,
        cifar_root=paths["cifar_root"],
        bp_split_path=paths["bp_split"],
        device=device,
        victim_net=args.bp_victim_net,
        checkpoint_name=args.bp_checkpoint_name,
        retrain_epochs=args.bp_retrain_epochs,
        retrain_bsize=args.bp_retrain_bsize,
        logger=LOGGER,
        )
    epochs = args.ns_victim_epochs
    if epochs is None:
        epochs = 2 if args.ns_profile == "smoke" else 200
    triggered_limit = args.ns_triggered_test_limit
    if triggered_limit is None and args.ns_profile == "smoke":
        triggered_limit = 500
    config = NSVictimConfig(
        epochs=epochs,
        batch_size=args.ns_victim_batch_size,
        num_workers=args.ns_victim_workers,
        seed=args.ns_victim_seed,
        checkpoint_interval=args.ns_victim_checkpoint_interval,
        triggered_test_limit=triggered_limit,
        resume=not args.ns_no_victim_resume,
    )
    return evaluate_ns_case(
        materialized=materialized,
        cifar_root=paths["cifar_root"],
        config=config,
        device=device,
        logger=LOGGER,
    )


# Purpose: Run benchmark orchestration from CLI args.
# Input: optional argument list.
# Output: benchmark run root path.
def main(argv=None) -> Path:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    requested_attack = args.attack_filter.upper()
    if args.ns_profile == "final" and requested_attack in {"ALL", "NS"} and not args.skip_retrain:
        final_errors = []
        if args.ns_victim_epochs not in {None, 200}:
            final_errors.append("--ns-victim-epochs must be 200")
        if args.ns_victim_batch_size != 128:
            final_errors.append("--ns-victim-batch-size must be 128")
        if args.ns_victim_seed != 62000:
            final_errors.append("--ns-victim-seed must be 62000")
        if args.seed != 52000:
            final_errors.append("--seed (CM inference seed) must be 52000")
        if args.ns_triggered_test_limit is not None:
            final_errors.append("--ns-triggered-test-limit must be unset")
        if final_errors:
            parser.error("final NS profile: " + "; ".join(final_errors))
    repo_dir = repo_root()
    device = resolve_device(args.device)
    if device.type != "cuda" and args.attack_filter.upper() in {"ALL", "BP"} and not args.skip_retrain:
        raise RuntimeError("BP retraining requires CUDA; submit benchmark/run_benchmark.sh on a GPU node.")

    paths = resolve_paths(repo_dir, args)
    run_id = args.run_id or default_run_id()
    run_root = paths["output_dir"] / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    case_results: List[Dict[str, object]] = []
    csv_path = run_root / "benchmark_results.csv"
    jsonl_path = run_root / "benchmark_results.jsonl"

    log_section(LOGGER, "1. DISCOVERING BENCHMARK CASES...")
    cases = discover_benchmark_cases(
        test_dir=paths["test_dir"],
        wb_config=paths["wb_config"],
        bp_config=paths["bp_config"],
        attack_filter=args.attack_filter,
        case_filter=args.case_filter,
        max_cases=args.max_cases,
    )
    summary = summarize_cases(cases)
    if args.ns_profile == "final":
        invalid_ns = [
            case.name
            for case in cases
            if case.attack == "NS"
            and (
                case.setup.get("profile") != "final"
                or int(case.setup.get("num_poison_train", 0)) != 500
                or int(case.setup.get("num_triggered_test", 0)) != 9000
            )
        ]
        if invalid_ns:
            raise ValueError(f"Final benchmark requested, but these NS cases are not final artifacts: {invalid_ns}")
    LOGGER.info("Repository: %s", repo_dir)
    LOGGER.info("Run root: %s", run_root)
    LOGGER.info("Case summary: %s", summary)
    write_json(run_root / "run_config.json", {"args": vars(args), "paths": {key: str(value) for key, value in paths.items()}, "cases": summary})

    purifier = None
    if not args.skip_purify:
        log_section(LOGGER, "2. LOADING CM PURIFIER...")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        purifier = CMPurifier.from_checkpoint(
            paths["checkpoint"],
            t_star=args.t_star,
            device=device,
            seed=args.seed,
        )
        schedule = purifier.schedule_statistics()
        LOGGER.info(
            "Loaded purifier on %s | t_star=%d | alpha=%.6f | sigma=%.6f | snr=%.6f | noise seed=%d",
            purifier.device,
            schedule["t_star"],
            schedule["alpha"],
            schedule["sigma"],
            schedule["snr"],
            schedule["seed"],
        )
    else:
        LOGGER.info("Skipping purification by request.")

    checkpoint_hash = "" if args.skip_purify else file_sha256(paths["checkpoint"])

    for case_index, case in enumerate(cases, start=1):
        log_section(LOGGER, f"CASE {case_index}/{len(cases)}: {case.name} ({case.attack})")
        reused = None if args.overwrite_artifacts else load_completed_materialized_case(case, run_root)
        if reused is not None and not args.skip_purify:
            prior_timing = reused[1]
            same_configuration = (
                prior_timing.t_star == int(purifier.t_star)
                and prior_timing.inference_seed == int(args.seed)
                and prior_timing.checkpoint_sha256 == checkpoint_hash
            )
            if not same_configuration:
                LOGGER.info("Existing purification configuration differs for %s; regenerating", case.name)
                reused = None
        timing = None
        if reused is not None:
            materialized, timing = reused
            LOGGER.info("Reusing completed materialization and purification for %s", case.name)
        else:
            if args.skip_purify:
                raise FileNotFoundError(
                    f"No completed reusable artifacts exist for {case.name}; rerun without --skip-purify."
                )
            materialized = materialize_case(
                case=case,
                paths=paths,
                run_root=run_root,
                overwrite=args.overwrite_artifacts,
            )
            purifier.noise_generator.manual_seed(args.seed)
            timing = purify_materialized_case(
                materialized=materialized,
                purifier=purifier,
                batch_size=args.batch_size,
                log_steps=args.log_steps,
                logger=LOGGER,
                checkpoint_sha256=checkpoint_hash,
            )

        result = None
        if not args.skip_retrain:
            log_section(LOGGER, f"RETRAINING AND EVALUATING {case.name}...")
            result = evaluate_case(
                materialized=materialized,
                repo_dir=repo_dir,
                paths=paths,
                args=args,
                device="cuda" if device.type == "cuda" else "cpu",
            )
        else:
            LOGGER.info("Skipping retrain/evaluation for %s by request.", case.name)

        result_record = {
            "case": case.name,
            "attack": case.attack,
            "target": target_descriptor(case),
            "result": result,
            "purification_timing": timing.to_dict() if timing else None,
        }
        append_jsonl(jsonl_path, result_record)
        case_results.append(result_record)
        row = build_result_row(case, result, timing)
        rows.append(row)
        write_results_csv(csv_path, rows)
        write_ns_aggregate(run_root, case_results)
        LOGGER.info("Updated benchmark CSV: %s", csv_path)

    log_section(LOGGER, "BENCHMARK COMPLETE")
    LOGGER.info("Results CSV: %s", csv_path)
    LOGGER.info("Results JSONL: %s", jsonl_path)
    return run_root


if __name__ == "__main__":
    main()
