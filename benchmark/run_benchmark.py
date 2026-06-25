"""Run CM purification benchmark over held-out WB and BP test cases."""

from __future__ import annotations

import argparse
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
from .wb import evaluate_wb_case


LOGGER = setup_logging("benchmark.run")


# Purpose: Build CLI parser for Slurm benchmark orchestration.
# Input: no arguments.
# Output: argparse.ArgumentParser with benchmark options.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark CM purification on WB/BP held-out poisoning cases.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--test-dir", type=str, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--attack-filter", type=str, default="all", choices=["all", "WB", "BP", "wb", "bp"])
    parser.add_argument("--case-filter", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--t-star", type=float, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
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
    for key in ["test_dir", "cifar_root", "wb_config", "bp_config", "bp_split"]:
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
    return f"target_c{int(case.setup['target class'])}_{int(case.setup['target index'])}"


# Purpose: Convert nested evaluator output into the required CSV row.
# Input: BenchmarkCase and optional evaluation result.
# Output: row dictionary matching the benchmark CSV schema.
def build_result_row(case: BenchmarkCase, result: Dict[str, object] | None) -> Dict[str, object]:
    if result is None:
        return {
            "Case": case.name,
            "Target": target_descriptor(case),
            "Attack": case.attack,
            "Clean Accuracy (Poison)": "",
            "Target Acc (Poison)": "",
            "Clean Acc (Purified)": "",
            "Target Acc (Purified)": "",
        }
    poison = result["poison"]
    purified = result["purified"]
    return {
        "Case": case.name,
        "Target": target_descriptor(case),
        "Attack": case.attack,
        "Clean Accuracy (Poison)": f"{float(poison['clean_acc']):.4f}",
        "Target Acc (Poison)": f"{float(poison['target_acc']):.4f}",
        "Clean Acc (Purified)": f"{float(purified['clean_acc']):.4f}",
        "Target Acc (Purified)": f"{float(purified['target_acc']):.4f}",
    }


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
    return materialize_bp_case(
        case=case,
        output_root=run_root,
        bp_split_path=paths["bp_split"],
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


# Purpose: Run benchmark orchestration from CLI args.
# Input: optional argument list.
# Output: benchmark run root path.
def main(argv=None) -> Path:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    repo_dir = repo_root()
    device = resolve_device(args.device)
    if device.type != "cuda" and args.attack_filter.upper() in {"ALL", "BP"} and not args.skip_retrain:
        raise RuntimeError("BP retraining requires CUDA; submit benchmark/run_benchmark.sh on a GPU node.")

    paths = resolve_paths(repo_dir, args)
    run_id = args.run_id or default_run_id()
    run_root = paths["output_dir"] / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
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
        LOGGER.info("Loaded purifier on %s with t_star=%d", purifier.device, purifier.t_star)
    else:
        LOGGER.info("Skipping purification by request.")

    for case_index, case in enumerate(cases, start=1):
        log_section(LOGGER, f"CASE {case_index}/{len(cases)}: {case.name} ({case.attack})")
        materialized = materialize_case(
            case=case,
            paths=paths,
            run_root=run_root,
            overwrite=args.overwrite_artifacts,
        )
        if not args.skip_purify:
            purify_materialized_case(
                materialized=materialized,
                purifier=purifier,
                batch_size=args.batch_size,
                log_steps=args.log_steps,
                logger=LOGGER,
            )
        elif not args.skip_retrain and not any(materialized.purified_train_dir.rglob("*.png")):
            raise FileNotFoundError(
                f"Purified train directory is empty for {case.name}: {materialized.purified_train_dir}. "
                "Run without --skip-purify before retraining purified data."
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
            append_jsonl(
                jsonl_path,
                {
                    "case": case.name,
                    "attack": case.attack,
                    "target": target_descriptor(case),
                    "result": result,
                },
            )
        else:
            LOGGER.info("Skipping retrain/evaluation for %s by request.", case.name)

        row = build_result_row(case, result)
        rows.append(row)
        write_results_csv(csv_path, rows)
        LOGGER.info("Updated benchmark CSV: %s", csv_path)

    log_section(LOGGER, "BENCHMARK COMPLETE")
    LOGGER.info("Results CSV: %s", csv_path)
    LOGGER.info("Results JSONL: %s", jsonl_path)
    return run_root


if __name__ == "__main__":
    main()
