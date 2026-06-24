"""Dataset discovery utilities for Algorithm 3 purification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class PurifyRecord:
    case_name: str
    source_path: Path
    relative_output_path: Path


@dataclass(frozen=True)
class TestCase:
    name: str
    case_dir: Path
    clean_dir: Path
    poison_dir: Path
    target_dir: Path
    poison_count: int


# Purpose: Return whether a path is a supported image file.
# Input: filesystem path.
# Output: True for supported image files, otherwise False.
def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


# Purpose: Discover held-out evaluation cases under the test dataset root.
# Input: root test directory containing case subdirectories.
# Output: sorted list of TestCase records with poison image counts.
def discover_test_cases(test_dir: str | Path) -> List[TestCase]:
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing test dataset directory: {test_dir}")

    cases: List[TestCase] = []
    for case_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        poison_dir = case_dir / "poisons"
        if not poison_dir.is_dir():
            continue
        poison_count = sum(1 for path in poison_dir.iterdir() if is_supported_image(path))
        cases.append(
            TestCase(
                name=case_dir.name,
                case_dir=case_dir,
                clean_dir=case_dir / "clean",
                poison_dir=poison_dir,
                target_dir=case_dir / "target",
                poison_count=poison_count,
            )
        )
    if not cases:
        raise ValueError(f"No test cases with poisons/ images found under {test_dir}")
    return cases


# Purpose: Build purification records for every poison image in every test case.
# Input: iterable of discovered TestCase objects.
# Output: sorted list of PurifyRecord objects.
def build_purify_records(cases: Iterable[TestCase]) -> List[PurifyRecord]:
    records: List[PurifyRecord] = []
    for case in cases:
        poison_paths = sorted(path for path in case.poison_dir.iterdir() if is_supported_image(path))
        for source_path in poison_paths:
            records.append(
                PurifyRecord(
                    case_name=case.name,
                    source_path=source_path,
                    relative_output_path=Path(case.name) / "poisons" / source_path.name,
                )
            )
    return records


# Purpose: Summarize test cases for logs and JSON output.
# Input: iterable of TestCase records.
# Output: dictionary with total cases, total poison images, and per-case counts.
def summarize_cases(cases: Iterable[TestCase]) -> Dict[str, object]:
    cases = list(cases)
    per_case = {case.name: case.poison_count for case in cases}
    return {
        "total_cases": len(cases),
        "total_poison_images": sum(per_case.values()),
        "cases": per_case,
    }

