"""Case discovery and metadata lookup for WB/BP benchmark runs."""

from __future__ import annotations

import fnmatch
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


WB_CASE_PATTERN = re.compile(r"^WB_c(?P<class_idx>[0-9]+)$")
BP_CASE_PATTERN = re.compile(r"^BP_c(?P<class_idx>[0-9]+)_g(?P<group_idx>[0-9]+)$")


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    attack: str
    class_idx: int
    group_idx: Optional[int]
    case_dir: Path
    poison_dir: Path
    clean_dir: Path
    target_dir: Path
    setup_index: int
    setup: Dict[str, object]


# Purpose: Read a pickle setup file from disk.
# Input: pickle path.
# Output: deserialized list of setup dictionaries.
def load_setup_pickle(path: str | Path) -> List[Dict[str, object]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing setup pickle: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


# Purpose: Parse a case directory name into attack/class/group fields.
# Input: case name such as WB_c0 or BP_c0_g1.
# Output: tuple of attack, class index, optional group index.
def parse_case_name(case_name: str) -> tuple[str, int, Optional[int]]:
    wb_match = WB_CASE_PATTERN.match(case_name)
    if wb_match:
        return "WB", int(wb_match.group("class_idx")), None
    bp_match = BP_CASE_PATTERN.match(case_name)
    if bp_match:
        return "BP", int(bp_match.group("class_idx")), int(bp_match.group("group_idx"))
    raise ValueError(f"Unsupported benchmark case name: {case_name}")


# Purpose: Return whether a case name should be included by a user filter.
# Input: case name and comma-separated filter string.
# Output: True when no filter is set or the case matches a literal/glob token.
def matches_case_filter(case_name: str, case_filter: str | None) -> bool:
    if not case_filter:
        return True
    tokens = [token.strip() for token in case_filter.split(",") if token.strip()]
    if not tokens:
        return True
    return any(case_name == token or fnmatch.fnmatch(case_name, token) for token in tokens)


# Purpose: Find the WB setup record corresponding to an eval case.
# Input: WB setup list and base class.
# Output: tuple of setup index and setup dictionary.
def find_wb_setup(wb_setups: List[Dict[str, object]], class_idx: int) -> tuple[int, Dict[str, object]]:
    for index, setup in enumerate(wb_setups):
        if (
            int(setup.get("base class")) == class_idx
            and not bool(setup.get("is_train", True))
            and setup.get("desc") == "eval"
        ):
            return index, setup
    raise KeyError(f"No WB eval setup found for class {class_idx}")


# Purpose: Find the BP setup record corresponding to an eval case.
# Input: BP setup list, base class, and eval group.
# Output: tuple of setup index and setup dictionary.
def find_bp_setup(
    bp_setups: List[Dict[str, object]],
    class_idx: int,
    group_idx: int,
) -> tuple[int, Dict[str, object]]:
    for index, setup in enumerate(bp_setups):
        if (
            int(setup.get("base class")) == class_idx
            and int(setup.get("batch_group")) == group_idx
            and not bool(setup.get("is_train", True))
        ):
            return index, setup
    raise KeyError(f"No BP eval setup found for class {class_idx}, group {group_idx}")


# Purpose: List all PNG poison files in a benchmark case.
# Input: BenchmarkCase object.
# Output: sorted poison image paths.
def list_case_poisons(case: BenchmarkCase) -> List[Path]:
    return sorted(path for path in case.poison_dir.glob("*.png") if path.is_file())


# Purpose: Build the expected WB poison filename for a base index.
# Input: base class and official CIFAR train index.
# Output: generated WB poison filename.
def wb_poison_name(class_idx: int, base_index: int) -> str:
    return f"wb_c{class_idx}_{base_index}.png"


# Purpose: Build the expected BP poison filename for a class-relative base index.
# Input: base class, group index, and class-relative BP base index.
# Output: generated BP poison filename.
def bp_poison_name(class_idx: int, group_idx: int, base_index: int) -> str:
    return f"bp_c{class_idx}_g{group_idx}_{base_index}.png"


# Purpose: Discover benchmark cases and attach original attack setup metadata.
# Input: test root, setup pickle paths, attack/case filters, and optional max cases.
# Output: sorted list of BenchmarkCase objects.
def discover_benchmark_cases(
    test_dir: str | Path,
    wb_config: str | Path,
    bp_config: str | Path,
    attack_filter: str = "all",
    case_filter: str | None = None,
    max_cases: int | None = None,
) -> List[BenchmarkCase]:
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing benchmark test directory: {test_dir}")

    wb_setups = load_setup_pickle(wb_config)
    bp_setups = load_setup_pickle(bp_config)
    requested_attack = attack_filter.upper()
    cases: List[BenchmarkCase] = []

    for case_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        try:
            attack, class_idx, group_idx = parse_case_name(case_dir.name)
        except ValueError:
            continue
        if requested_attack != "ALL" and requested_attack != attack:
            continue
        if not matches_case_filter(case_dir.name, case_filter):
            continue
        poison_dir = case_dir / "poisons"
        if not poison_dir.is_dir():
            continue
        if attack == "WB":
            setup_index, setup = find_wb_setup(wb_setups, class_idx)
        else:
            if group_idx is None:
                raise ValueError(f"BP case is missing group index: {case_dir.name}")
            setup_index, setup = find_bp_setup(bp_setups, class_idx, group_idx)
        cases.append(
            BenchmarkCase(
                name=case_dir.name,
                attack=attack,
                class_idx=class_idx,
                group_idx=group_idx,
                case_dir=case_dir,
                poison_dir=poison_dir,
                clean_dir=case_dir / "clean",
                target_dir=case_dir / "target",
                setup_index=setup_index,
                setup=setup,
            )
        )
        if max_cases is not None and len(cases) >= max_cases:
            break

    if not cases:
        raise ValueError(
            f"No benchmark cases found under {test_dir} for attack_filter={attack_filter!r}, "
            f"case_filter={case_filter!r}"
        )
    return cases


# Purpose: Summarize discovered cases by attack type.
# Input: iterable of BenchmarkCase objects.
# Output: dictionary of counts for logs and summary JSON.
def summarize_cases(cases: Iterable[BenchmarkCase]) -> Dict[str, object]:
    cases = list(cases)
    return {
        "total": len(cases),
        "WB": sum(1 for case in cases if case.attack == "WB"),
        "BP": sum(1 for case in cases if case.attack == "BP"),
        "names": [case.name for case in cases],
    }
