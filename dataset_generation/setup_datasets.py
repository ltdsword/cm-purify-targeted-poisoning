"""Compatibility wrapper for the dataset generation setup step.

The canonical setup logic lives in scripts/dataset_generation.py so the Slurm
runner and any manual setup call produce the same configs, image layout, and
BullseyePoison CIFAR split.
"""

from dataset_generation.scripts.dataset_generation import setup_clean_datasets


if __name__ == "__main__":
    setup_clean_datasets()
