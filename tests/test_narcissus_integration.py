import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from consistency_model.cm_purifier import ATTACK_TO_ID
from consistency_model.cm_purifier.dataset import parse_attack_type, parse_label
from dataset_generation.Narcissus.dataset_builder import (
    build_ns_setup_records,
    create_ns_setup_file,
    export_ns_cm_bank,
    export_ns_eval_cases,
    shuffled_indices_by_class,
    trigger_artifact_path,
    validate_ns_artifacts,
)
from dataset_generation.Narcissus.integration import (
    TriggerGenerationConfig,
    _load_checkpoint,
    _save_stage_checkpoint,
    apply_trigger_array,
    trigger_sha256,
    validate_trigger,
)


class NarcissusTriggerTests(unittest.TestCase):
    def test_pixel_trigger_application_preserves_shape_and_clips(self):
        image = np.full((32, 32, 3), 250, dtype=np.uint8)
        trigger = torch.full((3, 32, 32), 8.0 / 255.0)
        output = apply_trigger_array(image, trigger)
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(np.isfinite(output).all())
        self.assertEqual(int(output.max()), 255)
        validate_trigger(trigger, 8.0 / 255.0)

    def test_trigger_bound_is_enforced(self):
        with self.assertRaises(ValueError):
            validate_trigger(torch.full((3, 32, 32), 9.0 / 255.0), 8.0 / 255.0)

    def test_trigger_hash_is_deterministic(self):
        trigger = torch.linspace(-8 / 255, 8 / 255, 3 * 32 * 32).reshape(3, 32, 32)
        self.assertEqual(trigger_sha256(trigger), trigger_sha256(trigger.clone()))

    def test_stage_checkpoint_round_trip_and_configuration_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = TriggerGenerationConfig(
                target_class=2, kind="train", seed=12002,
                cifar_root=str(root), pood_root=str(root), output_path=str(root / "trigger.pt"),
                checkpoint_dir=str(root / "checkpoints"), synthesis_indices=(1, 2),
                surrogate_epochs=1, warmup_epochs=1, trigger_rounds=1,
                batch_size=1, num_workers=0, checkpoint_interval=1, device="cpu",
            )
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            checkpoint_path = root / "state.pt"
            _save_stage_checkpoint(checkpoint_path, config, "surrogate", 1, model, optimizer)
            loaded = _load_checkpoint(checkpoint_path, config)
            self.assertEqual(loaded["stage"], "surrogate")
            self.assertEqual(loaded["next_step"], 1)
            self.assertEqual(loaded["selected_indices"], [1, 2])
            incompatible = TriggerGenerationConfig(**{**config.__dict__, "seed": 999})
            with self.assertRaises(RuntimeError):
                _load_checkpoint(checkpoint_path, incompatible)


class NarcissusAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = np.repeat(np.arange(10), 5000)
        cls.shuffled = shuffled_indices_by_class(targets)
        cls.records = build_ns_setup_records(cls.shuffled)

    def test_locked_balanced_allocation(self):
        self.assertEqual(len(self.records), 10)
        all_pairs = []
        per_semantic_class = {label: 0 for label in range(10)}
        for record in self.records:
            self.assertEqual(len(record["eval_poison_indices"]), 500)
            for source_class, indices in record["cm_pair_source_indices"].items():
                self.assertEqual(len(indices), 100)
                per_semantic_class[int(source_class)] += len(indices)
                all_pairs.extend(indices)
        self.assertEqual(len(all_pairs), 10000)
        self.assertEqual(len(set(all_pairs)), 10000)
        self.assertEqual(per_semantic_class, {label: 1000 for label in range(10)})

    def test_no_cm_eval_overlap_and_disjoint_trigger_synthesis(self):
        for record in self.records:
            cm_indices = {
                index for values in record["cm_pair_source_indices"].values() for index in values
            }
            eval_indices = set(record["eval_poison_indices"])
            self.assertTrue(cm_indices.isdisjoint(eval_indices))
            self.assertTrue(
                set(record["train_trigger_synthesis_indices"]).isdisjoint(
                    record["eval_trigger_synthesis_indices"]
                )
            )

    def test_seed_namespaces(self):
        for target_class, record in enumerate(self.records):
            self.assertEqual(record["seeds"]["train_trigger"], 12000 + target_class)
            self.assertEqual(record["seeds"]["eval_trigger"], 22000 + target_class)
            self.assertEqual(record["seeds"]["eval_poison_selection"], 32000 + target_class)


class CMParserTests(unittest.TestCase):
    def test_ns_filename_parser_and_attack_id(self):
        name = "ns_c5_t2_12345.png"
        self.assertEqual(parse_attack_type(name), "ns")
        self.assertEqual(parse_label(name), 5)
        self.assertEqual(ATTACK_TO_ID["ns"], 3)


class LazyCIFARDataset:
    def __init__(self, train: bool):
        per_class = 5000 if train else 1000
        self.targets = np.repeat(np.arange(10), per_class).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        label = int(self.targets[index])
        array = np.full((32, 32, 3), (index + label) % 256, dtype=np.uint8)
        return Image.fromarray(array, mode="RGB"), label


class NarcissusSyntheticPipelineTests(unittest.TestCase):
    def test_smoke_bank_and_evaluation_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup_path = root / "configs" / "ns.json"
            train_set, test_set = LazyCIFARDataset(True), LazyCIFARDataset(False)
            create_ns_setup_file(train_set.targets, setup_path)
            for kind, value, seed in (("train", 1.0 / 255.0, 12002), ("eval", -1.0 / 255.0, 22002)):
                trigger = torch.full((3, 32, 32), value)
                metadata = {
                    "trigger_id": f"ns_{kind}_trigger_c02_seed_{seed}",
                    "seed": seed,
                    "sha256": trigger_sha256(trigger),
                }
                artifact_path = trigger_artifact_path(root, kind, 2, profile="smoke")
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"trigger": trigger, "metadata": metadata}, artifact_path)
            clean_dir = root / "train_smoke" / "clean"
            poison_dir = root / "train_smoke" / "poisons"
            test_root = root / "test_smoke"
            with patch(
                "dataset_generation.Narcissus.dataset_builder._load_cifar",
                return_value=(train_set, test_set),
            ):
                export_ns_cm_bank(
                    setup_path, root, clean_dir, poison_dir,
                    classes=[2], profile="smoke",
                )
                export_ns_eval_cases(
                    setup_path, root, test_root,
                    classes=[2], profile="smoke",
                )
            summary = validate_ns_artifacts(
                setup_path, root, clean_dir, poison_dir, test_root,
                classes=[2], profile="smoke",
            )
            self.assertEqual(summary["pairs"], 1000)
            self.assertEqual(summary["cases"]["NS_c2"]["num_poison_train"], 50)
            self.assertEqual(summary["cases"]["NS_c2"]["num_triggered_test"], 500)


if __name__ == "__main__":
    unittest.main()
