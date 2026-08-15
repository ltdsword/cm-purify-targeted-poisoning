import json
import logging
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from benchmark.cases import BenchmarkCase, discover_benchmark_cases, parse_case_name
from benchmark.materialize import MaterializedCase, purify_materialized_case
from benchmark.ns import NSVictimConfig, _accuracy, _attack_success, _atomic_save, _fingerprint, _load_checkpoint


class ConstantModel(torch.nn.Module):
    def __init__(self, predicted_class: int):
        super().__init__()
        self.predicted_class = predicted_class

    def forward(self, images):
        logits = torch.zeros((images.shape[0], 10), device=images.device)
        logits[:, self.predicted_class] = 1
        return logits


class FakePurifier:
    def __init__(self):
        self.device = torch.device("cpu")

    def purify_paths(self, source_paths, output_paths, batch_size):
        del batch_size
        for source, output in zip(source_paths, output_paths):
            output.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                image.convert("RGB").save(output)

    def schedule_statistics(self):
        return {"t_star": 200, "seed": 52000}


class BenchmarkNSTests(unittest.TestCase):
    def test_case_parser(self):
        self.assertEqual(parse_case_name("NS_c7"), ("NS", 7, None))

    def test_ns_discovery_without_legacy_pickles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "NS_c2"
            (case_dir / "poisons").mkdir(parents=True)
            (case_dir / "clean").mkdir()
            (case_dir / "metadata.json").write_text(
                json.dumps({"target_class": 2, "num_poison_train": 1, "num_triggered_test": 1, "profile": "smoke"}),
                encoding="utf-8",
            )
            (case_dir / "poison_indices.json").write_text("[12]", encoding="utf-8")
            cases = discover_benchmark_cases(root, root / "missing_wb", root / "missing_bp", attack_filter="NS")
            self.assertEqual([case.name for case in cases], ["NS_c2"])
            self.assertEqual(cases[0].setup["base indices"], [12])

    def test_metric_calculations(self):
        images = torch.zeros((4, 3, 32, 32))
        labels = torch.tensor([2, 2, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        model = ConstantModel(2)
        natural, count = _accuracy(model, loader, torch.device("cpu"))
        target, target_count = _accuracy(model, loader, torch.device("cpu"), target_class=2)
        asr, triggered_count = _attack_success(model, loader, torch.device("cpu"), target_class=2)
        self.assertEqual((natural, count), (50.0, 4))
        self.assertEqual((target, target_count), (100.0, 2))
        self.assertEqual((asr, triggered_count), (100.0, 4))

    def test_victim_checkpoint_compatibility_and_resume_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "state.pt"
            config = NSVictimConfig(epochs=2, num_workers=0, triggered_test_limit=500)
            fingerprint = _fingerprint(config, "poison", "NS_c2")
            _atomic_save({"config_fingerprint": fingerprint, "next_epoch": 1}, checkpoint_path)
            loaded = _load_checkpoint(checkpoint_path, fingerprint, torch.device("cpu"))
            self.assertEqual(loaded["next_epoch"], 1)
            equivalent = NSVictimConfig(epochs=2, num_workers=0, triggered_test_limit=9000, resume=False)
            self.assertEqual(fingerprint, _fingerprint(equivalent, "poison", "NS_c2"))
            with self.assertRaises(ValueError):
                _load_checkpoint(checkpoint_path, "different", torch.device("cpu"))

    def test_per_case_timing_serialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            poisoned = root / "poisoned_train" / "0"
            poisoned.mkdir(parents=True)
            for index in range(3):
                Image.new("RGB", (32, 32), color=(index, index, index)).save(poisoned / f"{index}.png")
            case = BenchmarkCase(
                name="NS_c0", attack="NS", class_idx=0, group_idx=None,
                case_dir=root, poison_dir=root / "poisons", clean_dir=root / "clean",
                target_dir=root / "target", setup_index=-1,
                setup={"target class": 0, "target index": -1, "base class": 0},
            )
            materialized = MaterializedCase(
                case=case, case_output_dir=root, poisoned_train_dir=root / "poisoned_train",
                purified_train_dir=root / "purified_train", purify_dir=root / "purify",
                target_dir=root / "target", poison_relpaths={}, bp_flat_base_indices=None,
                train_image_count=3, poison_image_count=1,
            )
            materialized.purified_train_dir.mkdir()
            stats = purify_materialized_case(
                materialized, FakePurifier(), batch_size=2, log_steps=10,
                logger=logging.getLogger("test"), checkpoint_sha256="abc",
            )
            self.assertEqual(stats.image_count, 3)
            self.assertGreater(stats.elapsed_seconds, 0)
            self.assertGreater(stats.images_per_second, 0)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["purification_timing"]["checkpoint_sha256"], "abc")


if __name__ == "__main__":
    unittest.main()
