from __future__ import annotations

import unittest
from pathlib import Path

from inference.vlbrandphish.domain_verification import registered_domain
from inference.vlbrandphish.types import Metrics


class ArchitectureContractsTest(unittest.TestCase):
    def test_metrics_are_named_and_deterministic(self) -> None:
        metrics = Metrics(tp=3, tn=4, fp=1, fn=2, brand_total_phish=3, brand_correct_phish=2)
        report = metrics.to_dict()

        self.assertEqual(report["tp"], 3)
        self.assertAlmostEqual(report["accuracy"], 0.7)
        self.assertAlmostEqual(report["brand_acc_on_phish"], 2 / 3)

    def test_domain_policy_is_self_contained(self) -> None:
        self.assertEqual(registered_domain("https://login.example.co.uk/path"), "example.co.uk")
        self.assertEqual(registered_domain("not-a-url.example"), "not-a-url.example")

    def test_training_configs_request_deterministic_backend(self) -> None:
        root = Path(__file__).resolve().parents[2]
        configs_dir = (
            root
            / "training"
            / "vlbrandphish"
            / "brand_recognition"
            / "configs"
        )
        for name in (
            "stage1_siglip.yaml",
            "stage1_router_logo2k.yaml",
            "stage2_trisig_ot_safe.yaml",
        ):
            config = (configs_dir / name).read_text(encoding="utf-8")
            self.assertIn("seed: 42", config)
            self.assertIn("deterministic: true", config)

    def test_workflow_oriented_source_layout(self) -> None:
        root = Path(__file__).resolve().parents[2]
        expected = (
            "inference/vlbrandphish/components/brand/encoders.py",
            "inference/vlbrandphish/components/brand/fusion.py",
            "inference/vlbrandphish/input_loading.py",
            "inference/vlbrandphish/logo_detection.py",
            "inference/vlbrandphish/model_loading.py",
            "inference/vlbrandphish/brand_recognition.py",
            "inference/vlbrandphish/domain_verification.py",
            "inference/vlbrandphish/evaluation.py",
            "inference/vlbrandphish/pipeline.py",
            "training/vlbrandphish/logo_detection/train.py",
            "training/vlbrandphish/brand_recognition/common/model.py",
            "training/vlbrandphish/brand_recognition/stage1/trainer.py",
            "training/vlbrandphish/brand_recognition/stage2/trainer.py",
        )
        for relative in expected:
            self.assertTrue((root / relative).is_file(), relative)

        legacy_modules = (
            "inference/vlbrandphish/runner.py",
            "training/vlbrandphish/brand/engine.py",
        )
        for relative in legacy_modules:
            self.assertFalse((root / relative).is_file(), relative)

if __name__ == "__main__":
    unittest.main()
