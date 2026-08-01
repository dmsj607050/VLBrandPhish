from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments._common import ExperimentImplementationRequired, run_all_methods
from experiments.overall_vlbp.run import main as overall_vlbp_main
from inference.vlbrandphish.decision.domain import registered_domain
from inference.vlbrandphish.runtime.types import Metrics


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
        configs_dir = root / "training" / "vlbrandphish" / "brand_recognition" / "configs"
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
            "cli.py",
            "inference/vlbrandphish/components/brand/encoders.py",
            "training/vlbrandphish/brand_recognition/common/model_blocks.py",
            "inference/vlbrandphish/data/detection_csv.py",
            "inference/vlbrandphish/data/testb_dataset.py",
            "inference/vlbrandphish/data/image_paths.py",
            "inference/vlbrandphish/detection/command.py",
            "inference/vlbrandphish/detection/detector.py",
            "inference/vlbrandphish/recognition/resources.py",
            "inference/vlbrandphish/recognition/features.py",
            "inference/vlbrandphish/decision/domain.py",
            "inference/vlbrandphish/decision/policy.py",
            "inference/vlbrandphish/evaluation/metrics.py",
            "inference/vlbrandphish/runtime/cache.py",
            "inference/vlbrandphish/runtime/diagnostics.py",
            "inference/vlbrandphish/runtime/progress.py",
            "inference/vlbrandphish/runtime/paths.py",
            "inference/vlbrandphish/runtime/types.py",
            "inference/vlbrandphish/run.py",
            "training/vlbrandphish/logo_detection/train.py",
            "training/vlbrandphish/brand_recognition/common/model.py",
            "training/vlbrandphish/brand_recognition/configs/stage1_siglip.yaml",
            "training/vlbrandphish/brand_recognition/stage1/trainer.py",
            "training/vlbrandphish/brand_recognition/stage2/trainer.py",
            "docs/EXPERIMENTS.md",
        )
        for relative in expected:
            self.assertTrue((root / relative).is_file(), relative)

        legacy_modules = (
            "inference/vlbrandphish/runner.py",
            "inference/vlbrandphish/data_contracts.py",
            "inference/vlbrandphish/logo_detection.py",
            "inference/vlbrandphish/brand_recognition.py",
            "inference/vlbrandphish/pipeline.py",
            "training/vlbrandphish/brand/engine.py",
            "src/vlbrandphish",
        )
        for relative in legacy_modules:
            self.assertFalse((root / relative).exists(), relative)

    def test_study_scaffolds_fail_before_writing_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "would-be-results"
            with self.assertRaises(ExperimentImplementationRequired):
                run_all_methods("screenshots", "samples.csv", str(output_dir))
            self.assertFalse(output_dir.exists())

    def test_study_cli_creates_no_output_for_an_unimplemented_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "would-be-study-results"
            with self.assertRaises(ExperimentImplementationRequired):
                overall_vlbp_main(
                    [
                        "--screenshot-dir",
                        "screenshots",
                        "--url-list",
                        "samples.csv",
                        "--out-dir",
                        str(output_dir),
                    ]
                )
            self.assertFalse(output_dir.exists())

    def test_reproduction_docs_match_the_flat_layout(self) -> None:
        root = Path(__file__).resolve().parents[2]
        reproducibility = (root / "docs" / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
        self.assertIn("python cli.py train-brand", reproducibility)
        self.assertIn("--include inference --include training --include scripts", reproducibility)
        self.assertNotIn("--include configs", reproducibility)
        self.assertNotIn("--include src", reproducibility)


if __name__ == "__main__":
    unittest.main()