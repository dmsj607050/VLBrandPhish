from __future__ import annotations

import unittest
from pathlib import Path

from vlbrandphish.decision.domains import registered_domain
from vlbrandphish.inference.types import Metrics


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
        for name in (
            "stage1_siglip.yaml",
            "stage1_router_logo2k.yaml",
            "stage2_trisig_ot_safe.yaml",
        ):
            config = (root / "configs" / "training" / name).read_text(encoding="utf-8")
            self.assertIn("seed: 42", config)
            self.assertIn("deterministic: true", config)

if __name__ == "__main__":
    unittest.main()
