from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from vlbrandphish.decision.domains import registered_domain
from vlbrandphish.inference.cache import _load_query_cache, _save_query_cache
from vlbrandphish.inference.types import Metrics, QueryFeatureData


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

    def test_query_cache_round_trip_preserves_contract(self) -> None:
        query_data = QueryFeatureData(
            query_feat=np.ones((1, 2, 3), dtype=np.float32),
            route_weights=np.zeros((1, 2, 3), dtype=np.float32),
            query_mask=np.asarray([[1, 0]], dtype=np.uint8),
            y_true=np.asarray([1], dtype=np.uint8),
            gt_brand_keys=["example"],
            screenshot_domains=["example.com"],
            resolved_image_paths=["sample.png"],
            resolve_stats={"resolve_fail": 0},
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "query.npz"
            _save_query_cache(str(path), query_data)
            loaded = _load_query_cache(str(path))

        np.testing.assert_array_equal(loaded.query_mask, query_data.query_mask)
        np.testing.assert_allclose(loaded.query_feat, query_data.query_feat)
        self.assertEqual(loaded.gt_brand_keys, ["example"])


if __name__ == "__main__":
    unittest.main()
