import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from inference.vlbrandphish.data.detection_csv import DetectionSample, read_detection_top3_csv, write_detection_top3_csv
from inference.vlbrandphish.data.testb_dataset import load_testb_samples
from inference.vlbrandphish.decision.policy import decide_with_threshold_and_url
from inference.vlbrandphish.recognition.features import precompute_query_features


class PipelineContractsTest(unittest.TestCase):
    def test_detection_csv_round_trip(self) -> None:
        sample = DetectionSample(
            image_path="example.png",
            url="https://login.example.test",
            sample_key="brand/example",
            gt_is_phishing=True,
            gt_brand="Example",
            bboxes=[(1.0, 2.0, 10.0, 12.0)],
            det_scores=[0.95],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.csv"
            write_detection_top3_csv([sample], str(path))
            loaded = read_detection_top3_csv(str(path))

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].sample_key, sample.sample_key)
        self.assertEqual(loaded[0].bboxes, sample.bboxes)

    def test_query_feature_precomputation_resolves_and_encodes_direct_image_path(self) -> None:
        class FakeEncoder:
            embed_dim = 2

            @staticmethod
            def encode_images(images, batch_size: int):
                self.assertEqual(len(images), 1)
                self.assertEqual(batch_size, 1)
                return torch.tensor([[0.6, 0.8]], dtype=torch.float32)

            @staticmethod
            def encode_query_images(images, batch_size: int):
                return torch.tensor([[0.6, 0.8]], dtype=torch.float32)

            @staticmethod
            def route_weights_from_query_features(features):
                return torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            Image.new("RGB", (16, 16), color="white").save(image_path)
            query = precompute_query_features(
                detections=[
                    DetectionSample(
                        image_path=str(image_path),
                        url="https://example.test",
                        sample_key="sample",
                        gt_is_phishing=False,
                        gt_brand="",
                        bboxes=[(0.0, 0.0, 8.0, 8.0)],
                        det_scores=[0.9],
                    )
                ],
                encoder=FakeEncoder(),
                benign_root=directory,
                phish_root=directory,
                progress=False,
            )

        self.assertEqual(query.query_mask.tolist(), [[1]])
        self.assertEqual(query.query_feat.shape, (1, 1, 2))
        self.assertAlmostEqual(float(query.query_feat[0, 0, 0]), 0.6)
        self.assertAlmostEqual(float(query.route_weights[0, 0, 2]), 0.5)
    def test_decision_respects_threshold(self) -> None:
        result = decide_with_threshold_and_url(
            score=0.59,
            threshold=0.60,
            screenshot_url="https://login.example.test",
            official_domains=[],
        )
        self.assertFalse(result.pred_is_phishing)
        self.assertEqual(result.reason, "score_below_threshold")


    def test_missing_or_empty_evaluation_roots_fail_fast(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_testb_samples("missing-benign", "missing-phish")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                load_testb_samples(directory, "")


if __name__ == "__main__":
    unittest.main()
