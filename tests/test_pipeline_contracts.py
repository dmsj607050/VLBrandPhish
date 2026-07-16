import tempfile
import unittest
from pathlib import Path

from vlbrandphish.pipeline.contracts import DetectionSample, read_detection_top3_csv, write_detection_top3_csv
from vlbrandphish.pipeline.decision import decide_with_threshold_and_url


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

    def test_decision_respects_threshold(self) -> None:
        result = decide_with_threshold_and_url(
            score=0.59,
            threshold=0.60,
            screenshot_url="https://login.example.test",
            official_domains=[],
        )
        self.assertFalse(result.pred_is_phishing)
        self.assertEqual(result.reason, "score_below_threshold")


if __name__ == "__main__":
    unittest.main()
