"""Regression checks for frozen paper runs; no model dependencies required."""
import argparse
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from experiments._shared.e2e import add_e2e_arguments, execute_experiment
from experiments.overall_phishintention import run as pi_run


class PaperProtocolTests(unittest.TestCase):
    def test_prepare_then_resume_rejects_historical_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            parser = argparse.ArgumentParser()
            add_e2e_arguments(parser)
            args = parser.parse_args(["--prepare-only", "--out-dir", str(output)])
            records = [{"sample_key": "one", "image_path": "one.png",
                        "url": "https://example.test", "gt_is_phishing": "0", "gt_brand": ""}]
            execute_experiment(experiment_name="test", experiment_dir=directory,
                               records=records, args=args)
            summary_file = output / "experiment_summary.json"
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            self.assertEqual(summary["fixed_decision"],
                             {"image_weight": 0.4, "text_weight": 0.6, "threshold": 0.4})
            self.assertEqual(summary["inference_protocol"]["topk"], 1)
            args.resume = True
            execute_experiment(experiment_name="test", experiment_dir=directory,
                               records=records, args=args)
            for old in [dict(summary, fixed_decision={"image_weight": 0.8,
                                                     "text_weight": 0.2, "threshold": 0.62}),
                        dict(summary, inference_protocol=dict(summary["inference_protocol"], topk=3)),
                        dict(summary, inference_protocol=dict(summary["inference_protocol"], max_logo_per_brand=1))]:
                summary_file.write_text(json.dumps(old), encoding="utf-8")
                previous = summary_file.read_bytes()
                with self.assertRaisesRegex(ValueError, "different or unrecorded"):
                    execute_experiment(experiment_name="test", experiment_dir=directory,
                                       records=records, args=args)
                self.assertEqual(summary_file.read_bytes(), previous)

    def test_phishintention_entrypoint_records_its_base_domain_policy(self):
        records = [{"sample_key": "one", "image_path": "one.png",
                    "url": "https://example.test", "gt_is_phishing": "0", "gt_brand": ""}]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(pi_run, "_load_phishintention_samples", return_value=records), \
                patch.object(pi_run, "_load_noncrp_keys", return_value=set()):
            pi_run.main(["--prepare-only", "--out-dir", directory])
            summary = json.loads((Path(directory) / "experiment_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["inference_protocol"]["url_policy_mode"], "phishintention")
            self.assertEqual(summary["inference_protocol"]["brand_text_mode"], "ckpt_text_proto_mean")
            self.assertEqual(summary["inference_protocol"]["topk"], 1)


if __name__ == "__main__":
    unittest.main()
