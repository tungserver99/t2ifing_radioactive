"""Regression checks for the one-command Stable Signature blind pipeline."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_owner_evaluator as owner_asset
from evaluate_blind_watermark import file_sha256, wilson_interval


ROOT = Path(__file__).resolve().parents[1]


class OneCommandTests(unittest.TestCase):
    def test_public_key_and_w4_owner_evaluation_are_wired_after_attack(self):
        source = (ROOT / "run_blind_quantization.sh").read_text(encoding="utf-8")
        self.assertIn('run_blind_suite.py', source)
        self.assertIn('"${1:-}" == "--direct"', source)
        self.assertFalse((ROOT / "run_blind_suite.sh").exists())
        self.assertIn('prepare_blind_environment.py', source)
        self.assertIn('${WMQ_ENV_MODE:-auto}', source)
        self.assertIn('WMQ_EXPECTED_PREFIX', source)
        self.assertIn('${WMQ_CONDA_ENV:-wmq}', source)
        self.assertIn('attack+=(--bits 4)', source)
        self.assertNotIn('attack+=(--bits 8 4)', source)
        self.assertNotIn('--exclude-method-bits natural_residual:8 reconstruction:4 fixed_ptq:4', source)
        self.assertIn('--steps 2000 --qat-steps 2000 --ft-steps 2000', source)
        self.assertIn('natural_teacher_rounding', source)
        self.assertIn('111010110101000001010111010011010100010000100111', source)
        self.assertLess(source.index('"${attack[@]}"'), source.index('evaluate_blind_watermark.py'))
        self.assertLess(source.index('selection_frozen.json'), source.index('evaluate_blind_watermark.py'))
        self.assertIn('--expected-extractor-sha256', source)
        self.assertIn('prepare_natural_images.py', source)
        self.assertIn('attack+=(--natural-images "$NATURAL_POOL")', source)
        self.assertIn('${WMQ_MODEL_ONLY:-0}', source)
        self.assertIn('${WMQ_HEAVY_ROOT:-$PROJECT_ROOT/output_artifacts}', source)
        self.assertNotIn('RUN_ROOT=', source)
        self.assertNotIn('$SCRIPT_DIR/wmq_runs', source)
        self.assertIn('$HEAVY_ROOT/models', source)
        self.assertIn('$HEAVY_ROOT/checkpoints', source)
        self.assertIn('$HEAVY_ROOT/images', source)
        self.assertIn('$HEAVY_ROOT/datasets', source)

    def test_owner_extractor_is_verified_on_download_and_reuse(self):
        payload = b"audited extractor bytes"
        expected = hashlib.sha256(payload).hexdigest()
        def download(url, destination):
            Path(destination).write_bytes(payload)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(owner_asset, "EXPECTED_EXTRACTOR_SHA256", expected), \
                patch("urllib.request.urlretrieve", side_effect=download):
            output = Path(folder) / "extractor.pt"
            with patch.object(sys, "argv", ["prepare_owner_evaluator", "--output", str(output)]):
                owner_asset.main()
                owner_asset.main()
            self.assertEqual(file_sha256(output), expected)
            output.write_bytes(b"changed")
            with patch.object(sys, "argv", ["prepare_owner_evaluator", "--output", str(output)]):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    owner_asset.main()

    def test_wilson_interval_does_not_report_false_precision_at_n100(self):
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, .4038315304, places=8)
        self.assertAlmostEqual(high, .5961684696, places=8)


if __name__ == "__main__":
    unittest.main()
