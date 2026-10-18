import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from wmq_bundle import bundle_result


ROOT = Path(__file__).resolve().parents[1]


class BundleTests(unittest.TestCase):
    def test_source_is_canonical_in_src_directory(self):
        for name in ("wmq_blind.py", "run_blind_suite.sh", "requirements-wmq.txt",
                     "tests/test_wmq_blind.py"):
            self.assertTrue((ROOT / name).is_file())
        self.assertFalse((ROOT / "BUNDLE_MANIFEST.json").exists())

    def test_result_bundle_keeps_original_and_compact_branch_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"; run.mkdir()
            (run / "manifest.json").write_text("{}")
            (run / "report.json").write_text('{"value": 1}')
            branch = run / "branches" / "method"; branch.mkdir(parents=True)
            (branch / "selection.json").write_text('{"selected": true}')
            (branch / "updates.json").write_text('{"large": true}')
            tradeoff = run / "tradeoff"; tradeoff.mkdir()
            (tradeoff / "points.csv").write_text("x\n1\n")
            destination = bundle_result(run)
            self.assertTrue((destination / "report.json").is_file())
            self.assertTrue((destination / "branches/method/selection.json").is_file())
            self.assertFalse((destination / "branches/method/updates.json").exists())
            self.assertTrue((run / "report.json").is_file())

    def test_suite_cli_defaults_match_documented_one_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            subprocess.run([sys.executable, str(ROOT / "run_blind_suite.py"),
                "--root", str(root), "--plan-only"], check=True, capture_output=True, text=True)
            plan = json.loads((root / "suite_plan.json").read_text())
            self.assertEqual(len(plan["runs"]), 3)
            self.assertEqual(plan["execution_mode"], "plan_only")
            self.assertEqual(plan["suite_status"], "not_executed")
            self.assertFalse(plan["results_available"])
            self.assertTrue(all(run["status"] == "planned_not_executed" for run in plan["runs"]))
            self.assertEqual({run["seed"] for run in plan["runs"]}, {3407})
            self.assertEqual({run["preserve_weight"] for run in plan["runs"]}, {.5, 2., 8.})
            for run in plan["runs"]:
                command = run["command"]
                indices = [i for i, value in enumerate(command) if value == "--min-ssim"]
                self.assertEqual(len(indices), 1)
                self.assertEqual(float(command[indices[0] + 1]), .8)
                for option in ("--natural-preservation", "--preserve-weight",
                               "--qat-semantic-preserve-weight"):
                    self.assertEqual(command.count(option), 1)
                excluded = command.index("--exclude-method-bits")
                self.assertEqual(set(command[excluded + 1:excluded + 4]),
                    {"natural_residual:8", "reconstruction:4", "fixed_ptq:4"})


if __name__ == "__main__":
    unittest.main()
