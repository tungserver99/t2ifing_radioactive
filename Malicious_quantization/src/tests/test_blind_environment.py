import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_blind_environment as setup


class EnvironmentTests(unittest.TestCase):
    def test_requirement_check_preserves_cuda_local_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "requirements.txt"
            req.write_text("--extra-index-url https://example.invalid\ntorch==2.7.1+cu128\ndiffusers==0.35.1\n")
            def version(name):
                if name == "diffusers":
                    raise metadata.PackageNotFoundError(name)
                return "2.7.1+cpu"
            with patch.object(setup.metadata, "version", side_effect=version):
                problems = setup.missing_requirements(req)
            self.assertEqual(len(problems), 2)
            self.assertIn("cu128", problems[0])
            self.assertIn("missing", problems[1])

    def test_current_install_uses_exact_interpreter_and_no_user_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "requirements.txt"
            req.write_text("diffusers==0.35.1\n")
            with patch.object(setup, "missing_requirements", side_effect=[["diffusers missing"], []]), \
                    patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
                setup.install_dependencies(req)
            calls = [c.args[0] for c in run.call_args_list]
            install = next(c for c in calls if "install" in c)
            self.assertEqual(install[:4], [sys.executable, "-m", "pip", "install"])
            self.assertIn("--no-user", install)
            self.assertNotIn("--break-system-packages", install)
            self.assertTrue(all(c[0] == sys.executable for c in calls))

    def test_ready_environment_does_not_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "requirements.txt"
            req.write_text("")
            with patch.object(setup, "missing_requirements", return_value=[]), \
                    patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
                setup.install_dependencies(req)
            self.assertFalse(any("install" in c.args[0] for c in run.call_args_list))

    def test_conda_mode_creates_and_enters_requested_env(self):
        calls = []
        def run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"envs": ["/opt/conda", "/opt/conda/envs/py"]}))
        argv = ["setup", "--conda", "/opt/conda/bin/conda", "--requirements", "req.txt", "--launcher", "run.sh",
                "--", "--output", "folder with spaces"]
        with patch.object(sys, "argv", argv), patch.object(setup.subprocess, "run", side_effect=run):
            with self.assertRaises(SystemExit) as result:
                setup.main()
        self.assertEqual(result.exception.code, 0)
        self.assertTrue(any(c[1:5] == ["create", "--yes", "--name", "wmq"] for c in calls))
        launch = calls[-1]
        self.assertEqual(launch[1:5], ["run", "--no-capture-output", "-n", "wmq"])
        self.assertEqual(launch[-3:], ["--", "--output", "folder with spaces"])

    def test_existing_requested_environment_is_not_recreated(self):
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps({"envs": ["/opt/conda/envs/wmq"]}))
        with patch.object(setup.subprocess, "run", return_value=result):
            self.assertEqual(setup.conda_prefix("conda", "wmq"), "/opt/conda/envs/wmq")
        with self.assertRaises(ValueError):
            setup.conda_prefix("conda", "base")

    def test_current_mode_relaunch_preserves_prefix_and_arguments(self):
        argv = ["setup", "--requirements", "req.txt", "--launcher", "run.sh", "--", "--bits", "4"]
        with patch.object(sys, "argv", argv), patch.object(setup, "install_dependencies") as install, \
                patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            with self.assertRaises(SystemExit) as result:
                setup.main()
        self.assertEqual(result.exception.code, 0)
        install.assert_called_once_with("req.txt")
        self.assertEqual(run.call_args.args[0], ["bash", "run.sh", "--bits", "4"])
        self.assertEqual(run.call_args.kwargs["env"]["WMQ_EXPECTED_PREFIX"], str(Path(sys.prefix).absolute()))
        self.assertEqual(run.call_args.kwargs["env"]["WMQ_BOOTSTRAP_READY"], "1")


if __name__ == "__main__":
    unittest.main()
