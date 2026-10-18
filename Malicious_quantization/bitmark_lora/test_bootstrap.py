"""Check environment isolation and CLI routing without GPU/package downloads."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap_env as bootstrap
import common


class BootstrapTests(unittest.TestCase):
    def test_colab_defaults_are_local_even_when_source_is_on_drive(self):
        with patch.dict(os.environ, {'COLAB_RELEASE_TAG': 'test'}, clear=True), \
             patch.object(common, 'HERE', Path('/content/drive/MyDrive/source/bitmark_lora')):
            self.assertEqual(common.runtime_root(), Path('/content/bitmark_runtime').resolve())
            self.assertEqual(common.require_local_runtime_path(common.runtime_root()), common.runtime_root())
            with self.assertRaisesRegex(ValueError, 'local /content'):
                common.require_local_runtime_path('/content/drive/MyDrive/.venv')

    def test_normal_linux_paths_unchanged(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(common, 'HERE', Path('/workspace/bitmark_lora')):
            self.assertEqual(common.runtime_root(), Path('/workspace/bitmark_lora').resolve())

    def test_private_uv_ignores_inherited_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)
            binary=cache/'bootstrap'/('uv-'+bootstrap.UV_VERSION)/'bin/uv'
            binary.parent.mkdir(parents=True);binary.write_text('fixture')
            with patch.dict(os.environ, {'UV_INTERNAL_BIN':'/usr/bin/uv'}), \
                 patch.object(bootstrap.subprocess, 'run', return_value=subprocess.CompletedProcess([],0,'uv 0.9.3\n','')):
                command,env=bootstrap.private_uv(cache)
            self.assertEqual(command,[str(binary)])
            self.assertNotIn('UV_INTERNAL_BIN',env)

    def test_private_uv_rejects_wrong_version_after_install(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(bootstrap.subprocess, 'run', return_value=subprocess.CompletedProcess([],0,'uv 0.12.22\n','')):
            with self.assertRaisesRegex(RuntimeError, 'Unexpected private uv'):
                bootstrap.private_uv(Path(tmp))

    def test_t4_rejected_before_install(self):
        with patch.object(bootstrap.subprocess, 'run', return_value=subprocess.CompletedProcess([],0,'7.5\n','')):
            with self.assertRaisesRegex(RuntimeError, 'T4 is unsupported'):
                bootstrap.check_colab_gpu()

    def test_a100_precheck(self):
        with patch.object(bootstrap.subprocess, 'run', return_value=subprocess.CompletedProcess([],0,'8.0\n','')):
            bootstrap.check_colab_gpu()

    def test_refuses_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'valuable.txt').write_text('keep')
            with self.assertRaisesRegex(RuntimeError, 'non-venv'):
                bootstrap.ensure_environment(root, root / 'cache')
            self.assertEqual((root / 'valuable.txt').read_text(), 'keep')

    def test_refuses_wrong_existing_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'pyvenv.cfg').write_text('home = /example')
            with patch.object(bootstrap, 'python_version', return_value='3.12'):
                with self.assertRaisesRegex(RuntimeError, 'requires 3.11'):
                    bootstrap.ensure_environment(root, root / 'cache')

    def test_existing_venv_does_not_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'pyvenv.cfg').write_text('home = /example')
            with patch.object(bootstrap, 'python_version', return_value='3.11'), \
                 patch.object(bootstrap.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)), \
                 patch.object(bootstrap, 'private_uv') as uv:
                self.assertEqual(bootstrap.ensure_environment(root, root / 'cache'), root / 'bin/python')
                uv.assert_not_called()

    def test_pip_failure_propagates(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                'BITMARK_VENV': str(Path(tmp) / 'venv'), 'BITMARK_BOOTSTRAP_CACHE': str(Path(tmp) / 'cache')}):
            with patch.object(bootstrap, 'ensure_environment', return_value=Path('/private/bin/python')), \
                 patch.object(bootstrap.platform, 'system', return_value='Linux'), \
                 patch.object(bootstrap.platform, 'machine', return_value='x86_64'), \
                 patch.object(bootstrap.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'pip')), \
                 patch.object(bootstrap.os, 'execv') as execute:
                with self.assertRaises(subprocess.CalledProcessError):
                    bootstrap.main(['--setup-only'])
                execute.assert_not_called()

    @unittest.skipUnless(sys.platform == 'linux', 'Bash launcher targets Linux')
    def test_plan_flag_anywhere_does_not_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, BITMARK_PYTHON=sys.executable, BITMARK_VENV=str(Path(tmp) / 'venv'))
            p = subprocess.run(['bash', str(bootstrap.HERE / 'run.sh'), '--ranks', '4', '16', '--plan-only'],
                               env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('lora_r16_wm', p.stdout)
            self.assertFalse((Path(tmp) / 'venv').exists())


if __name__ == '__main__':
    unittest.main()
