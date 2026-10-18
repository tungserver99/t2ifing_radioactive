"""Exercise default Bash dispatch without invoking downloads or GPU training."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class DefaultDispatchTests(unittest.TestCase):
    def test_root_launcher_is_regular_file_in_git(self):
        root = Path(__file__).resolve().parents[2]
        entry = root / 'run_blind_quantization.sh'
        self.assertTrue(entry.is_file())
        staged = subprocess.run(['git', 'ls-files', '--stage', '--', entry.name],
                                cwd=root, check=True, capture_output=True, text=True).stdout
        self.assertTrue(staged.startswith('100755 '), staged)

    def test_default_runs_both_in_order_and_stops_on_first_failure(self):
        git_bash = Path('C:/Program Files/Git/bin/bash.exe')
        shell = str(git_bash) if git_bash.exists() else shutil.which('bash')
        if not shell:
            self.skipTest('Bash unavailable')
        launcher = Path(__file__).resolve().parents[1] / 'run_blind_quantization.sh'
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            stub = folder / 'bash'
            stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$WMQ_TEST_LOG"\n'
                            'exit "${WMQ_TEST_EXIT:-0}"\n', encoding='utf-8', newline='\n')
            stub.chmod(0o755)
            log = folder / 'calls.txt'
            env = dict(os.environ, PATH=str(folder) + os.pathsep + os.environ['PATH'],
                       WMQ_TEST_LOG=log.as_posix(), WMQ_TEST_EXIT='0')
            shim_path = '$(cygpath -u "$1")' if os.name == 'nt' else '$1'
            command = [shell, '-c', f'export PATH="{shim_path}:$PATH"; exec /bin/bash "$2"',
                       'dispatch-test', folder.as_posix(), launcher.as_posix()]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = log.read_text().splitlines()
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[0].endswith('--profile transfer'), calls)
            self.assertTrue(calls[1].endswith('--watermark sleepermark'), calls)
            log.write_text('')
            env['WMQ_TEST_EXIT'] = '7'
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(len(log.read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
