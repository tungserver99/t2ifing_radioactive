"""Manifest creation on PBS needs neither a Git executable nor a Git checkout."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from semantic_native_lineage import code_identity


class NativeManifestTests(unittest.TestCase):
    def test_manifest_without_git_executable_or_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);harness=root/'Malicious_quantization/bitmark_lora'
            harness.mkdir(parents=True)
            for name in ('example.py','vendor/provenance.json'):
                path=harness/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('example')
            for name in ('main_native_margin.py','run_rssf_native_margin.sh','GPT_guide/rssf_native_margin_refinement_student_distill_codex_spec.md'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('example')
            with patch('semantic_native_lineage.HERE',harness),patch('subprocess.check_output',side_effect=FileNotFoundError('git')):
                first=code_identity()
                self.assertIsNone(first['git_head'])
                self.assertEqual(first,code_identity())
                (harness/'example.py').write_text('changed')
                self.assertNotEqual(first['files'],code_identity()['files'])

    def test_head_metadata_without_executable(self):
        from semantic_native_lineage import repository_head
        sha='a'*40
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);git=root/'.git';git.mkdir()
            (git/'HEAD').write_text(sha+'\n')
            self.assertEqual(repository_head(root),sha)
            (git/'HEAD').write_text('ref: refs/heads/main\n')
            (git/'refs/heads').mkdir(parents=True);(git/'refs/heads/main').write_text(sha)
            self.assertEqual(repository_head(root),sha)
            (git/'refs/heads/main').unlink()
            (git/'packed-refs').write_text('# pack-refs\n'+sha+' refs/heads/main\n')
            self.assertEqual(repository_head(root),sha)
            (git/'HEAD').write_text('invalid metadata')
            self.assertIsNone(repository_head(root))

    def test_worktree_metadata_and_missing_head(self):
        from semantic_native_lineage import repository_head
        sha='b'*40
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);checkout=root/'checkout';checkout.mkdir()
            self.assertIsNone(repository_head(checkout))
            common=root/'repository';admin=common/'worktrees/job';admin.mkdir(parents=True)
            (checkout/'.git').write_text('gitdir: ../repository/worktrees/job\n')
            (admin/'commondir').write_text('../..\n');(admin/'HEAD').write_text('ref: refs/heads/main\n')
            (common/'packed-refs').write_text(sha+' refs/heads/main\n')
            self.assertEqual(repository_head(checkout),sha)

if __name__=='__main__':unittest.main()
