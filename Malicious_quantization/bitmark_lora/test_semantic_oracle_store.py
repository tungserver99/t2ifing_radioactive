import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from PIL import Image
from semantic_oracle_store import OracleStore, load_source, verify_source_manifest, reject_overlap
from common import sha256, image_name


class OracleStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        (self.source/'config.json').write_text('{}')
        self.manifest = {'source_run': str(self.source.resolve()), 'files': {'config.json': sha256(self.source/'config.json')}, 'external_files': {}}
    def tearDown(self):
        self.tmp.cleanup()
    def store(self):
        return OracleStore(self.root/'output', {'target_shard_size': 1}, self.manifest, [0, 1], {'version': 'test'})
    def commit(self, store, index=0, image_id=0):
        Image.new('RGB', (2, 2)).save(store.output/'oracle_images'/image_name(image_id))
        store.commit_batch(index, [image_id], {f'delta_{image_id}': torch.zeros(4, 1, 2, 2, dtype=torch.float16), f'final_latent_{image_id}': torch.zeros(1, 1, 2, 2)}, [{'image_id': image_id, 'float_feasible': image_id == 0}], [])
    def test_snapshot_and_resume_never_rewrites(self):
        before = (self.source/'config.json').read_bytes()
        store = self.store()
        self.commit(store)
        marker = store.output/'batches/batch_00000/complete.json'
        stamp = marker.stat().st_mtime_ns
        resumed = self.store()
        self.assertTrue(resumed.has_batch(0, [0]))
        self.assertEqual(stamp, marker.stat().st_mtime_ns)
        self.assertEqual(before, (self.source/'config.json').read_bytes())
        verify_source_manifest(self.manifest)
        (self.source/'config.json').write_text('{"changed":true}')
        with self.assertRaises(RuntimeError): self.store()
    def test_interrupted_identity_initialization_resumes(self):
        output = self.root/'output'; output.mkdir()
        config_path = output/'config.json'; config_path.write_text(json.dumps({'target_shard_size': 1}))
        stamp = config_path.stat().st_mtime_ns
        (output/'source_manifest.json.interrupted.tmp').write_text('partial')
        store = self.store()
        self.assertEqual(stamp, config_path.stat().st_mtime_ns)
        self.assertTrue((store.output/'oracle_targets/protocol.json').is_file())
    def test_incomplete_recovery_and_export_retry(self):
        store = self.store()
        folder = store.output/'batches/batch_00000'
        folder.mkdir(parents=True)
        (folder/'metrics.json').write_text('interrupted')
        self.assertFalse(store.has_batch(0, [0]))
        self.commit(store)
        self.commit(store, 1, 1)
        store.export_targets()
        marker = store.output/'oracle_targets/complete.json'
        stamp = marker.stat().st_mtime_ns
        store.export_targets()
        self.assertEqual(stamp, marker.stat().st_mtime_ns)
        self.assertEqual(json.loads((store.output/'oracle_targets/shard_00000.json').read_text())['sample_ids'], [0])
    def test_corrupt_committed_png_rejected(self):
        store = self.store(); self.commit(store)
        (store.output/'oracle_images'/image_name(0)).write_bytes(b'broken')
        with self.assertRaises(RuntimeError): store.has_batch(0, [0])
    def test_nonfinite_and_overlapping_outputs(self):
        store = self.store()
        Image.new('RGB', (2, 2)).save(store.output/'oracle_images'/image_name(0))
        with self.assertRaises(ValueError):
            store.commit_batch(0, [0], {'delta_0': torch.full((4, 1, 2, 2), float('nan')), 'final_latent_0': torch.zeros(1, 1, 2, 2)}, [{'image_id': 0, 'float_feasible': False}], [])
        for output in [self.source, self.source/'oracle', self.root]:
            with self.assertRaises(ValueError): reject_overlap(output, self.source)
    def test_unrelated_output_and_incompatible_source(self):
        output = self.root/'output'; output.mkdir(); (output/'other').write_text('x')
        with self.assertRaises(ValueError): self.store()
        with self.assertRaises(ValueError): load_source(self.source)
    def test_load_source_read_only_and_optional_replay_validation(self):
        from semantic_trajectory import digest
        def write(name, value):
            path = self.source/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value)); return sha256(path)
        config = {'teacher_objective': 'semantic_sign_v2_trajectory', 'num_bits': 32, 'analytic_alpha': .01, 'inference_steps': 50, 'tail_unroll_calls': 4}
        write('config.json', config); prompt_hash = write('prompts.json', {'train': [{'text': 'test'}]})
        model = self.root/'model'; (model/'scheduler').mkdir(parents=True)
        (model/'scheduler/scheduler_config.json').write_text('{}')
        identity = {'scheduler/scheduler_config.json': sha256(model/'scheduler/scheduler_config.json')}
        write('assets.json', {'teacher_clean': str(model), 'teacher_clean_hashes': identity, 'prompt_sha256': prompt_hash})
        write('carrier/key.json', {}); write('carrier/complete.json', {'key.json': sha256(self.source/'carrier/key.json')})
        protocol = {'teacher_objective': config['teacher_objective'], 'tail_unroll_calls': 4, 'config_sha256': sha256(self.source/'config.json'), 'prompts_sha256': prompt_hash, 'clean_model_hash': digest(identity), 'saved_scheduler_sha256': identity['scheduler/scheduler_config.json'], 'native': {'num_inference_steps': 50}}
        write('teacher_trace/protocol.json', protocol); write('teacher_trace/index.json', {})
        write('teacher_trace/complete.json', {'files': {}})
        image_folder = self.source/'images/teacher_clean_train'; image_folder.mkdir(parents=True)
        Image.new('RGB', (2, 2)).save(image_folder/image_name(0))
        write('images/teacher_clean_train/image_hashes.json', [{'file_name': image_name(0), 'sha256': sha256(image_folder/image_name(0))}])
        write('images/teacher_clean_train/generation.json', {'model_hashes': identity, 'prompts_sha256': prompt_hash, 'steps': 50, 'split': 'train', 'native': True})
        (image_folder/'metadata.jsonl').write_text('{}\n')
        write('scores/teacher_clean_train.json', []); write('scores/teacher_clean_train.provenance.json', {})
        carrier = SimpleNamespace(signs=torch.ones(32), carrier_version='semantic_sign_v1')
        reader = SimpleNamespace(index={'0': 'shard.json'}, protocol=protocol)
        write('teacher_trace/shard.json', {'protocol_sha256': sha256(self.source/'teacher_trace/protocol.json'), 'tensor_sha256': sha256(self.source/'config.json'), 'records': {'0': {'metadata': {'png_sha256': sha256(image_folder/image_name(0))}}}})
        (self.source/'teacher_trace/shard.safetensors').write_bytes((self.source/'config.json').read_bytes())
        write('teacher_trace/complete.json', {'files': {name: sha256(self.source/'teacher_trace'/name) for name in ['protocol.json', 'index.json', 'shard.json', 'shard.safetensors']}})
        before = {str(p): p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        with patch('semantic_assets.snapshot_identity', return_value=identity), patch('semantic_carrier.SemanticCarrier.load', return_value=carrier), patch('semantic_trajectory.TraceReader', return_value=reader), patch('semantic_report.score_rows', return_value=[{'image_id': 0}]):
            result = load_source(self.source)
            self.assertEqual(before, {str(p): p.read_bytes() for p in self.source.rglob('*') if p.is_file()})
            self.assertIn('images/teacher_clean_train/metadata.jsonl', result['manifest']['files'])
            write('teacher_trace/replay.json', {'passed': False})
            with self.assertRaises(RuntimeError): load_source(self.source)


if __name__ == '__main__': unittest.main()
