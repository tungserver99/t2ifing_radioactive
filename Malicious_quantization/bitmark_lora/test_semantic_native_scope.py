"""PEFT suffix compaction must preserve physical adapter scope."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from semantic_teacher_train import adapter_targets


class AdapterScopeTests(unittest.TestCase):
    def model(self):
        model=torch.nn.Module()
        for name in ('block1','block2'):
            block=torch.nn.Module();block.to_q=torch.nn.Linear(2,2)
            block.conv=torch.nn.Conv2d(2,2,1);setattr(model,name,block)
        model.grouped=torch.nn.Conv2d(2,2,1,groups=2)
        return model

    @staticmethod
    def match(config,key):
        return any(key==suffix or key.endswith('.'+suffix) for suffix in config.target_modules)

    def test_compact_and_full_names_have_same_physical_scope(self):
        from semantic_sign_native_margin_refine import validate_adapter_scope
        model=self.model()
        full=SimpleNamespace(target_modules=adapter_targets(model))
        compact=SimpleNamespace(target_modules={'to_q','conv'})
        self.assertEqual(validate_adapter_scope(model,full,self.match),validate_adapter_scope(model,compact,self.match))

    def test_missing_and_grouped_convolution_rejected(self):
        from semantic_sign_native_margin_refine import validate_adapter_scope
        with self.assertRaisesRegex(RuntimeError,'missing=.*conv'):
            validate_adapter_scope(self.model(),SimpleNamespace(target_modules={'to_q'}),self.match)
        with self.assertRaisesRegex(RuntimeError,'extra=.*grouped'):
            validate_adapter_scope(self.model(),SimpleNamespace(target_modules={'to_q','conv','grouped'}),self.match)

    def test_pretraining_code_upgrade_preserves_history_and_rejects_training(self):
        from semantic_native_lineage import freeze_code_identity
        from common import write_json,read_json
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary);old={'git_head':None,'files':{'trainer.py':'old'}};new={**old,'files':{'trainer.py':'fixed'}}
            write_json(run/'code_manifest.json',old)
            with patch('semantic_native_lineage.code_identity',return_value=new):freeze_code_identity(run)
            self.assertEqual(read_json(run/'code_manifest.json'),new)
            history=list((run/'code_updates').glob('*/code_manifest.json'))
            self.assertEqual(len(history),1);self.assertEqual(read_json(history[0]),old)
            (run/'teacher_fp/native_margin_refine/checkpoint-1').mkdir(parents=True)
            with patch('semantic_native_lineage.code_identity',return_value=old):
                with self.assertRaisesRegex(RuntimeError,'training artifacts'):freeze_code_identity(run)

    def test_pretraining_protocol_hash_updated_but_other_inputs_immutable(self):
        from semantic_native_lineage import freeze_code_identity
        from semantic_sign_native_margin_refine import training_protocol
        from common import write_json,read_json,sha256
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)
            for name in ('config.json','assets.json','prompts.json','source_lineage.json',
                         'carrier/complete.json','carrier/key.json','images/teacher_clean_train/image_hashes.json'):
                write_json(run/name,{})
            old={'git_head':None,'files':{'trainer.py':'old'}};new={**old,'files':{'trainer.py':'fixed'}}
            write_json(run/'code_manifest.json',old)
            protocol_path=run/'teacher_fp/native_margin_refine/protocol.json'
            write_json(protocol_path,training_protocol(run))
            with patch('semantic_native_lineage.code_identity',return_value=new):freeze_code_identity(run)
            self.assertEqual(read_json(protocol_path),training_protocol(run))
            self.assertEqual(read_json(protocol_path)['code_manifest_json_sha256'],sha256(run/'code_manifest.json'))
            write_json(run/'config.json',{'changed':True})
            with patch('semantic_native_lineage.code_identity',return_value=old):
                with self.assertRaisesRegex(RuntimeError,'beyond code identity'):freeze_code_identity(run)
            self.assertEqual(read_json(run/'code_manifest.json'),new)

if __name__=='__main__':unittest.main()
