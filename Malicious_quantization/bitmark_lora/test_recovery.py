"""CPU recovery integration tests with small real safetensors checkpoints."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import read_json,write_json,sha256
from student import branches,training_command,checkpoint_hashes


class RecoveryTests(unittest.TestCase):
    def test_recover_and_reject_incomplete_or_changed_weights(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest('Needs torch/safetensors; run using the ML test environment')
        from recover_full_export import recover_branch
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run=Path(tmp); base=run/'base'; out=run/'students/full_wm'
            c={'ranks':[16],'adapter_scopes':['attention'],'lora_only':False,'paper_full_ema':False,
               'train_clean_controls':False,'train_n':1,'max_train_steps':2,'checkpoint_steps':[1,2],
               'learning_rate':1e-5,'batch_size':1,'precision':'no','seed':0}
            assets={'student':str(base)}
            write_json(run/'config.json',c); write_json(run/'assets.json',assets)
            image=run/'images/teacher_wm_train/image.png'; image.parent.mkdir(parents=True); image.write_bytes(b'fixture')
            images=[{'file_name':image.name,'text':'test','sha256':sha256(image)}]
            write_json(image.parent/'image_hashes.json',images)
            branch=next(b for b in branches(c) if b['name']=='full_wm')
            write_json(out/'protocol.json',{'branch':branch,'command':training_command(c,assets,run,branch),'training_images':images})
            (base/'unet').mkdir(parents=True)
            save_file({'weight':torch.ones(2,2)},str(base/'unet/diffusion_pytorch_model.safetensors'))
            for step in [1,2]:
                folder=out/f'snapshot-{step}/unet'; folder.mkdir(parents=True)
                write_json(folder/'config.json',{'fixture':True})
                save_file({'weight':torch.ones(2,2)+step},str(folder/'diffusion_pytorch_model.safetensors'))
            final=out/'snapshot-2/unet/diffusion_pytorch_model.safetensors'
            valid=final.read_bytes(); final.unlink()
            with self.assertRaises(FileNotFoundError): recover_branch(run,'full_wm')
            self.assertFalse((out/'complete.json').exists())
            save_file({'weight':torch.full((2,2),float('nan'))},str(final))
            with self.assertRaises(RuntimeError): recover_branch(run,'full_wm')
            final.write_bytes(valid)
            with patch('subprocess.run',side_effect=AssertionError('Recovery must never launch training')):
                recover_branch(run,'full_wm'); recover_branch(run,'full_wm')
            self.assertEqual(sha256(out/'unet/diffusion_pytorch_model.safetensors'),sha256(final))
            completed=read_json(out/'complete.json')
            self.assertEqual(completed['recovered_from_snapshot'],2)
            self.assertEqual(completed['checkpoints'],checkpoint_hashes(out))
            final.write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError): recover_branch(run,'full_wm')


if __name__=='__main__': unittest.main()
