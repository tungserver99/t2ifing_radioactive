import copy
import tempfile
import unittest
from pathlib import Path
import torch

class ResumeTests(unittest.TestCase):
    def test_full_state_matches_uninterrupted_adamw_and_rng(self):
        from semantic_checkpoint import save_training_state,restore_training_state
        torch.manual_seed(5)
        model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters(),lr=.01)
        rng=torch.Generator().manual_seed(8)
        def update():
            opt.zero_grad();x=torch.randn((3,2),generator=rng)
            model(x).square().mean().backward();opt.step()
        update();update()
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);identity={'config':'same'}
            save_training_state(folder,model,opt,None,rng,[3,1,2,0],2,2,identity)
            update();expected=copy.deepcopy(model.state_dict());state=copy.deepcopy(opt.state_dict())
            restored=torch.nn.Linear(2,1);other=torch.optim.AdamW(restored.parameters(),lr=.01)
            g=torch.Generator().manual_seed(999)
            metadata=restore_training_state(folder,restored,other,None,g,identity)
            self.assertEqual((metadata['step'],metadata['order'],metadata['offset']),(2,[3,1,2,0],2))
            other.zero_grad();x=torch.randn((3,2),generator=g);restored(x).square().mean().backward();other.step()
            for k,v in expected.items():torch.testing.assert_close(restored.state_dict()[k],v,rtol=0,atol=0)
            for k,v in state['state'].items():
                for n,t in v.items():torch.testing.assert_close(other.state_dict()['state'][k][n],t,rtol=0,atol=0)
            (folder/'training_state.pt').write_bytes(b'broken')
            with self.assertRaisesRegex(RuntimeError,'changed'):
                restore_training_state(folder,restored,other,None,g,identity)

    def test_legacy_sampler_reconstruction_matches_original_draws(self):
        from semantic_checkpoint import advance_sampler
        rng=torch.Generator().manual_seed(2);order=[];offset=0
        for step in range(7):
            if offset+4>len(order):order=torch.randperm(13,generator=rng).tolist();offset=0
            offset+=4;torch.randn((4,4,8,8),generator=rng);torch.randint(0,1000,(4,),generator=rng)
        other=torch.Generator().manual_seed(2)
        got_order,got_offset=advance_sampler(other,13,4,(4,4,8,8),1000,7)
        self.assertEqual((got_order,got_offset),(order,offset))
        self.assertTrue(torch.equal(rng.get_state(),other.get_state()))

    def test_continuation_reuses_verified_prep_and_preserves_source(self):
        from semantic_checkpoint import continuation_config,prepare_continuation
        from semantic_report import freeze_null
        from common import write_json,sha256,image_name
        from run import parse_args,config_from_args
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'old';source.mkdir();dest=Path(d)/'new';dest.mkdir()
            c=config_from_args(parse_args(['--teacher-objective','threshold_cfg_v1','--train-n','4','--calibration-n','100','--validation-n','4',
                  '--validation-generation-n','2'],method='semantic_fp'))
            c.pop('teacher_precision');c.pop('teacher_objective');c.pop('teacher_kappas')
            c.update(semantic_shift=.5,semantic_shift_sweep=None,lambda_preserve=1.,lambda_code=.1,lambda_perp=.01,lambda_cycle=.05)
            c['teacher_checkpoint_every']=500
            write_json(source/'config.json',c)
            write_json(source/'prompts.json',{'train':[{'text':'test'}]*4})
            write_json(source/'assets.json',{'prompt_sha256':sha256(source/'prompts.json')})
            carrier=source/'carrier';carrier.mkdir()
            torch.save({'mean':torch.zeros(2),'basis':torch.eye(2)},carrier/'pca.pt')
            write_json(carrier/'key.json',{'selected_pc_indices':[0,1],'sigmas':[1.,1.],'signs':[1.,-1.],
                                         'basis_sha256':sha256(carrier/'pca.pt')})
            write_json(carrier/'complete.json',{n:sha256(carrier/n) for n in ('pca.pt','key.json')})
            scores=[]
            for label,count in (('teacher_clean_train',4),('teacher_clean_validation',4),('teacher_clean_calibration',100)):
                folder=source/'images'/label;folder.mkdir(parents=True);entries=[]
                for i in range(count):
                    path=folder/image_name(i);path.write_bytes(b'image')
                    entries.append({'file_name':path.name,'sha256':sha256(path)})
                    if label=='teacher_clean_calibration':
                        scores.append({'file':path.name,'sha256':sha256(path),'score':i/2.,'coordinates':[float(i),0.]})
                write_json(folder/'image_hashes.json',entries)
            write_json(source/'scores/teacher_clean_calibration.json',scores)
            freeze_null(source,'teacher')
            candidate=source/'teacher_fp/candidates/shift_0.5';checkpoint=candidate/'checkpoint-1000'
            checkpoint.mkdir(parents=True)
            (checkpoint/'adapter_model.safetensors').write_bytes(b'adapter')
            write_json(checkpoint/'adapter_config.json',{})
            write_json(candidate/'protocol.json',{'config_sha256':sha256(source/'config.json'),
                'assets_sha256':sha256(source/'assets.json'),'carrier_sha256':sha256(carrier/'complete.json')})
            (candidate/'losses.jsonl').write_text(''.join('{"step":'+str(i)+'}\n' for i in range(1,1026)))
            before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            config=continuation_config(source)
            self.assertEqual(config['continuation']['checkpoints']['shift_0.5']['step'],1000)
            self.assertEqual(config['teacher_precision'],'bf16')
            self.assertEqual(config['teacher_train_steps'],3000)
            write_json(dest/'config.json',config);prepare_continuation(dest,config)
            self.assertEqual(before,{str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()})
            self.assertEqual(sha256(source/'carrier/complete.json'),sha256(dest/'carrier/complete.json'))
            self.assertEqual(sha256(source/'result/threshold_teacher.json'),sha256(dest/'result/threshold_teacher.json'))
            self.assertTrue((dest/'images/teacher_clean_train'/image_name(0)).exists())
            from semantic_checkpoint import reuse_preparation_config
            fresh=config_from_args(parse_args(['--teacher-objective','threshold_cfg_v1'],method='semantic_fp'))
            reused=reuse_preparation_config(fresh,source)
            self.assertEqual(reused['teacher_objective'],'threshold_cfg_v1')
            self.assertNotIn('continuation',reused)
            self.assertNotIn('semantic_shift',reused)
            other=Path(d)/'reuse';other.mkdir();write_json(other/'config.json',reused)
            prepare_continuation(other,reused)
            self.assertTrue((other/'preparation_reuse.json').exists())
            self.assertFalse((other/'teacher_fp').exists())
            self.assertEqual(sha256(other/'carrier/complete.json'),sha256(source/'carrier/complete.json'))
            self.assertEqual(before,{str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()})

    def test_incomplete_checkpoint_ignored_and_changed_identity_rejected(self):
        from semantic_checkpoint import latest_full_checkpoint,save_training_state,restore_training_state
        model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters());rng=torch.Generator()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);complete=root/'checkpoint-100';partial=root/'checkpoint-200'
            save_training_state(complete,model,opt,None,rng,[],0,100,{'run':1})
            partial.mkdir();(partial/'training_state.partial').write_bytes(b'incomplete')
            self.assertEqual(latest_full_checkpoint(root),complete)
            with self.assertRaisesRegex(RuntimeError,'identity changed'):
                restore_training_state(complete,model,opt,None,rng,{'run':2})

    def test_loss_prefix_tampering_is_rejected(self):
        import hashlib
        from semantic_checkpoint import align_loss_log
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'losses.jsonl';raw=b'{"step":1,"loss":0.1}\n'
            p.write_bytes(raw+b'{"step":2,"loss":0.2}\n')
            expected=hashlib.sha256(raw).hexdigest()
            align_loss_log(p,1,expected_hash=expected)
            p.write_bytes(b'{"step":1,"loss":9.9}\n')
            with self.assertRaisesRegex(RuntimeError,'prefix changed'):
                align_loss_log(p,1,expected_hash=expected)

    def test_loss_log_rolls_back_to_checkpoint_without_losing_archive(self):
        from semantic_checkpoint import align_loss_log
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'losses.jsonl';original='{"step":1}\n{"step":2}\n{"step":3}\n';p.write_text(original)
            align_loss_log(p,2)
            self.assertEqual(p.read_text(),'{"step":1}\n{"step":2}\n')
            archives=list(p.parent.glob('losses.before_resume.*.jsonl'))
            self.assertEqual(len(archives),1);self.assertEqual(archives[0].read_text(),original)

if __name__=='__main__':unittest.main()
