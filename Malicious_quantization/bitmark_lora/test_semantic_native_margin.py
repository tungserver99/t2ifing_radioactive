"""Small CPU contracts; actual-model integration gates execute on the GPU server."""
import unittest
import torch
from semantic_sign import detection_count, sign_hinge_loss, is_sign
from semantic_native_rollout import rgb8_ste, capped_gradient, dynamic_sign_margin_loss

class NativeMarginMath(unittest.TestCase):
    def test_dynamic_selection_and_gradient(self):
        x = torch.linspace(-.4, .1, 32).reshape(1, 32).requires_grad_()
        loss, stats = dynamic_sign_margin_loss(x, torch.ones(32))
        self.assertEqual(stats['selected_indices'], [list(range(31, 5, -1))])
        expected = (.02-x[0, 6:]).clamp_min(0).square().sum()/26
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(x.grad[0, :6].count_nonzero().item(), 0)
        ties = torch.zeros(1, 32, requires_grad=True)
        _, s = dynamic_sign_margin_loss(ties, torch.ones(32))
        self.assertEqual(s['selected_indices'], [list(range(26))])
        self.assertEqual(dynamic_sign_margin_loss(torch.ones(2,32),torch.ones(32))[0].item(), 0)

    def test_rgb8_and_cap(self):
        x=torch.tensor([-.8,0.,.8],requires_grad=True)
        q=rgb8_ste(x)
        torch.testing.assert_close((q+1)*127.5, ((x+1)*127.5).round())
        q.sum().backward(); torch.testing.assert_close(x.grad,torch.ones_like(x))
        gm=[torch.tensor([3.,4.])]; gf=[-10*gm[0]]
        combined, stats=capped_gradient(gm,gf)
        self.assertLessEqual(stats['alpha_F']*stats['n_F'], .5*stats['n_M']+1e-10)
        self.assertGreaterEqual(torch.dot(gm[0],combined[0]).item(),12.5-1e-5)
        self.assertFalse(capped_gradient([gm[0]*0],gf)[1]['update_applied'])

    def test_historical_and_verifier(self):
        self.assertEqual(detection_count(32,.01),24)
        self.assertTrue(is_sign({'teacher_objective':'semantic_sign_v3_native_margin_refine'}))
        clean=torch.arange(32.).reshape(1,32); current=-clean
        _,old=sign_hinge_loss(clean,current,torch.ones(32),.02,26)
        _,new=dynamic_sign_margin_loss(current,torch.ones(32))
        self.assertNotEqual(old['selected_indices'].tolist(),new['selected_indices'])

class NativeMarginIntegrationContracts(unittest.TestCase):
    def test_teacher_selection_uses_d26_then_pixel_then_step(self):
        from semantic_native_campaign import choose_teacher
        def row(step,d26,pixel,tpr=.9,clip=.01,lpips=999):
            return {'step':step,'metrics':{'detection_rate':tpr,'D_26':d26},
                    'quality':{'clip_drop':clip,'paired_pixel_mse':pixel,'paired_lpips':lpips}}
        rows=[row(50,.1,.3),row(100,.1,.2),row(200,.1,.2),row(300,0.,0.,tpr=.89)]
        self.assertEqual(choose_teacher(rows)['step'],100)
        self.assertIsNone(choose_teacher([row(50,0.,0.,clip=.011)]))

    def test_microbatch_equivalence(self):
        from semantic_native_rollout import accumulate_gradients,capped_gradient
        def run(micro):
            p=torch.tensor([.1,-.3],requires_grad=True)
            gm=[torch.zeros_like(p)];gf=[torch.zeros_like(p)]
            x=torch.arange(8.).reshape(4,2)/10
            for start in range(0,4,micro):
                batch=x[start:start+micro]
                lm=((p-batch)**2).mean();lf=((2*p+batch)**2).mean()
                accumulate_gradients(lm,[p],gm,len(batch)/4)
                accumulate_gradients(lf,[p],gf,len(batch)/4)
            combined,stats=capped_gradient(gm,gf)
            return gm,gf,combined,stats
        a,b=run(1),run(2)
        for i in range(3):torch.testing.assert_close(a[i][0],b[i][0])
        self.assertAlmostEqual(a[3]['alpha_F'],b[3]['alpha_F'],places=7)

    def test_live_history_prefix_and_ten_tail_gradients(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from semantic_native_rollout import live_rollout
        class Scheduler:
            def __init__(self):
                self.timesteps=torch.arange(51);self.timesteps[2]=self.timesteps[1]
                self.counter=0;self.num_inference_steps=50;self.ets=[]
            def scale_model_input(self,x,t):return x
            def step(self,output,t,latent):
                self.ets.append(output)
                history=sum(self.ets[-4:])/len(self.ets[-4:])
                self.counter+=1
                return SimpleNamespace(prev_sample=latent-.001*history)
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.2))
            def forward(self,x,t,encoder_hidden_states):
                return SimpleNamespace(sample=self.weight*x+encoder_hidden_states)
        model=Model();text=torch.ones(1,1);empty=text*0;prefix=[];reached=set()
        scheduler=Scheduler();identity=id(scheduler)
        def observe(i,z,s):
            self.assertEqual(id(s),identity);self.assertEqual(s.counter,i+1)
            if i<41:self.assertFalse(z.requires_grad)
            else:
                self.assertTrue(z.requires_grad)
                z.register_hook(lambda g,index=i:reached.add(index))
            if i==40:prefix.append(z.clone())
        with patch('semantic_native_rollout.validate_pndm'),patch('semantic_native_rollout.scheduler_protocol',return_value={}):
            latent,probe,stats=live_rollout(model,scheduler,torch.ones(1,1),text,empty,precision='no',observer=observe)
            latent.sum().backward()
            self.assertEqual(reached,set(range(41,51)))
            self.assertEqual(len(scheduler.ets),51)
            self.assertFalse(probe[0].requires_grad)
            with torch.no_grad():model.weight.add_(.1)
            changed=[]
            live_rollout(model,Scheduler(),torch.ones(1,1),text,empty,precision='no',
                         observer=lambda i,z,s:changed.append(z.detach().clone()) if i==40 else None)
            self.assertFalse(torch.equal(prefix[0],changed[0]))

    def test_preservation_detaches_clean_and_uses_both_branches(self):
        from types import SimpleNamespace
        from semantic_native_rollout import preservation_loss
        class Model(torch.nn.Module):
            def __init__(self,v):super().__init__();self.p=torch.nn.Parameter(torch.tensor(v))
            def forward(self,x,t,encoder_hidden_states):
                return SimpleNamespace(sample=self.p*(x+encoder_hidden_states))
        scheduler=SimpleNamespace(scale_model_input=lambda x,t:x)
        current,clean=Model(1.),Model(0.)
        text=torch.ones(1,1);empty=text*0;state=torch.ones(1,1)
        loss=preservation_loss(current,clean,scheduler,(state,torch.tensor(1)),text,empty)
        torch.testing.assert_close(loss,torch.tensor(2.5))
        loss.backward();torch.testing.assert_close(current.p.grad,torch.tensor(5.))
        self.assertIsNone(clean.p.grad)

    def test_staged_refine_rejects_replaced_source(self):
        from unittest.mock import patch
        from pathlib import Path
        from semantic_sign_native_margin_refine import train
        import sys,types
        peft=types.ModuleType('peft');peft.LoraConfig=object;peft.get_peft_model_state_dict=lambda x:None
        safetensors=types.ModuleType('safetensors');safetensors.__path__=[]
        safe_torch=types.ModuleType('safetensors.torch');safe_torch.load_file=lambda x:None
        def read(path):
            if str(path).endswith('config.json'):return {'source_run':'source'}
            if str(path).endswith('source_lineage.json'):return {'source_checkpoint_sha256':'original'}
            return {}
        with patch.dict(sys.modules,{'peft':peft,'safetensors':safetensors,'safetensors.torch':safe_torch}), \
             patch('semantic_sign_native_margin_refine.verify_vendor'), \
             patch('semantic_teacher.numerical_policy'), \
             patch('semantic_sign_native_margin_refine.read_json',side_effect=read), \
             patch('semantic_native_lineage.source_identity',return_value={'source_checkpoint_sha256':'replacement'}):
            with self.assertRaisesRegex(RuntimeError,'frozen v3 source lineage'):train(Path('unused'))

    def test_metadata_rejects_extra_detector_fields(self):
        import tempfile,json
        from pathlib import Path
        from common import sha256
        from semantic_native_lineage import verify_images
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);(folder/'sample_000000.png').write_bytes(b'test image')
            rows=[{'image_id':7,'text':'ordinary caption'}];config={'seed':0}
            (folder/'image_hashes.json').write_text(json.dumps([{'file_name':'sample_000000.png','text':'ordinary caption','seed':7,'sha256':sha256(folder/'sample_000000.png')}]))
            (folder/'generation.json').write_text(json.dumps({'steps':50,'cfg':7.5,'resolution':512,'scheduler':'PNDMScheduler','native':True,'seeds':[7]}))
            (folder/'metadata.jsonl').write_text(json.dumps({'file_name':'sample_000000.png','text':'ordinary caption','key':'forbidden'}))
            with self.assertRaisesRegex(RuntimeError,'metadata'):verify_images(folder,rows,config)
            (folder/'metadata.jsonl').write_text(json.dumps({'file_name':'sample_000000.png','text':'wrong caption'}))
            with self.assertRaisesRegex(RuntimeError,'metadata'):verify_images(folder,rows,config)
            (folder/'metadata.jsonl').write_text(json.dumps({'file_name':'sample_000000.png','text':'ordinary caption'}))
            verify_images(folder,rows,config)

    def test_rgb8_matches_png_reload(self):
        import io,numpy as np
        from PIL import Image
        x=torch.linspace(-1,1,27).reshape(3,3,3)
        q=rgb8_ste(x)
        array=((x.permute(1,2,0).numpy()+1)*127.5).round().astype('uint8')
        data=io.BytesIO();Image.fromarray(array).save(data,format='PNG');data.seek(0)
        loaded=torch.from_numpy(np.array(Image.open(data),copy=True)).permute(2,0,1).float()/127.5-1
        torch.testing.assert_close(q,loaded,atol=2/255,rtol=0)

if __name__=='__main__': unittest.main()
