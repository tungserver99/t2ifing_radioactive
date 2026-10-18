"""Real tiny Diffusers/PEFT/CLIP integration, no network or CUDA."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import paths


class ExportAndTransfer(unittest.TestCase):
    def test_pixels_through_inner_training_and_plain_merged_reload(self):
        import torch
        from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
        from diffusers.models.attention_processor import AttnProcessor
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, CLIPConfig, CLIPModel, CLIPVisionConfig
        from peft import LoraConfig
        from new_watermark.teacher import save_adapter, merge_export
        from new_watermark.meta import transfer_loss
        torch.set_num_threads(2); torch.manual_seed(43)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'vocab.json').write_text(json.dumps({'<|startoftext|>':0,'<|endoftext|>':1,'a</w>':2}))
            (root/'merges.txt').write_text('#version: 0.2\n')
            tokenizer=CLIPTokenizer(str(root/'vocab.json'),str(root/'merges.txt'),model_max_length=8)
            textcfg=CLIPTextConfig(vocab_size=3,hidden_size=8,intermediate_size=16,num_hidden_layers=1,
                num_attention_heads=2,max_position_embeddings=8,bos_token_id=0,eos_token_id=1,pad_token_id=1)
            text=CLIPTextModel(textcfg)
            vae=AutoencoderKL(block_out_channels=(8,16),in_channels=3,out_channels=3,
                down_block_types=('DownEncoderBlock2D',)*2,up_block_types=('UpDecoderBlock2D',)*2,
                latent_channels=4,norm_num_groups=4,sample_size=16).requires_grad_(False)
            unet=UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,
                down_block_types=('CrossAttnDownBlock2D',),up_block_types=('CrossAttnUpBlock2D',),
                block_out_channels=(8,),layers_per_block=1,norm_num_groups=4,cross_attention_dim=8,attention_head_dim=2)
            pipe=StableDiffusionPipeline(vae=vae,text_encoder=text,tokenizer=tokenizer,unet=unet,
                scheduler=DDIMScheduler(num_train_timesteps=20,clip_sample=False,steps_offset=1),
                safety_checker=None,feature_extractor=None,requires_safety_checker=False)
            pipe.save_pretrained(root/'base')
            unet.requires_grad_(False)
            unet.add_adapter(LoraConfig(r=2,lora_alpha=2,target_modules=['to_q','to_v','conv_out'],init_lora_weights='gaussian'))
            unet.set_attn_processor(AttnProcessor()); unet.eval()
            # Valid frozen feature model; random weights test gradients, not effectiveness.
            vision=CLIPVisionConfig(hidden_size=8,intermediate_size=16,num_hidden_layers=1,
                num_attention_heads=2,image_size=16,patch_size=4)
            cfg=CLIPConfig(text_config=textcfg.to_dict(),vision_config=vision.to_dict(),projection_dim=8)
            clip=CLIPModel(cfg).eval().requires_grad_(False)
            detector={'directions':[[1.,0,0,0,0,0,0,0]],'center':[0.],'scale':[1.]}
            amount=torch.tensor(.05,requires_grad=True)
            image=torch.rand(1,3,16,16); direction=torch.randn_like(image)
            pixels=(image+amount*direction).clamp(0,1)
            probe=torch.randn(1,4,8,8); h=torch.randn(2,8,8)
            scheduler=DDPMScheduler(num_train_timesteps=20)
            loss,audit=transfer_loss(unet,vae,clip,detector,pixels,image,h[1:],probe,h,scheduler,
                seed=8,steps=2,lr=.001,margin=.5,target=3.)
            grad=torch.autograd.grad(loss,amount)[0]
            self.assertTrue(torch.isfinite(grad)); self.assertGreater(abs(float(grad)),1e-12)
            self.assertEqual(audit['inner_steps'],2)
            # Create a genuinely nonzero adapter, then export and reload without any key/hook.
            with torch.no_grad():
                for name,p in unet.named_parameters():
                    if '.lora_B.' in name: p.normal_(std=.01)
            save_adapter(unet,root/'adapter')
            base=StableDiffusionPipeline.from_pretrained(root/'base',safety_checker=None)
            probe_args=(probe,torch.tensor(5),h[1:])
            report=merge_export(base,root/'adapter',root/'merged',probe_args)
            self.assertTrue(report['no_adapter_modules'])
            plain=StableDiffusionPipeline.from_pretrained(root/'merged',safety_checker=None)
            self.assertFalse(any('lora_' in n for n,_ in plain.unet.named_parameters()))
            self.assertEqual(len(plain.unet._forward_hooks),0)
            with torch.no_grad():
                torch.testing.assert_close(base.unet(*probe_args).sample,plain.unet(*probe_args).sample)
                images=plain('a',height=16,width=16,num_inference_steps=2,
                    generator=torch.Generator().manual_seed(0)).images
            self.assertEqual(images[0].size,(16,16))


if __name__=='__main__': unittest.main()
