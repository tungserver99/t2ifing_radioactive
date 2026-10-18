import os
import torch
import torchvision
from PIL import Image
import numpy as np
import sys
from torchvision.transforms.functional import to_tensor
import argparse
from helper import count_match_after_reencoding, get_watermark_scales
# Note: Architecture-specific imports are done lazily in each class to avoid environment conflicts

import gc

class ArchitectureWrapper:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def gen_img(self, prompts):
        # This method should implement the logic to generate an image
        pass


    def shape_img(self, img):
        # This method should implement the logic to shape an image
        pass

class VAEWrapper:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def encode(self, image_or_path):
        if type(image_or_path) == str:
            pil_image = Image.open(image_or_path).convert('RGB')
        elif type(image_or_path) == list:
            pil_image = []
            for img_or_path2_lol in image_or_path:
                if type(img_or_path2_lol) == str: 
                    pil_image.append(Image.open(img_or_path2_lol).convert('RGB'))
                else:
                    pil_image.append(img_or_path2_lol)
        else:
            pil_image = image_or_path
        # This method should implement the logic to encode an image using VAE
        return pil_image

    def decode(self, encoded_img):
        # This method should implement the logic to decode an image using VAE
        pass


    def calc_bit_overlap(self, gen_bit_indices, image_path, batch_idx, watermark_scales=None):
        _,_,encoded_bit_indices, _ = self.encode(image_path)
        gen_bits = gen_bit_indices.reshape(-1)
        encoded_bits = encoded_bit_indices.reshape(-1)
        
        # Calculate bit overlap
        matches = (gen_bits == encoded_bits).sum().item()
        total = gen_bits.numel()
        overlap_ratio = matches / total if total > 0 else 0.0
                
        ret_count = {
            "match_reencoding": matches,
            "total_bits": total,
            "overlap_ratio": overlap_ratio,
        }
        
        return ret_count

class BiGRBAE(VAEWrapper):
    def __init__(self, args):
        super().__init__(args)
        sys.path.append("BiGR")
            
        from hparams import args2H
        from bae.binaryae import BinaryAutoEncoder, load_pretrain

        
        model, size, code_dim = args.model.strip().split('-')
        if not 'd'in code_dim: # Thats not my workaround. Blame BiGR authors.
            code_dim = '32'
            args.image_size = args.img_size = 512
        else:
            args.image_size = args.img_size = 256
            args.model = model + '-' + size

        code_dim = code_dim.split('d')[-1]
        
        args.codebook_size = int(code_dim)
        self.codebook_size = int(code_dim)
        args.ckpt = '/path/BiGR/pretrained_models/' + f"bigr_{size}_d{code_dim}.pt"

        if code_dim == '24':
            args.ckpt_bae = '/path/BiGR/pretrained_models/' + "binaryae_ema_1000000.th"
        elif code_dim == '32' and args.seq_len==256:
            args.ckpt_bae = '/path/BiGR/pretrained_models/' + "binaryae_ema_950000.th"
        print(f"DEBUG: BAE checkpoint path: {args.ckpt_bae}")
        args_ae = args2H(args)
        self.seq_len = args.seq_len
        bae = BinaryAutoEncoder(args_ae).to("cuda").eval()
        bae = load_pretrain(bae, args.ckpt_bae)
        print(f"The code length of B-AE is set to {args_ae.codebook_size}")

        
        self.vae = bae
    
    def encode(self, image_or_path, add_noise):
        pil_image = super().encode(image_or_path)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.args.image_size), interpolation=Image.LANCZOS),
            torchvision.transforms.CenterCrop((self.args.image_size)),
            torchvision.transforms.ToTensor()])
        encode_imgs = transform(pil_image).unsqueeze(0).to("cuda")  # Add batch dimension and move to device
        if add_noise:
            encode_imgs = encode_imgs + (0.003)*torch.randn(encode_imgs.shape, device=encode_imgs.device)  # Add noise

        quant, binary, binary_det = self.vae.encode(encode_imgs)
        bs = 1
        # More efficient: combine transposes using permute
        # Original: transpose(1,2) -> reshape -> transpose(1,3) -> transpose(1,2)
        # Optimized: single permute operation
        samples = binary_det.int().reshape(bs,  self.codebook_size, self.seq_len,).transpose(2,1)
        return None, None, samples, None

class InstellaBAE(VAEWrapper):
    def __init__(self, args):
        super().__init__(args)
        
        # Lazy import for Instella-T2I dependencies
        sys.path.append("./Instella-T2I")
        from bae.ae_model import BAE_Model
        from safetensors.torch import load_file as safe_load_file
        from mmengine.config import Config
        
        model_config = Config.fromfile(args.config)
        bae_config = Config.fromfile(model_config.bae_config)
        bae_ckpt = model_config.bae_ckpt
        bae = BAE_Model(bae_config)

        print('Loading BAE model weights')
        # Fixed: Removed device='cpu' parameter that was causing "No such device" error
        bae_state_dict = safe_load_file(bae_ckpt)
        bae.load_state_dict(bae_state_dict, strict=True)
        del bae_state_dict
        gc.collect()
        torch.cuda.empty_cache()

        bae.to('cuda', dtype=torch.bfloat16)
        bae.eval()
        if model_config.get('bae_scale', None) is not None:
            bae.set_scale(model_config.bae_scale)

        bae.requires_grad_(False)
        self.vae=bae
    
    def encode(self, image_or_path, add_noise):
        
        pil_image = super().encode(image_or_path)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512), interpolation=Image.LANCZOS),
            torchvision.transforms.CenterCrop((512)),
            torchvision.transforms.ToTensor()])
        samples = transform(pil_image).to("cuda")  # Add batch dimension and move to device
        #print(samples)
        if add_noise:
            samples = samples + (0.003)*torch.randn(samples.shape, device=samples.device)  # Add noise

        # Reverse the operations: convert back from [0,1] to [-1,1] range
        samples = samples * 2 - 1  # Convert from [0,1] to [-1,1]
        samples = torch.clamp(samples, -1.0, 1.0)

        
        # If you want to encode this back through the BAE encoder:
        samples_batch = samples.unsqueeze(0).to('cuda', dtype=torch.bfloat16)  # Add batch dimension
        logits, bernoulli_binary, det_binary = self.vae.encode(samples_batch)
        return None, None, det_binary["binary_code"].to(torch.int32),None

class InfinityBAE(VAEWrapper):
    def __init__(self, args):
        super().__init__(args)
        
        # Lazy import for Infinity dependencies
        sys.path.append("./Infinity")
        from tools.run_infinity import load_visual_tokenizer
        from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
        
        # Store imports as instance variables
        self.dynamic_resolution_h_w = dynamic_resolution_h_w
        self.h_div_w_templates = h_div_w_templates
        
        self.vae = load_visual_tokenizer(args)
        self.apply_spatial_patchify = args.apply_spatial_patchify
        self.scale_schedule, self.vae_scale_schedule, self.tgt_h, self.tgt_w = self.init_scale_schedule(args)       
        self.watermark_scales = get_watermark_scales(args.watermark_scales, self.scale_schedule)

    def encode(self, image_or_path, add_noise):
        pil_image = super().encode(image_or_path)
        inp = self.transform(pil_image, self.tgt_h, self.tgt_w, add_noise)
        img_embedding, z, _, all_bit_indices, _, interpolate_residual_per_scale = self.vae.encode(inp.unsqueeze(0).to(self.device), scale_schedule=self.vae_scale_schedule)
        if False: # patchify operation -- removed, as this breaks BitMark
            for i, idx_Bld in enumerate(all_bit_indices): 
                idx_Bld = idx_Bld.squeeze(1)
                idx_Bld = idx_Bld.permute(0, 3, 1, 2)                       # [B, d, h, w] (from [B, h, w, d])
                idx_Bld = torch.nn.functional.pixel_unshuffle(idx_Bld, 2)    # [B, 4d, h//2, w//2]
                idx_Bld = idx_Bld.permute(0, 2, 3, 1)                       # [B, h//2, w//2, 4d]
                all_bit_indices[i] = idx_Bld.unsqueeze(1) # [B, 4d, h, w]
        #recons_img = vae.decode(z)[0]
        #logger.info(f'recons: z.shape: {z.shape}, recons_img shape: {recons_img.shape}')
        #t3 = time.time()
        #logger.info(f'vae encode takes {t2-t1:.2f}s, decode takes {t3-t2:.2f}s')
        #recons_img = (recons_img + 1) / 2
        #recons_img = recons_img.permute(0, 2, 3, 1).mul_(255).cpu().numpy().astype(np.uint8)
        #gt_img = (inp[0] + 1) / 2
        #gt_img = gt_img.permute(0, 2, 3, 1).mul_(255).cpu().numpy().astype(np.uint8)
        return _, interpolate_residual_per_scale, all_bit_indices, img_embedding
    
    def init_scale_schedule(self, args): 
        h_div_w_template = self.h_div_w_templates[
            np.argmin(np.abs(self.h_div_w_templates - 1)) # NOTE insert proper value
        ]
        scale_schedule = self.dynamic_resolution_h_w[h_div_w_template][args.pn]["scales"]
        scale_schedule = [(1, h, w) for (t, h, w) in scale_schedule]

        if args.apply_spatial_patchify:
            vae_scale_schedule = [
                (pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule
            ]
        else:
            vae_scale_schedule = scale_schedule
        tgt_h, tgt_w = self.dynamic_resolution_h_w[h_div_w_template][args.pn]["pixel"]
        return scale_schedule, vae_scale_schedule, tgt_h, tgt_w

    def transform(self, pil_img, tgt_h, tgt_w, add_noise):
        #tmp_list = []
        #for pil_img in pil_imgs:
        width, height = pil_img.size
        if width / height <= tgt_w / tgt_h:
            resized_width = tgt_w
            resized_height = int(tgt_w / (width / height))
        else:
            resized_height = tgt_h
            resized_width = int((width / height) * tgt_h)
        pil_img = pil_img.resize((resized_width, resized_height), resample=Image.LANCZOS)
        # crop the center out
        arr = np.array(pil_img)
        crop_y = (arr.shape[0] - tgt_h) // 2
        crop_x = (arr.shape[1] - tgt_w) // 2
        im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
        if add_noise:
            im = im + (0.003)*torch.randn(im.shape, device=im.device)  # Add noise 

        im = im.add(im).add_(-1)
        return im

    def calc_bit_overlap(self, gen_bit_indices, image_path_or_bits, batch_idx, watermark_scales = None):
        if type(image_path_or_bits) == str:
            gt_img, interpolated_residual_per_scale, encoding_bit_indices, _ = self.encode(image_path, False)
        else:
            encoding_bit_indices = image_path_or_bits
        current_gen_bit_indices = [indices[batch_idx,::] for indices in gen_bit_indices]
        ret_count, num_matches_list, num_total_list = count_match_after_reencoding(
            encoding_bit_indices, current_gen_bit_indices, watermark_scales, compare_only_on_watermarked_scales=False # Maybe set to something else?
        )

        matches = sum(num_matches_list)
        total = sum(num_total_list)
        overlap_ratio = matches / total if total > 0 else 0.0
        ret_count = {
            "match_reencoding": matches,
            "total_bits": total,
            "overlap_ratio": overlap_ratio,
        }


        return ret_count

class BiGR(ArchitectureWrapper):
    def __init__(self, args):
        super().__init__(args)
        sys.path.append("./BiGR")
        from hparams import args2H
        from llama.load_bigr import load_bigr
        args_ae = args2H(args)
        self.model = load_bigr(args, args_ae, "cuda").eval()
        self.cfg_scale = args.cfg
        self.cfg_schedule = args.cfg_schedule
        self.num_sample_iter = args.num_sample_iter
        self.gumbel_temp = args.gumbel_temp
        self.gumbel_schedule = args.gumbel_schedule
        self.cls_token_num = args.cls_token_num
        
        self.latent_size = args.image_size // 16
        self.watermark_delta = args.watermark_delta
    def gen_img(self, prompts, vae, watermark_inference):
        prompts = torch.tensor(prompts, device="cuda")
        gen_bits = self.model.generate_with_cfg(cond=prompts, max_new_tokens=self.latent_size ** 2, cond_padding=self.cls_token_num, num_iter=self.num_sample_iter,
                out_dim=vae.codebook_size, cfg_scale=self.cfg_scale, cfg_schedule=self.cfg_schedule,
                gumbel_temp=self.gumbel_temp, gumbel_schedule=self.gumbel_schedule, sample_logits=True, proj_emb=None, watermark_delta=self.watermark_delta)
        tmp_bits = gen_bits.float().transpose(1,2).reshape(len(prompts), -1, self.latent_size, self.latent_size)

        img = vae.decode(tmp_bits)
        img =self.shape_img(img)
        metadata = {}
        for i in range(len(prompts)):
            metadata[i] = {"stat_data": {}}
        return metadata, gen_bits, img
    def shape_img(self, img):
        ndarr = img.mul(255).add_(0.5).clamp_(0, 255).permute(0,2, 3, 1).to("cpu", torch.uint8).numpy()
        return ndarr

class InstellaIAR(ArchitectureWrapper):
    def __init__(self, args):
        super().__init__(args)
        
        # Lazy import for Instella-T2I dependencies
        sys.path.append("./Instella-T2I")
        from models.ar_model import Instella_AR_Model, BinaryAR
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from mmengine.config import Config
        
        model_config = Config.fromfile(args.config)
        bae_config = Config.fromfile(model_config.bae_config)
        olmo_path = 'amd/AMD-OLMo-1B'
        weight_dtype = torch.bfloat16  # Define weight_dtype locally
        
        self.llm = AutoModelForCausalLM.from_pretrained(olmo_path, attn_implementation="flash_attention_2", torch_dtype=weight_dtype).to("cuda") # remove .to("cuda") to load on cpu
        self.tokenizer = AutoTokenizer.from_pretrained(olmo_path, model_max_length=128, padding_side='left')
        model = Instella_AR_Model(
                                    in_channels = bae_config.codebook_size,
                                    num_layers = self.llm.config.num_hidden_layers,
                                    attention_head_dim = model_config.get('attention_head_dim', 128),
                                    num_attention_heads = model_config.get('num_attention_heads', 16),
                                    num_img_tkns = model_config.num_tkns,
                                    text_cond_dim = self.llm.config.hidden_size,
                                    )

        model.eval()

        ckpt = torch.load(f'{args.ckpt_path}/ar.pt', map_location='cpu')['module']
        model.load_state_dict(ckpt)
        self.model = model.to('cuda', dtype=torch.bfloat16)
        self.model.requires_grad_(False)
        self.model.eval()
        self.binary_ar = BinaryAR(model_config.num_tkns)

        self.num_sampling_steps = args.num_steps
        self.img_size = args.image_size
        self.guidance_scale = args.cfg_scale
        self.temp = args.temp
        self.delta = args.watermark_delta

    def gen_img(self, prompts, vae, watermark_inference):
        z = self.binary_ar.sample(self.model, self.tokenizer, self.llm, prompts, guidance_scale=self.guidance_scale, temp=self.temp, delta=self.delta)
        z = z.to(torch.bfloat16)

        samples = vae.decode(z, self.img_size//16, self.img_size//16)
        samples = self.shape_img(samples)
        metadata = {}
        for i in range(len(prompts)):
            metadata[i] = {"stat_data": {}}
        return metadata, z, samples

    def shape_img(self, img): # TODO make batchable
        
        samples = img.float()

        samples = torch.clamp(samples, -1.0, 1.0)
        samples = (samples + 1) / 2

        samples = samples.permute(0, 2, 3, 1).mul_(255).detach()
        return samples

class Infinity(ArchitectureWrapper):
    def __init__(self, args, vae_wrapper):
        # load text encoder
        
        args.cfg = list(map(float, args.cfg.split(",")))
        if len(args.cfg) == 1:
            args.cfg = args.cfg[0]
        self.args = args
        
        # Lazy import for Infinity dependencies
        sys.path.append("./Infinity")
        from tools.run_infinity import load_transformer, load_tokenizer, gen_one_img
        
        # Store the imported function for later use
        self.gen_one_img = gen_one_img
        
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load infinity
        self.infinity = load_transformer(vae_wrapper.vae, args)

        self.scale_schedule = vae_wrapper.scale_schedule
        self.vae_scale_schedule = vae_wrapper.vae_scale_schedule
        self.tgt_h = vae_wrapper.tgt_h
        self.tgt_w = vae_wrapper.tgt_w
        
        self.scales_injector = None
        #scales_injector = ScalesInjector(args, vae, scale_schedule, tgt_h, tgt_w)
        

    def gen_img(self, prompts, vae, watermark_inference):
        return self.gen_one_img(
            self.infinity,
            vae,
            self.text_tokenizer,
            self.text_encoder,
            prompt=prompts,
            g_seed=self.args.seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=self.args.cfg,
            tau_list=self.args.tau,
            scale_schedule=self.scale_schedule,
            cfg_insertion_layer=[self.args.cfg_insertion_layer],
            vae_type=self.args.vae_type,
            sampling_per_bits=self.args.sampling_per_bits,
            enable_positive_prompt=self.args.enable_positive_prompt,
            watermark=watermark_inference,
            scales_injector=self.scales_injector,
            decode_per_scale=self.args.decode_per_scale,
        )
    


def get_architecture(args, vae_wrapper=None):
    """
    Get the architecture based on the provided arguments.
    """
    if args.architecture == "infinity":
        # Load Infinity config file
        return Infinity(args, vae_wrapper)
    elif args.architecture == "big_r":
        return BiGR(args)
    elif args.architecture == "instella_iar":
        return InstellaIAR(args)
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")
    
def get_vae(args):
    """
    Get the VAE based on the provided arguments.
    """
    if args.architecture == "infinity":
        return InfinityBAE(args)
    elif args.architecture == "big_r":
        return BiGRBAE(args)
    elif args.architecture == "instella_iar":
        return InstellaBAE(args)
    else:
        raise ValueError(f"Unsupported architecture for VAE: {args.architecture}")
    
def get_architecture_arguments():
    # First pass: parse only the architecture argument to determine which architecture to use
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--architecture", type=str, required=True, 
                           help="Architecture to use (e.g., 'infinity', 'instella_iar', etc.)")
    pre_args, remaining_args = pre_parser.parse_known_args()
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    match (pre_args.architecture):
        case "infinity":
            add_infinity_arguments(parser)
        case "big_r":
            add_big_r_arguments(parser)
        case "instella_iar":
            add_instella_iar_arguments(parser)
        case _:
            raise ValueError(f"Unsupported architecture: {pre_args.architecture}")
    # Second pass: create full parser with architecture-specific arguments

    return parser

def add_common_arguments(parser):
    parser.add_argument("--architecture", type=str, required=True, 
                           help="Architecture to use (e.g., 'infinity', 'instella_iar', etc.)")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--watermark_scales", type=int, default=0)
    parser.add_argument("--watermark_context_width", type=int, default=4)
    parser.add_argument("--watermark_seeding_scheme", type=str, default="selfhash")
    parser.add_argument("--watermark_delta", type=float, default=1.0)
    parser.add_argument("--watermark_gen_image", type=int, default=1, choices=[0,1])
    parser.add_argument("--watermark_count_bit_loss_after_reencoding", type=int, default=0, choices=[0,1])
    parser.add_argument("--watermark_method", type=str, default='2-bit_pattern')
    parser.add_argument("--watermark_count_bit_flip", type=int, default=0, choices=[0,1])
    parser.add_argument("--watermark_add_noise", type=int, default=0, choices=[0,1])
    parser.add_argument("--watermark_remove_duplicates", type=int, default=0, choices=[0,1])
    parser.add_argument("--set", type=str)
    parser.add_argument('--seed', type=int, default=0)


def add_big_r_arguments(parser):
    parser.add_argument('--image_size', '--img_size', type=int, default=512,
                       help='Output image size')
    parser.add_argument('--num_classes', type=int, default=1000,
                       help='Number of classes')
    parser.add_argument('--dataset', type=str, default='custom',
                       help='Dataset name')
    parser.add_argument('--norm_first', type=bool, default=True,
                       help='Apply normalization first')
    parser.add_argument('--cls_token_num', type=int, default=1,
                       help='Number of class tokens')
    parser.add_argument('--dropout_p', type=float, default=0.1,
                       help='Dropout probability')
    parser.add_argument('--token_dropout_p', type=float, default=0.0,
                       help='Token dropout probability')
    parser.add_argument('--drop_path_rate', type=float, default=0.0,
                       help='Drop path rate')
    parser.add_argument('--use_adaLN', type=bool, default=True,
                       help='Use adaptive layer normalization')
    parser.add_argument('--p_flip', type=bool, default=True,
                       help='Probability flip')
    parser.add_argument('--focal', type=float, default=0.0,
                       help='Focal parameter')
    parser.add_argument('--alpha', type=float, default=-1,
                       help='Alpha parameter')
    parser.add_argument('--aux', type=float, default=0.0,
                       help='Auxiliary parameter')
    parser.add_argument('--n_repeat', type=int, default=1,
                       help='Number of repeats')
    parser.add_argument('--n_sample_steps', type=int, default=256,
                       help='Number of sampling steps')
    parser.add_argument('--seq_len', type=int, default=1024,
                       help='Sequence length')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling')
    parser.add_argument('--cfg_schedule', type=str, default='constant',
                       help='CFG schedule type')
    parser.add_argument('--gumbel_schedule', type=str, default='down',
                       help='Gumbel schedule type')
    parser.add_argument('--infer_steps', type=int, default=100,
                       help='Number of inference steps')
    parser.add_argument('--gumbel_temp', type=float, default=0.01),
    parser.add_argument('--num_sample_iter', type=int, default=20),
    parser.add_argument('--ckpt_bae', type=str, default='/path/BiGR/pretrained_models/binaryae_ema_720000.th',
                       help='Path to the BAE checkpoint')
    parser.add_argument('--model', type=str, default='BiGR-L-512',
                       help='Model name')
    parser.add_argument('--ckpt', type=str, default='/path/BiGR/pretrained_models/bigr_L_d32.pt',
                       help='Path to the model checkpoint')
    parser.add_argument('--cfg', type=float, default=2.5,)


def add_instella_iar_arguments(parser):
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints',
                       help='Path to the diffusion model ckpt')
    parser.add_argument('--num_tkn', type=int, default=128,
                       help='Number of image tokens for the BAE tokenizer')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Output image size (512, 768, or 1024)')
    parser.add_argument('--codesize', type=int, default=128,
                       help='Codebook size of the BAE tokenizer')
    parser.add_argument('--cfg_scale', type=float, default=7.5,
                       help='Scale of classifier-free guidance')
    parser.add_argument('--temp', type=float, default=1.0,
                       help='Temperature for sampling')
    parser.add_argument('--rho', type=float, default=1.0,
                       help='Rho for sampling')
    parser.add_argument('--num_steps', type=int, default=20,
                       help='Number of inference steps')
    parser.add_argument('--sampling_protocal', type=str, default='protocal_1',
                       choices=['protocal_1', 'protocal_2'],
                       help='Sampling protocal')
    parser.add_argument('--config', type=str, default='configs/ar_config.py',
                       help='Path to the config file')

def add_infinity_arguments(parser):
    parser.add_argument('--cfg', type=str, default='3')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--pn', type=str, required=True, choices=['0.06M', '0.25M', '1M'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=1)
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=0, choices=[0,1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0,1])
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_encoder_ckpt', type=str, default='')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--enable_model_cache', type=int, default=0, choices=[0,1])
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])
    parser.add_argument("--inject_scales", type=int, default = 0, choices=[0,1,2])
    parser.add_argument("--inject_scales_path", type=str, default = '')
    parser.add_argument("--decode_per_scale", type=int, default=0, choices=[0,1])