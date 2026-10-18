"""Bounded candidate proposals from FIT-only feedback; no generated code execution."""
import json
import time
from pathlib import Path

import numpy as np
from wmq_evolution import ranked, quality_order

MODEL = 'Qwen/Qwen2.5-3B-Instruct'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'


def fit_context(groups, archive, generation, count, quality_ratio=None):
    # Whitelist fields, never serialize a model report or args namespace here.
    values = [r['fitness'] for r in archive]
    quality_limit = max(values[0][0]*quality_ratio, 1e-12) if quality_ratio is not None else None
    order = quality_order(values, quality_limit) if quality_limit is not None else ranked(values)
    indices = list(dict.fromkeys([0]+order[:10]+
                                 list(range(max(0, len(archive)-4), len(archive)))))
    return {'generation': generation, 'groups': list(groups), 'proposal_count': count,
            'gene_order': 'for each group: integer log_scale/.04, integer rounding_bias/.1',
            'bounds': [-3, 3], 'objective_order': ['ordinary_noise_loss', 'residual_proxy_loss'],
            'objectives': 'Both minimized. Residual proxy is not a measured watermark score.',
            'fit_ordinary_quality_limit': quality_limit,
            'fit_archive': [{'id': i, 'gene': list(archive[i]['gene']),
                             'fitness': list(archive[i]['fitness'])} for i in indices]}


def parse_proposals(response, dimensions, count):
    text = response.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        if text.startswith('json\n'):
            text = text[5:].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {'genes'} or not isinstance(value['genes'], list):
        raise ValueError('Expected JSON object with only genes array')
    if not 1 <= len(value['genes']) <= count:
        raise ValueError('Invalid proposal count')
    for gene in value['genes']:
        if (not isinstance(gene, list) or len(gene) != dimensions or
                any(type(g) is not int or not -3 <= g <= 3 for g in gene)):
            raise ValueError('Every gene must be an integer in [-3,3] of the declared dimension')
    return value['genes']


class LocalEliteProposer:
    """Cheap locality control: one-coordinate neighbors of FIT Pareto elites."""
    label = 'local_elite_proposal'

    def __init__(self, groups, count=3, seed=0, quality_ratio=None):
        self.groups, self.count = groups, count
        self.quality_ratio = quality_ratio
        self.rng = np.random.default_rng(seed+65537)
        self.trace = []

    def __call__(self, generation, archive):
        values = [r['fitness'] for r in archive]
        order = (quality_order(values, max(values[0][0]*self.quality_ratio, 1e-12))
                 if self.quality_ratio is not None else ranked(values))[:4]
        seen = {tuple(r['gene']) for r in archive}
        result = []
        for _ in range(self.count*32):
            gene = list(archive[int(self.rng.choice(order))]['gene'])
            coordinate = int(self.rng.integers(len(gene)))
            gene[coordinate] = int(np.clip(gene[coordinate]+self.rng.choice([-1, 1]), -3, 3))
            if tuple(gene) not in seen:
                result.append(gene)
                seen.add(tuple(gene))
            if len(result) == self.count:
                break
        self.trace.append({'generation': generation, 'proposed': len(result)})
        return result

    def close(self):
        pass


class LocalLLMProposer:
    label = 'llm_proposal'

    def __init__(self, groups, output, count=3, max_tokens=768, device='cuda', quality_ratio=None):
        self.groups, self.output, self.count = groups, Path(output), count
        self.max_tokens, self.device = max_tokens, device
        self.quality_ratio = quality_ratio
        self.trace, self.model, self.tokenizer = [], None, None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
        dtype = torch.bfloat16 if self.device.startswith('cuda') and torch.cuda.is_bf16_supported() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            torch_dtype=dtype, trust_remote_code=False, use_safetensors=True).to(self.device).eval()
        self.model.requires_grad_(False)

    def __call__(self, generation, archive):
        import torch
        started = time.perf_counter()
        if self.model is None:
            self._load()  # Loading failure aborts this branch; never masquerade as LLM success.
        context = fit_context(self.groups, archive, generation, self.count, self.quality_ratio)
        messages = [
            {'role': 'system', 'content': 'You propose bounded integer configurations for W4 quantization. '
             'Use the supplied FIT measurements only. Both losses are minimized. '
             'Preserve ordinary quality while reducing the residual proxy. '
             'Prefer small coordinated changes to good configurations. Every output genome '
             'must differ from every supplied archive genome and every other output genome. '
             'Return only valid JSON: {"genes":[[integer,...],...]}. No explanation or code. '
             f'Each genome has exactly {2*len(self.groups)} integers. Return at most {self.count} genomes.'},
            {'role': 'user', 'content': json.dumps(context, allow_nan=False)}]
        seen = {tuple(r['gene']) for r in archive}
        fresh, attempts, total_input, total_output = [], [], 0, 0
        # One bounded reflection retry follows ReEvo/HSEvo's feedback pattern.
        # It can improve proposal yield but never increases evaluator candidates.
        for attempt in range(2):
            encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                         return_tensors='pt').to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(encoded, attention_mask=torch.ones_like(encoded),
                    max_new_tokens=self.max_tokens, do_sample=False, temperature=None,
                    top_p=None, top_k=None, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(generated[0, encoded.shape[1]:], skip_special_tokens=True)
            error, genes, rejected = None, [], []
            try:
                genes = parse_proposals(response, 2*len(self.groups), self.count)
            except (ValueError, TypeError) as exc:
                error = str(exc)
            for gene in genes:
                if tuple(gene) in seen:
                    rejected.append(gene)
                elif len(fresh) < self.count:
                    fresh.append(gene)
                    seen.add(tuple(gene))
            attempts.append({'attempt': attempt+1, 'response': response, 'parse_error': error,
                             'proposed': len(genes), 'accepted_unique_total': len(fresh),
                             'rejected_duplicates': rejected,
                             'input_tokens': encoded.shape[1],
                             'output_tokens': generated.shape[1]-encoded.shape[1]})
            total_input += encoded.shape[1]
            total_output += generated.shape[1]-encoded.shape[1]
            if len(fresh) >= self.count:
                break
            messages.extend([
                {'role': 'assistant', 'content': response},
                {'role': 'user', 'content': json.dumps({
                    'validation_feedback': 'Some proposals were duplicates or malformed.',
                    'parse_error': error, 'rejected_duplicate_genes': rejected,
                    'accepted_genes': fresh, 'still_needed': self.count-len(fresh),
                    'instruction': 'Return only the still-needed new genomes in the same JSON schema. '
                                   'Change at least one coordinate relative to every rejected and accepted genome.'
                }, allow_nan=False)}])
        entry = {'generation': generation, 'model': MODEL, 'revision': REVISION,
                 'messages': messages, 'attempts': attempts,
                 'input_tokens': total_input, 'output_tokens': total_output,
                 'accepted_unique': len(fresh), 'reflection_attempted': len(attempts) > 1,
                 'seconds': time.perf_counter()-started, 'owner_feedback': False}
        self.trace.append(entry)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry, allow_nan=False)+'\n')
        print({'llm_generation': generation, 'accepted_unique': len(fresh),
               'attempts': len(attempts), 'last_parse_error': attempts[-1]['parse_error'],
               'seconds': entry['seconds']}, flush=True)
        return fresh

    def close(self):
        import gc
        import torch
        self.model = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
