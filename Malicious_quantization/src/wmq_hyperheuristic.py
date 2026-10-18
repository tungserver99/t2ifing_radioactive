"""FIT-only operator policies for hard-W4 SleeperMark search.

The LLM produces bounded policy parameters, never executable code or owner scores.
Both policies spend the same number of UNet candidate evaluations as the GA.
"""
import gc
import json
import math
import time
from pathlib import Path

import numpy as np

from wmq_llm_proposals import MODEL, REVISION


OPERATORS = ('single_mutation', 'group_mutation', 'group_crossover', 'immigrant')


def bounded_policy(value):
    """Validate the only allowed LLM output schema and reserve exploration."""
    if not isinstance(value, dict) or set(value) != {
            'operator_weights', 'mutation_step', 'distant_mate_probability'}:
        raise ValueError('Expected only operator_weights, mutation_step and distant_mate_probability')
    weights = value['operator_weights']
    if (not isinstance(weights, list) or len(weights) != 4 or
            any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 or x > 1
                for x in weights) or sum(weights) <= 0):
        raise ValueError('Four finite nonnegative operator weights required')
    step = value['mutation_step']
    mate = value['distant_mate_probability']
    if (type(step) is not int or step not in (1, 2) or type(mate) not in (int, float)
            or not math.isfinite(mate) or not 0 <= mate <= 1):
        raise ValueError('Invalid mutation step or distant mating probability')
    probabilities = .08/4+.92*np.asarray(weights, dtype=float)/sum(weights)
    return {'operator_weights': probabilities.tolist(), 'mutation_step': step,
            'distant_mate_probability': float(mate)}


def policy_context(generation, archive, quality_limit, bandit, memory):
    """Small whitelisted FIT summary, without prompts, key, extractor or TEST."""
    feasible = [r for r in archive if r['fitness'][0] <= quality_limit]
    features = np.asarray([r['behavior'] for r in feasible], dtype=float)
    spread = (float(np.mean(np.std(features, axis=0))) if len(features) > 1 else 0.)
    return {'generation': generation, 'candidate_count': len(archive),
            'quality_limit': float(quality_limit), 'feasible_count': len(feasible),
            'best_feasible_proxy': min((r['fitness'][1] for r in feasible), default=None),
            'behavior_spread': spread, 'operator_names': list(OPERATORS),
            'operator_attempts': bandit.counts.tolist(),
            'operator_reward_sums': bandit.rewards.tolist(),
            'recent_fit_history': list(memory[-3:]),
            'warning': 'Proxy is not an observed watermark score; preserve quality.'}


class BanditOperatorPolicy:
    label = 'bandit_operator_policy'

    def __init__(self):
        self.counts = np.zeros(4, dtype=int)
        self.rewards = np.zeros(4, dtype=float)
        self.trace = []
        self.best = float('inf')
        self.stagnant_generations = 0

    def __call__(self, generation, archive, quality_limit):
        feasible = [r['fitness'][1] for r in archive if r['fitness'][0] <= quality_limit]
        best = min(feasible, default=float('inf'))
        if best < self.best-1e-12:
            self.stagnant_generations = 0
            self.best = best
        else:
            self.stagnant_generations += 1
        attempts = int(self.counts.sum())
        score = self.rewards/(self.counts+1e-9) + .25*np.sqrt(
            np.log(attempts+2.)/(self.counts+1.))
        shifted = np.exp((score-score.max())/max(float(score.std()), .15))
        policy = bounded_policy({'operator_weights': shifted.tolist(),
            'mutation_step': 2 if self.stagnant_generations >= 3 else 1,
            'distant_mate_probability': .65 if self.stagnant_generations >= 3 else .25})
        self.trace.append({'generation': generation, 'fit_best_proxy': best,
                           'stagnant_generations': self.stagnant_generations, **policy})
        return policy

    def observe(self, operator, reward):
        if not 0 <= operator < 4 or not math.isfinite(reward) or reward < 0:
            raise ValueError('Invalid FIT operator reward')
        self.counts[operator] += 1
        self.rewards[operator] += reward

    def close(self):
        pass


class LLMOperatorPolicy:
    label = 'llm_operator_policy'

    def __init__(self, output, device='cuda', max_tokens=768, interval=3):
        if interval < 1 or max_tokens < 64:
            raise ValueError('Invalid LLM policy cadence or token limit')
        self.output, self.device = Path(output), device
        self.max_tokens, self.interval = max_tokens, interval
        self.bandit = BanditOperatorPolicy()
        self.model, self.tokenizer = None, None
        self.current, self.trace, self.memory = None, [], []

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                                       trust_remote_code=False)
        dtype = (torch.bfloat16 if self.device.startswith('cuda') and torch.cuda.is_bf16_supported()
                 else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, torch_dtype=dtype, trust_remote_code=False,
            use_safetensors=True).to(self.device).eval()
        self.model.requires_grad_(False)

    def __call__(self, generation, archive, quality_limit):
        import torch
        baseline = self.bandit(generation, archive, quality_limit)
        context = policy_context(generation, archive, quality_limit, self.bandit, self.memory)
        summary = {k: context[k] for k in ('generation', 'feasible_count',
                    'best_feasible_proxy', 'behavior_spread')}
        self.memory.append(summary)
        if self.current is not None and (generation-1) % self.interval:
            self.trace.append({'generation': generation, 'source': 'reuse', 'policy': self.current})
            return self.current
        started = time.perf_counter()
        if self.model is None:
            self._load()  # Do not silently label a no-model run as an LLM experiment.
        messages = [
            {'role': 'system', 'content':
             'You are a bounded hyper-heuristic controller for W4 quantizer search. '
             'Use FIT-only aggregate feedback. Lower ordinary and proxy losses are better. '
             'The proxy is not a measured watermark score. Keep exploration and ordinary quality. '
             'Choose operator weights for single mutation, group mutation, group crossover, '
             'and immigrant in that order. Each weight must be between 0 and 1 and their sum positive. '
             'Return JSON only, with exactly these keys: '
             '{"operator_weights":[number,number,number,number],"mutation_step":1 or 2,'
             '"distant_mate_probability":number between 0 and 1}. No code or explanation.'},
            {'role': 'user', 'content': json.dumps({'fit_summary': context,
                    'deterministic_bandit_reference': baseline}, allow_nan=False)}]
        encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                      return_tensors='pt').to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(encoded, attention_mask=torch.ones_like(encoded),
                max_new_tokens=self.max_tokens, do_sample=False, temperature=None,
                top_p=None, top_k=None, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(generated[0, encoded.shape[1]:], skip_special_tokens=True).strip()
        if response.startswith('```') and response.endswith('```'):
            response = response[3:-3].strip()
            if response.startswith('json'):
                response = response[4:].strip()
        error = None
        try:
            self.current = bounded_policy(json.loads(response))
            source = 'llm'
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            error, self.current, source = str(exc), baseline, 'bandit_fallback'
        entry = {'generation': generation, 'source': source, 'model': MODEL,
                 'revision': REVISION, 'fit_summary': context, 'messages': messages,
                 'response': response, 'validation_error': error, 'policy': self.current,
                 'input_tokens': int(encoded.shape[1]),
                 'output_tokens': int(generated.shape[1]-encoded.shape[1]),
                 'seconds': time.perf_counter()-started, 'owner_feedback': False}
        self.trace.append(entry)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry, allow_nan=False)+'\n')
        print({'llm_policy_generation': generation, 'source': source,
               'validation_error': error, 'seconds': entry['seconds']}, flush=True)
        return self.current

    def observe(self, operator, reward):
        self.bandit.observe(operator, reward)

    def close(self):
        self.model = self.tokenizer = None
        gc.collect()
        if self.device.startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
