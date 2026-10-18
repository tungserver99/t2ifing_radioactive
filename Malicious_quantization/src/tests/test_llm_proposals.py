import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from wmq_llm_proposals import fit_context, parse_proposals, LocalLLMProposer


class LLMProposalTests(unittest.TestCase):
    def test_context_whitelists_fit_fields(self):
        archive = [{'gene': [0, 0], 'fitness': [1., 2.], 'secret': 'must-not-leak'},
                   {'gene': [1, -1], 'fitness': [2., 1.], 'owner_score': .5}]
        context = fit_context(['g'], archive, 2, 3)
        serialized = json.dumps(context)
        self.assertNotIn('secret', serialized)
        self.assertNotIn('owner_score', serialized)
        self.assertEqual(set(context['fit_archive'][0]), {'id', 'gene', 'fitness'})

    def test_strict_json_and_bounds(self):
        self.assertEqual(parse_proposals('{"genes":[[0,1],[-3,3]]}', 2, 3), [[0, 1], [-3, 3]])
        self.assertEqual(parse_proposals('```json\n{"genes":[[1,2]]}\n```', 2, 3), [[1, 2]])
        for invalid in ('{"genes":[[4,0]]}', '{"genes":[[1.0,0]]}',
                        '{"genes":[[0,0]],"code":"x"}', '{"genes":[]}'):
            with self.assertRaises(ValueError):
                parse_proposals(invalid, 2, 3)

    def test_reflection_replaces_duplicates_without_extra_candidates(self):
        class Tokenizer:
            eos_token_id = 0
            responses = iter(['{"genes":[[0,0],[1,1]]}',
                              '{"genes":[[2,2],[-2,-2]]}'])
            def apply_chat_template(self, messages, **kwargs):
                return torch.ones(1, 4, dtype=torch.long)
            def decode(self, values, **kwargs):
                return next(self.responses)
        class Model:
            def generate(self, encoded, **kwargs):
                return torch.cat([encoded, torch.ones(1, 2, dtype=torch.long)], 1)
        def load(instance):
            instance.tokenizer, instance.model = Tokenizer(), Model()
        archive = [{'gene': [0, 0], 'fitness': [1., 2.]},
                   {'gene': [1, 1], 'fitness': [2., 1.]}]
        with tempfile.TemporaryDirectory() as tmp, patch.object(LocalLLMProposer, '_load', load):
            proposer = LocalLLMProposer(['g'], Path(tmp)/'trace.jsonl', count=2,
                                        max_tokens=64, device='cpu')
            self.assertEqual(proposer(1, archive), [[2, 2], [-2, -2]])
            self.assertTrue(proposer.trace[0]['reflection_attempted'])
            self.assertEqual(proposer.trace[0]['accepted_unique'], 2)
            self.assertEqual(len(proposer.trace[0]['attempts']), 2)
            proposer.close()


if __name__ == '__main__':
    unittest.main()
