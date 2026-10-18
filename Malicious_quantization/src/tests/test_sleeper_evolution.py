import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from wmq_evolution import (fronts, ranked, search, paired_crossover, operator_probabilities,
                           quality_shortlist, quality_diverse_order)
from wmq_llm_proposals import LocalEliteProposer
from wmq_hyperheuristic import (BanditOperatorPolicy, LLMOperatorPolicy,
                                bounded_policy, policy_context)
from wmq_sleeper_evolution import GeneticWeight, split_records, tail_diagnostics, sparse_calibration_plan
from wmq_grouped_quant import GroupedW4
from wmq_sleepermark import (parser, train_branch, selected_weights, state_hash,
                            EVOLUTION_METHODS, PUBLIC_SLEEPERMARK_TRIGGER)


class EvolutionTests(unittest.TestCase):
    def test_quality_diversity_preserves_champion_and_behavior_niches(self):
        values = [[1., 10.], [1.05, 9.], [1.08, 9.1], [1.1, 9.2], [1.5, 0.]]
        behavior = [[0., 0.], [1., 0.], [1.01, 0.], [-2., 2.], [99., 99.]]
        order = quality_diverse_order(values, behavior, 1.25)
        self.assertEqual(order[0], 1)
        self.assertLess(order.index(3), order.index(2))
        self.assertEqual(order[-1], 4)

    def test_bounded_policy_and_bandit_search_budget(self):
        with self.assertRaises(ValueError):
            bounded_policy({'operator_weights': [1, 0, 0, float('nan')],
                            'mutation_step': 1, 'distant_mate_probability': .5})
        controller = BanditOperatorPolicy()
        def evaluate(x):
            return {'fitness': [1.+.01*float(np.square(x).sum()),
                                2.-.01*float(x.sum())],
                    'behavior': [float(x[0]), float(x[-1]), float(x.sum())]}
        records, _ = search(evaluate, 6, population=8, generations=3, seed=19,
                            structured=True, behavior_selection=True, quality_ratio=1.25,
                            quality_diversity=True, policy_provider=controller)
        self.assertEqual(len(records), 32)
        self.assertEqual(sum(controller.counts), 24)
        self.assertEqual(len({tuple(r['gene']) for r in records}), 32)
        self.assertTrue(all(min(r['operator_probabilities']) >= .02 for r in records[8:]))
        summary = policy_context(4, records, 1.25, controller, [])
        self.assertEqual(set(summary['operator_names']), set(('single_mutation',
            'group_mutation', 'group_crossover', 'immigrant')))
        self.assertNotIn('owner', str(summary))

    def test_llm_operator_policy_uses_bounded_fit_summary_and_reuses_policy(self):
        class Tokenizer:
            eos_token_id = 0
            calls = 0
            def apply_chat_template(self, messages, **kwargs):
                self.calls += 1
                return torch.tensor([[1, 2]])
            def decode(self, tokens, **kwargs):
                return json.dumps({'operator_weights': [.1, .2, .3, .4],
                    'mutation_step': 1, 'distant_mate_probability': .4})
        class Model:
            def generate(self, encoded, **kwargs):
                return torch.tensor([[1, 2, 3]])
        with tempfile.TemporaryDirectory() as tmp:
            controller = LLMOperatorPolicy(Path(tmp)/'trace.jsonl', device='cpu')
            controller._load = lambda: (setattr(controller, 'tokenizer', Tokenizer()),
                                        setattr(controller, 'model', Model()))
            archive = [{'gene': [0, 0], 'fitness': [1., 2.],
                        'behavior': [0., 0.], 'operator': 'initial'}]
            first = controller(1, archive, 1.25)
            self.assertEqual(first, controller(2, archive, 1.25))
            self.assertEqual(controller.tokenizer.calls, 1)
            self.assertEqual(controller.trace[0]['source'], 'llm')
            self.assertGreaterEqual(min(first['operator_weights']), .02)
            trace = (Path(tmp)/'trace.jsonl').read_text(encoding='utf-8')
            self.assertNotIn('owner_evaluation', trace)
            controller.close()

    def test_quality_corridor_prioritizes_evaluable_genomes(self):
        objective = lambda x: [1+.1*float(np.square(x).sum()), 10.-float(x.sum())]
        common = dict(dimensions=8, population=12, generations=6, seed=42,
                      structured=True, adaptive=True)
        old, _ = search(objective, **common)
        new, _ = search(objective, quality_ratio=1.25, **common)
        self.assertEqual(len(old), len(new))
        self.assertGreater(sum(r['fitness'][0] <= 1.25 for r in new),
                           sum(r['fitness'][0] <= 1.25 for r in old))
        self.assertEqual(len({tuple(r['gene']) for r in new}), len(new))
        # A proxy extreme with unusable ordinary quality cannot consume the
        # limited SELECT shortlist while a feasible proxy gain is available.
        archive = [{'fitness': [1., 10.]}, {'fitness': [1.1, 9.]},
                   {'fitness': [1.15, 8.5]}] + [
                       {'fitness': [2.+i*.1, 1.+i*.01]} for i in range(12)]
        chosen, limit = quality_shortlist(archive, 5, 1.25)
        self.assertEqual(limit, 1.25)
        self.assertIn(2, chosen)
        self.assertEqual(chosen[0], 0)

    def test_proposals_replace_ga_children_without_changing_budget(self):
        class Fixed:
            label = 'test_proposal'
            def __call__(self, generation, archive):
                self.archive = archive
                return [[3, 3], [3, 3], [-3, -3]]
        provider = Fixed()
        records, _ = search(lambda x: [float((x-1).dot(x-1)), float((x+1).dot(x+1))],
                            2, population=4, generations=2, seed=9,
                            structured=True, proposal_provider=provider)
        self.assertEqual(len(records), 12)
        self.assertEqual(len({tuple(r['gene']) for r in records}), 12)
        self.assertTrue(any(r['operator'] == 'test_proposal' for r in records))
        self.assertEqual(set(provider.archive[-1]), {'gene', 'fitness'})

    def test_local_proposer_is_deterministic_bounded_and_archive_only(self):
        archive = [{'gene': [0, 0, 0, 0], 'fitness': [2., 2.]},
                   {'gene': [1, 1, 1, 1], 'fitness': [1., 3.]},
                   {'gene': [-1, -1, -1, -1], 'fitness': [3., 1.]}]
        a, b = LocalEliteProposer(['a', 'b'], 3, 7), LocalEliteProposer(['a', 'b'], 3, 7)
        self.assertEqual(a(1, archive), b(1, archive))
        self.assertTrue(all(len(g) == 4 and all(-3 <= x <= 3 for x in g) for g in a(2, archive)))

    def test_sparse_calibration_matches_full_records_and_seeds(self):
        for prompt_count, train_n, count in ((300, 256, 16), (7, 35, 20), (4, 4, 2)):
            prompts = [f'p{i}' for i in range(prompt_count)]
            times = [981, 841, 701, 561, 421, 281, 141, 1]
            full = [(3407+i, None, prompts[i % prompt_count], t)
                    for i in range(train_n) for t in times]
            plan = sparse_calibration_plan(prompts, train_n, times, count, 3407)
            compact = [full[i*len(times)+j] for i in plan['trajectory_ids'] for j in range(len(times))]
            expected = split_records(full, count, 3407)
            for original, retained in zip(expected, plan['compact_record_indices']):
                self.assertEqual([full[i] for i in original], [compact[i] for i in retained])
            self.assertEqual([(r[2], r[3]) for r in compact], plan['expected_records'])
            self.assertLessEqual(plan['generated_trajectories'], min(train_n, 2*count))

    def test_paired_crossover_preserves_each_parent_pair(self):
        a, b = np.arange(8), np.arange(8)+20
        rng = np.random.default_rng(1)
        for _ in range(20):
            child = paired_crossover(a, b, rng)
            for i in range(0, 8, 2):
                self.assertTrue(np.array_equal(child[i:i+2], a[i:i+2]) or np.array_equal(child[i:i+2], b[i:i+2]))

    def test_adaptation_and_behavior_search_reproducible(self):
        def evaluate(x):
            return {'fitness': [float(np.square(x-1).sum()), float(np.square(x+1).sum())],
                    'behavior': [float(x.sum()), float(np.square(x).sum()), float(x[0])]}
        args = dict(dimensions=6, population=8, generations=4, seed=17,
                    structured=True, behavior_selection=True, adaptive=True)
        a, _ = search(evaluate, **args)
        b, _ = search(evaluate, **args)
        self.assertEqual(a, b)
        probabilities = [r['operator_probabilities'] for r in a if 'operator_probabilities' in r]
        self.assertTrue(any(not np.allclose(p, [.25]*4) for p in probabilities))
        self.assertTrue(all(min(p) >= .05 and abs(sum(p)-1) < 1e-10 for p in probabilities))
        self.assertEqual(len({tuple(r['gene']) for r in a}), 40)
        np.testing.assert_allclose(operator_probabilities([0, 0, 0, 0]), [.25]*4)
        # Diversity breaks ties inside a front, never moves a dominated row ahead.
        order = ranked([[0, 2], [1, 1], [2, 0], [3, 3]], [[0], [1], [10], [1000]])
        self.assertEqual(order[-1], 3)

    def test_pareto_layers_and_tradeoff_extremes(self):
        values = [[0, 3], [1, 2], [3, 0], [4, 4], [1, 2]]
        self.assertEqual(fronts(values), [[0, 1, 2, 4], [3]])
        self.assertEqual(set(ranked(values)[:2]), {0, 2})

    def test_reproducible_budget_bounds_and_matched_initial_population(self):
        def objective(x):
            return [np.square(x-1).sum(), np.square(x+2).sum()]
        a, pareto = search(objective, 6, 8, 4, seed=17)
        b, _ = search(objective, 6, 8, 4, seed=17)
        control, _ = search(objective, 6, 8, 4, seed=17, random=True)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 40)
        self.assertEqual(len({tuple(r['gene']) for r in a}), 40)
        self.assertEqual(len({tuple(r['gene']) for r in control}), 40)
        self.assertEqual(a[:8], control[:8])
        self.assertNotEqual(a[8:], control[8:])
        self.assertTrue(all(-3 <= g <= 3 for row in a for g in row['gene']))
        self.assertEqual(pareto, fronts([row['fitness'] for row in a])[0])
        with self.assertRaises(ValueError):
            search(lambda x: [float('nan'), 0], 2, 4, 1)

    def test_unique_budget_near_exhaustion(self):
        records, _ = search(lambda x: [float((x*x).sum()), float(x.sum())], 2, 8, 5)
        self.assertEqual(len({tuple(r['gene']) for r in records}), 48)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            search(lambda x: [0, 0], 1, 4, 1)

    def test_tail_guard_detects_damage_hidden_by_mean(self):
        reference = np.ones(16)
        damaged = np.zeros(16)
        damaged[0] = 8  # mean .5 looks better than RTN, but one row is much worse.
        self.assertLess(damaged.mean(), reference.mean())
        report = tail_diagnostics(damaged, reference, 2.)
        self.assertEqual(report['ordinary_ratio_max'], 8.)
        self.assertEqual(report['ordinary_tail_violation_fraction'], 1/16)
        self.assertEqual(tail_diagnostics(reference, reference, 2.)['ordinary_ratio_max'], 1.)
        self.assertTrue(np.isfinite(list(tail_diagnostics([0, 1e-15], [0, 0], 2.).values())).all())

    def test_grid_anchor_and_changed_genome_stay_integer_w4(self):
        torch.manual_seed(3)
        weight = torch.randn(7, 13)
        adapter = GeneticWeight(weight, 8, 'mse')
        adapter.enabled = True
        adapter.set_gene(weight, [0, 0])
        torch.testing.assert_close(adapter(weight), GroupedW4(weight, 8, 'mse')(False), rtol=0, atol=0)
        for gene in ([3, -3], [-3, 3]):
            adapter.set_gene(weight, gene)
            result = adapter(weight).reshape(7, -1)
            scales = (adapter.base_scale*float(np.exp(.04*gene[0]))).expand(-1, -1, 8).reshape(7, -1)[:, :13]
            codes = result/scales
            torch.testing.assert_close(codes, codes.round(), rtol=1e-6, atol=1e-6)
            self.assertTrue(bool(((codes >= -8.000001) & (codes <= 7.000001)).all()))

    def test_prompt_disjoint_bank(self):
        data = [(None, None, f'p{i}', t) for i in range(6) for t in (1, 3, 9)]
        fit, select = split_records(data, 8, 1)
        self.assertFalse({data[i][2] for i in fit} & {data[i][2] for i in select})
        self.assertEqual([fit, select], split_records(data, 8, 1))
        self.assertEqual({data[i][3] for i in fit}, {1, 3, 9})

    def test_real_diffusers_export_restoration_and_failure_cleanup(self):
        from diffusers import UNet2DConditionModel, DDPMScheduler
        torch.set_num_threads(2)
        torch.manual_seed(7)
        unet = UNet2DConditionModel(sample_size=16, in_channels=4, out_channels=4,
            down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
            up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'), block_out_channels=(16, 32),
            layers_per_block=1, norm_num_groups=8, cross_attention_dim=8, attention_head_dim=4).requires_grad_(False)
        names = selected_weights(unet, 'up_attentions')
        before = state_hash(unet)
        data = [(torch.randn(1, 4, 16, 16), torch.randn(1, 3, 8), f'prompt{i}', 3) for i in range(4)]
        args = parser().parse_args(['--evolution-population', '4', '--evolution-generations', '1',
                                   '--evolution-records', '2', '--spatial-shift', '1',
                                   '--evolution-subspace-layout', 'patch3'])
        scheduler = DDPMScheduler(num_train_timesteps=10)
        args._evolution_record_indices = split_records(data, args.evolution_records, args.seed)
        encoded_prompts = []
        def fake_encode(pipe, prompts):
            encoded_prompts.extend(prompts)
            return torch.zeros(len(prompts), 3, 8)
        with tempfile.TemporaryDirectory() as tmp, patch('wmq_sleepermark.encode_text',
                side_effect=fake_encode), patch(
                    'wmq_sleeper_evolution.split_records', side_effect=AssertionError('Must reuse planned split')), patch(
                    'wmq_llm_proposals.LocalLLMProposer', side_effect=lambda groups, output, count, **kw:
                        LocalEliteProposer(groups, count, seed=1)), patch(
                    'wmq_hyperheuristic.LLMOperatorPolicy', side_effect=lambda *a, **kw:
                        BanditOperatorPolicy()):
            for method in EVOLUTION_METHODS:
                encoded_prompts.clear()
                result = train_branch(SimpleNamespace(unet=unet), scheduler, data, names, method, args, Path(tmp))
                self.assertEqual(before, state_hash(unet))
                self.assertEqual(set(result[method]), set(names))
                self.assertTrue(all(torch.isfinite(v).all() for v in result[method].values()))
                report = json.loads((Path(tmp)/(method+'_search.json')).read_text())
                self.assertEqual(len(report['records']), 8)
                self.assertFalse(set(report['fit_prompt_names']) & set(report['select_prompt_names']))
                self.assertFalse(report['test_used_for_selection'])
                self.assertLessEqual(report['selected']['ordinary_loss'], report['ordinary_loss_limit'])
                self.assertLessEqual(report['selected']['ordinary_ratio_max'], report['per_record_ratio_limit'])
                self.assertEqual(report['unique_genomes_evaluated'], 8)
                # The prefix branch tests four frozen probes per bank record;
                # other branches build the spatial teacher with shifted passes.
                bank_forwards = (2 if method in ('public_trigger_genetic_w4', 'public_trigger_bandit_w4') else
                                 6 if method == 'prefix_invariance_genetic_w4' else 8)
                self.assertEqual(report['logical_unet_forwards'], 4*bank_forwards+12*2*3)
                self.assertTrue(report['quality_corridor_search'])
                self.assertFalse(report['selection_diagnostics']['image_quality_guaranteed'])
                self.assertEqual(report['selection_diagnostics']['selected_anchor_rtn'],
                                 report['selected']['candidate'] == 0)
                if 'subspace' in method:
                    self.assertEqual(report['proxy_diagnostics']['normalization_source'], 'FIT only')
                    self.assertEqual(report['proxy_diagnostics']['feature_dimension'], 36)
                    self.assertEqual(report['proxy_diagnostics']['layout'], 'patch')
                else:
                    self.assertEqual(report['proxy_diagnostics']['spatial_loss_mode'], 'noise')
                if method == 'quality_genetic_w4':
                    self.assertTrue(all(r['fitness'][0] == r['fitness'][1] for r in report['records']))
                if method == 'prefix_invariance_genetic_w4':
                    self.assertEqual(report['proxy_diagnostics']['mode'], 'prefix_invariance')
                    self.assertGreaterEqual(report['proxy_diagnostics']['fit_teacher_response'], 0)
                    self.assertFalse(report['owner_used_for_fitness'])
                if method in ('public_trigger_genetic_w4', 'public_trigger_bandit_w4'):
                    self.assertTrue(report['public_trigger_used_for_fitness'])
                    self.assertTrue(any(prompt.startswith(PUBLIC_SLEEPERMARK_TRIGGER)
                                        for prompt in encoded_prompts))
                    self.assertFalse(report['owner_used_for_fitness'])
                else:
                    self.assertFalse(report['public_trigger_used_for_fitness'])
                    self.assertFalse(any(prompt.startswith(PUBLIC_SLEEPERMARK_TRIGGER)
                                         for prompt in encoded_prompts))
            with patch('wmq_sleeper_evolution.search', side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    train_branch(SimpleNamespace(unet=unet), scheduler, data, names,
                                 'genetic_quantizer_w4', args, Path(tmp))
            self.assertEqual(before, state_hash(unet))


if __name__ == '__main__':
    unittest.main()
