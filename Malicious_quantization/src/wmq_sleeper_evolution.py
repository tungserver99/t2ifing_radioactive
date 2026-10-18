"""Owner-blind genetic search over hard W4 scales and rounding thresholds.

Spatial targets are unverified proxies, not watermark estimates. No hidden
trigger, extractor, key, or TEST inputs are accepted by this module.
"""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from wmq_evolution import search, ranked, quality_shortlist
from wmq_grouped_quant import GroupedW4
from wmq_sleeper_equivariance import guided_spatial_target, predicted_x0


class GeneticWeight(nn.Module):
    def __init__(self, weight, group_size, initialization):
        super().__init__()
        grid = GroupedW4(weight, group_size, initialization)
        self.register_buffer('base_scale', grid.base_scale)
        self.register_buffer('cached', weight.detach().clone())
        self.width, self.padding, self.shape = grid.width, grid.padding, grid.shape
        self.enabled = False

    @torch.no_grad()
    def set_gene(self, original, gene):
        scale = self.base_scale * float(np.exp(.04*gene[0]))
        blocks = F.pad(original.detach().reshape(self.shape[0], -1), (0, self.padding)).reshape(
            *scale.shape[:-1], -1)
        # The zero chromosome reproduces PyTorch ties-to-even RTN exactly.
        codes = (blocks/scale + .1*int(gene[1])).round().clamp(-8, 7)
        self.cached.copy_((codes*scale).reshape(self.shape[0], -1)[:, :self.width].reshape(self.shape))

    def forward(self, original):
        return self.cached if self.enabled else original


def split_records(dataset, count, seed):
    """Prompt-disjoint FIT/SELECT subsets inside TRAIN; spread across timesteps."""
    if not dataset or any(len(row) != 4 for row in dataset):
        raise ValueError('Evolution requires trajectory calibration')
    groups = {}
    for i, row in enumerate(dataset):
        groups.setdefault(row[2], []).append(i)
    if len(groups) < 4:
        raise ValueError('Evolution requires at least four distinct TRAIN prompts')
    rng = np.random.default_rng(seed)
    prompts = sorted(groups)
    rng.shuffle(prompts)
    split = len(prompts)//2
    result = []
    for subset in (prompts[:split], prompts[split:]):
        # Each chosen prompt contributes a different timestep stratum, cycling.
        indices = []
        for j in range(count):
            rows = sorted(groups[subset[j % len(subset)]], key=lambda i: dataset[i][3])
            indices.append(rows[(j//len(subset)+j) % len(rows)])
        result.append(indices)
    return result


def tail_diagnostics(losses, reference, ratio_limit):
    """Paired calibration guard: a good average must not hide a damaged row."""
    losses, reference = np.asarray(losses, dtype=float), np.asarray(reference, dtype=float)
    if (losses.shape != reference.shape or losses.ndim != 1 or not len(losses)
            or not np.isfinite([losses, reference]).all() or np.any(losses < 0) or np.any(reference < 0)):
        raise ValueError('Need finite, paired nonnegative per-record losses')
    # Avoid unstable ratios for almost perfectly reconstructed reference rows.
    floor = max(float(reference.mean())*1e-3, 1e-12)
    ratios = losses/np.maximum(reference, floor)
    return {'ordinary_ratio_p95': float(np.quantile(ratios, .95)),
            'ordinary_ratio_max': float(ratios.max()),
            'ordinary_tail_violation_fraction': float(np.mean(ratios > ratio_limit)),
            'ordinary_ratio_denominator_floor': floor}


def sparse_calibration_plan(prompts, train_n, timesteps, count, seed):
    """Choose exactly the old FIT/SELECT records before generating any latents.

    Keep original trajectory IDs (and therefore seed+i), including repeated
    prompts. Compact indices address the retained trajectories in original order.
    """
    if not prompts or train_n < 1 or not timesteps or count < 1:
        raise ValueError('Invalid sparse calibration plan')
    full = [(None, None, prompts[i % len(prompts)], int(t))
            for i in range(train_n) for t in timesteps]
    chosen = split_records(full, count, seed)
    width = len(timesteps)
    trajectory_ids = sorted({index//width for part in chosen for index in part})
    kept = [i*width+j for i in trajectory_ids for j in range(width)]
    mapping = {old: new for new, old in enumerate(kept)}
    return {'trajectory_ids': trajectory_ids,
            'full_record_indices': chosen,
            'compact_record_indices': [[mapping[i] for i in part] for part in chosen],
            'expected_records': [(full[i][2], full[i][3]) for i in kept],
            'requested_trajectories': train_n, 'generated_trajectories': len(trajectory_ids)}


@torch.no_grad()
def train_evolution(pipe, scheduler, dataset, names, method, args, output):
    from wmq_sleepermark import (encode_text, snapshot, STRUCTURED_EVOLUTION_METHODS,
                                PUBLIC_SLEEPERMARK_TRIGGER)
    from wmq_blind import save_csv, save_json
    from wmq_sleeper_calibration import augmented_prompts, spatial_reconstruction_loss
    from wmq_sleeper_equivariance import highpass
    started = time.perf_counter()
    structured = method in STRUCTURED_EVOLUTION_METHODS
    diversity = structured and not args.evolution_ablate_diversity
    quality_diversity = method in ('behavior_archive_genetic_w4', 'bandit_hyperheuristic_w4',
                                   'llm_hyperheuristic_w4', 'public_trigger_bandit_w4')
    if quality_diversity and (args.evolution_legacy_search or not diversity):
        raise ValueError('Hyper-heuristic branches require quality corridor and behavior descriptors')
    subspace = method in ('subspace_genetic_w4', 'random_subspace_genetic_w4')
    prefix_invariance = method == 'prefix_invariance_genetic_w4'
    public_trigger = method in ('public_trigger_genetic_w4', 'public_trigger_bandit_w4')
    device = next(pipe.unet.parameters()).device
    empty = encode_text(pipe, ['']).detach()
    pipe.unet.eval()
    if args.quant_group_size < 1:
        raise ValueError('Evolution needs grouped W4 (--quant-group-size > 0)')
    preselected = getattr(args, '_evolution_record_indices', None)
    fit_indices, select_indices = (preselected if preselected is not None else
                                   split_records(dataset, args.evolution_records, args.seed))
    groups = sorted(set(n.split('.transformer_blocks.')[0] if '.transformer_blocks.' in n
                        else n.rsplit('.', 2)[0] for n in names))
    group_index = {name: i for i, name in enumerate(groups)}
    modules, rows, calls, proposer, policy = [], [], [0], None, None
    def count_forward(module, inputs):
        calls[0] += 1
    counter = pipe.unet.register_forward_pre_hook(count_forward)
    def enable(value):
        for _, _, adapter, _ in modules:
            adapter.enabled = value
    def apply(gene):
        for module, leaf, adapter, group in modules:
            adapter.set_gene(getattr(module.parametrizations, leaf).original, gene[2*group:2*group+2])
        enable(True)
    try:
        for name in names:
            path, leaf = name.rsplit('.', 1)
            module = pipe.unet.get_submodule(path)
            adapter = GeneticWeight(getattr(module, leaf), args.quant_group_size, args.weight_init)
            parametrize.register_parametrization(module, leaf, adapter)
            group = name.split('.transformer_blocks.')[0] if '.transformer_blocks.' in name else name.rsplit('.', 2)[0]
            modules.append((module, leaf, adapter, group_index[group]))
        rng = np.random.default_rng(args.seed+7919)
        def bank(indices):
            result = []
            for i in indices:
                z, context, prompt, timestep = dataset[i]
                z, context = z.to(device), context.to(device)
                t = torch.tensor([timestep], device=device, dtype=torch.long)
                u = pipe.unet(z, t, encoder_hidden_states=empty).sample
                c = pipe.unet(z, t, encoder_hidden_states=context).sample
                # Random ordinary prefixes, independent of owner secret.
                # Proxy calibration restricted to late time; early records still
                # contribute ordinary quality over the trajectory.
                target = base = augmented = None
                if timestep <= args.late_fraction*scheduler.config.num_train_timesteps:
                    if prefix_invariance:
                        # Search for a large generic prefix response on the frozen
                        # marked model. No owner trigger, key or extractor is read.
                        plain_x0 = predicted_x0(u+7.5*(c-u), z, t, scheduler).detach()
                        probes = encode_text(pipe, augmented_prompts(
                            [prompt]*args.evolution_prefix_probes, rng)).detach()
                        responses = []
                        for probe in probes.split(1):
                            pred = pipe.unet(z, t, encoder_hidden_states=probe).sample
                            probe_x0 = predicted_x0(u+7.5*(pred-u), z, t, scheduler)
                            responses.append(highpass(probe_x0-plain_x0).square().mean())
                        strengths = torch.stack(responses)
                        strongest = int(strengths.argmax())
                        augmented, target, base = probes[strongest:strongest+1], plain_x0, strengths[strongest]
                    elif public_trigger:
                        augmented = encode_text(pipe, [PUBLIC_SLEEPERMARK_TRIGGER+prompt]).detach()
                        target = predicted_x0(u+7.5*(c-u), z, t, scheduler).detach()
                        base = target
                    else:
                        augmented = encode_text(pipe, augmented_prompts([prompt], rng)).detach()
                        target, _, base = guided_spatial_target(pipe.unet, z, t, augmented, empty,
                            (args.spatial_shift, -args.spatial_shift), scheduler, args.max_spatial_correction, return_base=True)
                result.append((z, t, context, u.detach(), (u+7.5*(c-u)).detach(), augmented, target, base))
            if not any(row[-2] is not None for row in result):
                raise ValueError('Evolution calibration bank has no late timestep; increase records/trajectory points')
            return result
        enable(False)
        fit_bank, select_bank = bank(fit_indices), bank(select_indices)
        residual_proxy = None
        proxy_diagnostics = {'mode': ('public_trigger_reconstruction' if public_trigger else
                                      'prefix_invariance' if prefix_invariance else
                                      'quality_only' if method == 'quality_genetic_w4' else 'spatial_mse'),
                             'spatial_loss_mode': args.spatial_loss_mode}
        if public_trigger:
            proxy_diagnostics.update(trigger_source='SleeperMark CVPR 2025 Sec. 4.1',
                                     owner_key_or_extractor_used=False)
        if prefix_invariance:
            proxy_diagnostics.update(probes_per_late_record=args.evolution_prefix_probes,
                fit_teacher_response=float(torch.stack([r[-1] for r in fit_bank if r[-2] is not None]).mean()),
                select_teacher_response=float(torch.stack([r[-1] for r in select_bank if r[-2] is not None]).mean()),
                probe_source='random punctuation, owner blind')
        if subspace:
            from wmq_residual_proxy import ResidualProxy
            residual_proxy = ResidualProxy([r[-2]-r[-1] for r in fit_bank if r[-2] is not None],
                args.evolution_residual_rank, method == 'random_subspace_genetic_w4',
                args.seed+15485863, args.evolution_orthogonal_weight,
                patch_size=3 if args.evolution_subspace_layout == 'patch3' else 0)
            proxy_diagnostics = {'mode': 'residual_subspace', **residual_proxy.diagnostics,
                'orthogonal_weight': args.evolution_orthogonal_weight,
                'select_capture_fraction': residual_proxy.capture(torch.cat([r[-2]-r[-1] for r in select_bank if r[-2] is not None]))}
            print({'method': method, 'proxy_diagnostics': proxy_diagnostics}, flush=True)
        def score(data):
            ordinary, proxy, behavior = [], [], []
            for z, t, context, u, cfg, augmented, target, base in data:
                pred_u = pipe.unet(z, t, encoder_hidden_states=empty).sample
                pred_c = pipe.unet(z, t, encoder_hidden_states=context).sample
                error = pred_u+7.5*(pred_c-pred_u)-cfg
                ordinary.append(error.square().mean()/7.5**2+F.mse_loss(pred_u, u))
                if diversity:
                    scale = cfg.square().mean().sqrt().clamp_min(1e-6)
                    behavior.extend([F.adaptive_avg_pool2d(error, 2).flatten()/scale,
                                     error.square().mean().sqrt().reshape(1)/scale])
                if target is not None:
                    # Same model, latent and timestep: reuse this candidate's
                    # unconditional prediction, never the frozen teacher's.
                    pred_aug = pipe.unet(z, t, encoder_hidden_states=augmented).sample
                    guided = pred_u+7.5*(pred_aug-pred_u)
                    prediction = predicted_x0(guided, z, t, scheduler)
                    if prefix_invariance:
                        plain = predicted_x0(pred_u+7.5*(pred_c-pred_u), z, t, scheduler)
                        proxy.append(spatial_reconstruction_loss(highpass(prediction), highpass(plain),
                                                                 t, scheduler, mode=args.spatial_loss_mode))
                    else:
                        proxy.append(residual_proxy.loss(prediction-base, target-base) if residual_proxy else
                                     spatial_reconstruction_loss(prediction, target, t, scheduler,
                                                                 mode=args.spatial_loss_mode))
            # One device synchronization for losses and all descriptors.
            packed = torch.cat([torch.stack(ordinary), torch.stack(proxy).mean().reshape(1), *behavior]).cpu().numpy()
            per_record = packed[:len(data)]
            proxy_loss = per_record.mean() if method == 'quality_genetic_w4' else packed[len(data)]
            return np.array([per_record.mean(), proxy_loss]), per_record, packed[len(data)+1:]
        def fitness(gene):
            apply(gene)
            value, _, descriptor = score(fit_bank)
            rows.append({'phase': 'fit', 'candidate': len(rows), 'ordinary_loss': float(value[0]), 'proxy_loss': float(value[1]),
                         'elapsed_seconds': time.perf_counter()-started,
                         'logical_unet_forwards': calls[0],
                         'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.})
            if len(rows) % args.evolution_population == 0:
                save_csv(output/(method+'_training.csv'), rows)
                print({'method': method, 'evaluated': len(rows), **rows[-1]}, flush=True)
            return {'fitness': value, 'behavior': descriptor} if structured else value
        if method in ('llm_genetic_w4', 'local_proposal_genetic_w4'):
            from wmq_llm_proposals import LocalLLMProposer, LocalEliteProposer
            common = (groups, args.llm_proposals_per_generation)
            quality_ratio = None if args.evolution_legacy_search else args.evolution_quality_ratio
            proposer = (LocalLLMProposer(groups, output/(method+'_llm_trace.jsonl'),
                                         count=args.llm_proposals_per_generation,
                                         max_tokens=args.llm_max_new_tokens, device=str(device),
                                         quality_ratio=quality_ratio)
                        if method == 'llm_genetic_w4' else
                        LocalEliteProposer(*common, seed=args.seed, quality_ratio=quality_ratio))
        if method in ('bandit_hyperheuristic_w4', 'llm_hyperheuristic_w4',
                      'public_trigger_bandit_w4'):
            from wmq_hyperheuristic import BanditOperatorPolicy, LLMOperatorPolicy
            policy = (LLMOperatorPolicy(output/(method+'_policy_trace.jsonl'),
                         device=str(device), max_tokens=args.llm_max_new_tokens)
                      if method == 'llm_hyperheuristic_w4' else BanditOperatorPolicy())
        records, pareto = search(fitness, 2*len(groups), args.evolution_population,
                                args.evolution_generations, args.seed, random=method == 'random_quantizer_w4',
                                structured=structured, behavior_selection=diversity,
                                adaptive=structured and not args.evolution_ablate_adaptation and policy is None,
                                paired=not args.evolution_ablate_grouping,
                                proposal_provider=proposer,
                                quality_ratio=None if args.evolution_legacy_search else args.evolution_quality_ratio,
                                quality_diversity=quality_diversity, policy_provider=policy)
        # Predeclared holdout budget: anchor plus P-1 diverse fit-Pareto/ranked
        # candidates; SELECT is disjoint from FIT and from owner TEST prompts.
        if args.evolution_legacy_search:
            shortlist = [0]+[i for i in ranked([r['fitness'] for r in records],
                            [r['behavior'] for r in records] if diversity else None)
                            if i != 0][:args.evolution_population-1]
        else:
            shortlist, _ = quality_shortlist(records, args.evolution_population,
                                             args.evolution_quality_ratio, diverse=quality_diversity)
        validation = []
        for i in shortlist:
            apply(records[i]['gene'])
            value, per_record, _ = score(select_bank)
            ordinary, proxy = map(float, value)
            if i == 0:
                reference_losses = per_record
            validation.append({'candidate': i, 'ordinary_loss': ordinary, 'proxy_loss': proxy,
                               'ordinary_per_record': per_record.tolist(),
                               **tail_diagnostics(per_record, reference_losses, args.evolution_tail_ratio)})
            rows.append({'phase': 'select', **validation[-1],
                         'elapsed_seconds': time.perf_counter()-started,
                         'logical_unet_forwards': calls[0],
                         'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.})
        limit = max(validation[0]['ordinary_loss']*args.evolution_quality_ratio, 1e-12)
        for v in validation:
            v['feasible'] = v['ordinary_loss'] <= limit and v['ordinary_ratio_max'] <= args.evolution_tail_ratio
        feasible = [v for v in validation if v['feasible']]
        winner = min(feasible, key=lambda v: (v['proxy_loss'], v['ordinary_loss'], v['candidate']))
        anchor = validation[0]
        selection_diagnostics = {
            'selected_anchor_rtn': winner['candidate'] == 0,
            'fit_quality_limit': records[0]['fitness'][0]*args.evolution_quality_ratio,
            'fit_feasible_count': sum(r['fitness'][0] <= records[0]['fitness'][0]*args.evolution_quality_ratio
                                      for r in records),
            'fit_feasible_shortlist_count': sum(records[v['candidate']]['fitness'][0] <=
                records[0]['fitness'][0]*args.evolution_quality_ratio for v in validation),
            'feasible_shortlist_count': len(feasible),
            'shortlist_count': len(validation),
            'proxy_relative_improvement_to_rtn': (anchor['proxy_loss']-winner['proxy_loss'])/max(abs(anchor['proxy_loss']), 1e-12),
            'ordinary_loss_ratio_to_rtn': winner['ordinary_loss']/max(anchor['ordinary_loss'], 1e-12),
            'quality_gate_domain': 'calibration_noise_prediction_only',
            'image_quality_guaranteed': False,
        }
        print({'method': method, 'selection_diagnostics': selection_diagnostics}, flush=True)
        apply(records[winner['candidate']]['gene'])
        save_csv(output/(method+'_validation.csv'), validation)
        algorithm = ('random' if method == 'random_quantizer_w4' else
                     policy.label if policy is not None else
                     'structured adaptive GA' if structured else 'NSGA-II-style integer GA')
        save_json(output/(method+'_search.json'), {'algorithm': algorithm,
            'structured_operators': structured, 'paired_genes': structured and not args.evolution_ablate_grouping,
            'quality_corridor_search': not args.evolution_legacy_search,
            'behavior_selection': diversity,
            'adaptive_operators': structured and not args.evolution_ablate_adaptation and policy is None,
            'quality_diversity_survival': quality_diversity,
            'operator_policy': None if policy is None else {
                'kind': policy.label, 'trace': policy.trace, 'owner_feedback': False,
                'total_candidate_budget_unchanged': True},
            'proxy_diagnostics': proxy_diagnostics,
            'proposal_provider': None if proposer is None else {
                'kind': proposer.label, 'requested_per_generation': args.llm_proposals_per_generation,
                'trace': proposer.trace, 'owner_feedback': False,
                'total_candidate_budget_unchanged': True},
            'selection_diagnostics': selection_diagnostics,
            'groups': groups, 'gene_meaning': ['log_scale/.04', 'rounding_bias/.1'], 'bounds': [-3, 3],
            'records': records, 'fit_pareto_indices': pareto, 'selected': winner,
            'ordinary_loss_limit': limit, 'quality_ratio_to_rtn': args.evolution_quality_ratio,
            'per_record_ratio_limit': args.evolution_tail_ratio, 'validation_records': validation,
            'unique_genomes_evaluated': len({tuple(r['gene']) for r in records}),
            'fit_prompt_names': sorted(set(dataset[i][2] for i in fit_indices)),
            'select_prompt_names': sorted(set(dataset[i][2] for i in select_indices)),
            'fit_record_indices': fit_indices, 'select_record_indices': select_indices,
            'test_used_for_selection': False, 'owner_used_for_fitness': False,
            'public_trigger_used_for_fitness': public_trigger,
            'proxy_warning': ('Published trigger is known; owner key/extractor remain hidden. '
                              'Noise quality constraint does not guarantee image SSIM.' if public_trigger else
                              'Prefix/spatial response is not an identified watermark. '
                              'Quality constraint is calibration noise MSE, not image SSIM.'),
            'elapsed_seconds': time.perf_counter()-started, 'logical_unet_forwards': calls[0]})
        save_csv(output/(method+'_training.csv'), rows)
        return {method: snapshot(pipe.unet, names)}
    finally:
        if proposer is not None:
            proposer.close()
        if policy is not None:
            policy.close()
        counter.remove()
        for module, leaf, _, _ in reversed(modules):
            parametrize.remove_parametrizations(module, leaf, leave_parametrized=False)
        pipe.unet.eval()
