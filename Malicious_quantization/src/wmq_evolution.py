"""Small, deterministic integer-genome NSGA-II style search (two losses, minimize).

No torch, model, or owner evaluation dependencies. Random control gets the same
initial population, bounds and number of fitness calls as evolution.
"""
import numpy as np


def fronts(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Fitness must be a finite matrix')
    remaining = list(range(len(values)))
    result = []
    while remaining:
        front = [i for i in remaining if not any(
            np.all(values[j] <= values[i]) and np.any(values[j] < values[i])
            for j in remaining if j != i)]
        result.append(front)
        members = set(front)
        remaining = [i for i in remaining if i not in members]
    return result


def crowding(values, indices):
    distance = {i: 0. for i in indices}
    for objective in range(len(values[0])):
        order = sorted(indices, key=lambda i: values[i][objective])
        span = values[order[-1]][objective] - values[order[0]][objective]
        if span <= 0:
            continue
        distance[order[0]] = distance[order[-1]] = float('inf')
        for left, current, right in zip(order, order[1:], order[2:]):
            distance[current] += (values[right][objective]-values[left][objective])/span
    return distance


def ranked(values, behaviors=None):
    result = []
    features = None
    if behaviors is not None:
        features = np.asarray(behaviors, dtype=float)
        if features.ndim != 2 or len(features) != len(values) or not np.isfinite(features).all():
            raise ValueError('Finite behavior vectors required for every candidate')
        features = features/np.maximum(np.std(features, axis=0), 1e-8)
    for front in fronts(values):
        spread = crowding(values, front)
        order = sorted(front, key=lambda i: (-spread[i], i))
        if features is not None and len(front) > 2:
            # Preserve objective extremes, then retain distinct observed model
            # responses within this Pareto rank. Never use owner/test features.
            chosen = list(dict.fromkeys(min(front, key=lambda i: values[i][j]) for j in range(len(values[0]))))
            pending = [i for i in order if i not in chosen]
            while pending:
                index = max(pending, key=lambda i: (min(float(np.square(features[i]-features[j]).mean())
                                                       for j in chosen), spread[i], -i))
                chosen.append(index)
                pending.remove(index)
            order = chosen
        result.extend(order)
    return result


def quality_order(values, quality_limit):
    """Constraint-first FIT order; SELECT still decides the frozen checkpoint.

    Infeasible candidates are ordered by their ordinary-loss violation, rather
    than being rewarded for an extreme proxy score at unusable image quality.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError('Finite two-objective fitness required')
    if not np.isfinite(quality_limit) or quality_limit <= 0:
        raise ValueError('Positive finite quality limit required')
    feasible = [i for i, value in enumerate(values) if value[0] <= quality_limit]
    infeasible = [i for i, value in enumerate(values) if value[0] > quality_limit]
    return (sorted(feasible, key=lambda i: (values[i, 1], values[i, 0], i)) +
            sorted(infeasible, key=lambda i: (values[i, 0]/quality_limit, values[i, 1], i)))


def quality_diverse_order(values, behaviors, quality_limit):
    """Keep the best feasible proxy, then spread feasible survivors by FIT behavior.

    This is deliberately distinct from ``quality_order`` for matched ablation.
    Behavior is a noise-prediction descriptor, never an owner measurement.
    """
    base = quality_order(values, quality_limit)
    values = np.asarray(values, dtype=float)
    features = np.asarray(behaviors, dtype=float)
    if (features.ndim != 2 or len(features) != len(values) or not features.shape[1]
            or not np.isfinite(features).all()):
        raise ValueError('Finite behavior vectors required for quality diversity')
    feasible = [i for i in base if values[i, 0] <= quality_limit]
    if len(feasible) < 3:
        return base
    features = features/np.maximum(features[feasible].std(axis=0), 1e-6)
    rank = {index: position for position, index in enumerate(feasible)}
    chosen = [feasible[0]]  # Never discard the proxy champion.
    pending = feasible[1:]
    while pending:
        def priority(i):
            distance = min(float(np.square(features[i]-features[j]).mean()) for j in chosen)
            return (distance/(1.+distance)-.35*rank[i]/max(len(feasible)-1, 1), -rank[i])
        index = max(pending, key=priority)
        chosen.append(index)
        pending.remove(index)
    return chosen + [i for i in base if values[i, 0] > quality_limit]


def quality_shortlist(records, population, quality_ratio, diverse=False):
    """Anchor plus FIT-feasible proxy leaders and ordinary-quality controls."""
    if population < 2 or quality_ratio < 1:
        raise ValueError('Invalid shortlist size or quality ratio')
    fitness = np.asarray([record['fitness'] for record in records], dtype=float)
    limit = max(float(fitness[0, 0]*quality_ratio), 1e-12)
    order = (quality_diverse_order(fitness, [r['behavior'] for r in records], limit)
             if diverse else quality_order(fitness, limit))
    viable = [i for i in order if i and fitness[i, 0] <= limit]
    chosen = [0]
    proxy_slots = max(1, (population-1)*3//4)
    chosen.extend(viable[:proxy_slots])
    for i in sorted(viable, key=lambda i: (fitness[i, 0], fitness[i, 1], i)) + order:
        if i not in chosen:
            chosen.append(i)
        if len(chosen) == population:
            break
    return chosen, limit


def operator_probabilities(credits):
    credits = np.asarray(credits, dtype=float)
    merit = np.maximum(credits, 0)+.05
    return .2/len(merit)+.8*merit/merit.sum()


def paired_crossover(a, b, rng, group_size=2):
    if len(a) % group_size:
        raise ValueError('Genome must contain complete crossover groups')
    return np.where(np.repeat(rng.random(len(a)//group_size) < .5, group_size), a, b)


def search(evaluate, dimensions, population=12, generations=12, seed=0, random=False,
           structured=False, behavior_selection=False, adaptive=False, paired=True,
           proposal_provider=None, quality_ratio=None, quality_diversity=False,
           policy_provider=None):
    """Integer genes in [-3,3]. Zero anchor is measured, not assumed optimal.

    Elitist parent+offspring selection, Pareto rank/crowding tournaments,
    uniform crossover and discrete mutation. Exactly P*(G+1) distinct genomes
    evaluated; reject budgets larger than the finite search space.
    """
    if dimensions < 1 or population < 4 or generations < 1:
        raise ValueError('Need dimensions>0, population>=4, generations>=1')
    if population*(generations+1) > 7**dimensions:
        raise ValueError('Unique evaluation budget exceeds the integer search space')
    if structured and paired and dimensions % 2:
        raise ValueError('Paired search needs an even number of genes')
    if quality_ratio is not None and (not np.isfinite(quality_ratio) or quality_ratio < 1):
        raise ValueError('Quality ratio must be finite and at least one')
    if quality_diversity and (quality_ratio is None or not behavior_selection):
        raise ValueError('Quality diversity needs quality ratio and behavior descriptors')
    if policy_provider is not None and (not structured or random or proposal_provider is not None
                                        or quality_ratio is None or not behavior_selection):
        raise ValueError('Operator policy needs structured, quality-constrained behavior search')
    rng = np.random.default_rng(seed)
    records = []
    seen = set()
    credits = np.zeros(4)
    operator_names = ('single_mutation', 'group_mutation', 'group_crossover', 'immigrant')

    def unseen(gene):
        # Duplicates arise from crossover, saturated mutation and elitism.
        # Try random immigrants without spending a model forward. Near space
        # exhaustion, deterministic enumeration guarantees termination.
        for _ in range(64):
            if tuple(gene) not in seen:
                return gene
            gene = rng.integers(-3, 4, dimensions)
        for code in range(len(seen)+1):
            gene = np.empty(dimensions, dtype=int)
            for j in range(dimensions):
                gene[j] = code % 7-3
                code //= 7
            if tuple(gene) not in seen:
                return gene
        raise RuntimeError('No unseen chromosome despite valid budget')

    def measure(gene, generation, operator='initial', probabilities=None):
        duplicate = tuple(gene) in seen
        gene = unseen(gene)
        seen.add(tuple(gene))
        evaluated = evaluate(gene.copy())
        behavior = evaluated.get('behavior') if isinstance(evaluated, dict) else None
        score = np.asarray(evaluated['fitness'] if isinstance(evaluated, dict) else evaluated, dtype=float)
        if score.shape != (2,) or not np.isfinite(score).all():
            raise ValueError('Nonfinite or malformed candidate fitness')
        record = {'gene': gene.tolist(), 'fitness': score.tolist(), 'generation': generation,
                  'operator': 'duplicate_restart' if duplicate else operator}
        if behavior_selection:
            vector = np.asarray(behavior, dtype=float)
            if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                raise ValueError('Behavior selection requires finite nonempty descriptors')
            record['behavior'] = vector.tolist()
        if probabilities is not None:
            record['operator_probabilities'] = probabilities.tolist()
        records.append(record)
        return len(records)-1

    def population_order(indices):
        values = [records[i]['fitness'] for i in indices]
        return (quality_diverse_order(values, [records[i]['behavior'] for i in indices], fit_quality_limit)
                if quality_diversity else
                quality_order(values, fit_quality_limit) if quality_ratio is not None else
                ranked(values, [records[i]['behavior'] for i in indices]
                       if behavior_selection else None))

    initial = rng.integers(-3, 4, size=(population, dimensions))
    initial[0] = 0
    if quality_ratio is not None:
        # Start near the feasible RTN anchor. Dense random W4 perturbations
        # mostly waste evaluations outside the ordinary-quality corridor.
        initial[1:] = 0
        for gene in initial[1:]:
            coordinates = rng.choice(dimensions, size=min(dimensions, int(rng.integers(1, 3))), replace=False)
            gene[coordinates] = rng.choice([-1, 1], size=len(coordinates))
    live = [measure(gene, 0) for gene in initial]
    fit_quality_limit = max(records[0]['fitness'][0]*quality_ratio, 1e-12) if quality_ratio is not None else None
    for generation in range(1, generations+1):
        policy = None
        if policy_provider is not None:
            archive = [{'gene': list(r['gene']), 'fitness': list(r['fitness']),
                        'behavior': list(r['behavior']), 'operator': r['operator'],
                        'operator_reward': r.get('operator_reward', 0.)} for r in records]
            policy = policy_provider(generation, archive, fit_quality_limit)
            p = np.asarray(policy.get('operator_weights'), dtype=float)
            if (p.shape != (4,) or not np.isfinite(p).all() or np.any(p < .02)
                    or not np.isclose(p.sum(), 1., atol=1e-6)
                    or policy.get('mutation_step') not in (1, 2)
                    or not isinstance(policy.get('distant_mate_probability'), (float, int))
                    or not 0. <= policy['distant_mate_probability'] <= 1.):
                raise ValueError('Invalid bounded operator policy')
        proposed = []
        if proposal_provider is not None:
            # Copy only FIT measurements; provider cannot mutate archive or see SELECT/TEST.
            archive = [{'gene': list(r['gene']), 'fitness': list(r['fitness'])} for r in records]
            seen_batch = set(seen)
            for value in proposal_provider(generation, archive):
                array = np.asarray(value)
                if (array.shape != (dimensions,) or array.dtype.kind not in 'iu'
                        or np.any(array < -3) or np.any(array > 3)):
                    raise ValueError('Proposal must be a bounded integer genome')
                gene = array.astype(int)
                if tuple(gene) not in seen_batch:
                    proposed.append(gene)
                    seen_batch.add(tuple(gene))
                if len(proposed) >= population:
                    break
        order = population_order(live)
        priority = {live[index]: rank for rank, index in enumerate(order)}
        def tournament_index():
            a, b = rng.choice(live, 2, replace=False)
            return min((a, b), key=priority.get)
        children = []
        for gene in proposed:
            children.append(measure(gene, generation, proposal_provider.label))
        for _ in range(population-len(children)):
            op = None
            probabilities = (p if policy is not None else
                             operator_probabilities(credits) if adaptive else np.full(4, .25))
            parent = None
            if random:
                gene = rng.integers(-3, 4, dimensions)
            elif structured:
                op = int(rng.choice(4, p=probabilities))
                parent = tournament_index()
                gene = np.asarray(records[parent]['gene']).copy()
                group_size = 2 if paired else 1
                if op == 0:
                    delta = (rng.choice([-policy['mutation_step'], policy['mutation_step']])
                             if policy is not None else rng.choice([-2, -1, 1, 2]))
                    gene[rng.integers(dimensions)] += delta
                elif op == 1:
                    start = int(rng.integers(dimensions//group_size))*group_size
                    deltas = (rng.choice([-policy['mutation_step'], policy['mutation_step']], group_size)
                              if policy is not None else rng.choice([-2, -1, 1, 2], group_size))
                    gene[start:start+group_size] += deltas
                elif op == 2:
                    if (policy is not None and behavior_selection and
                            rng.random() < policy['distant_mate_probability']):
                        features = np.asarray([records[i]['behavior'] for i in live], dtype=float)
                        features /= np.maximum(features.std(axis=0), 1e-6)
                        parent_pos = live.index(parent)
                        mate_pos = max((j for j in range(len(live)) if j != parent_pos),
                                       key=lambda j: (float(np.square(features[j]-features[parent_pos]).mean()), -j))
                        mate = live[mate_pos]
                    else:
                        mate = tournament_index()
                    other = np.asarray(records[mate]['gene'])
                    gene = paired_crossover(gene, other, rng, group_size)
                else:
                    if quality_ratio is not None and rng.random() < .8:
                        gene = np.zeros(dimensions, dtype=int)
                        coordinates = rng.choice(dimensions, size=min(dimensions, int(rng.integers(1, 4))), replace=False)
                        gene[coordinates] = rng.choice([-2, -1, 1, 2], size=len(coordinates))
                    else:
                        gene = rng.integers(-3, 4, dimensions)
                gene = np.clip(gene, -3, 3)
            else:
                a, b = (np.asarray(records[tournament_index()]['gene']) for _ in range(2))
                gene = np.where(rng.random(dimensions) < .5, a, b)
                mutation = rng.random(dimensions) < max(1/dimensions, .1)
                mutation[rng.integers(dimensions)] = True
                gene = np.clip(gene + mutation*rng.choice([-2, -1, 1, 2], dimensions), -3, 3)
            child = measure(gene, generation, operator_names[op] if op is not None else 'random' if random else 'uniform',
                            probabilities if structured else None)
            children.append(child)
            if adaptive and parent is not None and records[child]['operator'] != 'duplicate_restart':
                before, after = np.asarray(records[parent]['fitness']), np.asarray(records[child]['fitness'])
                if quality_ratio is not None:
                    reward = float((after[0] <= fit_quality_limit and
                                    (before[0] > fit_quality_limit or after[1] < before[1])) or
                                   (before[0] > fit_quality_limit and after[0] < before[0]))
                else:
                    reward = float(np.any(after < before) and not (np.all(before <= after) and np.any(before < after)))
                credits[op] = .9*credits[op]+.1*reward
                records[child]['operator_reward'] = reward
            if policy_provider is not None and op is not None:
                previous = [r['fitness'][1] for r in records[:-1]
                            if r['fitness'][0] <= fit_quality_limit]
                after = records[child]['fitness']
                reward = (min(1., max(0., (min(previous)-after[1])/max(abs(min(previous)), 1e-12)))
                          if previous and after[0] <= fit_quality_limit else 0.)
                if records[child]['operator'] == 'duplicate_restart':
                    reward = 0.
                policy_provider.observe(op, reward)
                records[child]['operator_reward'] = reward
        pool = live + children
        live = [pool[i] for i in population_order(pool)[:population]]
    return records, [i for i in fronts([r['fitness'] for r in records])[0]]
