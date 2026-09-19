"""Independent scientific checks of stored results, never reruns training.

Run with iclr_20260919/.venv/bin/python. Missing experiment arms are incomplete,
not successes. The verifier writes only artifact_verification.json and its own
brief Markdown report. It does not import the experiment implementations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
CHECKS = []
FILES = {}


class Incomplete(Exception):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_file(path):
    path = Path(path)
    if not path.exists():
        raise Incomplete(f'Missing {path.relative_to(WORKSPACE)}')
    FILES[str(path.relative_to(WORKSPACE))] = digest(path)
    return path


def read_json(path):
    path = record_file(path)
    try:
        obj = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise Incomplete(f'JSON may still be being written: {path.name}: {exc}')
    finite_tree(obj)
    return obj


def finite_tree(obj):
    if isinstance(obj, float):
        assert math.isfinite(obj), f'Non-finite JSON scalar {obj}'
    elif isinstance(obj, dict):
        for value in obj.values():
            finite_tree(value)
    elif isinstance(obj, list):
        for value in obj:
            finite_tree(value)


def npz(path):
    with np.load(record_file(path), allow_pickle=False) as payload:
        out = {key: payload[key] for key in payload.files}
    for key, value in out.items():
        if np.issubdtype(value.dtype, np.number):
            assert np.isfinite(value).all(), f'Nonfinite {path}:{key}'
    return out


def tensor(path):
    out = torch.load(record_file(path), map_location='cpu', weights_only=True)
    assert isinstance(out, torch.Tensor) and out.ndim == 1
    a = out.double().numpy()
    assert np.isfinite(a).all() and (a > 0).all(), f'Invalid costs in {path}'
    return a


def close(a, b, label='', atol=2e-10, rtol=2e-10):
    aa, bb = np.asarray(a), np.asarray(b)
    assert aa.shape == bb.shape, f'{label}: shape {aa.shape} != {bb.shape}'
    assert np.isfinite(aa).all() and np.isfinite(bb).all(), f'{label}: nonfinite'
    assert np.allclose(aa, bb, atol=atol, rtol=rtol), (
        f'{label}: max error {np.max(np.abs(aa-bb))}')


def check(name, fn):
    begin = datetime.now(timezone.utc)
    try:
        details = fn()
        result = dict(name=name, status='pass', details=details)
    except Incomplete as exc:
        result = dict(name=name, status='incomplete', reason=str(exc))
    except Exception as exc:
        result = dict(name=name, status='fail', reason=str(exc), traceback=traceback.format_exc())
    result['seconds'] = (datetime.now(timezone.utc)-begin).total_seconds()
    CHECKS.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def bootstrap(values, samples=1000, seed=492190, rng=None):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1, 'Bootstrap input must be one value per geometry'
    rng = np.random.default_rng(seed) if rng is None else rng
    boot = []
    for start in range(0, samples, 100):
        ix = rng.integers(len(values), size=(min(100, samples-start), len(values)))
        boot.extend(values[ix].mean(axis=1))
    return np.quantile(boot, [.025, .975])


def stat_check(reported, values, label=''):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1
    assert reported['n_geometry'] == len(values), f'{label}: wrong sampling unit'
    close(reported['mean'], math.fsum(values)/len(values), label+' mean')
    close(reported['ci95'], bootstrap(values), label+' CI')


def verify_history():
    summary = read_json(ROOT/'historical_recomputed/summary.json')
    assert len(summary['rows']) == 8
    rng = np.random.default_rng(summary['bootstrap_seed'])
    for row in summary['rows']:
        folder = WORKSPACE/'strong_hosts_20260919/evidence'/row['host']
        costs = {mode: np.stack([tensor(folder/f'{mode}-seed{s}'/f"{row['case']}-costs.pt")
                                for s in row['training_seeds']])
                 for mode in ('route', 'random', 'learned')}
        n = row['test_instances']
        assert all(v.shape == (3, n) for v in costs.values())
        for mode, v in costs.items():
            close(row['means'][mode], v.mean(), mode)
            close(row['per_seed_means'][mode], v.mean(axis=1), mode)
        close(row['native_mean'], tensor(folder/f"native-{row['case']}-costs.pt").mean())
        ids = rng.integers(n, size=(summary['bootstrap_samples'], n))
        base = costs['route']
        for mode in ('random', 'learned'):
            rec = row['comparisons'][mode]
            improvement = base.mean(axis=0)-costs[mode].mean(axis=0)
            close(rec['improvement_percent'], 100*improvement.mean()/base.mean())
            close(rec['per_seed_improvement_percent'], 100*(base.mean(1)-costs[mode].mean(1))/base.mean(1))
            sampled = 100*improvement[ids].sum(axis=1)/base.mean(axis=0)[ids].sum(axis=1)
            close(rec['paired_instance_ci'], np.quantile(sampled, [.025, .975]))
    for name, expected in summary['input_sha256'].items():
        assert digest(WORKSPACE/name) == expected, f'Historical input changed: {name}'
    assert digest(ROOT/'recompute_historical.py') == summary['script_sha256']
    return dict(rows=8, cost_vectors=len(summary['input_sha256']), paired_bootstrap_verified=True,
                bootstrap_unit='geometry, conditional on three fitted seeds sharing warmup',
                percentage_estimand='ratio of mean cost differences to mean baseline cost')


def expanded_tour_q(x, paths):
    """Independent verification by expanding every component order to n vertices."""
    n, m = len(x), len(paths)
    best = [math.inf]*(m-1)
    count = 0
    for order in itertools.permutations(range(1, m)):
        tour = list(paths[0])
        for component in order:
            tour.extend(paths[component])
        assert len(tour) == n and len(set(tour)) == n
        edges = list(zip(tour, tour[1:]+tour[:1]))
        # This validates original-node edges; it does not use contracted W.
        value = math.fsum(math.hypot(*(x[a]-x[b])) for a, b in edges)
        best[order[0]-1] = min(best[order[0]-1], value)
        count += 1
    return np.array(best), count


def mask_from_paths(paths, n):
    pred = {node for path in paths for node in path[1:]}
    source_members = set(paths[0])
    return np.array([node in pred or node in source_members for node in range(n)])


def verify_stress():
    folder = ROOT/'counterfactual_stress'
    summary = read_json(folder/'summary.json')
    assert len(summary['cells']) == 6 and summary['not_on_policy'] and summary['fixed_source']
    total = 0
    independent_forests = 0
    max_q_error = 0.
    for cell in summary['cells']:
        tag = cell['distribution']+str(cell['n'])
        x = npz(folder/f'{tag}-coordinates.npz')['coordinates']
        records = read_json(folder/f'{tag}-records.json')
        assert len(records) == len(x) == cell['instances'] == 256
        assert hashlib.sha256(x.tobytes()).hexdigest() == cell['coordinates_sha256']
        selection = set(np.random.default_rng(cell['seed']+190).choice(len(x), 8, replace=False).tolist())
        floors, conflicts, near, original = [], [], [], []
        for i, rec in enumerate(records):
            paths, partner = rec['paths'], rec['partner_paths']
            assert len(paths) == len(partner) == 6
            assert all(len(path) for path in paths+partner)
            assert sorted(v for p in paths for v in p) == list(range(len(x[i])))
            assert sorted(v for p in partner for v in p) == list(range(len(x[i])))
            assert paths[0] == partner[0]
            assert [p[0] for p in paths] == [p[0] for p in partner]
            assert sorted(p[-1] for p in paths) == sorted(p[-1] for p in partner)
            assert np.array_equal(mask_from_paths(paths, len(x[i])), mask_from_paths(partner, len(x[i])))
            q = np.asarray(rec['conditional_q'])
            assert q.shape == (2, 5) and np.isfinite(q).all()
            regret = q-q.min(axis=1, keepdims=True)
            floor = regret.mean(axis=0).min()
            close(rec['alias_floor'], floor)
            assert rec['disjoint_optima'] == bool(floor > 1e-9)
            source = paths[0][-1]
            target = min(range(5), key=lambda j: math.hypot(*(x[i, source]-x[i, paths[j+1][0]])))
            close(rec['nearest_mean_regret'], regret[:, target].mean())
            tour = [v for p in paths for v in p]
            cost = math.fsum(math.hypot(*(x[i,a]-x[i,b])) for a,b in zip(tour, tour[1:]+tour[:1]))
            close(rec['original_tour_cost'], cost)
            assert q[0].min() <= cost+1e-10
            close(rec['control_floor'], 0.)
            if i in selection:
                for state, forest in enumerate((paths, partner)):
                    independent, orders = expanded_tour_q(x[i], forest)
                    close(independent, q[state], tag+f' record {i} state {state}', atol=1e-11)
                    max_q_error = max(max_q_error, float(np.max(np.abs(independent-q[state]))))
                    assert orders == 120
                    independent_forests += 1
            floors.append(floor); conflicts.append(floor > 1e-9)
            near.append(regret[:,target].mean()); original.append(cost)
        close(cell['mean_bayes_regret'], np.mean(floors))
        close(cell['conflict_fraction'], np.mean(conflicts))
        close(cell['nearest_mean_regret'], np.mean(near))
        close(cell['max_bayes_regret'], max(floors))
        close(cell['mean_original_tour_cost'], np.mean(original))
        rng = np.random.default_rng(cell['seed']+900000)
        close(cell['bayes_regret_ci'], bootstrap(floors, 4000, rng=rng))
        close(cell['conflict_fraction_ci'], bootstrap(conflicts, 4000, rng=rng))
        total += len(records)
    assert total == 1536
    return dict(geometry_pairs=total, all_legality_masks_and_summary_statistics_verified=True,
                independently_expanded_forests=independent_forests,
                full_node_tours_checked=independent_forests*120, max_completion_q_error=max_q_error,
                bootstrap_unit='whole geometry pair', positive_regret_tolerance=1e-9)


def verify_certificate():
    cert = read_json(ROOT/'witness_interval_certificate.json')
    source = WORKSPACE/'strong_hosts_20260919/evidence/endpoint-aliasing.json'
    assert digest(source) == cert['input_sha256']
    assert digest(ROOT/'verify_witness_interval.py') == cert['script_sha256']
    case = json.loads(source.read_text(), parse_float=Decimal)
    with localcontext() as context:
        context.prec = 65
        x = case['coordinates']; n = len(x)
        d = [[sum((a-b)**2 for a,b in zip(u,v)).sqrt() for v in x] for u in x]
        qs = []
        for fixed in case['partial_edges']:
            q = {}
            for rest in itertools.permutations(range(1,n)):
                tour = (0,)+rest
                succ = dict(zip(tour, tour[1:]+tour[:1]))
                if not all(succ[a] == b for a,b in fixed):
                    continue
                cost = sum(d[a][b] for a,b in succ.items())
                a = succ[0]; q[a] = min(q.get(a,cost),cost)
            qs.append(q)
        actions = sorted(qs[0])
        regret = min(sum(q[a] for q in qs) for a in actions)/2-sum(min(q.values()) for q in qs)/2
        assert Decimal(cert['bayes_regret_lower']) <= regret <= Decimal(cert['bayes_regret_upper'])
        for k,q in enumerate(qs):
            for action,value in q.items():
                low,high = cert['conditional_q_intervals'][k][str(action)]
                assert Decimal(low) <= value <= Decimal(high)
    return dict(independent_arithmetic='65-digit Decimal distance sums and full-node enumeration',
                interval_contains_independent_result=True, decimal_regret=str(regret),
                certified_interval=[cert['bayes_regret_lower'],cert['bayes_regret_upper']])


def diagnostic_data(folder, name):
    paths = sorted((folder/'data').glob(name+'_s*.npz'))
    if len(paths) != 3:
        raise Incomplete(f'{name}: expected three diagnostic test shards, found {len(paths)}')
    shards = [npz(path) for path in paths]
    data = {key: np.concatenate([shard[key] for shard in shards],axis=0)
            if key in ('x','tails','q','internal') else shards[0][key] for key in shards[0]}
    return data


def dp_q(x, heads, tails):
    """Held-Karp recursion, independent from permutation-based label generators."""
    m = len(heads)
    dist = lambda i,j: math.hypot(*(x[tails[i]]-x[heads[j]]))
    @lru_cache(None)
    def remaining(current, mask):
        if mask == 0:
            return dist(current,0)
        return min(dist(current,j)+remaining(j,mask^(1<<j)) for j in range(1,m) if mask&(1<<j))
    all_remaining = (1<<m)-2
    return np.array([dist(0,j)+remaining(j,all_remaining^(1<<j)) for j in range(1,m)])


def verify_diagnostic_data():
    folder = ROOT/'exact_diagnostic/results'
    train = npz(folder/'data/train.npz')['x']
    val = npz(folder/'data/validation.npz')['x']
    fingerprints = lambda x: {hashlib.sha256(a.tobytes()).digest() for a in x}
    training, validation = fingerprints(train), fingerprints(val)
    assert len(training) == len(train) and len(validation) == len(val)
    assert not training&validation
    tests = set(); checked = 0; max_error = 0.
    for name in ('iid_n9','cluster_n9','ood_n7','ood_n11','ood_n13'):
        data = diagnostic_data(folder,name)
        hashes = fingerprints(data['x'])
        assert len(hashes) == len(data['x'])
        assert not hashes&(training|validation|tests)
        tests |= hashes
        sample = np.random.default_rng(32198).choice(len(data['x']),12,replace=False)
        for geom in sample:
            for state in range(2):
                got = dp_q(data['x'][geom].astype(float),data['heads'],data['tails'][geom,state])
                close(got,data['q'][geom,state],name+' DP labels',atol=2e-12)
                max_error = max(max_error,float(np.max(np.abs(got-data['q'][geom,state]))))
                checked += 1
    return dict(train_geometry=len(training),validation_geometry=len(validation),
                test_geometry=len(tests), geometry_disjoint=True,
                independent_held_karp_forests=checked,max_q_error=max_error)


def verify_model_arrays(folder,name,model,data,reported,all_pairings=False):
    suffix = 'all_pairings_' if all_pairings else ''
    payload = npz(folder/f'{name}_{suffix}{model}.npz')
    q = data['q']; regret = q-q.min(axis=-1,keepdims=True)
    actions = payload['actions']
    assert actions.shape == q.shape[:2]
    assert np.issubdtype(actions.dtype,np.integer) and actions.min()>=0 and actions.max()<q.shape[-1]
    expected = np.take_along_axis(regret,actions[...,None],axis=-1).squeeze(-1)
    close(payload['regret'],expected,model+' stored regret')
    assert np.array_equal(payload['accuracy'],expected<1e-8)
    assert np.array_equal(actions,payload['pred'].argmin(axis=-1))
    if model.startswith('blind'):
        assert np.array_equal(payload['pred'],np.repeat(payload['pred'][:,0:1],q.shape[1],axis=1))
        assert np.all(expected.mean(axis=1)>=regret.mean(axis=1).min(axis=-1)-1e-12)
    stat_check(reported['regret'],expected.mean(axis=1),model+' regret')
    stat_check(reported['accuracy'],(expected<1e-8).mean(axis=1),model+' accuracy')
    return payload


def verify_diagnostic_models():
    folder = ROOT/'exact_diagnostic/results'
    summary = read_json(folder/'summary.json')
    expected_names = {f'{kind}_s{seed}' for kind in ('blind','aware') for seed in summary['protocol']['args']['seeds']}
    names = {m['name'] for m in summary['models']}
    if names != expected_names:
        raise Incomplete(f'Conditional model arms incomplete: present {sorted(names)}, expected {sorted(expected_names)}')
    assert len(names) == len(summary['models']) == 6
    assert len({m['parameters'] for m in summary['models']}) == 1
    for model in summary['models']:
        history = read_json(folder/(model['name']+'_training.json'))
        assert history[-1]['step'] == summary['protocol']['args']['steps']
        best = min(history,key=lambda h:h['val_regret'])
        assert best['step'] == model['best_step']
        close(best['val_regret'],model['validation_regret'])
        record_file(folder/(model['name']+'.pt'))
    files = 0
    for name,rec in summary['datasets'].items():
        data = diagnostic_data(folder,name); q = data['q']; r = q-q.min(axis=-1,keepdims=True)
        assert rec['n_geometry'] == len(q) and rec['n_forests'] == 2*len(q)
        stat_check(rec['pair_bayes_lower_bound'],r.mean(axis=1).min(axis=-1),name+' floor')
        optimal = r<1e-9
        stat_check(rec['conflict_rate'],~(optimal[:,0]&optimal[:,1]).any(axis=-1),name+' conflict')
        base = npz(folder/f'{name}_baselines.npz')
        close(base['pair_lower_bound'],r.mean(axis=1).min(axis=-1))
        nearest = np.linalg.norm(data['x'][:,data['heads'][1:]]-data['x'][:,0:1],axis=-1).argmin(axis=-1)
        nearest_r = np.take_along_axis(r,np.repeat(nearest[:,None,None],2,axis=1),axis=-1).squeeze(-1)
        close(base['nearest_regret'],nearest_r)
        stat_check(rec['nearest_regret'],nearest_r.mean(axis=1))
        gr = np.take_along_axis(r,base['greedy_actions'][...,None],axis=-1).squeeze(-1)
        close(base['greedy_regret'],gr)
        stat_check(rec['greedy_completion_regret'],gr.mean(axis=1))
        assert set(rec['models']) == names
        for model in names:
            verify_model_arrays(folder,name,model,data,rec['models'][model]); files += 1
    return dict(completed_models=6,model_dataset_arrays=files,checkpoint_selection='validation only, checked against history',
                identical_parameter_count=summary['models'][0]['parameters'],
                bootstrap_unit='geometry, averaging paired states within unit',nan_check='all loaded JSON numbers and numeric arrays finite')


def verify_extended():
    folder = ROOT/'exact_diagnostic/results'
    summary = read_json(folder/'extended_summary.json')
    models = read_json(folder/'summary.json')['models']
    count = 0
    for name,rec in summary['all_pairings'].items():
        data = npz(folder/'data'/f'{name}_all_pairings.npz')
        assert data['tails'].shape[1] == math.factorial(int(data['k'])) == rec['pairings_per_geometry']
        for g in (0,17,301):
            assert len({tuple(t) for t in data['tails'][g]}) == rec['pairings_per_geometry']
        q = data['q']; r = q-q.min(-1,keepdims=True)
        stat_check(rec['exact_blind_conditional_bayes_regret'],r.mean(1).min(-1))
        assert set(rec['models']) == {m['name'] for m in models}
        for model in models:
            verify_model_arrays(folder,name,model['name'],data,rec['models'][model['name']],True); count += 1
    for name,arms in summary['wrong_pairing'].items():
        data = diagnostic_data(folder,name); r = data['q']-data['q'].min(-1,keepdims=True)
        for model,rec in arms.items():
            correct = npz(folder/f'{name}_{model}.npz')
            wrong = npz(folder/f'{name}_wrong_pairing_{model}.npz')
            assert np.array_equal(wrong['actions'],correct['actions'][:,::-1])
            actual = np.take_along_axis(r,wrong['actions'][...,None],axis=-1).squeeze(-1)
            close(wrong['regret'],actual)
            difference = actual.mean(1)-correct['regret'].mean(1)
            close(wrong['paired_delta'],difference)
            stat_check(rec['wrong_pairing_regret'],actual.mean(1))
            stat_check(rec['wrong_minus_correct'],difference)
    assert summary['permutation_equivariance_max_abs_error'] < 1e-5
    return dict(all_pairing_model_arrays=count,full_observation_class_bayes_risk_verified=True,
                corrupted_pairing_actions_and_effects_verified=True)


def verify_assignment():
    folder = ROOT/'exact_diagnostic/results'
    summary = read_json(folder/'assignment_summary.json')
    for name,rec in summary['datasets'].items():
        data = diagnostic_data(folder,name); q=data['q']; r=q-q.min(-1,keepdims=True)
        output = npz(folder/f'{name}_assignment.npz')
        assert np.all(output['estimated_completion'] <= q+1e-8)
        assert np.array_equal(output['actions'],output['estimated_completion'].argmin(-1))
        actual = np.take_along_axis(r,output['actions'][...,None],-1).squeeze(-1)
        close(output['regret'],actual)
        stat_check(rec['regret'],actual.mean(1));stat_check(rec['accuracy'],(actual<1e-8).mean(1))
        close(rec['seconds_per_forest'],rec['cpu_seconds']/actual.size)
    return dict(datasets=len(summary['datasets']),all_assignment_scores_lower_bound_exact_q=True,
                action_regrets_recomputed_from_exact_q=True,unit='geometry')


def verify_shuffled():
    folder = ROOT/'exact_diagnostic/results'
    summary = read_json(folder/'shuffled_summary.json')
    if len(summary['models']) != 3:
        raise Incomplete('Expected three completed shuffled-correspondence models')
    formal=read_json(folder/'summary.json')
    assert {m['seed'] for m in summary['models']} == set(formal['protocol']['args']['seeds'])
    for model in summary['models']:
        history=read_json(folder/(model['name']+'_training.json'))
        assert history[-1]['step']==formal['protocol']['args']['steps']
        assert model['parameters']==formal['models'][0]['parameters']
        best=min(history,key=lambda h:h['validation_regret'])
        assert best['step']==model['best_step']
        close(best['validation_regret'],model['validation_regret'])
    for name,arms in summary['datasets'].items():
        data=diagnostic_data(folder,name)
        assert set(arms)=={m['name'] for m in summary['models']}
        for model,rec in arms.items():
            verify_model_arrays(folder,name,model,data,rec)
    return dict(seeds=3,meaning='separately trained random-correspondence control; post-pilot experiment')


def combined_check(reported, seed_by_geometry):
    values=np.asarray(seed_by_geometry,dtype=float)
    assert values.ndim==2 and values.shape[0]==reported['n_training_seeds']==3
    stat_check(reported,values.mean(axis=0))
    close(reported['seed_means'],values.mean(axis=1))
    close(reported['seed_sd'],values.mean(axis=1).std(ddof=1))
    rng=np.random.default_rng(190919992)
    boot=[]
    for _ in range(1500):
        selected_seeds=rng.integers(values.shape[0],size=values.shape[0])
        selected_geometries=rng.integers(values.shape[1],size=values.shape[1])
        # Average the Cartesian seed-by-geometry sample; states were averaged
        # before resampling. This independently checks the reported two-way CI.
        seed_average=values[selected_seeds].mean(axis=0)
        boot.append(seed_average[selected_geometries].mean())
    close(reported['hierarchical_ci95'],np.quantile(boot,[.025,.975]))


def verify_consolidated_diagnostic():
    folder=ROOT/'exact_diagnostic/results'
    summary=read_json(ROOT/'exact_diagnostic/summary.json')
    assert summary['status']=='complete' and len(summary['models'])==9
    seeds=summary['training_seeds']; comparisons=0
    for name,rec in summary['datasets'].items():
        data=diagnostic_data(folder,name)
        conditional_optimal_length=data['q'].min(axis=-1)+data['internal']
        assert (conditional_optimal_length>0).all()
        arrays={}
        for method in ('blind','aware','shuffled'):
            raw=np.stack([npz(folder/f'{name}_{method}_s{s}.npz')['regret'] for s in seeds])
            arrays[method]=raw.mean(axis=2)
            combined_check(rec['models'][method],arrays[method])
            combined_check(rec['models'][method]['percent_optimal_complete_tour_length'],
                           (100*raw/conditional_optimal_length[None]).mean(axis=2))
            comparisons+=2
        arrays['wrong_pairing']=np.stack([npz(folder/f'{name}_wrong_pairing_aware_s{s}.npz')['regret'].mean(axis=1) for s in seeds])
        combined_check(rec['models']['wrong_pairing'],arrays['wrong_pairing'])
        for key,left,right in [('blind_minus_aware','blind','aware'),('shuffled_minus_aware','shuffled','aware'),('wrong_minus_correct','wrong_pairing','aware')]:
            combined_check(rec[key],arrays[left]-arrays[right]);comparisons+=1
        close(rec['aware_relative_regret_reduction_vs_blind'],1-arrays['aware'].mean()/arrays['blind'].mean())
    for name,rec in summary['all_pairings'].items():
        data=npz(folder/'data'/f'{name}_all_pairings.npz')
        regret=data['q']-data['q'].min(axis=-1,keepdims=True)
        bayes=regret.mean(axis=1).min(axis=-1)
        stat_check(rec['exact_blind_conditional_bayes_regret'],bayes)
        for method in ('blind','aware'):
            array=np.stack([npz(folder/f'{name}_all_pairings_{method}_s{s}.npz')['regret'].mean(axis=1) for s in seeds])
            combined_check(rec['models'][method],array);comparisons+=1
            if method=='aware':
                combined_check(rec['blind_bayes_minus_aware'],bayes[None]-array);comparisons+=1
    return dict(combined_statistics_checked=comparisons,seed_and_geometry_bootstrap_verified=True,
                normalized_regret_denominator='Optimal completion preserving the particular forest, not unconstrained TSP OPT',
                caveat='Only three independently trained seeds; the bootstrap is not evidence of broad seed-population robustness.')


def verify_information_curve():
    folder=ROOT/'exact_diagnostic/results'
    summary=read_json(folder/'information_curve.json')
    for name,rec in summary['datasets'].items():
        data=npz(folder/'data'/f'{name}_all_pairings.npz')
        q=data['q']; regret=q-q.min(axis=-1,keepdims=True); tails=data['tails']
        saved=npz(folder/f'{name}_information_curve.npz')['bayes_regret']
        columns=[]
        for exposed,entry in enumerate(rec['curve']):
            groups={}
            for state,association in enumerate(tails[0]):
                groups.setdefault(tuple(association[1:exposed+1]),[]).append(state)
            assert len(groups)==entry['observation_groups']
            risk=np.zeros(len(q))
            for states in groups.values():
                risk+=regret[:,states].sum(axis=1).min(axis=-1)/q.shape[1]
            columns.append(risk)
            stat_check(entry['bayes_regret'],risk)
            if exposed:
                stat_check(entry['reduction_from_previous'],columns[-2]-risk)
        actual=np.stack(columns,axis=1)
        close(actual,saved)
        assert np.all(np.diff(actual,axis=1)<=1e-12)
        assert np.all(actual[:,2:]==0)
    return dict(datasets=len(summary['datasets']),observation_partitions_independently_reconstructed=True,
                information_risk_pointwise_nonincreasing=True,two_partners_determine_third=True)


def verify_adapter(folder):
    config=read_json(folder/'config.json')
    audit=folder/'final_test_audit.pt'
    sets=torch.load(record_file(audit if audit.exists() else folder/'final_test_data.pt'),map_location='cpu',weights_only=True)
    manifest=read_json(folder/'frozen_checkpoint_manifest.json')
    data_manifest=read_json(folder/'final_test_manifest.json')
    aggregate=read_json(folder/'aggregate.json')
    expected={f'{mode}-{kind}-seed{seed}' for mode in ('route','random') for kind in ('blind','aware') for seed in (12031,12037,12041)}
    assert set(manifest['checkpoints'])==expected
    costs={name:{} for name in sets}; conditionals={name:{} for name in sets}; parameters=set(); training_data_hashes=set()
    for name,data in sets.items():
        x=data['x'].double().numpy(); tour=data['pi'].numpy()
        assert x.ndim==3 and x.shape[-1]==2 and np.isfinite(x).all()
        assert tour.shape==x.shape[:2] and np.array_equal(np.sort(tour,axis=1),np.broadcast_to(np.arange(x.shape[1]),tour.shape))
        assert hashlib.sha256(data['x'].numpy().tobytes()).hexdigest()==data_manifest[name]['sha']
        xy=np.take_along_axis(x,tour[...,None],axis=1)
        true_teacher=np.linalg.norm(xy-np.roll(xy,1,axis=1),axis=-1).sum(axis=1)
        close(data['teacher_cost'].double().numpy(),true_teacher,'independent teacher tour cost',atol=5e-6,rtol=2e-7)
        close(data_manifest[name]['native_mean'],data['native_cost'].double().numpy().mean(),atol=2e-6)
        close(data_manifest[name]['teacher_mean'],true_teacher.mean(),atol=2e-6)
    for run in sorted(expected):
        sub=folder/run; report=read_json(sub/'test_summary.json');train=read_json(sub/'training_summary.json')
        history=read_json(sub/'history.json')
        assert train['steps']==config['steps'] and history[-1]['step']==config['steps']
        best=min(history,key=lambda h:h['normalized_validation'])
        assert best['step']==report['selected_step']==train['best_step']
        assert digest(sub/'best.pt')==manifest['checkpoints'][run]
        ckpt=torch.load(record_file(sub/'best.pt'),map_location='cpu',weights_only=True)
        assert ckpt['step']==report['selected_step']
        assert (ckpt['seed'],ckpt['aware'],ckpt['mode'])==(report['seed'],report['aware'],report['mode'])
        weights=list(ckpt['residual'].values())+list(ckpt.get('decoder',{}).values())
        assert all(torch.isfinite(weight).all() for weight in weights)
        assert sum(weight.numel() for weight in weights)==train['parameters']
        training_manifest=read_json(sub/'training_manifest.json')
        training_data_hashes.add(training_manifest['data_sha'])
        # Check only code that controls the run. Later-added diagnostic scripts
        # need not have existed at all seeds' launch times.
        core=['component_adapter.py','strong_adapter.py','run_pilot.py']
        if 'decoder' in ckpt:core+=['run_secondary.py','secondary_component.py']
        for filename in core:
            assert digest(record_file(folder.parent/filename))==training_manifest['sources'][filename]
        parameters.add(train['parameters'])
        arm=report['mode']+('-aware' if report['aware'] else '-blind')
        for name,data in sets.items():
            value=tensor(sub/f'{name}-costs.pt'); raw=report['test'][name]
            assert len(value)==len(data['x'])
            close(raw['mean'],value.mean(),run+' mean')
            close(raw['native_relative_percent'],(value/data['native_cost'].double().numpy()-1).mean()*100,atol=1e-4)
            close(raw['teacher_relative_percent'],(value/data['teacher_cost'].double().numpy()-1).mean()*100,atol=1e-4)
            cond=torch.load(record_file(sub/f'{name}-conditional.pt'),map_location='cpu',weights_only=True)
            assert cond['generator_seed']==9417320 and cond['mode']==report['mode']
            ll=cond['nll'].double().numpy(); correct=cond['correct'].double().numpy()
            assert ll.shape==correct.shape==(3,len(value)) and np.isfinite(ll).all() and (ll>=0).all()
            assert np.isin(correct,[0.,1.]).all()
            close(raw['conditional']['accuracy'],correct.mean(),atol=5e-8)
            close(raw['conditional']['nll'],ll.mean(),atol=2e-6)
            conditionals[name].setdefault(arm,[]).append((ll.mean(),correct.mean(),report['selected_step']))
            costs[name].setdefault(arm,[]).append(value)
    assert len(parameters)==1,'Matched arms have unequal parameter counts'
    assert len(training_data_hashes)==1,'Matched arms were trained on different stored data'
    rng=np.random.default_rng(880192)
    for name,methods in costs.items():
        methods={arm:np.stack(values) for arm,values in methods.items()}
        for arm,values in methods.items():
            rec=aggregate['summary'][name][arm]
            assert values.shape==(3,len(sets[name]['x'])) and rec['seeds']==3
            close(rec['mean'],values.mean());close(rec['seed_sd'],values.mean(1).std(ddof=1))
            cond=np.array(conditionals[name][arm])
            close(rec['conditional_nll'],cond[:,0].mean(),atol=2e-6)
            close(rec['conditional_accuracy'],cond[:,1].mean(),atol=5e-8)
            assert rec['selected_steps']==cond[:,2].tolist()
        for treatment,base in [('random-aware','random-blind'),('random-aware','route-aware'),('route-aware','route-blind'),('random-aware','native'),('route-aware','native')]:
            value=methods[treatment]
            reference=(np.broadcast_to(sets[name]['native_cost'].double().numpy(),value.shape) if base=='native' else methods[base])
            perseed=(reference-value)/reference*100
            pergeometry=perseed.mean(axis=0)
            rec=aggregate['comparisons'][name][treatment+' vs '+base]
            close(rec['improvement_percent'],pergeometry.mean())
            close(rec['per_seed_improvement_percent'],perseed.mean(1))
            close(rec['paired_instance_ci95'],bootstrap(pergeometry,5000,rng=rng))
            close(rec['win_fraction'],(value.mean(0)<reference.mean(0)).mean())
    return dict(completed_arms=12,test_sets=len(sets),finite_positive_costs=True,
                selected_checkpoints_match_frozen_hashes=True,
                identical_parameter_count=next(iter(parameters)),conditional_vectors_checked=48,
                checkpoint_weights_finite_and_count_verified=True,training_core_source_hashes_verified=True,
                teacher_tours_independently_costed=sum(len(d['x']) for d in sets.values()),
                bootstrap_unit='paired geometry after averaging three seed-specific relative effects',
                percentage_estimand='mean instance-relative cost improvement, distinct from historical ratio of means')


def verify_adapter_posthoc(folder):
    rows=read_json(folder/'source_rng_variability.json')
    assert len(rows)==24
    for rec in rows:
        arm='random-'+('aware' if rec['aware'] else 'blind')+'-seed'+str(rec['training_seed'])
        values=np.array([tensor(folder/arm/f"{rec['dataset']}-source-rng{s}-costs.pt").mean() for s in rec['source_seeds']])
        close(rec['means'],values);close(rec['average'],values.mean());close(rec['source_seed_sd'],values.std(ddof=1))
    latency=read_json(folder/'latency_benchmark.json')
    for rec in latency['records']:
        for timing in ('end_to_end','cached_encoder','encoding'):
            if timing not in rec:continue
            v=np.array(rec[timing]['repetitions_seconds'])
            assert len(v)==5 and (v>0).all()
            close(rec[timing]['median_seconds'],np.median(v))
    sensitivity=read_json(folder/'pairing_sensitivity.json')
    for rec in sensitivity:
        close(rec['nll_increase'],rec['shuffled_tail_nll']-rec['correct_nll'])
        assert 0<=rec['argmax_change_fraction']<=1
    return dict(source_rng_rows=len(rows),source_rng_raw_vectors=len(rows)*3,latency_rows=len(latency['records']),
                sensitivity_rows=len(sensitivity),sensitivity_limit='Scalar arithmetic only; raw perturbed logits are not supplied.')


def verify_tables():
    stress=read_json(ROOT/'counterfactual_stress/summary.json')
    historical=read_json(ROOT/'historical_recomputed/summary.json')
    st=record_file(ROOT/'paper/stress_table.tex').read_text()
    ht=record_file(ROOT/'paper/historical_table.tex').read_text()
    for row in stress['cells']:
        low,high=row['bayes_regret_ci']
        key=f"{row['distribution'].capitalize()} & {row['n']} & {100*row['conflict_fraction']:.1f} & {row['mean_bayes_regret']:.4f} & [{low:.4f}, {high:.4f}]"
        assert key in st,'Stress table not generated from current summary: '+key
    for row in historical['rows']:
        if row['case']=='test200-aug8':continue
        label={'test200':'U200','cluster200':'C200','test500':'U500'}[row['case']]
        m=row['means'];excess=-row['comparisons']['learned']['improvement_percent']
        key=f"{row['host'].upper()} & {label} & {m['route']:.4f} & {m['random']:.4f} & {m['learned']:.4f} & {excess:.2f}"
        assert key in ht,'Historical table mismatch: '+key
    source=record_file(ROOT/'build_paper_assets.py').read_text()
    assert 'counterfactual_stress/summary.json' in source and 'historical_recomputed/summary.json' in source
    for name in ('stress','historical','witness'):
        for extension in ('pdf','png'):
            assert record_file(ROOT/'paper/figures'/f'{name}.{extension}').stat().st_size>1000
    return dict(numerical_table_rows=12,all_table_values_match_stored_summaries=True,
                figure_data_provenance='Reviewed plotting code reads these exact summaries and original witness coordinates; hashes recorded.',
                limitation='This check validates provenance and tables, not independent extraction of numeric values from figure pixels.')


def verify_new_tables():
    summary=read_json(ROOT/'exact_diagnostic/summary.json')
    table=record_file(ROOT/'paper/conditional_table.tex').read_text()
    for key,label in [('iid_n9','Uniform 9'),('cluster_n9','Clustered 9'),('ood_n7','Uniform 7'),('ood_n11','Uniform 11'),('ood_n13','Uniform 13')]:
        row=summary['datasets'][key]
        values=[row['models'][m]['mean'] for m in ('blind','shuffled','aware')]+[row['assignment_regret']['mean'],row['nearest_regret']['mean']]
        expected=label+' & '+' & '.join(f'{v:.5f}' for v in values)
        assert expected in table,'Conditional table mismatch: '+expected
    source=record_file(ROOT/'build_new_results.py').read_text()
    assert "S['all_pairings'][k]['models'][method]" in source
    assert "S['all_pairings'][key]['exact_blind_conditional_bayes_regret']" in source
    assert "S['information_curve']['datasets'][key]['curve']" in source
    for ext in ('pdf','png'):
        assert record_file(ROOT/f'paper/figures/conditional.{ext}').stat().st_size>1000
    return dict(numerical_rows=5,conditional_table_matches_summary=True,
                figure_provenance='Reviewed plot uses all-six-matching learned results and Bayes risk from the same family; information curve comes from its verified summary.',
                interval_scope='The plot uses geometry-only intervals conditional on three fitted seeds; the main IID difference uses crossed seed/geometry intervals.')


def verify_host_table():
    table=record_file(ROOT/'paper/adapter_table.tex').read_text()
    for host,phase,folder in [('ICAM','Frozen','results'),('AM','Frozen','results_am'),('ICAM','Decoder','results_decoder_icam'),('AM','Decoder','results_decoder_am')]:
        base=ROOT/'state_adapter'/folder
        agg=read_json(base/'aggregate.json');meta=read_json(base/'final_test_manifest.json')
        for case,label in [('uniform100','U100'),('cluster100','C100'),('uniform200','U200'),('cluster200','C200')]:
            values=[meta[case]['native_mean']]+[agg['summary'][case][m]['mean'] for m in ('route-blind','route-aware','random-blind','random-aware')]
            expected=f'{host} & {phase} & {label} & '+' & '.join(f'{v:.3f}' for v in values)
            assert expected in table,'Host table mismatch: '+expected
    return dict(numerical_rows=16,all_48_arms_required=True,all_table_values_match_stored_summaries=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifact_verification.json')
    args=parser.parse_args()
    for name,fn in [('historical_recomputation',verify_history),('large_stress',verify_stress),
                    ('witness_certificate',verify_certificate),('diagnostic_data',verify_diagnostic_data),
                    ('diagnostic_models',verify_diagnostic_models),('diagnostic_all_pairings',verify_extended),
                    ('assignment_baseline',verify_assignment),('shuffled_control',verify_shuffled),
                    ('consolidated_diagnostic_statistics',verify_consolidated_diagnostic),
                    ('progressive_information_curve',verify_information_curve),
                    ('existing_paper_tables_and_figures',verify_tables),
                    ('controlled_task_table_and_figure',verify_new_tables)]:
        check(name,fn)
    for suite in ('results','results_am','results_decoder_icam','results_decoder_am'):
        folder=ROOT/'state_adapter'/suite
        check('host_adaptation:'+str(folder.relative_to(ROOT)),lambda folder=folder:verify_adapter(folder))
        if (folder/'source_rng_variability.json').exists():
            check('host_posthoc:'+str(folder.relative_to(ROOT)),lambda folder=folder:verify_adapter_posthoc(folder))
    check('official_host_manuscript_table',verify_host_table)
    status='fail' if any(c['status']=='fail' for c in CHECKS) else ('incomplete' if any(c['status']=='incomplete' for c in CHECKS) else 'pass')
    report=dict(status=status,generated_at=datetime.now(timezone.utc).isoformat(),
                scope='Independent stored-artifact scientific consistency audit; no training rerun; missing arms never treated as passed.',
                checks=CHECKS,checked_file_sha256=FILES,
                verifier_sha256=digest(Path(__file__)),software=dict(numpy=np.__version__,torch=torch.__version__))
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# Artifact verification', '',f'Overall status: **{status}**. Completed checks and incomplete work are distinguished.', '']
    for item in CHECKS:
        lines.append(f"- **{item['status']}** — {item['name']}"+(f": {item['reason']}" if 'reason' in item else ''))
    lines+=['','The JSON report records checked file hashes and independent numerical checks. This audit does not establish model convergence, broad novelty, or conference acceptance readiness.']
    (args.output.parent/'ARTIFACT_VERIFICATION.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(overall_status=status,checks=len(CHECKS),files=len(FILES))),flush=True)


if __name__=='__main__':
    main()
