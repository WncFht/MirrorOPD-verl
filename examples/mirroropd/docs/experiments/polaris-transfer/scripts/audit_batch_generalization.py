#!/usr/bin/env python3
"""Audit frozen inputs, matched exposure, finite updates and paired evaluations."""
import argparse
import collections
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args()
    root = a.data
    bank = read(root / 'bank.json')
    rich = sum((read(root / f'rich-shard-{i}.json') for i in range(4)), [])
    all_inputs = bank + rich
    assert len(bank) == 160 and len(rich) == 448
    assert len({r['id'] for r in all_inputs}) == 608
    assert len({' '.join(r['prompt'].split()) for r in all_inputs}) == 608
    byid = {r['id']: r for r in all_inputs}
    assert {r['id'] for r in bank if r['split'] == 'train'} == set(range(64))
    assert {r['id'] for r in bank if r['split'] == 'heldout'} == set(range(64, 160))
    assert all(0 < len(r['response_ids']) <= 1024 for r in all_inputs)
    arms = []
    for d in sorted(root.iterdir()):
        if not (d / 'training.json').exists():
            continue
        protocol = read(d / 'protocol.json')
        updates = read(d / 'training.json')
        steps = protocol['steps']
        assert read(d / 'complete.json') == {'steps': steps, 'finished': True}
        assert [r['step'] for r in updates] == list(range(1, steps + 1))
        visits = collections.Counter()
        for row in updates:
            assert row['normalizer'] == 16384
            assert len(row['prompt_ids']) == len(set(row['prompt_ids'])) == 16
            assert row['tokens'] == sum(len(byid[i]['response_ids']) for i in row['prompt_ids'])
            assert all(math.isfinite(row[k]) for k in ['loss', 'grad_norm'])
            if 'parameter_step_norm' in row:
                assert math.isfinite(row['parameter_step_norm']) and row['parameter_step_norm'] > 0
            visits.update(row['prompt_ids'])
        if protocol.get('rich'):
            expected = set(range(64)) | {r['id'] for r in rich}
            nvisits = 4
        elif protocol['schedule'] == 'single':
            expected, nvisits = set(range(16)), steps
        else:
            expected, nvisits = set(range(64)), steps // 4
        assert set(visits) == expected and set(visits.values()) == {nvisits}
        for step in range(16, steps + 1, 16):
            rows = read(d / f'fixed-{step}.json')
            assert len(rows) == 160 and {r['id'] for r in rows} == set(range(160))
            for row in rows:
                assert row['length'] == len(byid[row['id']]['response_ids'])
                assert all(math.isfinite(v) for band in row['stats'].values() for v in band.values())
        fresh = read(d / f'fresh-{steps}.json')
        assert len(fresh) == 64
        assert {(r['id'], r['seed']) for r in fresh} == {(i, s) for i in range(64, 96) for s in [65101, 65102]}
        assert all(r['mode'] == 'student' and r['source'].startswith('fresh-') for r in fresh)
        assert all(r['prompt_ids'] == byid[r['id']]['prompt_ids'] and len(r['response_ids']) <= 1024 for r in fresh)
        arms.append(dict(arm=d.name, updates=steps, unique_train_prompts=len(visits),
                         visits_per_prompt=nvisits, actual_tokens=sum(r['tokens'] for r in updates),
                         normalizer=16384, complete=True))
    assert len(arms) == 14 and sum(r['updates'] for r in arms) == 1504
    warm = read(root / 'baseline-warm-polaris/fixed.json')
    parent = read(root / 'polaris-blocked/fixed-128.json')
    assert [(r['id'], r['stats']) for r in warm] == [(r['id'], r['stats']) for r in parent]
    warm_fresh = read(root / 'baseline-warm-polaris/fresh.json')
    parent_fresh = read(root / 'polaris-blocked/fresh-128.json')
    assert [(r['id'], r['seed'], r['response_ids'], r['stats']) for r in warm_fresh] == [(r['id'], r['seed'], r['response_ids'], r['stats']) for r in parent_fresh]
    offload_loss = read(root / 'polaris-rich-lr1/training.json')[0]['loss']
    original_loss = read(root / 'polaris-roundrobin/training.json')[0]['loss']
    assert abs(offload_loss - original_loss) < 1e-6
    geometries = read(root / 'batch-gradient-geometry.json')
    geometries += sum((read(root / f'fixed-gradient-case-{i}.json') for i in range(5)), [])
    geometries += read(root / 'coverage-gradient.json') + read(root / 'coverage-gradient-64.json')
    for row in geometries:
        assert row['parameter_count'] == 1777088000
        matrix = row['cosine']
        for i in range(len(matrix)):
            assert abs(matrix[i][i] - 1) < 1e-8
            for j in range(len(matrix)):
                assert math.isfinite(matrix[i][j]) and abs(matrix[i][j]) <= 1 + 1e-8
                assert abs(matrix[i][j] - matrix[j][i]) < 1e-8
    prod = read(root / 'latest-run-summary.json')['E48']
    cloud = read(root / 'cloud-audit.json')
    source = read(root / 'run/source.json')
    assert prod['updates'] == 4096 and prod['outer'] == 256
    assert cloud['state'] == 'finished' and source['exit_code'] == 0
    out = dict(passed=True, frozen_prompts=160, independent_added_prompts=448,
               unique_prompts=608, arms=arms, total_updates=1504,
               gradient_cases=len(geometries), full_parameter_count=1777088000,
               teacher_offload_initial_loss_difference=offload_loss-original_loss,
               warm_baseline_identical=True, E48_complete=True,
               notes=['One training seed per recipe; bootstrap intervals quantify prompt sampling only.',
                      'Discarded shared-GPU OOM attempt is excluded; completed retry starts from original student.',
                      'Frozen bank checks preserve original trajectories; no checksums are generated.'])
    (root / 'verification.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
