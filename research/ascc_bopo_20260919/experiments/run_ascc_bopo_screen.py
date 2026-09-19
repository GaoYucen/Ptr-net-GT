"""Run the preregistered four-arm screen sequentially; stop on any failed arm."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backbone', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    arms = [('route', 'bopo'), ('learned', 'bopo'),
            ('route', 'reinforce'), ('learned', 'reinforce')]
    manifest = {'role': 'one_seed_bounded_screen_not_paper_main_experiment',
                'seed': 1234, 'steps_per_arm': 200, 'arms': [],
                'started_unix': time.time(), 'status': 'running',
                'pid': os.getpid(), 'working_tree': str(root)}
    write(args.out / 'manifest.json', manifest)
    for source, objective in arms:
        name = f'{source}-{objective}-seed1234'
        cmd = [sys.executable, str(root / 'experiments/train_ascc_bopo.py'),
               '--out', str(args.out / name), '--source', source,
               '--objective', objective, '--backbone', str(args.backbone),
               '--steps', '200', '--batch', '4', '--rollouts', '16', '--filtered', '8',
               '--validation-size', '128', '--test-size', '256', '--eval-rollouts', '8',
               '--eval-batch', '8', '--eval-every', '100', '--log-every', '25',
               '--max-seconds', '600', '--memory-fraction', '0.22']
        entry = {'name': name, 'command': cmd, 'started_unix': time.time(), 'status': 'running'}
        manifest['arms'].append(entry)
        write(args.out / 'manifest.json', manifest)
        print(f'START {name}', flush=True)
        with (args.out / f'{name}.log').open('w') as log:
            try:
                result = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=900)
                entry['exit_code'] = result.returncode
                entry['status'] = 'completed' if result.returncode == 0 else 'failed'
            except subprocess.TimeoutExpired:
                entry.update(exit_code=None, status='timeout')
        entry['finished_unix'] = time.time()
        if entry['status'] != 'completed':
            manifest.update(status='failed', finished_unix=time.time())
            write(args.out / 'manifest.json', manifest)
            raise SystemExit(f'Queue stopped after {name}: {entry["status"]}')
        write(args.out / 'manifest.json', manifest)
        print(f'FINISH {name}', flush=True)
    manifest.update(status='completed', finished_unix=time.time())
    write(args.out / 'manifest.json', manifest)
    subprocess.run([sys.executable, str(root / 'experiments/summarize_ascc_bopo.py'),
                    str(args.out)], check=True)


if __name__ == '__main__':
    main()
