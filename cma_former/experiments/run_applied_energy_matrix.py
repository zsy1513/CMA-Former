"""Resumable orchestration of the experiment matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CORE_MODELS = [
    'cma_former', 'patchtst', 'dlinear', 'nbeatsx', 'tft',
    'itransformer', 'temporal_gat', 'single_market_patchtst',
    'concatenated_transformer',
]
ABLATIONS = [
    'cma_no_cross', 'cma_topology_prior', 'cma_fixed_prior',
    'cma_shuffled_prior', 'cma_late_future',
]


def run(command):
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/applied_energy.yaml')
    parser.add_argument('--out', default='results/applied_energy')
    parser.add_argument('--stage', choices=['smoke', 'core', 'ablation', 'all'],
                        default='smoke')
    args = parser.parse_args()
    runner = str(Path(__file__).with_name('run_applied_energy.py'))
    base = [sys.executable, runner, '--config', args.config, '--resume']

    if args.stage == 'smoke':
        run(base + ['--out', 'results/applied_energy_smoke', '--models',
                    'cma_former', '--seeds', '42', '--epochs', '2'])
        return
    if args.stage in ('core', 'all'):
        run(base + ['--out', args.out, '--models', *CORE_MODELS])
    if args.stage in ('ablation', 'all'):
        run(base + ['--out', args.out, '--models', *ABLATIONS])


if __name__ == '__main__':
    main()
