"""Compare two saved prediction runs array by array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('left')
    parser.add_argument('right')
    parser.add_argument('--atol', type=float, default=0.0)
    args = parser.parse_args()
    left = Path(args.left)
    right = Path(args.right)
    a = np.load(left / 'predictions.npz')
    b = np.load(right / 'predictions.npz')
    if set(a.files) != set(b.files):
        raise SystemExit(f'array keys differ: {a.files} vs {b.files}')
    report = {}
    passed = True
    for key in a.files:
        same_shape = a[key].shape == b[key].shape
        if not same_shape:
            maximum = float('inf')
            equal = False
        elif np.issubdtype(a[key].dtype, np.number):
            maximum = float(np.max(np.abs(
                a[key].astype(float) - b[key].astype(float))))
            equal = bool(maximum <= args.atol)
        else:
            maximum = None
            equal = bool(np.array_equal(a[key], b[key]))
        report[key] = {
            'shape_equal': same_shape,
            'max_absolute_difference': maximum,
            'within_tolerance': equal,
        }
        passed &= equal
    metrics_equal = json.load(open(left / 'metrics.json', encoding='utf-8')) == \
        json.load(open(right / 'metrics.json', encoding='utf-8'))
    report['metrics_json_equal'] = metrics_equal
    passed &= metrics_equal
    report['passed'] = bool(passed)
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
