#!/usr/bin/env python3
"""
Step 66b (GPU) — Spawn threshold sweep on Split-CIFAR-100.

Validates GPU results match CPU baseline. Establishes GPU timing baseline.
Sweeps spawn_thresh in {0.5, 0.7, 0.8, 0.9, 0.95}.

Expected CPU baseline (run_step66b_threshold_sweep.py):
  0.5 -> 2.8%, 0.7 -> 8.8%, 0.8 -> 19.2%, 0.9 -> 30.9%, 0.95 -> 32.3%
"""

import sys
import time
import torch
import numpy as np

sys.path.insert(0, 'B:/M/avir/research/fluxcore')
from fluxcore_torch import TorchCodebook

CACHE_PATH   = 'C:/Users/Admin/cifar100_resnet18_features.npz'
N_TASKS      = 20
CLASSES_TASK = 5
D_EMBED      = 512


def load_data():
    data = np.load(CACHE_PATH)
    X_tr, y_tr = data['X_train'], data['y_train']
    X_te, y_te = data['X_test'],  data['y_test']
    splits = []
    for t in range(N_TASKS):
        c0 = t * CLASSES_TASK
        c1 = c0 + CLASSES_TASK
        splits.append((
            torch.tensor(X_tr[np.isin(y_tr, range(c0, c1))], dtype=torch.float32),
            torch.tensor(y_tr[np.isin(y_tr, range(c0, c1))]),
            torch.tensor(X_te[np.isin(y_te, range(c0, c1))], dtype=torch.float32),
            torch.tensor(y_te[np.isin(y_te, range(c0, c1))]),
        ))
    return splits


def run(splits, spawn_thresh):
    cb = TorchCodebook(D_EMBED, spawn_thresh=spawn_thresh)
    acc_matrix = [[None] * N_TASKS for _ in range(N_TASKS)]
    t0 = time.time()

    for task_id in range(N_TASKS):
        X_t, y_t = splits[task_id][0], splits[task_id][1]
        cb.step_batch_train(X_t, y_t)

        for et in range(task_id + 1):
            X_te, y_te = splits[et][2], splits[et][3]
            preds = cb.classify_batch(X_te, k=1)
            acc_matrix[et][task_id] = sum(p == int(g) for p, g in zip(preds, y_te.tolist())) / len(y_te)

    elapsed = time.time() - t0
    final = [acc_matrix[i][N_TASKS - 1] for i in range(N_TASKS)]
    aa    = sum(final) / N_TASKS
    fgt   = [max(0., acc_matrix[i][i] - acc_matrix[i][N_TASKS - 1]) for i in range(N_TASKS - 1)]
    return {'aa': aa, 'forgetting': sum(fgt) / len(fgt), 'cb': len(cb), 'elapsed': elapsed}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.6)  # cap at 60% VRAM, leave room for other processes
    print('=' * 70)
    print(f'  Step 66b (GPU) -- Spawn Threshold Sweep')
    print(f'  Split-CIFAR-100, 20 tasks, ResNet-18 features')
    print(f'  Device: {device} ({torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"})')
    print('=' * 70)

    print('\nLoading features...')
    splits = load_data()

    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    cpu_baseline = {0.5: 2.8, 0.7: 8.8, 0.8: 19.2, 0.9: 30.9, 0.95: 32.3}

    results = []
    for thresh in thresholds:
        r = run(splits, thresh)
        results.append((thresh, r))
        delta = r['aa'] * 100 - cpu_baseline[thresh]
        print(f'  thresh={thresh:.2f}: AA={r["aa"]*100:.1f}% '
              f'(CPU baseline {cpu_baseline[thresh]:.1f}%, delta {delta:+.1f}pp) '
              f'F={r["forgetting"]*100:.1f}pp  cb={r["cb"]}  t={r["elapsed"]:.1f}s')

    print('\n' + '=' * 70)
    print('  SUMMARY')
    print('=' * 70)
    print(f'  {"Thresh":>7} | {"AA":>7} | {"CPU base":>9} | {"Delta":>7} | {"Forgetting":>10} | {"CB":>7} | {"Time":>6}')
    print('  ' + '-' * 65)
    for thresh, r in results:
        delta = r['aa'] * 100 - cpu_baseline[thresh]
        print(f'  {thresh:>7.2f} | {r["aa"]*100:>6.1f}% | {cpu_baseline[thresh]:>8.1f}% | '
              f'{delta:>+6.1f}pp | {r["forgetting"]*100:>9.1f}pp | {r["cb"]:>7} | {r["elapsed"]:>5.1f}s')

    print()
    best = max(results, key=lambda x: x[1]['aa'])
    print(f'  Best: thresh={best[0]} -> AA={best[1]["aa"]*100:.1f}%')
    print('=' * 70)


if __name__ == '__main__':
    main()
