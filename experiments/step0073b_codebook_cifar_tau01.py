#!/usr/bin/env python3
"""
Step 73b — AtomicFold CIFAR-100, tau=0.01 slice only.

Spec: skip remaining P-MNIST taus, run CIFAR-100 with
tau=0.01 x lr=[0.001, 0.005, 0.01, 0.05] only.

P-MNIST results already collected (4 configs, tau=0.01):
  lr=0.001: AA=65.6%, Fgt=0.0pp, CB=1075, 661s
  lr=0.005: AA=53.7%, Fgt=0.0pp, CB=966,  617s
  lr=0.010: AA=48.6%, Fgt=0.0pp, CB=883,  570s
  lr=0.050: AA=32.5%, Fgt=0.2pp, CB=644,  425s

Baseline (CIFAR-100): 33.5% AA / 12.6pp F (Step 71 FullGrad, 1-NN)
"""

import sys, time
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, 'B:/M/avir/research/fluxcore')
from atomic_fold import AtomicFold

CIFAR_CACHE = 'C:/Users/Admin/cifar100_resnet18_features.npz'

TAU  = 0.01
LRS  = [0.001, 0.005, 0.01, 0.05]


def batch_classify(af, X_norm):
    if af.n == 0:
        return [None] * len(X_norm)
    sims     = X_norm @ af.V.T
    weighted = sims * af.kappa.unsqueeze(0)
    weights  = F.softmax(weighted / af.tau, dim=1)
    unique_labels = sorted(set(af.labels))
    label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
    scores = torch.zeros(len(X_norm), len(unique_labels), device=af.V.device)
    for j, lbl in enumerate(af.labels):
        scores[:, label_to_idx[lbl]] += weights[:, j]
    return [unique_labels[i] for i in scores.argmax(dim=1).tolist()]


def calibrate_spawn_energy(X_norm_first100, labels_first100, tau, lr):
    af = AtomicFold(X_norm_first100.shape[1], tau=tau, lr=lr, spawn_energy=None)
    energies = []
    for i in range(len(X_norm_first100)):
        r = X_norm_first100[i]
        if af.n > 0:
            energies.append(af.energy(r))
        af.step(r, label=int(labels_first100[i]))
    return float(torch.tensor(energies).median()) if energies else 1.0


def run_sequential(splits, n_tasks, tau, lr, spawn_energy, verbose=False):
    d = splits[0][0].shape[1]
    af = AtomicFold(d, tau=tau, lr=lr, spawn_energy=spawn_energy)
    acc_matrix = [[None] * n_tasks for _ in range(n_tasks)]
    t0 = time.time()

    for task_id in range(n_tasks):
        X_t, y_t = splits[task_id][0], splits[task_id][1]
        X_n = F.normalize(X_t, dim=1)
        for i in range(len(X_n)):
            af.step(X_n[i], label=int(y_t[i]))
        for et in range(task_id + 1):
            X_te, y_te = splits[et][2], splits[et][3]
            preds = batch_classify(af, F.normalize(X_te, dim=1))
            acc_matrix[et][task_id] = sum(p == int(g) for p, g in zip(preds, y_te.tolist())) / len(y_te)
        if verbose:
            cur_aa = sum(acc_matrix[et][task_id] for et in range(task_id+1)) / (task_id+1)
            print(f'    task {task_id+1}/{n_tasks}  interim_AA={cur_aa*100:.1f}%  CB={af.n}', flush=True)

    elapsed = time.time() - t0
    final = [acc_matrix[i][n_tasks - 1] for i in range(n_tasks)]
    aa  = sum(final) / n_tasks
    fgt = [max(0., acc_matrix[i][i] - acc_matrix[i][n_tasks - 1]) for i in range(n_tasks - 1)]
    return {'aa': aa, 'forgetting': sum(fgt) / len(fgt), 'elapsed': elapsed, **af.stats()}


def load_cifar(device):
    data   = np.load(CIFAR_CACHE)
    X_tr, y_tr = data['X_train'], data['y_train']
    X_te, y_te = data['X_test'],  data['y_test']
    N_TASKS = 20; CLASSES_TASK = 5
    splits = []
    for t in range(N_TASKS):
        c0, c1 = t * CLASSES_TASK, (t + 1) * CLASSES_TASK
        splits.append((
            torch.tensor(X_tr[np.isin(y_tr, range(c0, c1))], dtype=torch.float32, device=device),
            torch.tensor(y_tr[np.isin(y_tr, range(c0, c1))]),
            torch.tensor(X_te[np.isin(y_te, range(c0, c1))], dtype=torch.float32, device=device),
            torch.tensor(y_te[np.isin(y_te, range(c0, c1))]),
        ))
    return splits, N_TASKS


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.6)
    print(f'Device: {device} ({torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"})')
    print('Baseline  — CIFAR-100: 33.5% AA / 12.6pp F  (Step 71 FullGrad 1-NN)')
    print(f'Sweep: tau={TAU}, lr={LRS}')

    print('\nLoading CIFAR-100...')
    splits, n_tasks = load_cifar(device)
    cal_x = F.normalize(splits[0][0][:100], dim=1)
    cal_y = splits[0][1][:100]

    print(f'\n{"="*70}')
    print(f'  CIFAR-100 (20 tasks, tau={TAU})  |  AtomicFold  |  Step 73b')
    print(f'{"="*70}')
    print(f'  {"Config":<28} | {"AA":>7} | {"Fgt":>8} | {"CB":>7} | {"kappa_mean":>10} | {"Time":>7}')
    print('  ' + '-'*72)

    best = None
    for lr in LRS:
        spawn_e = calibrate_spawn_energy(cal_x, cal_y, TAU, lr)
        r = run_sequential(splits, n_tasks, TAU, lr, spawn_e, verbose=True)
        tag = f'tau={TAU:.3f} lr={lr:.3f}'
        print(f'  {tag:<28} | {r["aa"]*100:>6.1f}% | {r["forgetting"]*100:>7.1f}pp | {r["n_prototypes"]:>7} | {r["kappa_mean"]:>10.3f} | {r["elapsed"]:>6.0f}s', flush=True)
        if best is None or r['aa'] > best['aa']:
            best = {**r, 'lr': lr, 'spawn_e': spawn_e}

    print(f'\n  Best: lr={best["lr"]}, spawn_e={best["spawn_e"]:.3f}')
    print(f'    AA={best["aa"]*100:.1f}%  F={best["forgetting"]*100:.1f}pp  CB={best["n_prototypes"]}  kappa_mean={best["kappa_mean"]:.3f}')
    print('\nStep 73b done.')


if __name__ == '__main__':
    main()
