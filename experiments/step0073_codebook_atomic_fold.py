#!/usr/bin/env python3
"""
Step 73 — AtomicFold benchmark.

AtomicFold: unified soft training + soft inference, per-prototype kappa confidence,
energy-gated spawning. Tests whether co-evolved soft attention improves over
hard-trained FullGradCodebook (Step 71: 33.5% CIFAR-100, 84.1% P-MNIST).

Protocol:
  1. Calibrate spawn_energy from first 100 training samples (median energy).
  2. Sweep tau x lr on P-MNIST first (faster), then CIFAR-100.
  3. Report AA, forgetting, CB size, kappa stats.

Baselines:
  CIFAR-100: 33.5% AA, 12.6pp F (Step 71 FullGrad, 1-NN)
  P-MNIST:   84.1% AA, 11.39pp F (Step 72 FullGrad, 1-NN)
"""

import sys, time
import torch
import torch.nn.functional as F
import numpy as np
import torchvision, torchvision.transforms as transforms

sys.path.insert(0, 'B:/M/avir/research/fluxcore')
from atomic_fold import AtomicFold

CIFAR_CACHE  = 'C:/Users/Admin/cifar100_resnet18_features.npz'
MNIST_PATH   = 'C:/Users/Admin/mnist_data'

TAUS    = [0.01, 0.05, 0.1, 0.2]
LRS     = [0.001, 0.005, 0.01, 0.05]

# ── Batch classify (bypasses slow Python _vote loop) ─────────────────────────

def batch_classify(af, X_norm):
    """Vectorized classify over X_norm (pre-normalized, device tensor). Returns list of labels."""
    if af.n == 0:
        return [None] * len(X_norm)
    sims    = X_norm @ af.V.T                                        # (batch, n)
    weighted = sims * af.kappa.unsqueeze(0)                          # (batch, n)
    weights  = F.softmax(weighted / af.tau, dim=1)                   # (batch, n)
    unique_labels = sorted(set(af.labels))
    label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
    n_classes = len(unique_labels)
    scores = torch.zeros(len(X_norm), n_classes, device=af.V.device)
    for j, lbl in enumerate(af.labels):
        scores[:, label_to_idx[lbl]] += weights[:, j]
    return [unique_labels[i] for i in scores.argmax(dim=1).tolist()]


# ── spawn_energy calibration ──────────────────────────────────────────────────

def calibrate_spawn_energy(X_norm_first100, labels_first100, tau, lr):
    """Run first 100 steps with spawn_energy=None, return median energy seen."""
    af = AtomicFold(X_norm_first100.shape[1], tau=tau, lr=lr, spawn_energy=None)
    energies = []
    for i in range(len(X_norm_first100)):
        r = X_norm_first100[i]
        if af.n > 0:
            energies.append(af.energy(r))
        af.step(r, label=int(labels_first100[i]))
    if not energies:
        return 1.0
    return float(torch.tensor(energies).median())


# ── Sequential eval ───────────────────────────────────────────────────────────

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
        fn = lambda X, _af=af: batch_classify(_af, X)
        for et in range(task_id + 1):
            X_te, y_te = splits[et][2], splits[et][3]
            preds = fn(F.normalize(X_te, dim=1))
            acc_matrix[et][task_id] = sum(p == int(g) for p, g in zip(preds, y_te.tolist())) / len(y_te)
        if verbose:
            cur_aa = sum(acc_matrix[et][task_id] for et in range(task_id+1)) / (task_id+1)
            print(f'    task {task_id+1}/{n_tasks}  interim_AA={cur_aa*100:.1f}%  CB={af.n}', flush=True)

    elapsed = time.time() - t0
    final = [acc_matrix[i][n_tasks - 1] for i in range(n_tasks)]
    aa  = sum(final) / n_tasks
    fgt = [max(0., acc_matrix[i][i] - acc_matrix[i][n_tasks - 1]) for i in range(n_tasks - 1)]
    return {
        'aa': aa,
        'forgetting': sum(fgt) / len(fgt),
        'elapsed': elapsed,
        **af.stats(),
    }


# ── P-MNIST data ──────────────────────────────────────────────────────────────

def load_pmn(device):
    tf = transforms.Compose([transforms.ToTensor()])
    mnist_tr = torchvision.datasets.MNIST(MNIST_PATH, train=True,  download=True, transform=tf)
    mnist_te = torchvision.datasets.MNIST(MNIST_PATH, train=False, download=True, transform=tf)
    X_tr = mnist_tr.data.float().reshape(len(mnist_tr), -1) / 255.0
    y_tr = mnist_tr.targets
    X_te = mnist_te.data.float().reshape(len(mnist_te), -1) / 255.0
    y_te = mnist_te.targets

    rng_proj = torch.Generator(); rng_proj.manual_seed(42)
    proj = torch.randn(784, 384, generator=rng_proj).to(device)
    proj = F.normalize(proj, dim=0)

    MN_TASKS = 10; MN_TRAIN = 6000
    splits = []
    for t in range(MN_TASKS):
        perm  = torch.randperm(784, generator=torch.Generator().manual_seed(t * 7919))
        Xtr_p = (X_tr[:, perm].to(device) @ proj)
        Xte_p = (X_te[:, perm].to(device) @ proj)
        rng_s = torch.Generator(); rng_s.manual_seed(t)
        idx = []
        for c in range(10):
            ci = (y_tr == c).nonzero(as_tuple=True)[0]
            ci = ci[torch.randperm(len(ci), generator=rng_s)[:MN_TRAIN // 10]]
            idx.append(ci)
        idx = torch.cat(idx)
        splits.append((Xtr_p[idx], y_tr[idx], Xte_p, y_te))
    return splits, MN_TASKS


# ── CIFAR-100 data ────────────────────────────────────────────────────────────

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


# ── Sweep runner ──────────────────────────────────────────────────────────────

def run_sweep(splits, n_tasks, benchmark_name, calibration_x, calibration_y):
    print(f'\n{"=" * 70}')
    print(f'  {benchmark_name}  |  AtomicFold  |  Step 73')
    print(f'{"=" * 70}')
    print(f'  {"Config":<28} | {"AA":>7} | {"Fgt":>8} | {"CB":>7} | {"kappa_mean":>10} | {"Time":>7}')
    print('  ' + '-' * 72)

    best = None
    for tau in TAUS:
        for lr in LRS:
            spawn_e = calibrate_spawn_energy(calibration_x, calibration_y, tau, lr)
            r = run_sequential(splits, n_tasks, tau, lr, spawn_e)
            tag = f'tau={tau:.3f} lr={lr:.3f}'
            print(f'  {tag:<28} | {r["aa"]*100:>6.1f}% | {r["forgetting"]*100:>7.1f}pp | {r["n_prototypes"]:>7} | {r["kappa_mean"]:>10.3f} | {r["elapsed"]:>6.0f}s',
                  flush=True)
            if best is None or r['aa'] > best['aa']:
                best = {**r, 'tau': tau, 'lr': lr, 'spawn_e': spawn_e}

    print(f'\n  Best: tau={best["tau"]}, lr={best["lr"]}, spawn_e={best["spawn_e"]:.3f}')
    print(f'    AA={best["aa"]*100:.1f}%  F={best["forgetting"]*100:.1f}pp  CB={best["n_prototypes"]}  kappa_mean={best["kappa_mean"]:.3f}')
    return best


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.6)
    print(f'Device: {device} ({torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"})')
    print('Baselines  — CIFAR-100: 33.5% AA / 12.6pp F  |  P-MNIST: 84.1% AA / 11.4pp F')

    # ── P-MNIST ───────────────────────────────────────────────────────────────
    print('\nLoading P-MNIST...')
    pmn_splits, pmn_tasks = load_pmn(device)
    cal_x = F.normalize(pmn_splits[0][0][:100], dim=1)
    cal_y = pmn_splits[0][1][:100]
    run_sweep(pmn_splits, pmn_tasks, 'P-MNIST (10 tasks, 6K/task, d=384)', cal_x, cal_y)

    # ── CIFAR-100 ─────────────────────────────────────────────────────────────
    print('\nLoading CIFAR-100...')
    cif_splits, cif_tasks = load_cifar(device)
    cal_x = F.normalize(cif_splits[0][0][:100], dim=1)
    cal_y = cif_splits[0][1][:100]
    run_sweep(cif_splits, cif_tasks, 'CIFAR-100 (20 tasks, spawn=energy-gated, d=512)', cal_x, cal_y)

    print('\nStep 73 done.')


if __name__ == '__main__':
    main()
