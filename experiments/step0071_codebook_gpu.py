#!/usr/bin/env python3
"""
Step 71 (GPU) — Gradient-derived update rule for FluxCore codebook.

Baseline: spawn=0.95, always-attractive (Step 66b GPU best = ~32.3% AA)

Version 1 — Full gradient (softmax over all prototypes):
  probs = softmax(sims / temp)
  v_i -= lr * (probs_i - y_i) * (r - v_i * cos(r, v_i))   for all i
  Attractive for correct class, repulsive for wrong classes.

Version 2 — Winner-only bipolar (atomic, minimal):
  if label(winner) == label(input): v_winner += lr * r   (current rule)
  if label(winner) != label(input): v_winner -= lr * r   (fold reads its own label)
  Same operation, same magnitude, sign from stored label — no softmax needed.

Test: Split-CIFAR-100, spawn=0.95, same ResNet-18 features as Step 66b.
"""

import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, 'B:/M/avir/research/fluxcore')

CACHE_PATH    = 'C:/Users/Admin/cifar100_resnet18_features.npz'
N_TASKS       = 20
CLASSES_TASK  = 5
D_EMBED       = 512
SPAWN_THRESH  = 0.95
MERGE_THRESH  = 0.95
LR            = 0.015
SOFTMAX_TEMP  = 0.1


class GradTorchCodebook:
    """Codebook with configurable gradient update rule. CUDA-accelerated."""

    def __init__(self, d, spawn_thresh=SPAWN_THRESH, mode='attractive',
                 device=None):
        self.d            = d
        self.spawn_thresh = spawn_thresh
        self.mode         = mode
        self.device       = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.vectors      = torch.empty((0, d), dtype=torch.float32, device=self.device)
        self.labels       = []
        self.n_spawned    = 0
        self.n_merged     = 0

    def step(self, r, label=None):
        """r: unit-normalized (d,) tensor on device."""
        if len(self.vectors) == 0:
            self._spawn(r, label)
            return

        sims   = self.vectors @ r          # (N,)
        winner = int(sims.argmax())
        max_sim = float(sims[winner])

        if max_sim < self.spawn_thresh:
            self._spawn(r, label)
            return

        if self.mode == 'attractive':
            self._update(winner, r, sign=+1)

        elif self.mode == 'bipolar':
            sign = +1 if self.labels[winner] == label else -1
            self._update(winner, r, sign=sign)

        elif self.mode == 'full_grad':
            s = sims / SOFTMAX_TEMP
            s = s - s.max()
            probs = s.exp() / (s.exp().sum() + 1e-15)         # (N,)
            y = torch.tensor(
                [1.0 if self.labels[i] == label else 0.0 for i in range(len(self.labels))],
                dtype=torch.float32, device=self.device)
            errors = probs - y                                  # (N,)
            active = errors.abs() > 1e-4
            if active.any():
                V_act  = self.vectors[active]                   # (M, d)
                e_act  = errors[active].unsqueeze(1)            # (M, 1)
                cos_act = sims[active].unsqueeze(1)             # (M, 1)
                perp   = r.unsqueeze(0) - V_act * cos_act      # (M, d)  tangent
                self.vectors[active] = self.vectors[active] - LR * e_act * perp
                norms = self.vectors[active].norm(dim=1, keepdim=True).clamp(min=1e-15)
                self.vectors[active] = self.vectors[active] / norms

    def _update(self, i, r, sign):
        v = self.vectors[i] + sign * LR * r
        n = v.norm()
        if n > 1e-15:
            self.vectors[i] = v / n

    def _spawn(self, r_norm, label):
        if len(self.vectors) > 0:
            abs_sims = (self.vectors @ r_norm).abs()
            best_i   = int(abs_sims.argmax())
            if float(abs_sims[best_i]) > MERGE_THRESH:
                fused = self.vectors[best_i] + r_norm
                self.vectors[best_i] = fused / (fused.norm() + 1e-15)
                self.n_merged += 1
                return
        self.vectors = torch.cat([self.vectors, r_norm.unsqueeze(0)], dim=0)
        self.labels.append(label)
        self.n_spawned += 1

    def classify_batch(self, X, k=1):
        if len(self.vectors) == 0:
            return [None] * len(X)
        X_n  = F.normalize(X, dim=1)
        sims = X_n @ self.vectors.T        # (n, N)
        if k == 1:
            return [self.labels[w] for w in sims.argmax(dim=1).tolist()]
        top_k = sims.topk(k, dim=1).indices
        preds = []
        for i in range(len(X)):
            votes = [self.labels[j] for j in top_k[i].tolist()]
            preds.append(max(set(votes), key=votes.count))
        return preds


def load_data():
    data = np.load(CACHE_PATH)
    X_tr, y_tr = data['X_train'], data['y_train']
    X_te, y_te = data['X_test'],  data['y_test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits = []
    for t in range(N_TASKS):
        c0, c1 = t * CLASSES_TASK, (t + 1) * CLASSES_TASK
        splits.append((
            torch.tensor(X_tr[np.isin(y_tr, range(c0, c1))], dtype=torch.float32, device=device),
            torch.tensor(y_tr[np.isin(y_tr, range(c0, c1))]),
            torch.tensor(X_te[np.isin(y_te, range(c0, c1))], dtype=torch.float32, device=device),
            torch.tensor(y_te[np.isin(y_te, range(c0, c1))]),
        ))
    return splits


def run(splits, mode, label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cb = GradTorchCodebook(D_EMBED, mode=mode, device=device)
    acc_matrix = [[None] * N_TASKS for _ in range(N_TASKS)]
    t0 = time.time()

    for task_id in range(N_TASKS):
        X_t, y_t = splits[task_id][0], splits[task_id][1]
        X_n = F.normalize(X_t, dim=1)
        for i in range(len(X_n)):
            cb.step(X_n[i], label=int(y_t[i]))

        for et in range(task_id + 1):
            X_te, y_te = splits[et][2], splits[et][3]
            preds = cb.classify_batch(X_te, k=1)
            acc_matrix[et][task_id] = sum(p == int(g) for p, g in zip(preds, y_te.tolist())) / len(y_te)

    elapsed = time.time() - t0
    final = [acc_matrix[i][N_TASKS - 1] for i in range(N_TASKS)]
    aa    = sum(final) / N_TASKS
    fgt   = [max(0., acc_matrix[i][i] - acc_matrix[i][N_TASKS - 1]) for i in range(N_TASKS - 1)]
    return {
        'label': label, 'mode': mode, 'aa': aa,
        'forgetting': sum(fgt) / len(fgt), 'cb_size': len(cb.vectors),
        'elapsed': elapsed, 'final': final,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.6)  # cap at 60% VRAM, leave room for other processes
    print('=' * 70)
    print('  Step 71 (GPU) -- Gradient-Derived Update Rule')
    print('  Split-CIFAR-100, spawn=0.95, ResNet-18 features')
    print(f'  Device: {device} ({torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"})')
    print('  One variable: update rule (attractive / bipolar / full_grad)')
    print('=' * 70)

    print('\nLoading features...')
    splits = load_data()

    modes = [
        ('attractive', 'Attractive (baseline, Step 66b)'),
        ('bipolar',    'Bipolar winner-only (atomic)'),
        ('full_grad',  'Full gradient (softmax, all prototypes)'),
    ]
    results = []
    for mode, lbl in modes:
        print(f'\n  Running {lbl}...')
        r = run(splits, mode, lbl)
        results.append(r)
        print(f'    AA={r["aa"]*100:.1f}%  F={r["forgetting"]*100:.1f}pp  '
              f'cb={r["cb_size"]}  t={r["elapsed"]:.1f}s')

    print('\n' + '=' * 70)
    print('  STEP 71 SUMMARY -- Gradient Update Rule Comparison')
    print('=' * 70)
    print(f'  {"Mode":<35} | {"AA":>7} | {"Forgetting":>10} | {"CB":>7} | {"Time":>6}')
    print('  ' + '-' * 68)
    for r in results:
        print(f'  {r["label"]:<35} | {r["aa"]*100:>6.1f}% | '
              f'{r["forgetting"]*100:>9.1f}pp | {r["cb_size"]:>7} | {r["elapsed"]:>5.1f}s')

    base = results[0]
    print(f'\n  Published baselines: Fine-tune ~6% | EWC ~33% | SI ~36% | DER++ ~51%')
    print(f'\n  AA gain over attractive baseline:')
    for r in results[1:]:
        delta = (r['aa'] - base['aa']) * 100
        print(f'    {r["label"]}: {delta:+.1f}pp AA, '
              f'{(r["forgetting"] - base["forgetting"]) * 100:+.1f}pp forgetting')

    best = max(results, key=lambda r: r['aa'])
    print(f'\n  Best: {best["label"]} -> AA={best["aa"]*100:.1f}%  '
          f'F={best["forgetting"]*100:.1f}pp')

    if best['aa'] > base['aa'] + 0.03:
        print('  VERDICT: Gradient-derived update improves accuracy.')
        if best['forgetting'] < base['forgetting']:
            print('           AND reduces forgetting. Both axes improved.')
    else:
        print('  VERDICT: Gradient update does not significantly improve AA.')
        print('           Classify as readout limitation, not update rule.')
    print('=' * 70)


if __name__ == '__main__':
    main()
