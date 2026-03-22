#!/usr/bin/env python3
"""
Step 71: Gradient-derived update rule for FluxCore codebook.

Baseline: spawn=0.95, always-attractive update (Step 66b best = 32.3% AA, 12.5pp F)

Version 1 — Full gradient (softmax over all prototypes):
  probs = softmax(sims)
  v_i -= lr * (probs_i - y_i) * (r - v_i * cos(r, v_i))  for all i
  Attractive for correct class, repulsive for wrong classes.

Version 2 — Winner-only bipolar (atomic, minimal):
  if label(winner) == label(input): v_winner += lr * r  (current rule)
  if label(winner) != label(input): v_winner -= lr * r  (fold reads its own label)
  Same operation, same magnitude, sign from stored label — no softmax needed.

Test: Split-CIFAR-100, spawn=0.95, same ResNet-18 features as Step 66b.
"""

import sys
import time
import math
import numpy as np

CACHE_PATH = 'C:/Users/Admin/cifar100_resnet18_features.npz'
N_TASKS = 20
CLASSES_TASK = 5
D_EMBED = 512
MERGE_THRESH = 0.95
SPAWN_THRESH = 0.95
LR = 0.015
SOFTMAX_TEMP = 0.1   # temperature for full-gradient version


def load_data():
    data = np.load(CACHE_PATH)
    X_tr, y_tr = data['X_train'], data['y_train']
    X_te, y_te = data['X_test'],  data['y_test']
    splits = []
    for t in range(N_TASKS):
        c0 = t * CLASSES_TASK
        tm = np.isin(y_tr, range(c0, c0 + CLASSES_TASK))
        em = np.isin(y_te, range(c0, c0 + CLASSES_TASK))
        splits.append((X_tr[tm], y_tr[tm], X_te[em], y_te[em]))
    return splits


class GradCodebook:
    """Codebook with configurable update rule."""

    def __init__(self, d, spawn_thresh=SPAWN_THRESH, mode='attractive'):
        """
        mode: 'attractive' (baseline), 'bipolar' (winner-only), 'full_grad'
        """
        self.d = d
        self.spawn_thresh = spawn_thresh
        self.mode = mode
        self.vectors = np.empty((0, d), dtype=np.float32)
        self.labels = []
        self.n_spawned = 0
        self.n_merged = 0

    def step(self, r, label=None):
        if len(self.vectors) == 0:
            self._spawn(r, label)
            return

        sims   = self.vectors @ r   # (N,) cosines
        winner = int(np.argmax(sims))

        if sims[winner] < self.spawn_thresh:
            self._spawn(r, label)
            return

        if self.mode == 'attractive':
            # Baseline: always attract winner
            self._update(winner, r, sign=+1)

        elif self.mode == 'bipolar':
            # Winner-only bipolar: sign from stored label
            sign = +1 if self.labels[winner] == label else -1
            self._update(winner, r, sign=sign)

        elif self.mode == 'full_grad':
            # Full gradient: update all prototypes
            # probs = softmax(sims / temp)
            s = sims / SOFTMAX_TEMP
            s -= s.max()
            exps = np.exp(s)
            probs = exps / exps.sum()
            # y_i = 1 if label[i] == label else 0
            y = np.array([1.0 if self.labels[i] == label else 0.0
                          for i in range(len(self.labels))], dtype=np.float32)
            # delta_v_i = (probs_i - y_i) * (r - v_i * cos_i)
            # gradient descent: v_i -= lr * delta_v_i
            errors = probs - y   # (N,)
            # Only update vectors with non-negligible gradient
            active = np.abs(errors) > 1e-4
            if active.any():
                perp = r[np.newaxis, :] - self.vectors[active] * sims[active, np.newaxis]
                self.vectors[active] -= LR * (errors[active, np.newaxis] * perp)
                norms = np.linalg.norm(self.vectors[active], axis=1, keepdims=True) + 1e-15
                self.vectors[active] /= norms

    def _update(self, i, r, sign):
        v = self.vectors[i] + sign * LR * r
        n = np.linalg.norm(v)
        if n > 1e-15:
            self.vectors[i] = v / n

    def _spawn(self, r, label):
        new_v = r / (np.linalg.norm(r) + 1e-15)
        if len(self.vectors) > 0:
            abs_sims = np.abs(self.vectors @ new_v)
            bi = int(np.argmax(abs_sims))
            if abs_sims[bi] > MERGE_THRESH:
                fused = self.vectors[bi] + new_v
                self.vectors[bi] = fused / (np.linalg.norm(fused) + 1e-15)
                self.n_merged += 1
                return
        self.vectors = np.vstack([self.vectors, new_v[np.newaxis, :]])
        self.labels.append(label)
        self.n_spawned += 1

    def classify_batch(self, X, k=1):
        if len(self.vectors) == 0:
            return [None] * len(X)
        sims = X @ self.vectors.T
        if k == 1:
            return [self.labels[i] for i in np.argmax(sims, axis=1)]
        top_k = np.argpartition(sims, -k, axis=1)[:, -k:]
        preds = []
        for i in range(len(X)):
            votes = [self.labels[j] for j in top_k[i]]
            preds.append(max(set(votes), key=votes.count))
        return preds


def run(splits, mode, label):
    cb = GradCodebook(D_EMBED, mode=mode)
    acc_matrix = [[None]*N_TASKS for _ in range(N_TASKS)]
    t0 = time.time()
    for task_id in range(N_TASKS):
        X_t, y_t = splits[task_id][0], splits[task_id][1]
        for i in range(len(X_t)):
            cb.step(X_t[i], label=int(y_t[i]))
        for et in range(task_id + 1):
            X_te, y_te = splits[et][2], splits[et][3]
            preds = cb.classify_batch(X_te)
            acc_matrix[et][task_id] = sum(p==g for p,g in zip(preds,y_te))/len(y_te)
    elapsed = time.time() - t0
    final = [acc_matrix[i][N_TASKS-1] for i in range(N_TASKS)]
    aa    = sum(final) / N_TASKS
    fgt   = [max(0., acc_matrix[i][i] - acc_matrix[i][N_TASKS-1]) for i in range(N_TASKS-1)]
    return {
        'label': label, 'mode': mode, 'aa': aa,
        'forgetting': sum(fgt)/len(fgt), 'cb_size': len(cb.vectors),
        'elapsed': elapsed, 'final': final,
    }


def main():
    print('='*70)
    print('  Step 71 -- Gradient-Derived Update Rule')
    print('  Split-CIFAR-100, spawn=0.95, ResNet-18 features')
    print('  One variable: update rule (attractive / bipolar / full_grad)')
    print('='*70)

    print('\nLoading features...')
    splits = load_data()

    modes = [
        ('attractive', 'Attractive (baseline, Step 66b)'),
        ('bipolar',    'Bipolar winner-only (atomic)'),
        ('full_grad',  'Full gradient (softmax, all prototypes)'),
    ]
    results = []
    for mode, label in modes:
        print(f'\n  Running {label}...')
        r = run(splits, mode, label)
        results.append(r)
        print(f'    AA={r["aa"]*100:.1f}%  F={r["forgetting"]*100:.1f}pp  '
              f'cb={r["cb_size"]}  t={r["elapsed"]:.0f}s')

    print('\n' + '='*70)
    print('  STEP 71 SUMMARY -- Gradient Update Rule Comparison')
    print('='*70)
    print(f'\n  {"Mode":<35} | {"AA":>7} | {"Forgetting":>10} | {"CB":>7} | {"Time":>6}')
    print('  ' + '-'*68)
    for r in results:
        print(f'  {r["label"]:<35} | {r["aa"]*100:>6.1f}% | '
              f'{r["forgetting"]*100:>9.1f}pp | {r["cb_size"]:>7} | {r["elapsed"]:>5.0f}s')

    base = results[0]
    print(f'\n  Published baselines: Fine-tune ~6% | EWC ~33% | SI ~36% | DER++ ~51%')
    print(f'\n  AA gain over attractive baseline:')
    for r in results[1:]:
        delta = (r["aa"] - base["aa"]) * 100
        print(f'    {r["label"]}: {delta:+.1f}pp AA, '
              f'{(r["forgetting"]-base["forgetting"])*100:+.1f}pp forgetting')

    print()
    best = max(results, key=lambda r: r['aa'])
    print(f'  Best: {best["label"]} -> AA={best["aa"]*100:.1f}%  '
          f'F={best["forgetting"]*100:.1f}pp')

    if best['aa'] > base['aa'] + 0.03:
        print('  VERDICT: Gradient-derived update improves accuracy.')
        if best['forgetting'] < base['forgetting']:
            print('           AND reduces forgetting. Both axes improved.')
    else:
        print('  VERDICT: Gradient update does not significantly improve AA.')
        print('           Classify as readout limitation, not update rule.')
    print('='*70)


if __name__ == '__main__':
    main()
