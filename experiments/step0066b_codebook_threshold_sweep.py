#!/usr/bin/env python3
"""
Step 66b: Spawn threshold sweep on Split-CIFAR-100 (frozen ResNet-18 features).

Same setup as Step 66. Only variable: spawn_thresh in {0.7, 0.8, 0.9, 0.95}.

Question: calibration vs fundamental limitation.
  - If higher threshold recovers AA (30%+, low forgetting) -> CALIBRATION.
  - If even optimal threshold gives poor results (best ~10%) -> FUNDAMENTAL.

Implementation note:
  Uses numpy-accelerated codebook layer for feasibility. Same algorithm as
  fluxcore_manytofew.py's codebook: spawn threshold, additive update + normalize,
  merge at spawn time (|cos| > merge_thresh). Matrix layer omitted — classification
  depends only on codebook, not matrix dynamics.

  spawn: if max_cosine(r, codebook) < spawn_thresh -> add normalize(r) to codebook
  update: codebook[winner] = normalize(codebook[winner] + lr * r)
  merge: at spawn, if |cos(new, existing)| > merge_thresh -> fuse, discard new
  classify: argmax(r @ codebook.T) -> codebook label
"""

import sys
import time
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────────────

N_TASKS      = 20
CLASSES_TASK = 5
D_EMBED      = 512
N_TRAIN_CLS  = 500
N_TEST_CLS   = 100
N_TRAIN_TASK = N_TRAIN_CLS * CLASSES_TASK   # 2500
N_TEST_TASK  = N_TEST_CLS  * CLASSES_TASK   # 500

MERGE_THRESH   = 0.95
LR_CODEBOOK    = 0.015
CACHE_PATH     = 'C:/Users/Admin/cifar100_resnet18_features.npz'
THRESHOLDS     = [0.7, 0.8, 0.9, 0.95]

REPORT_TASKS   = 5   # report per-task accuracy for tasks 0..4


# ─── Numpy codebook (same algorithm as fluxcore_manytofew.py) ─────────────────

class NumpyCodebook:
    """
    Vectorized codebook implementing the fold's spawn/update/merge rules.
    Same algorithm as ManyToFewKernel's codebook layer, numpy-accelerated.
    """

    def __init__(self, d, spawn_thresh, merge_thresh=0.95, lr=0.015):
        self.d            = d
        self.spawn_thresh = spawn_thresh
        self.merge_thresh = merge_thresh
        self.lr           = lr
        self.vectors      = np.empty((0, d), dtype=np.float32)  # (N, d) unit vecs
        self.labels       = []
        self.n_spawned    = 0
        self.n_merged     = 0

    def step(self, r, label=None):
        """
        r: (d,) unit-normalized numpy vector.
        Returns winner index (after potential spawn/merge).
        """
        if len(self.vectors) == 0:
            self._spawn(r, label)
            return 0

        sims = self.vectors @ r   # (N,) — cosines (unit vectors)
        winner = int(np.argmax(sims))
        max_sim = float(sims[winner])

        if max_sim < self.spawn_thresh:
            winner = self._spawn(r, label)
        else:
            # Additive update on winner
            v = self.vectors[winner] + self.lr * r
            self.vectors[winner] = v / (np.linalg.norm(v) + 1e-15)

        return winner

    def _spawn(self, r, label):
        new_v = r / (np.linalg.norm(r) + 1e-15)
        n = len(self.vectors)

        if n > 0:
            # Merge check: |cos(new, existing)| > merge_thresh
            abs_sims = np.abs(self.vectors @ new_v)   # (N,)
            best_i   = int(np.argmax(abs_sims))
            if abs_sims[best_i] > self.merge_thresh:
                # Fuse into existing (keep existing label)
                fused = self.vectors[best_i] + new_v
                self.vectors[best_i] = fused / (np.linalg.norm(fused) + 1e-15)
                self.n_merged += 1
                return best_i

        # No merge — add new vector
        self.vectors = np.vstack([self.vectors, new_v[np.newaxis, :]])
        self.labels.append(label)
        self.n_spawned += 1
        return len(self.vectors) - 1

    def classify_batch(self, X):
        """X: (n, d) unit vectors. Returns (n,) predicted labels."""
        if len(self.vectors) == 0:
            return [None] * len(X)
        sims    = X @ self.vectors.T    # (n, N)
        winners = np.argmax(sims, axis=1)  # (n,)
        return [self.labels[w] for w in winners]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_features():
    print(f'  Loading cached features from {CACHE_PATH}...')
    data = np.load(CACHE_PATH)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']


def make_task_splits(X_train, y_train, X_test, y_test):
    splits = []
    for t in range(N_TASKS):
        cls_start = t * CLASSES_TASK
        cls_end   = cls_start + CLASSES_TASK
        task_cls  = list(range(cls_start, cls_end))
        tr_mask   = np.isin(y_train, task_cls)
        te_mask   = np.isin(y_test,  task_cls)
        splits.append((X_train[tr_mask], y_train[tr_mask],
                       X_test[te_mask],  y_test[te_mask]))
    return splits


# ─── Single run ───────────────────────────────────────────────────────────────

def run_threshold(splits, spawn_thresh):
    cb = NumpyCodebook(D_EMBED, spawn_thresh, MERGE_THRESH, LR_CODEBOOK)
    acc_matrix = [[None] * N_TASKS for _ in range(N_TASKS)]
    t0 = time.time()

    for task_id in range(N_TASKS):
        t_task = time.time()
        X_t, y_t, _, _ = splits[task_id]

        spawned_before = cb.n_spawned
        for i in range(len(X_t)):
            cb.step(X_t[i], label=int(y_t[i]))
        new_spawns = cb.n_spawned - spawned_before

        train_time = time.time() - t_task
        print(f'    T{task_id:02d}: {new_spawns:4d} new spawns, '
              f'total_cb={len(cb.vectors):5d}, train={train_time:.1f}s')

        # Evaluate on all tasks seen so far
        for eval_task in range(task_id + 1):
            X_te, y_te = splits[eval_task][2], splits[eval_task][3]
            preds   = cb.classify_batch(X_te)
            correct = sum(1 for p, g in zip(preds, y_te) if p == g)
            acc_matrix[eval_task][task_id] = correct / len(y_te)

    elapsed = time.time() - t0

    # Final metrics
    final_accs = [acc_matrix[i][N_TASKS - 1] for i in range(N_TASKS)]
    aa = sum(final_accs) / N_TASKS

    forgetting_vals = []
    for i in range(N_TASKS - 1):
        at_train = acc_matrix[i][i]
        final    = acc_matrix[i][N_TASKS - 1]
        forgetting_vals.append(max(0.0, at_train - final))
    avg_f = sum(forgetting_vals) / len(forgetting_vals)

    return {
        'thresh':   spawn_thresh,
        'aa':       aa,
        'forgetting': avg_f,
        'cb_size':  len(cb.vectors),
        'spawned':  cb.n_spawned,
        'merged':   cb.n_merged,
        'elapsed':  elapsed,
        'final_accs': final_accs,
        'acc_matrix': acc_matrix,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('  Step 66b -- Spawn Threshold Sweep on Split-CIFAR-100')
    print(f'  20 tasks | 5 classes/task | frozen ResNet-18 -> R^512')
    print(f'  Thresholds: {THRESHOLDS} | merge_thresh=0.95 | lr=0.015')
    print(f'  (numpy-accelerated codebook, same algorithm as fluxcore_manytofew.py)')
    print('=' * 70)

    print('\nLoading features...')
    X_train, y_train, X_test, y_test = load_features()
    splits = make_task_splits(X_train, y_train, X_test, y_test)
    print(f'  {N_TASKS} task splits. Train: {X_train.shape}  Test: {X_test.shape}')

    results = []

    for thresh in THRESHOLDS:
        print(f'\n{"-"*70}')
        print(f'  spawn_thresh = {thresh}')
        print(f'{"-"*70}')
        r = run_threshold(splits, thresh)
        results.append(r)
        print(f'  DONE: AA={r["aa"]*100:.1f}%  F={r["forgetting"]*100:.1f}pp  '
              f'cb={r["cb_size"]}  elapsed={r["elapsed"]:.1f}s')

    # ─── Summary ──────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('  STEP 66b SUMMARY -- Spawn Threshold Sweep')
    print('=' * 70)

    print(f'\n  {"thresh":>8} | {"AA":>8} | {"Forgetting":>10} | '
          f'{"CB size":>8} | {"Spawned":>8} | {"Merged":>7} | {"Time":>6}')
    print('  ' + '-' * 70)
    for r in results:
        print(f'  {r["thresh"]:>8.2f} | {r["aa"]*100:>7.1f}% | '
              f'{r["forgetting"]*100:>9.1f}pp | {r["cb_size"]:>8} | '
              f'{r["spawned"]:>8} | {r["merged"]:>7} | {r["elapsed"]:>5.0f}s')

    print(f'\n  Per-task accuracy (tasks 0-{REPORT_TASKS-1}) for each threshold:')
    header = f'  {"Task":>6} |' + ''.join(f'  thresh={t}' for t in THRESHOLDS)
    print(header)
    print('  ' + '-' * (8 + 12 * len(THRESHOLDS)))
    for i in range(REPORT_TASKS):
        row = f'  T{i:02d}    |'
        for r in results:
            row += f'  {r["final_accs"][i]*100:>7.1f}%  '
        print(row)

    print(f'\n  All-task final accuracy:')
    for i in range(N_TASKS):
        row = f'  T{i:02d}    |'
        for r in results:
            row += f'  {r["final_accs"][i]*100:>7.1f}%  '
        print(row)

    print(f'\n  Published baselines (Split-CIFAR-100, 20-task):')
    print(f'    Fine-tune: ~6% AA | EWC: ~33% | SI: ~36% | DER++: ~51% | Joint: ~67%')

    best = max(results, key=lambda r: r['aa'])
    print(f'\n  Best FluxCore: {best["aa"]*100:.1f}% AA at spawn_thresh={best["thresh"]}')

    print(f'\n  Classification:')
    if best['aa'] > 0.25:
        print(f'    CALIBRATION — higher threshold recovers competitive AA.')
        print(f'    spawn_thresh=0.5 was miscalibrated for ResNet-18 geometry.')
    elif best['aa'] > 0.10:
        print(f'    PARTIAL — threshold helps but readout may also limit performance.')
        print(f'    Next step: compare with stronger readout (e.g., per-task threshold).')
    else:
        print(f'    FUNDAMENTAL — even optimal threshold gives <10% AA.')
        print(f'    Nearest-prototype cannot separate dense ResNet-18 features.')

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
