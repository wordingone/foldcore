#!/usr/bin/env python3
"""
Step 65: Permuted-MNIST continual learning benchmark with FluxCore.

Protocol (standard in CL literature):
  - 10 sequential tasks. Each task applies a fixed random pixel permutation to MNIST.
  - Training: 6,000 training images per task (600/class, stratified sample).
  - Evaluation: full 10,000 test images per task, batch numpy inference.
  - No replay of previous tasks.
  - Final metric: Average Accuracy (AA) and Forgetting (F) after all 10 tasks.

Note on sampling: published CL baselines use all 60K training images. This benchmark
uses 6K/task for runtime feasibility (Python codebook matching is O(N*d) per step).
The results are comparable as a proof-of-mechanism. Step 66 (CIFAR-100) uses frozen
ResNet features which enable full-dataset training.

Embedding: flatten 28x28 → R^784, random projection to R^384 (numpy, fast).
Classifier: nearest-prototype via numpy batch cosine (snapshot codebook after each task).
"""

import sys
import math
import random
import time

import numpy as np

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from fluxcore_manytofew import ManyToFewKernel, _vec_cosine

# ─── Config ───────────────────────────────────────────────────────────────────

N_TASKS       = 10
D_OUT         = 384
N_TRAIN_TASK  = 6000    # 600 per class per task (10 classes)
N_TEST_TASK   = 10000   # all test images
N_CLASSES     = 10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES


# ─── MNIST loading ─────────────────────────────────────────────────────────────

def load_mnist():
    import torchvision, torchvision.transforms as transforms
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=True,  download=True, transform=tf)
    test_ds  = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=False, download=True, transform=tf)
    X_train = train_ds.data.numpy().reshape(len(train_ds), -1).astype(np.float32) / 255.0
    y_train = train_ds.targets.numpy()
    X_test  = test_ds.data.numpy().reshape(len(test_ds), -1).astype(np.float32) / 255.0
    y_test  = test_ds.targets.numpy()
    return X_train, y_train, X_test, y_test


def stratified_sample(X, y, n_per_class, seed):
    """Return n_per_class samples per class, shuffled."""
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(N_CLASSES):
        cls_idx = np.where(y == c)[0]
        chosen = rng.choice(cls_idx, n_per_class, replace=False)
        idx.extend(chosen)
    rng.shuffle(idx)
    return X[idx], y[idx]


# ─── Projection (numpy) ───────────────────────────────────────────────────────

def make_projection_matrix(d_in=784, d_out=D_OUT, seed=12345):
    """Frozen random Gaussian projection. Returns numpy array (d_out, d_in)."""
    rng = np.random.RandomState(seed)
    P = rng.randn(d_out, d_in).astype(np.float32) / math.sqrt(d_in)
    return P


def project_batch(X_flat, P, perm):
    """Project a batch of flat images with permutation. Returns unit-normalized R^d_out."""
    # X_flat: (n, 784) numpy
    # perm: list of 784 ints
    X_perm = X_flat[:, perm]              # (n, 784) — apply permutation
    proj = X_perm @ P.T                   # (n, d_out)
    norms = np.linalg.norm(proj, axis=1, keepdims=True) + 1e-15
    return (proj / norms).astype(np.float32)


# ─── Task permutations ────────────────────────────────────────────────────────

def make_permutation(n=784, seed=None):
    perm = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(perm)
    return np.array(perm, dtype=np.int64)


# ─── Numpy batch classify (fast evaluation) ───────────────────────────────────

def batch_classify(X_embed, codebook_np, cb_labels):
    """
    X_embed: (n, d) unit vectors (numpy)
    codebook_np: (N, d) unit vectors (numpy snapshot)
    cb_labels: list of N labels

    Returns list of predicted labels.
    """
    if len(cb_labels) == 0:
        return [None] * len(X_embed)
    sims = X_embed @ codebook_np.T     # (n, N) — cosines (unit vectors)
    winners = np.argmax(sims, axis=1)  # (n,)
    return [cb_labels[w] for w in winners]


# ─── Benchmark ────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('  Step 65 — Permuted-MNIST Continual Learning (FluxCore)')
    print(f'  {N_TASKS} tasks | {N_TRAIN_TASK} train / {N_TEST_TASK} test per task | d={D_OUT}')
    print(f'  spawn_thresh=0.5 | n_matrix=8 | no replay')
    print('=' * 70)

    t0 = time.time()

    print('\nLoading MNIST...')
    X_train_full, y_train_full, X_test_full, y_test_full = load_mnist()

    print('Building projection matrix R^784 -> R^384...')
    P = make_projection_matrix()

    # Task permutations (task 0 = identity)
    perms = [np.arange(784, dtype=np.int64)]
    for t in range(1, N_TASKS):
        perms.append(make_permutation(784, seed=t * 1000))

    # Pre-embed all test data for each task (fast numpy, one time)
    print('Pre-embedding test sets...')
    test_embeds = []   # list of (N_TEST_TASK, D_OUT) arrays
    for t in range(N_TASKS):
        emb = project_batch(X_test_full, P, perms[t])
        test_embeds.append(emb)
        print(f'  Task {t} test embed: {emb.shape}')

    # FluxCore kernel
    kernel = ManyToFewKernel(
        n_matrix=8, k=4, d=D_OUT, seed=42, proj_seed=999,
        tau=0.3, k_couple=5,
        spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015,
    )

    # Per-task accuracy matrix
    acc_matrix = [[None] * N_TASKS for _ in range(N_TASKS)]

    print('\n' + '-' * 70)

    for task_id in range(N_TASKS):
        t_task = time.time()
        print(f'\n  Task {task_id} — Training ({N_TRAIN_TASK} samples, {TRAIN_PER_CLS}/class)...')

        # Sample training data
        X_t, y_t = stratified_sample(X_train_full, y_train_full,
                                      TRAIN_PER_CLS, seed=task_id * 1337)
        # Embed training data (batch, fast)
        X_t_emb = project_batch(X_t, P, perms[task_id])  # (N_TRAIN_TASK, D_OUT)

        spawned_before = kernel.total_spawned
        # Stream through FluxCore (Python kernel, sequential)
        for i in range(len(X_t_emb)):
            r = X_t_emb[i].tolist()
            y = int(y_t[i])
            kernel.step(r=r, label=y)

        new_spawns = kernel.total_spawned - spawned_before
        print(f'  Task {task_id}: {new_spawns} new spawns, '
              f'total cb={len(kernel.codebook)}, '
              f'train_time={time.time()-t_task:.1f}s')

        # Snapshot codebook for batch numpy inference
        if kernel.codebook:
            cb_np = np.array(kernel.codebook, dtype=np.float32)  # (N, D_OUT)
            cb_labels = list(kernel.cb_labels)
        else:
            cb_np = np.zeros((0, D_OUT), dtype=np.float32)
            cb_labels = []

        # Evaluate on all tasks seen so far
        t_eval = time.time()
        for eval_task in range(task_id + 1):
            preds = batch_classify(test_embeds[eval_task], cb_np, cb_labels)
            correct = sum(1 for p, g in zip(preds, y_test_full) if p == g)
            acc = correct / N_TEST_TASK
            acc_matrix[eval_task][task_id] = acc
        eval_time = time.time() - t_eval
        print(f'  Evaluation ({task_id+1} tasks): {eval_time:.1f}s', end='  |')
        for eval_task in range(task_id + 1):
            print(f' T{eval_task}={acc_matrix[eval_task][task_id]:.3f}', end='')
        print()

    elapsed = time.time() - t0

    # ─── Summary ──────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('  STEP 65 SUMMARY — Permuted-MNIST')
    print('=' * 70)

    print(f'\n  FluxCore: n_matrix=8, k=4, d={D_OUT}, spawn=0.5, lr_cb=0.015')
    print(f'  Codebook: {len(kernel.codebook)} vectors, '
          f'{kernel.total_spawned} spawned, {kernel.total_merged} merged')

    final_accs = [acc_matrix[i][N_TASKS - 1] for i in range(N_TASKS)]
    avg_accuracy = sum(final_accs) / N_TASKS

    print(f'\n  Per-task accuracy after all {N_TASKS} tasks:')
    for i, acc in enumerate(final_accs):
        tag = "base" if i == 0 else f"perm{i}"
        print(f'    Task {i:2d} ({tag:6s}): {acc:.4f}  ({acc*100:.1f}%)')

    print(f'\n  Average Accuracy (AA): {avg_accuracy:.4f}  ({avg_accuracy*100:.1f}%)')

    # Forgetting
    forgetting_vals = []
    for i in range(N_TASKS - 1):
        at_train = acc_matrix[i][i]
        final    = acc_matrix[i][N_TASKS - 1]
        forgetting_vals.append(max(0.0, at_train - final))
    avg_forgetting = sum(forgetting_vals) / len(forgetting_vals)

    print(f'\n  Forgetting (avg drop from peak): {avg_forgetting:.4f}  ({avg_forgetting*100:.1f}pp)')

    print(f'\n  Published baselines (Permuted-MNIST, per BENCHMARK_PLAN.md):')
    print(f'    SGD  (fine-tuning, catastrophic forgetting): ~65% AA')
    print(f'    EWC  (Kirkpatrick 2017):                     ~85% AA')
    print(f'    SI   (Zenke 2017):                           ~86% AA')
    print(f'    FluxCore (this run, 6K train/task):          {avg_accuracy*100:.1f}% AA')

    print(f'\n  Runtime: {elapsed:.1f}s')

    # Full accuracy matrix
    print('\n  Full accuracy matrix:')
    print('  Task |' + ''.join(f'  AfterT{j}' for j in range(N_TASKS)))
    print('  ' + '-' * (7 + 10 * N_TASKS))
    for i in range(N_TASKS):
        row = f'   T{i:2d} |'
        for j in range(N_TASKS):
            v = acc_matrix[i][j]
            row += f'    {v:.3f}' if v is not None else '      --'
        print(row)

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
