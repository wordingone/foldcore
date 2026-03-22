#!/usr/bin/env python3
"""
Step 66: Split-CIFAR-100 continual learning benchmark with FluxCore.

Protocol (standard in CL literature):
  - 20 sequential tasks. Each task contains 5 classes from CIFAR-100.
  - Classes split in order: task 0 = classes 0-4, task 1 = classes 5-9, ..., task 19 = classes 95-99.
  - Training: 500 samples/class x 5 classes = 2500 train per task.
  - Evaluation: 100 samples/class x 5 classes = 500 test per task (all tasks seen so far).
  - No replay of previous tasks.
  - Final metric: Average Accuracy (AA) and Forgetting (F) after all 20 tasks.

Embedding: Frozen ResNet-18, avgpool output -> R^512. Unit-normalized.
  - One-time extraction with torch (CUDA if available).
  - Features cached in numpy array for fast iteration.
Classifier: nearest-prototype via numpy batch cosine (snapshot codebook after each task).

Published baselines (Split-CIFAR-100, 20-task):
  Fine-tune (SGD):     ~6% AA  (catastrophic forgetting)
  EWC:                ~33% AA
  SI:                 ~36% AA
  LwF:                ~27% AA
  DER++:              ~51% AA
  Joint (upper bound): ~67% AA

FluxCore expected: near-zero forgetting (prototypes persist), per-task accuracy limited
by nearest-prototype readout vs gradient-optimized decision boundaries.
"""

import sys
import time

import numpy as np

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from fluxcore_manytofew import ManyToFewKernel

# ─── Config ───────────────────────────────────────────────────────────────────

N_TASKS      = 20
N_CLASSES    = 100
CLASSES_TASK = 5      # classes per task
D_EMBED      = 512    # ResNet-18 avgpool dim
N_TRAIN_CLS  = 500    # CIFAR-100 train samples per class
N_TEST_CLS   = 100    # CIFAR-100 test samples per class
N_TRAIN_TASK = N_TRAIN_CLS * CLASSES_TASK   # 2500
N_TEST_TASK  = N_TEST_CLS  * CLASSES_TASK   # 500


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_features(cache_path='C:/Users/Admin/cifar100_resnet18_features.npz'):
    """
    Load or compute frozen ResNet-18 features for CIFAR-100.

    Returns:
        X_train: (50000, 512) float32 unit-normalized embeddings
        y_train: (50000,) int labels 0-99
        X_test:  (10000, 512) float32 unit-normalized embeddings
        y_test:  (10000,) int labels 0-99
    """
    import os
    if os.path.exists(cache_path):
        print(f'  Loading cached features from {cache_path}...')
        data = np.load(cache_path)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']

    print('  Extracting ResNet-18 features (one-time)...')
    import torch
    import torchvision
    import torchvision.transforms as transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')

    # Standard CIFAR normalization
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_ds = torchvision.datasets.CIFAR100('C:/Users/Admin/cifar100_data',
                                              train=True, download=True, transform=tf)
    test_ds  = torchvision.datasets.CIFAR100('C:/Users/Admin/cifar100_data',
                                              train=False, download=True, transform=tf)

    # Load frozen ResNet-18, remove final FC layer
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()   # output avgpool features (512-dim)
    model = model.to(device)
    model.eval()

    def extract(dataset, label='train'):
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False,
                                              num_workers=0)
        feats, labels = [], []
        with torch.no_grad():
            for batch_idx, (imgs, ys) in enumerate(loader):
                imgs = imgs.to(device)
                f = model(imgs).cpu().numpy().astype(np.float32)
                feats.append(f)
                labels.append(ys.numpy())
                if (batch_idx + 1) % 20 == 0:
                    print(f'    {label}: {(batch_idx+1)*256} / {len(dataset)}')
        X = np.concatenate(feats, axis=0)
        y = np.concatenate(labels, axis=0)
        # Unit normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-15
        return (X / norms).astype(np.float32), y

    X_train, y_train = extract(train_ds, 'train')
    X_test,  y_test  = extract(test_ds,  'test')

    np.savez(cache_path, X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test)
    print(f'  Features saved to {cache_path}')
    return X_train, y_train, X_test, y_test


# ─── Task splits ──────────────────────────────────────────────────────────────

def make_task_splits(X_train, y_train, X_test, y_test):
    """
    Split into 20 tasks of 5 classes each (classes in order).

    Returns list of (X_tr, y_tr, X_te, y_te) per task.
    """
    splits = []
    for t in range(N_TASKS):
        cls_start = t * CLASSES_TASK
        cls_end   = cls_start + CLASSES_TASK
        task_classes = list(range(cls_start, cls_end))

        tr_mask = np.isin(y_train, task_classes)
        te_mask = np.isin(y_test,  task_classes)

        splits.append((X_train[tr_mask], y_train[tr_mask],
                       X_test[te_mask],  y_test[te_mask]))
    return splits


# ─── Numpy batch classify ─────────────────────────────────────────────────────

def batch_classify(X_embed, codebook_np, cb_labels):
    """
    X_embed:     (n, d) unit vectors
    codebook_np: (N, d) unit vectors
    cb_labels:   list of N labels

    Returns list of predicted labels.
    """
    if len(cb_labels) == 0:
        return [None] * len(X_embed)
    sims    = X_embed @ codebook_np.T     # (n, N)
    winners = np.argmax(sims, axis=1)     # (n,)
    return [cb_labels[w] for w in winners]


# ─── Benchmark ────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('  Step 66 -- Split-CIFAR-100 Continual Learning (FluxCore)')
    print(f'  {N_TASKS} tasks | {CLASSES_TASK} classes/task | '
          f'{N_TRAIN_TASK} train / {N_TEST_TASK} test per task')
    print(f'  Embedding: frozen ResNet-18 -> R^{D_EMBED} (unit-normalized)')
    print(f'  spawn_thresh=0.5 | n_matrix=8 | no replay')
    print('=' * 70)

    t0 = time.time()

    # Feature extraction (cached after first run)
    print('\nLoading/extracting features...')
    X_train, y_train, X_test, y_test = extract_features()
    print(f'  Train: {X_train.shape}  Test: {X_test.shape}')

    # Task splits
    splits = make_task_splits(X_train, y_train, X_test, y_test)
    print(f'  {N_TASKS} task splits created ({CLASSES_TASK} classes each)')

    # Pre-select test embeddings per task
    test_embeds = [(s[2], s[3]) for s in splits]   # (X_te, y_te) per task

    # FluxCore kernel (d=512 for ResNet-18 avgpool)
    kernel = ManyToFewKernel(
        n_matrix=8, k=4, d=D_EMBED, seed=42, proj_seed=999,
        tau=0.3, k_couple=5,
        spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015,
    )

    # Per-task accuracy matrix [eval_task][after_task]
    acc_matrix = [[None] * N_TASKS for _ in range(N_TASKS)]

    print('\n' + '-' * 70)

    for task_id in range(N_TASKS):
        t_task = time.time()
        cls_start = task_id * CLASSES_TASK
        cls_end   = cls_start + CLASSES_TASK
        print(f'\n  Task {task_id:2d} (classes {cls_start}-{cls_end-1})'
              f' -- Training ({N_TRAIN_TASK} samples)...')

        X_t, y_t = splits[task_id][0], splits[task_id][1]

        spawned_before = kernel.total_spawned
        # Stream through FluxCore (sequential, label=class index)
        for i in range(len(X_t)):
            kernel.step(r=X_t[i].tolist(), label=int(y_t[i]))

        new_spawns = kernel.total_spawned - spawned_before
        train_time = time.time() - t_task
        print(f'  Task {task_id:2d}: {new_spawns} new spawns, '
              f'total cb={len(kernel.codebook)}, '
              f'train_time={train_time:.1f}s')

        # Snapshot codebook for batch numpy inference
        if kernel.codebook:
            cb_np     = np.array(kernel.codebook, dtype=np.float32)
            cb_labels = list(kernel.cb_labels)
        else:
            cb_np     = np.zeros((0, D_EMBED), dtype=np.float32)
            cb_labels = []

        # Evaluate on all tasks seen so far
        t_eval = time.time()
        for eval_task in range(task_id + 1):
            X_te, y_te = test_embeds[eval_task]
            preds   = batch_classify(X_te, cb_np, cb_labels)
            correct = sum(1 for p, g in zip(preds, y_te) if p == g)
            acc     = correct / len(y_te)
            acc_matrix[eval_task][task_id] = acc

        eval_time = time.time() - t_eval
        print(f'  Evaluation ({task_id+1} tasks): {eval_time:.1f}s', end='  |')
        for eval_task in range(task_id + 1):
            print(f' T{eval_task}={acc_matrix[eval_task][task_id]:.3f}', end='')
        print()

    elapsed = time.time() - t0

    # ─── Summary ──────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('  STEP 66 SUMMARY -- Split-CIFAR-100')
    print('=' * 70)

    print(f'\n  FluxCore: n_matrix=8, k=4, d={D_EMBED}, spawn=0.5, lr_cb=0.015')
    print(f'  Codebook: {len(kernel.codebook)} vectors, '
          f'{kernel.total_spawned} spawned, {kernel.total_merged} merged')

    final_accs = [acc_matrix[i][N_TASKS - 1] for i in range(N_TASKS)]
    avg_accuracy = sum(final_accs) / N_TASKS

    print(f'\n  Per-task accuracy after all {N_TASKS} tasks:')
    for i, acc in enumerate(final_accs):
        cls_start = i * CLASSES_TASK
        print(f'    Task {i:2d} (cls {cls_start:2d}-{cls_start+4}): '
              f'{acc:.4f}  ({acc*100:.1f}%)')

    print(f'\n  Average Accuracy (AA): {avg_accuracy:.4f}  ({avg_accuracy*100:.1f}%)')

    # Forgetting
    forgetting_vals = []
    for i in range(N_TASKS - 1):
        at_train = acc_matrix[i][i]
        final    = acc_matrix[i][N_TASKS - 1]
        forgetting_vals.append(max(0.0, at_train - final))
    avg_forgetting = sum(forgetting_vals) / len(forgetting_vals)

    print(f'\n  Forgetting (avg drop from peak): '
          f'{avg_forgetting:.4f}  ({avg_forgetting*100:.1f}pp)')

    print(f'\n  Published baselines (Split-CIFAR-100, 20-task):')
    print(f'    Fine-tune (catastrophic forgetting):  ~6% AA')
    print(f'    EWC  (Kirkpatrick 2017):             ~33% AA')
    print(f'    SI   (Zenke 2017):                   ~36% AA')
    print(f'    LwF  (Li & Hoiem 2016):              ~27% AA')
    print(f'    DER++ (Buzzega 2020):                ~51% AA')
    print(f'    Joint (upper bound):                 ~67% AA')
    print(f'    FluxCore (this run):          {avg_accuracy*100:.1f}% AA')

    print(f'\n  Runtime: {elapsed:.1f}s')

    # Full accuracy matrix
    print('\n  Full accuracy matrix (rows=eval_task, cols=after_task):')
    header = '  Task |' + ''.join(f' AfterT{j:02d}' for j in range(N_TASKS))
    print(header)
    print('  ' + '-' * (7 + 9 * N_TASKS))
    for i in range(N_TASKS):
        row = f'   T{i:2d} |'
        for j in range(N_TASKS):
            v = acc_matrix[i][j]
            row += f'  {v:.3f}' if v is not None else '     --'
        print(row)

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
