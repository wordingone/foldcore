#!/usr/bin/env python3
"""
Step 107 -- Unified Process (S1) on CIFAR-100. Full 10-task.
[researcher] mail 1258.

Same process() function from Step 106. Stress-test: ~38% accuracy means
~62% of self-supervised spawns are mislabeled. Does S1 survive?

Config: sp=0.95 (CIFAR-100 calibrated), k=5.
Condition 1: labeled training, frozen eval (should reproduce ~38.3% AA from Step 100b).
Condition 2: labeled training, self-supervised eval spawning.

Kill: condition 2 AA < condition 1 AA - 5pp (wider margin than P-MNIST).
"""

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ─── Config ───────────────────────────────────────────────────────────────────

N_TASKS      = 10
CLASSES_TASK = 10
D_EMBED      = 512
SPAWN_THRESH = 0.95
K            = 5
CACHE_PATH   = '/mnt/c/Users/Admin/cifar100_resnet18_features.npz'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─── Unified Process (identical to Step 106) ─────────────────────────────────

def process(V, labels, r, label=None, k=K, spawn_thresh=SPAWN_THRESH):
    """SAME code path for training and inference."""
    r = F.normalize(r, dim=0)

    if V.shape[0] == 0:
        V      = r.unsqueeze(0)
        labels = torch.tensor([label if label is not None else 0], device=DEVICE)
        return 0, V, labels

    sims = V @ r

    # Top-k readout (always)
    n_classes = int(labels.max().item()) + 1
    scores = torch.zeros(n_classes, device=DEVICE)
    for c in range(n_classes):
        cs = sims[labels == c]
        if cs.shape[0] == 0:
            continue
        scores[c] = cs.topk(min(k, cs.shape[0])).values.sum()
    prediction = int(scores.argmax().item())

    # Spawn if novel (always)
    if sims.max().item() < spawn_thresh:
        use_label = label if label is not None else prediction
        V      = torch.cat([V, r.unsqueeze(0)])
        labels = torch.cat([labels, torch.tensor([use_label], device=DEVICE)])

    return prediction, V, labels


def process_batch_eval_frozen(V, labels, X_te, k=K):
    """Condition 1: frozen codebook, batched top-k."""
    X_te    = F.normalize(X_te, dim=1)
    sims    = X_te @ V.T                              # (n, N)
    n       = X_te.shape[0]
    n_cls   = int(labels.max().item()) + 1
    scores  = torch.zeros(n, n_cls, device=DEVICE)
    for c in range(n_cls):
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        cs   = sims[:, mask]
        keff = min(k, cs.shape[1])
        scores[:, c] = cs.topk(keff, dim=1).values.sum(dim=1)
    return scores.argmax(dim=1).cpu()


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_splits():
    print(f"Loading ResNet-18 features from {CACHE_PATH}...", flush=True)
    data  = np.load(CACHE_PATH)
    X_tr  = data['X_train']
    y_tr  = data['y_train']
    X_te  = data['X_test']
    y_te  = data['y_test']
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}, D={X_tr.shape[1]}", flush=True)
    splits = []
    for t in range(N_TASKS):
        c0, c1   = t * CLASSES_TASK, (t + 1) * CLASSES_TASK
        mask_tr  = np.isin(y_tr, range(c0, c1))
        mask_te  = np.isin(y_te, range(c0, c1))
        X_tr_t   = F.normalize(torch.tensor(X_tr[mask_tr], dtype=torch.float32, device=DEVICE), dim=1)
        y_tr_t   = torch.tensor(y_tr[mask_tr], dtype=torch.long)
        X_te_t   = F.normalize(torch.tensor(X_te[mask_te], dtype=torch.float32, device=DEVICE), dim=1)
        y_te_t   = torch.tensor(y_te[mask_te], dtype=torch.long)
        splits.append((X_tr_t, y_tr_t, X_te_t, y_te_t))
        print(f"  Task {t}: classes {c0}-{c1-1}, train={mask_tr.sum()}, test={mask_te.sum()}", flush=True)
    return splits


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_aa(mat, n_tasks):
    vals = [mat[t][n_tasks - 1] for t in range(n_tasks)
            if mat[t][n_tasks - 1] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def compute_fgt(mat, n_tasks):
    vals = []
    for t in range(n_tasks - 1):
        if mat[t][t] is not None and mat[t][n_tasks - 1] is not None:
            vals.append(max(0.0, mat[t][t] - mat[t][n_tasks - 1]))
    return sum(vals) / len(vals) if vals else 0.0


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Step 107 -- Unified Process (S1), CIFAR-100", flush=True)
    print(f"sp={SPAWN_THRESH}, k={K}, DEVICE={DEVICE}", flush=True)
    print(f"Baseline cond1 target: ~38.3% AA (Step 100b)", flush=True)
    print(f"Kill: cond2 < cond1 - 5pp", flush=True)
    print(flush=True)

    splits = load_splits()
    print(flush=True)

    mat_c1 = [[None] * N_TASKS for _ in range(N_TASKS)]
    mat_c2 = [[None] * N_TASKS for _ in range(N_TASKS)]

    V_train      = torch.empty(0, D_EMBED, device=DEVICE)
    labels_train = torch.empty(0, dtype=torch.long, device=DEVICE)

    for task_id in range(N_TASKS):
        X_tr_t, y_tr_t, _, _ = splits[task_id]

        # --- Train ---
        t_train = time.time()
        cb_before = V_train.shape[0]
        for i in range(len(X_tr_t)):
            _, V_train, labels_train = process(
                V_train, labels_train, X_tr_t[i], label=int(y_tr_t[i].item()))
        train_s = time.time() - t_train

        # --- Eval condition 1: frozen codebook ---
        t_eval = time.time()
        for eval_task in range(task_id + 1):
            _, _, X_te_t, y_te_t = splits[eval_task]
            preds = process_batch_eval_frozen(V_train, labels_train, X_te_t)
            acc = (preds == y_te_t).float().mean().item()
            mat_c1[eval_task][task_id] = acc

        # --- Eval condition 2: self-supervised spawning ---
        V_eval      = V_train.clone()
        labels_eval = labels_train.clone()
        for eval_task in range(task_id + 1):
            _, _, X_te_t, y_te_t = splits[eval_task]
            preds_list = []
            for i in range(X_te_t.shape[0]):
                pred, V_eval, labels_eval = process(
                    V_eval, labels_eval, X_te_t[i], label=None)
                preds_list.append(pred)
            preds = torch.tensor(preds_list, dtype=torch.long)
            acc   = (preds == y_te_t).float().mean().item()
            mat_c2[eval_task][task_id] = acc
        eval_s   = time.time() - t_eval
        spawned  = V_eval.shape[0] - V_train.shape[0]
        cb_after = V_train.shape[0]

        aa_c1 = sum(mat_c1[t][task_id] for t in range(task_id + 1)) / (task_id + 1)
        aa_c2 = sum(mat_c2[t][task_id] for t in range(task_id + 1)) / (task_id + 1)
        print(f"  T{task_id}: train={train_s:.1f}s eval={eval_s:.1f}s "
              f"cb={cb_before}->{cb_after} c1={aa_c1*100:.1f}% "
              f"c2={aa_c2*100:.1f}% spawned={spawned}", flush=True)

    elapsed = time.time() - t0

    aa_c1  = compute_aa(mat_c1, N_TASKS)
    fgt_c1 = compute_fgt(mat_c1, N_TASKS)
    aa_c2  = compute_aa(mat_c2, N_TASKS)
    fgt_c2 = compute_fgt(mat_c2, N_TASKS)
    delta  = aa_c2 - aa_c1

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 107 FINAL SUMMARY -- Unified Process (S1), CIFAR-100", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  S1 compliant: YES (one function, no train/eval branch)", flush=True)
    print(f"  Cond 1 (frozen):  AA={aa_c1*100:.1f}%  fgt={fgt_c1*100:.1f}pp  "
          f"(target: ~38.3%)", flush=True)
    print(f"  Cond 2 (spawning): AA={aa_c2*100:.1f}%  fgt={fgt_c2*100:.1f}pp  "
          f"delta={delta*100:+.1f}pp", flush=True)

    if delta < -0.05:
        verdict = f"DISPROVED -- cond2 degrades by {abs(delta)*100:.1f}pp > 5pp threshold"
    elif delta >= 0:
        verdict = f"PASSES -- self-supervised spawning helps (+{delta*100:.1f}pp)"
    else:
        verdict = f"PASSES -- self-supervised spawning neutral ({delta*100:.1f}pp, within 5pp)"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
