#!/usr/bin/env python3
"""
Step 100 -- Top-K Class Vote on CIFAR-100.
[researcher] mail 1237. Push harder: does top-k generalize beyond P-MNIST?

Split-CIFAR-100: 10 tasks, 10 classes each.
ResNet-18 features (same as Step 71 baseline, 33.5% AA fold).
Config: lr=0.001, spawn_thresh=0.7, k_vote={1,3,5,10}.

Kill criterion: best top-k (k>1) AA <= 1-NN AA on CIFAR-100 -> DEAD.
Secondary: 1-NN with cosine spawn must beat 33.5% fold baseline.
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
LR           = 0.001
SPAWN_THRESH = 0.95  # cosine threshold (calibrated for ResNet-18 features)
K_VOTE_VALS  = [1, 3, 5, 10]
CACHE_PATH   = '/mnt/c/Users/Admin/cifar100_resnet18_features.npz'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─── TopKFold ─────────────────────────────────────────────────────────────────

class TopKFold:
    def __init__(self, d, lr=0.001, spawn_thresh=0.7):
        self.V      = torch.empty(0, d, device=DEVICE)
        self.labels = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.lr     = lr
        self.spawn_thresh = spawn_thresh
        self.d      = d
        self.n_spawned = 0

    def step(self, r, label):
        r = F.normalize(r, dim=0)
        if self.V.shape[0] == 0 or (self.V @ r).max().item() < self.spawn_thresh:
            self.V      = torch.cat([self.V, r.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([label], device=DEVICE)])
            self.n_spawned += 1
            return
        sims   = self.V @ r
        winner = sims.argmax().item()
        self.V[winner] = F.normalize(
            self.V[winner] + self.lr * (r - self.V[winner]), dim=0)

    def eval_batch(self, R, k_vals):
        """R: (n, d) unit-normalized GPU tensor. Returns dict k -> preds (CPU)."""
        R    = F.normalize(R, dim=1)
        sims = R @ self.V.T                   # (n, N)
        n    = len(R)
        n_cls = int(self.labels.max().item()) + 1
        results = {}
        for k in k_vals:
            scores = torch.zeros(n, n_cls, device=DEVICE)
            for c in range(n_cls):
                mask = (self.labels == c)
                if mask.sum() == 0:
                    continue
                class_sims = sims[:, mask]
                k_eff      = min(k, class_sims.shape[1])
                scores[:, c] = class_sims.topk(k_eff, dim=1).values.sum(dim=1)
            results[k] = scores.argmax(dim=1).cpu()
        return results


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_splits():
    print(f"Loading ResNet-18 features from {CACHE_PATH}...", flush=True)
    data = np.load(CACHE_PATH)
    X_tr, y_tr = data['X_train'], data['y_train']
    X_te, y_te = data['X_test'],  data['y_test']
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}, D={X_tr.shape[1]}",
          flush=True)
    splits = []
    for t in range(N_TASKS):
        c0, c1 = t * CLASSES_TASK, (t + 1) * CLASSES_TASK
        mask_tr = np.isin(y_tr, range(c0, c1))
        mask_te = np.isin(y_te, range(c0, c1))
        X_tr_t = torch.tensor(X_tr[mask_tr], dtype=torch.float32, device=DEVICE)
        y_tr_t = torch.tensor(y_tr[mask_tr], dtype=torch.long)
        X_te_t = torch.tensor(X_te[mask_te], dtype=torch.float32, device=DEVICE)
        y_te_t = torch.tensor(y_te[mask_te], dtype=torch.long)
        # L2 normalize features
        X_tr_t = F.normalize(X_tr_t, dim=1)
        X_te_t = F.normalize(X_te_t, dim=1)
        splits.append((X_tr_t, y_tr_t, X_te_t, y_te_t))
        print(f"  Task {t}: classes {c0}-{c1-1}, "
              f"train={mask_tr.sum()}, test={mask_te.sum()}", flush=True)
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
    print("Step 100 -- Top-K Class Vote, CIFAR-100", flush=True)
    print(f"N_TASKS={N_TASKS}, CLASSES_TASK={CLASSES_TASK}, DEVICE={DEVICE}",
          flush=True)
    print(f"Config: lr={LR}, spawn_thresh={SPAWN_THRESH}", flush=True)
    print(f"k_vote sweep: {K_VOTE_VALS}", flush=True)
    print(flush=True)

    splits = load_splits()
    print(flush=True)

    model = TopKFold(D_EMBED, lr=LR, spawn_thresh=SPAWN_THRESH)

    acc = {k: [[None] * N_TASKS for _ in range(N_TASKS)]
           for k in K_VOTE_VALS}

    for task_id in range(N_TASKS):
        X_tr_t, y_tr_t, _, _ = splits[task_id]
        print(f"=== Task {task_id} ===", flush=True)
        t_task = time.time()

        cb_before   = model.V.shape[0]
        labels_list = y_tr_t.tolist()
        for i in range(len(X_tr_t)):
            model.step(X_tr_t[i], labels_list[i])
        cb_after = model.V.shape[0]
        print(f"  cb: {cb_before}->{cb_after} (+{cb_after - cb_before})",
              flush=True)
        print(f"  Train: {time.time() - t_task:.1f}s", flush=True)

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            _, _, X_te_t, y_te_t = splits[eval_task]
            preds = model.eval_batch(X_te_t, K_VOTE_VALS)
            for k in K_VOTE_VALS:
                acc[k][eval_task][task_id] = \
                    (preds[k] == y_te_t).float().mean().item()
        print(f"  Eval ({task_id+1} tasks): {time.time()-t_eval:.1f}s",
              flush=True)

        k1  = acc[1][task_id][task_id]
        k3  = acc[3][task_id][task_id]
        k10 = acc[10][task_id][task_id]
        if k1 is not None:
            print(f"  Peek: k=1={k1*100:.1f}% k=3={k3*100:.1f}% "
                  f"k=10={k10*100:.1f}%", flush=True)
        print(flush=True)

    elapsed = time.time() - t0

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("=" * 65, flush=True)
    print("STEP 100 SUMMARY -- Top-K Class Vote CIFAR-100", flush=True)
    print(f"lr={LR}, sp={SPAWN_THRESH}, DEVICE={DEVICE}", flush=True)
    print("=" * 65, flush=True)

    print(f"\n{'k':>4} | {'AA':>7} {'Forgetting':>11} | {'vs k=1':>8}",
          flush=True)
    print("-" * 40, flush=True)

    aa_vals  = {k: compute_aa(acc[k], N_TASKS)  for k in K_VOTE_VALS}
    fgt_vals = {k: compute_fgt(acc[k], N_TASKS) for k in K_VOTE_VALS}
    aa_1nn   = aa_vals[1]

    for k in K_VOTE_VALS:
        delta  = aa_vals[k] - aa_1nn
        marker = " <-- 1-NN baseline" if k == 1 else ""
        print(f"{k:>4} | {aa_vals[k]*100:>6.1f}% {fgt_vals[k]*100:>10.1f}pp | "
              f"{delta*100:>+7.1f}pp{marker}", flush=True)

    print(flush=True)
    print(f"Final codebook: {model.V.shape[0]} vectors", flush=True)
    print(f"Fold baseline (Step 71): 33.5% AA", flush=True)
    print(f"1-NN with cosine spawn:  {aa_1nn*100:.1f}% AA", flush=True)
    print(f"vs fold baseline: {(aa_1nn - 0.335)*100:+.1f}pp", flush=True)

    best_topk_aa = max(aa_vals[k] for k in K_VOTE_VALS if k > 1)
    best_k       = max((k for k in K_VOTE_VALS if k > 1), key=lambda k: aa_vals[k])

    print(flush=True)
    print("KILL CRITERION:", flush=True)
    print(f"  Best top-k (k={best_k}): {best_topk_aa*100:.1f}%", flush=True)
    print(f"  1-NN baseline:           {aa_1nn*100:.1f}%", flush=True)
    if best_topk_aa <= aa_1nn:
        print(f"  VERDICT: KILLED -- top-k ({best_topk_aa*100:.1f}%) "
              f"<= 1-NN ({aa_1nn*100:.1f}%)", flush=True)
    else:
        print(f"  VERDICT: PASSES "
              f"+{(best_topk_aa - aa_1nn)*100:.1f}pp over 1-NN", flush=True)
        if fgt_vals[best_k] > 0.05:
            print(f"  FORGETTING: FAIL ({fgt_vals[best_k]*100:.1f}pp > 5pp)",
                  flush=True)
        else:
            print(f"  FORGETTING: PASS ({fgt_vals[best_k]*100:.1f}pp <= 5pp)",
                  flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
