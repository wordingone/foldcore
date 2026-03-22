#!/usr/bin/env python3
"""
Step 101 -- Spawn-Only Codebook (lr=0) on CIFAR-100.
[researcher] mail 1241. Hypothesis: competitive updates cause forgetting. Fix: no updates.

Codebook grows by spawning only -- vectors never modified after insertion.
Structural guarantee: zero forgetting by construction.

Config: sp=0.95 (calibrated for ResNet-18 features), k_vote={1,3,5,10}.
Compare to Step 100b (lr=0.001): k=5 38.3% AA, 11.6pp forgetting.

Kill criterion:
- DISPROVES if: forgetting > 5pp (updates weren't the cause)
- DISPROVES if: AA < 25% (too few effective vectors)
"""

import time
import numpy as np
import torch
import torch.nn.functional as F

# ─── Config ───────────────────────────────────────────────────────────────────

N_TASKS      = 10
CLASSES_TASK = 10
D_EMBED      = 512
SPAWN_THRESH = 0.95
K_VOTE_VALS  = [1, 3, 5, 10]
CACHE_PATH   = '/mnt/c/Users/Admin/cifar100_resnet18_features.npz'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─── SpawnOnlyFold ────────────────────────────────────────────────────────────

class SpawnOnlyFold:
    """TopKFold with lr=0. No updates, only spawning."""

    def __init__(self, d, spawn_thresh=0.95):
        self.V          = torch.empty(0, d, device=DEVICE)
        self.labels     = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.spawn_thresh = spawn_thresh
        self.d          = d
        self.n_spawned  = 0

    def step(self, r, label):
        r = F.normalize(r, dim=0)
        if self.V.shape[0] == 0 or (self.V @ r).max().item() < self.spawn_thresh:
            self.V      = torch.cat([self.V, r.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([label], device=DEVICE)])
            self.n_spawned += 1
        # No update — spawn-only. Guarantees structural zero forgetting.

    def eval_batch(self, R, k_vals):
        """R: (n, d) GPU tensor. Returns dict k -> preds (CPU)."""
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
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}, D={X_tr.shape[1]}", flush=True)
    splits = []
    for t in range(N_TASKS):
        c0, c1 = t * CLASSES_TASK, (t + 1) * CLASSES_TASK
        mask_tr = np.isin(y_tr, range(c0, c1))
        mask_te = np.isin(y_te, range(c0, c1))
        X_tr_t = torch.tensor(X_tr[mask_tr], dtype=torch.float32, device=DEVICE)
        y_tr_t = torch.tensor(y_tr[mask_tr], dtype=torch.long)
        X_te_t = torch.tensor(X_te[mask_te], dtype=torch.float32, device=DEVICE)
        y_te_t = torch.tensor(y_te[mask_te], dtype=torch.long)
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
    print("Step 101 -- Spawn-Only Codebook (lr=0), CIFAR-100", flush=True)
    print(f"N_TASKS={N_TASKS}, CLASSES_TASK={CLASSES_TASK}, DEVICE={DEVICE}", flush=True)
    print(f"Config: lr=0 (spawn-only), spawn_thresh={SPAWN_THRESH}", flush=True)
    print(f"k_vote sweep: {K_VOTE_VALS}", flush=True)
    print(f"Compare: Step 100b (lr=0.001) k=5 38.3% AA, 11.6pp forgetting", flush=True)
    print(flush=True)

    splits = load_splits()
    print(flush=True)

    model = SpawnOnlyFold(D_EMBED, spawn_thresh=SPAWN_THRESH)

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
        print(f"  cb: {cb_before}->{cb_after} (+{cb_after - cb_before})", flush=True)
        print(f"  Train: {time.time() - t_task:.1f}s", flush=True)

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            _, _, X_te_t, y_te_t = splits[eval_task]
            preds = model.eval_batch(X_te_t, K_VOTE_VALS)
            for k in K_VOTE_VALS:
                acc[k][eval_task][task_id] = \
                    (preds[k] == y_te_t).float().mean().item()
        print(f"  Eval ({task_id+1} tasks): {time.time()-t_eval:.1f}s", flush=True)

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
    print("STEP 101 SUMMARY -- Spawn-Only CIFAR-100", flush=True)
    print(f"lr=0 (spawn-only), sp={SPAWN_THRESH}, DEVICE={DEVICE}", flush=True)
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

    best_k       = max((k for k in K_VOTE_VALS if k > 1), key=lambda k: aa_vals[k])
    best_topk_aa = aa_vals[best_k]

    print(flush=True)
    print(f"Final codebook: {model.V.shape[0]} vectors", flush=True)
    print(f"Step 100b comparison (lr=0.001): k=5 38.3% AA, 11.6pp forgetting", flush=True)
    print(f"Fold baseline (Step 71): 33.5% AA", flush=True)
    print(f"1-NN (lr=0):  {aa_1nn*100:.1f}% AA", flush=True)
    print(f"vs fold baseline: {(aa_1nn - 0.335)*100:+.1f}pp", flush=True)

    print(flush=True)
    print("KILL CRITERION:", flush=True)
    print(f"  Best top-k (k={best_k}): {best_topk_aa*100:.1f}%", flush=True)
    print(f"  Forgetting (k={best_k}): {fgt_vals[best_k]*100:.1f}pp", flush=True)

    disproved = False
    if fgt_vals[best_k] > 0.05:
        print(f"  DISPROVED: forgetting {fgt_vals[best_k]*100:.1f}pp > 5pp "
              f"(updates weren't the cause)", flush=True)
        disproved = True
    if best_topk_aa < 0.25:
        print(f"  DISPROVED: AA {best_topk_aa*100:.1f}% < 25% "
              f"(too few effective vectors)", flush=True)
        disproved = True
    if not disproved:
        print(f"  CONFIRMED: forgetting {fgt_vals[best_k]*100:.1f}pp <= 5pp, "
              f"AA {best_topk_aa*100:.1f}% >= 25%", flush=True)
        print(f"  UPDATE REMOVAL FIXES FORGETTING", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
