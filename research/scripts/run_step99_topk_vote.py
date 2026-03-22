#!/usr/bin/env python3
"""
Step 99 -- Top-K Class Vote on P-MNIST, full 10-task run.
[researcher] mail 1235. Final readout candidate.

TopKFold: per-class sum of top-k cosine similarities.
Why it might beat 1-NN: dense local class coverage wins over single champion.
Satisfies C11+C12: all factors positive, monotonic, input-conditional.

Train ONCE (lr=0.001, sp=0.7 from Step 98 best config).
Evaluate with k_vote in {1, 3, 5, 10} from same codebook.
k_vote=1 == 1-NN (sanity check -- must match).

Kill criterion: best classify_topk (k>1) AA <= classify_1nn AA -> DEAD.
Baseline (fold Step 65): 56.7% AA, 0.0pp forgetting.
"""

import sys
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

# ─── Config ───────────────────────────────────────────────────────────────────

N_TASKS       = int(sys.argv[1]) if len(sys.argv) > 1 else 10
D_OUT         = 384
N_TRAIN_TASK  = 6000
N_TEST_TASK   = 10000
N_CLASSES     = 10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES
SEED          = 42
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

# Best config from Step 98
LR           = 0.001
SPAWN_THRESH = 0.7
K_VOTE_VALS  = [1, 3, 5, 10]


# ─── TopKFold ─────────────────────────────────────────────────────────────────

class TopKFold:
    def __init__(self, d, lr=0.01, spawn_thresh=0.7):
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
        """
        Vectorized eval for multiple k values.
        R: (n, d) unit-normalized GPU tensor.
        Returns dict: k -> predicted labels (CPU long tensor).
        """
        R    = F.normalize(R, dim=1)
        sims = R @ self.V.T                  # (n, N)
        n    = len(R)
        n_cls = int(self.labels.max().item()) + 1

        # For each k, build class score matrix
        results = {}
        for k in k_vals:
            scores = torch.zeros(n, n_cls, device=DEVICE)
            for c in range(n_cls):
                mask = (self.labels == c)
                if mask.sum() == 0:
                    continue
                class_sims = sims[:, mask]              # (n, N_c)
                k_eff      = min(k, class_sims.shape[1])
                topk_sum   = class_sims.topk(k_eff, dim=1).values.sum(dim=1)
                scores[:, c] = topk_sum
            results[k] = scores.argmax(dim=1).cpu()

        return results


# ─── MNIST + embedding ────────────────────────────────────────────────────────

def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST(
        'C:/Users/Admin/mnist_data', train=True,  download=True)
    te = torchvision.datasets.MNIST(
        'C:/Users/Admin/mnist_data', train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_tr, tr.targets.numpy(), X_te, te.targets.numpy()


def make_projection(d_in=784, d_out=D_OUT, seed=12345):
    rng = np.random.RandomState(seed)
    P   = rng.randn(d_out, d_in).astype(np.float32) / math.sqrt(d_in)
    return torch.from_numpy(P).to(DEVICE)


def make_permutation(seed):
    perm = list(range(784))
    random.Random(seed).shuffle(perm)
    return perm


def embed(X_flat_np, perm, P):
    X_t = torch.from_numpy(X_flat_np[:, perm]).to(DEVICE)
    return F.normalize(X_t @ P.T, dim=1)


def stratified_sample(X, y, n_per_class, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(N_CLASSES):
        chosen = rng.choice(np.where(y == c)[0], n_per_class, replace=False)
        idx.extend(chosen.tolist())
    rng.shuffle(idx)
    return X[idx], y[idx]


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
    print(f"Step 99 -- Top-K Class Vote, P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, D_OUT={D_OUT}, DEVICE={DEVICE}", flush=True)
    print(f"Config: lr={LR}, spawn_thresh={SPAWN_THRESH} (cosine)", flush=True)
    print(f"k_vote sweep: {K_VOTE_VALS}  (k=1 should == 1-NN)", flush=True)
    print(flush=True)

    print("Loading MNIST...", flush=True)
    X_tr, y_tr, X_te, y_te = load_mnist()
    P      = make_projection()
    y_te_t = torch.from_numpy(y_te).long()

    perms = [list(range(784))]
    for t in range(1, N_TASKS):
        perms.append(make_permutation(seed=t * 1000))

    print("Pre-embedding test sets...", flush=True)
    test_embeds = [embed(X_te, perms[t], P) for t in range(N_TASKS)]
    print(f"  Done: {N_TASKS} tasks x {N_TEST_TASK} samples", flush=True)
    print(flush=True)

    model = TopKFold(D_OUT, lr=LR, spawn_thresh=SPAWN_THRESH)

    # acc[k][eval_task][train_task]
    acc = {k: [[None] * N_TASKS for _ in range(N_TASKS)]
           for k in K_VOTE_VALS}

    for task_id in range(N_TASKS):
        print(f"=== Task {task_id} ===", flush=True)
        t_task = time.time()

        X_sub, y_sub = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS,
                                         seed=task_id * 1337)
        X_emb       = embed(X_sub, perms[task_id], P)
        labels_list = y_sub.tolist()

        cb_before = model.V.shape[0]
        for i in range(len(X_emb)):
            model.step(X_emb[i], labels_list[i])
        cb_after = model.V.shape[0]
        print(f"  cb: {cb_before}->{cb_after} (+{cb_after - cb_before})", flush=True)
        print(f"  Train: {time.time() - t_task:.1f}s", flush=True)

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            preds = model.eval_batch(test_embeds[eval_task], K_VOTE_VALS)
            for k in K_VOTE_VALS:
                acc[k][eval_task][task_id] = \
                    (preds[k] == y_te_t).float().mean().item()

        print(f"  Eval ({task_id+1} tasks): {time.time()-t_eval:.1f}s", flush=True)
        # Peek
        k1  = acc[1][task_id][task_id]
        k3  = acc[3][task_id][task_id]
        k10 = acc[10][task_id][task_id]
        if k1 is not None:
            print(f"  Peek: k=1={k1*100:.1f}% k=3={k3*100:.1f}% "
                  f"k=10={k10*100:.1f}%", flush=True)
        print(flush=True)

    elapsed = time.time() - t0

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("=" * 70, flush=True)
    print("STEP 99 SUMMARY -- Top-K Class Vote P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, lr={LR}, sp={SPAWN_THRESH}, DEVICE={DEVICE}",
          flush=True)
    print("=" * 70, flush=True)

    print(f"\n{'k':>4} | {'AA':>7} {'Forgetting':>11} | {'vs k=1':>8}",
          flush=True)
    print("-" * 40, flush=True)

    aa_vals  = {}
    fgt_vals = {}
    for k in K_VOTE_VALS:
        aa_vals[k]  = compute_aa(acc[k], N_TASKS)
        fgt_vals[k] = compute_fgt(acc[k], N_TASKS)

    aa_1nn = aa_vals[1]
    for k in K_VOTE_VALS:
        delta = aa_vals[k] - aa_1nn
        marker = " <-- 1-NN baseline" if k == 1 else ""
        print(f"{k:>4} | {aa_vals[k]*100:>6.1f}% {fgt_vals[k]*100:>10.1f}pp | "
              f"{delta*100:>+7.1f}pp{marker}", flush=True)

    print(flush=True)
    print(f"Final codebook size: {model.V.shape[0]}", flush=True)
    print(f"Total spawned: {model.n_spawned}", flush=True)
    print(flush=True)
    print(f"Baseline (fold Step 65): 56.7% AA, 0.0pp forgetting", flush=True)
    print(f"1-NN with cosine spawn:  {aa_1nn*100:.1f}% AA "
          f"(calibration vs fold baseline)", flush=True)
    print(f"vs fold baseline (56.7%): {(aa_1nn - 0.567)*100:+.1f}pp", flush=True)

    # Kill criterion
    best_topk_aa = max(aa_vals[k] for k in K_VOTE_VALS if k > 1)
    best_k       = max((k for k in K_VOTE_VALS if k > 1), key=lambda k: aa_vals[k])
    print(flush=True)
    print("KILL CRITERION:", flush=True)
    print(f"  Best top-k (k={best_k}): {best_topk_aa*100:.1f}%", flush=True)
    print(f"  1-NN baseline:           {aa_1nn*100:.1f}%", flush=True)
    if best_topk_aa <= aa_1nn:
        print(f"  VERDICT: KILLED -- top-k ({best_topk_aa*100:.1f}%) "
              f"<= 1-NN ({aa_1nn*100:.1f}%)", flush=True)
        print(f"  READOUT LINE: DEAD (3 candidates, all fail kill criterion)",
              flush=True)
    else:
        print(f"  VERDICT: PASSES "
              f"+{(best_topk_aa - aa_1nn)*100:.1f}pp over 1-NN", flush=True)
        if fgt_vals[best_k] > 0.05:
            print(f"  FORGETTING: FAIL ({fgt_vals[best_k]*100:.1f}pp > 5pp)",
                  flush=True)
        else:
            print(f"  FORGETTING: PASS ({fgt_vals[best_k]*100:.1f}pp <= 5pp)",
                  flush=True)

    # Sanity check: k=1 vs k=3 for task 0
    print(flush=True)
    print("Sanity check (k=1 == 1-NN):", flush=True)
    match = abs(aa_vals[1] - aa_1nn) < 1e-6
    print(f"  k=1 AA={aa_vals[1]*100:.3f}% == 1-NN: {'YES' if match else 'NO'}",
          flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
