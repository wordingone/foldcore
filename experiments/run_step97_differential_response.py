#!/usr/bin/env python3
"""
Step 97 -- Differential Response on P-MNIST.
Leo mail 1231. tau x lr sweep, classify_diff vs classify_1nn.

DifferentialResponse: codebook update IS the classification signal.
delta_i serves as both update and readout -- collapses read/write separation.

classify_diff: argmin/argmax of per-class response magnitude.
classify_1nn:  1-NN baseline over same codebook (kill bar).

Kill criterion: classify_diff AA <= classify_1nn AA -> KILLED.
Also fails if forgetting > 5pp.

Baseline (fold Step 65): 56.7% AA, 0.0pp forgetting.

Usage:
  python run_step97_differential_response.py [N_TASKS]
  N_TASKS default: 10. Use 2 for Tier 1 sanity check.
"""

import sys
import os
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
SPAWN_THRESH  = -2.0
SEED          = 42
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

TAU_VALS = [0.01, 0.05, 0.1, 0.5, 1.0]
LR_VALS  = [0.001, 0.01, 0.1]
CONFIGS  = [(tau, lr) for tau in TAU_VALS for lr in LR_VALS]


# ─── DifferentialResponse ─────────────────────────────────────────────────────

class DifferentialResponse:
    def __init__(self, d, tau=0.1, lr=0.01, spawn_thresh=SPAWN_THRESH):
        self.V      = torch.empty(0, d, device=DEVICE)
        self.labels = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.tau    = tau
        self.lr     = lr
        self.spawn_thresh = spawn_thresh
        self.d      = d
        self.n_spawned = 0

    def energy(self, r):
        if self.V.shape[0] == 0:
            return float('inf')
        return -torch.logsumexp(self.V @ r / self.tau, dim=0).item()

    def step(self, r, label):
        """Learn: compute differential, apply as update."""
        r = F.normalize(r, dim=0)
        if self.V.shape[0] == 0 or self.energy(r) > self.spawn_thresh:
            self.V      = torch.cat([self.V, r.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([label], device=DEVICE)])
            self.n_spawned += 1
            return
        sims    = self.V @ r
        weights = F.softmax(sims / self.tau, dim=0)
        deltas  = weights.unsqueeze(1) * (r.unsqueeze(0) - self.V)
        self.V  = F.normalize(self.V + self.lr * deltas, dim=1)

    def classify_batch(self, R):
        """
        Batch classification. R: (n, d) unit-normalized GPU tensor.
        Returns (pred_argmin, pred_argmax, pred_1nn) as CPU long tensors.

        Efficient delta magnitude: ||w*(r-V)||^2 = w^2 * (2 - 2*sim)
        since both r and V are unit-normalized.
        """
        R    = F.normalize(R, dim=1)           # (n, d)
        sims = R @ self.V.T                    # (n, N)
        weights = F.softmax(sims / self.tau, dim=1)  # (n, N)

        # ||r - V_j||^2 = 2 - 2*sims[i,j]  (unit vectors)
        dist_sq  = (2.0 - 2.0 * sims).clamp(min=0.0)  # (n, N)
        mags     = weights * dist_sq.sqrt()             # (n, N)

        n_cls    = int(self.labels.max().item()) + 1
        per_cls  = torch.zeros(len(R), n_cls, device=DEVICE)
        per_cls.scatter_add_(1, self.labels.unsqueeze(0).expand(len(R), -1), mags)

        pred_argmin = per_cls.argmin(dim=1).cpu()
        pred_argmax = per_cls.argmax(dim=1).cpu()
        pred_1nn    = self.labels[sims.argmax(dim=1)].cpu()

        return pred_argmin, pred_argmax, pred_1nn


# ─── MNIST + embedding ────────────────────────────────────────────────────────

def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST(
        os.environ.get('MNIST_DATA', './data'), train=True,  download=True)
    te = torchvision.datasets.MNIST(
        os.environ.get('MNIST_DATA', './data'), train=False, download=True)
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
    """X_flat_np: (n, 784) numpy. Returns (n, D_OUT) normalized GPU tensor."""
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


# ─── Main benchmark ───────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Step 97 -- Differential Response, P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, D_OUT={D_OUT}, DEVICE={DEVICE}", flush=True)
    print(f"Configs: {len(CONFIGS)} (tau x lr), spawn_thresh={SPAWN_THRESH}",
          flush=True)
    print(flush=True)

    print("Loading MNIST...", flush=True)
    X_tr, y_tr, X_te, y_te = load_mnist()

    P = make_projection()
    y_te_t = torch.from_numpy(y_te).long()

    perms = [list(range(784))]
    for t in range(1, N_TASKS):
        perms.append(make_permutation(seed=t * 1000))

    print("Pre-embedding test sets...", flush=True)
    test_embeds = [embed(X_te, perms[t], P) for t in range(N_TASKS)]
    print(f"  Done: {N_TASKS} tasks x {N_TEST_TASK} samples", flush=True)
    print(flush=True)

    # One model per config
    models = {cfg: DifferentialResponse(D_OUT, tau=cfg[0], lr=cfg[1])
              for cfg in CONFIGS}

    # results[cfg][readout][eval_task][train_task] = accuracy
    results = {
        cfg: {
            'argmin': [[None] * N_TASKS for _ in range(N_TASKS)],
            'argmax': [[None] * N_TASKS for _ in range(N_TASKS)],
            '1nn':    [[None] * N_TASKS for _ in range(N_TASKS)],
        }
        for cfg in CONFIGS
    }

    for task_id in range(N_TASKS):
        print(f"=== Task {task_id} ===", flush=True)
        t_task = time.time()

        X_sub, y_sub = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS,
                                         seed=task_id * 1337)
        X_emb = embed(X_sub, perms[task_id], P)   # (6000, D_OUT)
        labels_list = y_sub.tolist()

        # Train each config
        for cfg in CONFIGS:
            m  = models[cfg]
            cb_before = m.V.shape[0]
            for i in range(len(X_emb)):
                m.step(X_emb[i], labels_list[i])
            cb_after = m.V.shape[0]
            print(f"  tau={cfg[0]:.3f} lr={cfg[1]:.3f}: "
                  f"cb {cb_before}->{cb_after} (+{cb_after - cb_before})",
                  flush=True)

        print(f"  Train time: {time.time() - t_task:.1f}s", flush=True)

        # Evaluate all configs on all seen tasks
        t_eval = time.time()
        for cfg in CONFIGS:
            m = models[cfg]
            if m.V.shape[0] == 0:
                continue
            for eval_task in range(task_id + 1):
                emb = test_embeds[eval_task]
                p_min, p_max, p_1nn = m.classify_batch(emb)
                results[cfg]['argmin'][eval_task][task_id] = \
                    (p_min == y_te_t).float().mean().item()
                results[cfg]['argmax'][eval_task][task_id] = \
                    (p_max == y_te_t).float().mean().item()
                results[cfg]['1nn'][eval_task][task_id]    = \
                    (p_1nn == y_te_t).float().mean().item()

        print(f"  Eval time: {time.time() - t_eval:.1f}s", flush=True)
        # Quick peek at one config after each task
        cfg0 = (0.1, 0.01)
        r0   = results[cfg0]
        aa_peek = r0['argmin'][task_id][task_id]
        nn_peek = r0['1nn'][task_id][task_id]
        if aa_peek is not None:
            print(f"  Peek (tau=0.1 lr=0.01): "
                  f"task{task_id} argmin={aa_peek*100:.1f}% "
                  f"1nn={nn_peek*100:.1f}%", flush=True)
        print(flush=True)

    elapsed = time.time() - t0

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("=" * 78, flush=True)
    print("STEP 97 SUMMARY -- Differential Response P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, DEVICE={DEVICE}", flush=True)
    print("=" * 78, flush=True)

    print(f"\n{'tau':>6} {'lr':>6} | "
          f"{'AA_min':>7} {'AA_max':>7} {'AA_1nn':>7} | "
          f"{'Fgt_min':>8} {'Fgt_max':>8} | "
          f"{'CB':>6} {'diff_wins':>10}", flush=True)
    print("-" * 78, flush=True)

    best_cfg    = None
    best_aa_diff = 0.0
    best_readout = 'argmin'

    rows = []
    for tau, lr in CONFIGS:
        cfg = (tau, lr)
        m   = models[cfg]
        r   = results[cfg]

        aa_min = compute_aa(r['argmin'], N_TASKS)
        aa_max = compute_aa(r['argmax'], N_TASKS)
        aa_1nn = compute_aa(r['1nn'],    N_TASKS)
        fgt_min = compute_fgt(r['argmin'], N_TASKS)
        fgt_max = compute_fgt(r['argmax'], N_TASKS)
        cb_size = m.V.shape[0]

        aa_diff  = max(aa_min, aa_max)
        readout  = 'argmin' if aa_min >= aa_max else 'argmax'
        diff_wins = aa_diff > aa_1nn

        if aa_diff > best_aa_diff:
            best_aa_diff = aa_diff
            best_cfg     = cfg
            best_readout = readout

        rows.append((tau, lr, aa_min, aa_max, aa_1nn, fgt_min, fgt_max,
                     cb_size, diff_wins, readout))

        print(f"{tau:>6.3f} {lr:>6.3f} | "
              f"{aa_min*100:>6.1f}% {aa_max*100:>6.1f}% {aa_1nn*100:>6.1f}% | "
              f"{fgt_min*100:>7.1f}pp {fgt_max*100:>7.1f}pp | "
              f"{cb_size:>6} {'YES' if diff_wins else 'no':>10}",
              flush=True)

    print(flush=True)
    print(f"Baseline (fold Step 65): 56.7% AA, 0.0pp forgetting", flush=True)

    # Kill criterion
    print(flush=True)
    print("KILL CRITERION:", flush=True)
    if best_cfg:
        r        = results[best_cfg]
        aa_diff  = best_aa_diff
        aa_1nn   = compute_aa(r['1nn'], N_TASKS)
        fgt_best = compute_fgt(r[best_readout], N_TASKS)

        print(f"  Best diff ({best_readout}) AA: {aa_diff*100:.1f}%  "
              f"(tau={best_cfg[0]}, lr={best_cfg[1]})", flush=True)
        print(f"  1-NN AA (same codebook):     {aa_1nn*100:.1f}%", flush=True)

        if aa_diff <= aa_1nn:
            print(f"  VERDICT: KILLED -- diff ({aa_diff*100:.1f}%) <= 1-NN ({aa_1nn*100:.1f}%)",
                  flush=True)
        else:
            print(f"  VERDICT: PASSES +{(aa_diff - aa_1nn)*100:.1f}pp over 1-NN",
                  flush=True)

        if fgt_best > 0.05:
            print(f"  FORGETTING: FAIL ({fgt_best*100:.1f}pp > 5pp)", flush=True)
        else:
            print(f"  FORGETTING: PASS ({fgt_best*100:.1f}pp <= 5pp)", flush=True)

        vs_baseline = aa_diff - 0.567
        print(f"  vs fold baseline (56.7%): {vs_baseline*100:+.1f}pp", flush=True)

    # Count configs where diff beats 1nn
    n_wins = sum(1 for row in rows if row[8])
    print(f"\n  Configs where diff beats 1-NN: {n_wins}/{len(CONFIGS)}", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
