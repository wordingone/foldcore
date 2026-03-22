#!/usr/bin/env python3
"""
Step 98 -- Neighborhood Coherence Readout on P-MNIST.
[researcher] mail 1233.

CoherenceFold: coherence-weighted cosine readout.
Fixes Step 97 anti-correlation (C11) and codebook starvation.

Spawn by cosine threshold (not energy) -- avoids starvation.
classify_coherence: cos(v,r) * coherence(v) -- all positive factors.
classify_1nn: raw 1-NN baseline (kill bar).

Kill criterion: best classify_coherence AA <= best classify_1nn AA -> KILLED.
Baseline (fold Step 65): 56.7% AA, 0.0pp forgetting.

Usage:
  python run_step98_coherence_readout.py [N_TASKS]
  N_TASKS default: 10. Use 2 for Tier 1.
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

K_COHERENCE_VALS  = [3, 5, 10]
LR_VALS           = [0.001, 0.01, 0.1]
SPAWN_THRESH_VALS = [0.3, 0.5, 0.7]
CONFIGS = [(k, lr, sp)
           for k  in K_COHERENCE_VALS
           for lr in LR_VALS
           for sp in SPAWN_THRESH_VALS]


# ─── CoherenceFold ────────────────────────────────────────────────────────────

class CoherenceFold:
    def __init__(self, d, k_coherence=5, lr=0.01, spawn_thresh=0.5):
        self.V          = torch.empty(0, d, device=DEVICE)
        self.labels     = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.coherence  = torch.empty(0, device=DEVICE)
        self.k          = k_coherence
        self.lr         = lr
        self.spawn_thresh = spawn_thresh
        self.d          = d
        self.n_spawned  = 0

    def step(self, r, label):
        """One operation: classify + update + coherence refresh."""
        r = F.normalize(r, dim=0)
        if self.V.shape[0] == 0:
            self._spawn(r, label)
            return
        sims    = self.V @ r
        max_sim = sims.max().item()
        if max_sim < self.spawn_thresh:
            self._spawn(r, label)
            return
        winner = sims.argmax().item()
        self.V[winner] = F.normalize(
            self.V[winner] + self.lr * (r - self.V[winner]), dim=0)
        self._refresh_coherence(winner)

    def _spawn(self, r, label):
        self.V         = torch.cat([self.V, r.unsqueeze(0)])
        self.labels    = torch.cat([self.labels,
                                    torch.tensor([label], device=DEVICE)])
        self.coherence = torch.cat([self.coherence,
                                    torch.tensor([0.5], device=DEVICE)])
        self.n_spawned += 1

    def _refresh_coherence(self, idx):
        """Coherence = mean cosine to k-nearest same-class neighbors."""
        mask       = (self.labels == self.labels[idx])
        mask[idx]  = False
        if mask.sum() == 0:
            self.coherence[idx] = 0.5
            return
        class_sims = self.V[mask] @ self.V[idx]
        k          = min(self.k, class_sims.shape[0])
        topk_sims  = class_sims.topk(k).values
        self.coherence[idx] = topk_sims.mean()

    def classify_coherence_batch(self, R):
        """Coherence-weighted: cos(v,r) * coherence(v). R: (n, d) GPU tensor."""
        R        = F.normalize(R, dim=1)
        sims     = R @ self.V.T                        # (n, N)
        weighted = sims * self.coherence.unsqueeze(0)  # (n, N)
        return self.labels[weighted.argmax(dim=1)].cpu()

    def classify_1nn_batch(self, R):
        """Raw 1-NN baseline."""
        R    = F.normalize(R, dim=1)
        sims = R @ self.V.T
        return self.labels[sims.argmax(dim=1)].cpu()


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


# ─── Main benchmark ───────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Step 98 -- Neighborhood Coherence Readout, P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, D_OUT={D_OUT}, DEVICE={DEVICE}", flush=True)
    print(f"Configs: {len(CONFIGS)} (k_coherence x lr x spawn_thresh)", flush=True)
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

    models = {cfg: CoherenceFold(D_OUT, k_coherence=cfg[0], lr=cfg[1],
                                  spawn_thresh=cfg[2])
              for cfg in CONFIGS}

    results = {
        cfg: {
            'coh': [[None] * N_TASKS for _ in range(N_TASKS)],
            '1nn': [[None] * N_TASKS for _ in range(N_TASKS)],
        }
        for cfg in CONFIGS
    }

    for task_id in range(N_TASKS):
        print(f"=== Task {task_id} ===", flush=True)
        t_task = time.time()

        X_sub, y_sub = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS,
                                         seed=task_id * 1337)
        X_emb       = embed(X_sub, perms[task_id], P)
        labels_list = y_sub.tolist()

        for cfg in CONFIGS:
            m         = models[cfg]
            cb_before = m.V.shape[0]
            for i in range(len(X_emb)):
                m.step(X_emb[i], labels_list[i])
            cb_after = m.V.shape[0]
            print(f"  k={cfg[0]:2d} lr={cfg[1]:.3f} sp={cfg[2]:.1f}: "
                  f"cb {cb_before}->{cb_after} (+{cb_after - cb_before})",
                  flush=True)

        print(f"  Train time: {time.time() - t_task:.1f}s", flush=True)

        t_eval = time.time()
        for cfg in CONFIGS:
            m = models[cfg]
            if m.V.shape[0] == 0:
                continue
            for eval_task in range(task_id + 1):
                emb   = test_embeds[eval_task]
                p_coh = m.classify_coherence_batch(emb)
                p_1nn = m.classify_1nn_batch(emb)
                results[cfg]['coh'][eval_task][task_id] = \
                    (p_coh == y_te_t).float().mean().item()
                results[cfg]['1nn'][eval_task][task_id] = \
                    (p_1nn == y_te_t).float().mean().item()

        print(f"  Eval time: {time.time() - t_eval:.1f}s", flush=True)

        # Peek at sp=0.5, lr=0.01, k=5
        peek_cfg = (5, 0.01, 0.5)
        r_peek   = results[peek_cfg]
        coh_peek = r_peek['coh'][task_id][task_id]
        nn_peek  = r_peek['1nn'][task_id][task_id]
        if coh_peek is not None:
            print(f"  Peek (k=5 lr=0.01 sp=0.5): "
                  f"task{task_id} coh={coh_peek*100:.1f}% "
                  f"1nn={nn_peek*100:.1f}% "
                  f"cb={models[peek_cfg].V.shape[0]}",
                  flush=True)
        print(flush=True)

    elapsed = time.time() - t0

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("=" * 80, flush=True)
    print("STEP 98 SUMMARY -- Neighborhood Coherence Readout P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, DEVICE={DEVICE}", flush=True)
    print("=" * 80, flush=True)

    print(f"\n{'k':>3} {'lr':>6} {'sp':>4} | "
          f"{'AA_coh':>7} {'AA_1nn':>7} | "
          f"{'Fgt_coh':>8} {'Fgt_1nn':>8} | "
          f"{'CB':>6} {'coh_wins':>9}", flush=True)
    print("-" * 80, flush=True)

    rows = []
    for k, lr, sp in CONFIGS:
        cfg    = (k, lr, sp)
        m      = models[cfg]
        r      = results[cfg]
        aa_coh = compute_aa(r['coh'], N_TASKS)
        aa_1nn = compute_aa(r['1nn'], N_TASKS)
        fg_coh = compute_fgt(r['coh'], N_TASKS)
        fg_1nn = compute_fgt(r['1nn'], N_TASKS)
        cb     = m.V.shape[0]
        wins   = aa_coh > aa_1nn
        rows.append((k, lr, sp, aa_coh, aa_1nn, fg_coh, fg_1nn, cb, wins))
        print(f"{k:>3} {lr:>6.3f} {sp:>4.1f} | "
              f"{aa_coh*100:>6.1f}% {aa_1nn*100:>6.1f}% | "
              f"{fg_coh*100:>7.1f}pp {fg_1nn*100:>7.1f}pp | "
              f"{cb:>6} {'YES' if wins else 'no':>9}",
              flush=True)

    # Best configs
    best_coh_row = max(rows, key=lambda r: r[3])
    best_1nn_row = max(rows, key=lambda r: r[4])

    print(flush=True)
    print(f"Baseline (fold Step 65): 56.7% AA, 0.0pp forgetting", flush=True)
    print(f"Best coherence: k={best_coh_row[0]} lr={best_coh_row[1]} "
          f"sp={best_coh_row[2]} -> {best_coh_row[3]*100:.1f}% AA",
          flush=True)
    print(f"Best 1-NN:      k={best_1nn_row[0]} lr={best_1nn_row[1]} "
          f"sp={best_1nn_row[2]} -> {best_1nn_row[4]*100:.1f}% AA",
          flush=True)

    # Kill criterion
    print(flush=True)
    print("KILL CRITERION:", flush=True)
    best_aa_coh = best_coh_row[3]
    best_aa_1nn = best_1nn_row[4]
    # Compare best coh vs best 1nn (across all configs)
    if best_aa_coh <= best_1nn_row[4]:
        print(f"  VERDICT: KILLED -- coherence ({best_aa_coh*100:.1f}%) "
              f"<= 1-NN ({best_aa_1nn*100:.1f}%)", flush=True)
    else:
        print(f"  VERDICT: PASSES "
              f"+{(best_aa_coh - best_aa_1nn)*100:.1f}pp over best 1-NN",
              flush=True)

    # Forgetting at best coherence config
    fg_at_best = best_coh_row[5]
    if fg_at_best > 0.05:
        print(f"  FORGETTING: FAIL ({fg_at_best*100:.1f}pp > 5pp)", flush=True)
    else:
        print(f"  FORGETTING: PASS ({fg_at_best*100:.1f}pp <= 5pp)", flush=True)

    vs_baseline = best_aa_coh - 0.567
    print(f"  vs fold baseline (56.7%): {vs_baseline*100:+.1f}pp", flush=True)

    n_wins = sum(1 for row in rows if row[8])
    print(f"  Configs where coh beats 1-NN: {n_wins}/{len(CONFIGS)}", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
