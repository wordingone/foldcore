#!/usr/bin/env python3
"""
Step 103 -- Resonance Equation. Non-codebook substrate test.
Leo mail 1247.

One state variable z evolves under:
  dz = alpha * autonomous(z) + beta * energy_grad(z)
  z = normalize(z + dt * dz)

autonomous(z) = normalize(z) - z  (normalization attractor)
energy_grad(z) = softmax(patterns @ z / 0.1) @ patterns - z  (Hopfield pull)

Training and inference use SAME dynamics. Partial Separation 1 collapse.

Sweep: n_steps={1,5,10,20} x alpha={0.0,0.5,1.0} x beta={0.5,1.0} = 24 configs.
n_steps=1, alpha=0.0 ~= 1-NN (sanity check).
P-MNIST, sp=0.7.

Kill: resonance <= 1-NN on same patterns for ALL configs.
Bar to beat: top-k 91.8%, 1-NN 86.8%.
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
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

SPAWN_THRESH  = 0.7
DT            = 0.1
TEMP          = 0.1   # softmax temperature for energy_grad

N_STEPS_VALS  = [1, 5, 10, 20]
ALPHA_VALS    = [0.0, 0.5, 1.0]
BETA_VALS     = [0.5, 1.0]


# ─── ResonanceSubstrate ───────────────────────────────────────────────────────

class ResonanceSubstrate:
    def __init__(self, d, n_steps=10, alpha=1.0, beta=1.0, spawn_thresh=0.7):
        self.patterns    = torch.empty(0, d, device=DEVICE)
        self.labels      = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.d           = d
        self.n_steps     = n_steps
        self.alpha       = alpha
        self.beta        = beta
        self.spawn_thresh = spawn_thresh
        self.n_spawned   = 0

    def _energy_grad(self, z):
        """z: (d,). Returns gradient pulling z toward softmax-weighted pattern centroid."""
        sims    = self.patterns @ z                      # (N,)
        weights = F.softmax(sims / TEMP, dim=0)          # (N,)
        target  = (weights.unsqueeze(1) * self.patterns).sum(0)  # (d,)
        return target - z

    def _autonomous(self, z):
        """z: (d,). Drives z toward unit sphere."""
        return F.normalize(z, dim=0) - z

    def _run_dynamics(self, z):
        """Evolve z for n_steps. z: (d,)."""
        for _ in range(self.n_steps):
            dz = (self.alpha * self._autonomous(z)
                  + self.beta * self._energy_grad(z))
            z  = F.normalize(z + DT * dz, dim=0)
        return z

    def step(self, r, label):
        r = F.normalize(r, dim=0)
        if self.patterns.shape[0] == 0 or \
                (self.patterns @ r).max().item() < self.spawn_thresh:
            self.patterns = torch.cat([self.patterns, r.unsqueeze(0)])
            self.labels   = torch.cat([self.labels,
                                       torch.tensor([label], device=DEVICE)])
            self.n_spawned += 1
            return
        z       = self._run_dynamics(r.clone())
        nearest = (self.patterns @ z).argmax().item()
        if self.labels[nearest].item() == label:
            self.patterns[nearest] = F.normalize(
                self.patterns[nearest] + 0.01 * (z - self.patterns[nearest]), dim=0)

    def _energy_grad_batch(self, Z):
        """Z: (n, d). Batched energy gradient."""
        sims    = Z @ self.patterns.T                    # (n, N)
        weights = F.softmax(sims / TEMP, dim=1)          # (n, N)
        target  = weights @ self.patterns                 # (n, d)
        return target - Z

    def _autonomous_batch(self, Z):
        return F.normalize(Z, dim=1) - Z

    def eval_batch(self, R):
        """
        Batched eval: resonance and 1-NN on same patterns.
        R: (n, d) GPU tensor.
        Returns: res_preds (CPU), nn_preds (CPU)
        """
        R  = F.normalize(R, dim=1)
        N  = self.patterns.shape[0]
        if N == 0:
            zeros = torch.zeros(len(R), dtype=torch.long)
            return zeros, zeros

        # 1-NN: no dynamics
        nn_preds = self.labels[( R @ self.patterns.T).argmax(dim=1)].cpu()

        # Resonance: run dynamics on batch
        Z = R.clone()
        for _ in range(self.n_steps):
            dZ = (self.alpha * self._autonomous_batch(Z)
                  + self.beta * self._energy_grad_batch(Z))
            Z  = F.normalize(Z + DT * dZ, dim=1)
        res_preds = self.labels[(Z @ self.patterns.T).argmax(dim=1)].cpu()

        return res_preds, nn_preds


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

def compute_aa(mat):
    vals = [mat[t][N_TASKS - 1] for t in range(N_TASKS)
            if mat[t][N_TASKS - 1] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def compute_fgt(mat):
    vals = []
    for t in range(N_TASKS - 1):
        if mat[t][t] is not None and mat[t][N_TASKS - 1] is not None:
            vals.append(max(0.0, mat[t][t] - mat[t][N_TASKS - 1]))
    return sum(vals) / len(vals) if vals else 0.0


# ─── Run one config ───────────────────────────────────────────────────────────

def run_config(cfg_id, n_steps, alpha, beta,
               X_tr, y_tr, test_embeds, perms, P, y_te_t):
    label = f"cfg{cfg_id:02d} n_steps={n_steps} alpha={alpha} beta={beta}"
    print(f"\n{'='*65}", flush=True)
    print(f"CONFIG {cfg_id:02d}: {label}", flush=True)
    print(f"{'='*65}", flush=True)

    model = ResonanceSubstrate(D_OUT, n_steps=n_steps, alpha=alpha,
                                beta=beta, spawn_thresh=SPAWN_THRESH)

    acc_res = [[None] * N_TASKS for _ in range(N_TASKS)]
    acc_nn  = [[None] * N_TASKS for _ in range(N_TASKS)]

    for task_id in range(N_TASKS):
        t_task = time.time()
        X_sub, y_sub = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS,
                                         seed=task_id * 1337)
        X_emb       = embed(X_sub, perms[task_id], P)
        labels_list = y_sub.tolist()

        cb_before = model.patterns.shape[0]
        for i in range(len(X_emb)):
            model.step(X_emb[i], labels_list[i])
        cb_after = model.patterns.shape[0]

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            rp, np_ = model.eval_batch(test_embeds[eval_task])
            acc_res[eval_task][task_id] = (rp  == y_te_t).float().mean().item()
            acc_nn [eval_task][task_id] = (np_ == y_te_t).float().mean().item()

        tr_t = t_eval - t_task
        ev_t = time.time() - t_eval
        r0 = acc_res[task_id][task_id]
        n0 = acc_nn [task_id][task_id]
        print(f"  T{task_id}: cb {cb_before}->{cb_after} "
              f"train={tr_t:.1f}s eval={ev_t:.1f}s "
              f"res={r0*100:.1f}% nn={n0*100:.1f}%", flush=True)

    aa_r  = compute_aa(acc_res)
    aa_n  = compute_aa(acc_nn)
    fgt_r = compute_fgt(acc_res)
    fgt_n = compute_fgt(acc_nn)
    delta = aa_r - aa_n

    verdict = "DISPROVED" if aa_r <= aa_n else "PASSES"
    print(f"  RESULT: res={aa_r*100:.1f}% fgt={fgt_r*100:.1f}pp | "
          f"nn={aa_n*100:.1f}% fgt={fgt_n*100:.1f}pp | "
          f"delta={delta*100:+.1f}pp | {verdict}", flush=True)

    return {
        'cfg_id': cfg_id, 'n_steps': n_steps, 'alpha': alpha, 'beta': beta,
        'aa_res': aa_r, 'fgt_res': fgt_r,
        'aa_nn':  aa_n, 'fgt_nn':  fgt_n,
        'delta': delta, 'verdict': verdict,
        'cb_size': model.patterns.shape[0],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 103 -- Resonance Equation, P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, D_OUT={D_OUT}, DEVICE={DEVICE}", flush=True)
    print(f"sp={SPAWN_THRESH}, dt={DT}, temp={TEMP}", flush=True)
    print(f"Sweep: n_steps={N_STEPS_VALS} x alpha={ALPHA_VALS} x beta={BETA_VALS}", flush=True)
    print(f"Total configs: {len(N_STEPS_VALS)*len(ALPHA_VALS)*len(BETA_VALS)}", flush=True)
    print(f"Kill: resonance <= 1-NN for ALL configs", flush=True)
    print(f"Bar to beat: top-k 91.8%, 1-NN 86.8%", flush=True)
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

    results = []
    cfg_id  = 0
    for n_steps in N_STEPS_VALS:
        for alpha in ALPHA_VALS:
            for beta in BETA_VALS:
                r = run_config(cfg_id, n_steps, alpha, beta,
                               X_tr, y_tr, test_embeds, perms, P, y_te_t)
                results.append(r)
                cfg_id += 1

    elapsed = time.time() - t0

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print("STEP 103 FINAL SUMMARY -- Resonance Equation", flush=True)
    print(f"Baselines: top-k(5) 91.8%, 1-NN 86.8%, fgt 0.0pp", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"\n{'cfg':>4} {'n_st':>5} {'alp':>5} {'bet':>5} | "
          f"{'res AA':>7} {'fgt':>6} | {'nn AA':>7} {'fgt':>6} | "
          f"{'delta':>7} | {'verdict':>10}", flush=True)
    print("-" * 80, flush=True)

    passes = [r for r in results if r['verdict'] == 'PASSES']
    for r in results:
        print(f"{r['cfg_id']:>4} {r['n_steps']:>5} {r['alpha']:>5.1f} {r['beta']:>5.1f} | "
              f"{r['aa_res']*100:>6.1f}% {r['fgt_res']*100:>5.1f}pp | "
              f"{r['aa_nn']*100:>6.1f}% {r['fgt_nn']*100:>5.1f}pp | "
              f"{r['delta']*100:>+6.1f}pp | "
              f"{r['verdict']:>10}", flush=True)

    print(flush=True)
    if passes:
        best = max(passes, key=lambda r: r['aa_res'])
        print(f"VERDICT: PASSES -- {len(passes)}/{len(results)} configs beat 1-NN", flush=True)
        print(f"Best: n_steps={best['n_steps']} alpha={best['alpha']} beta={best['beta']}: "
              f"res {best['aa_res']*100:.1f}% vs nn {best['aa_nn']*100:.1f}% "
              f"({best['delta']*100:+.1f}pp)", flush=True)
        vs_topk = best['aa_res'] - 0.918
        print(f"vs top-k bar (91.8%): {vs_topk*100:+.1f}pp", flush=True)
    else:
        print(f"VERDICT: DISPROVED -- resonance <= 1-NN on all {len(results)} configs",
              flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
