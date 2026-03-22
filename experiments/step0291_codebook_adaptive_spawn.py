#!/usr/bin/env python3
"""
Step 291 -- Adaptive spawn threshold vs fixed on P-MNIST.

Spec. Frozen frame test: can the spawn criterion emerge from the
state distribution rather than being a hardcoded external hyperparameter?

Hypothesis: If adaptive threshold (derived from the distribution of
nearest-neighbor similarities already seen) matches or beats fixed threshold,
the spawn criterion is a state-derived property, not an external design choice.
This collapses one frozen frame toward S1.

Adaptive rule: theta = mean(1 - max_sim) + 1*std(1 - max_sim)
  where distances are tracked online (Welford), no external memory.
  Spawn if (1 - max_sim) > theta  <=>  max_sim < 1 - theta
  Interpretation: spawn only if this input is >1 sigma more novel than average.
  Initially (no history): fall back to fixed=0.7 until n_seen >= 10.

Kill criterion (Spec):
  If adaptive AA < fixed AA - 5pp: state distribution alone insufficient.
  If within 5pp or better: frozen frame is collapsible -> push to CIFAR-100.

Baseline: Step 99 fixed sp=0.7 -> 91.8% AA, 0.0pp forgetting.
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

LR            = 0.001
FIXED_THRESH  = 0.7
K_VOTE        = 3          # best k from Step 99
N_WARMUP      = 10         # min samples before adaptive threshold kicks in


# ─── Welford online mean/std ──────────────────────────────────────────────────

class Welford:
    """Online mean and std (Welford's algorithm). No stored history."""
    def __init__(self):
        self.n   = 0
        self.mean = 0.0
        self.M2   = 0.0

    def update(self, x):
        self.n += 1
        delta  = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self):
        if self.n < 2:
            return 0.0
        return math.sqrt(self.M2 / (self.n - 1))

    @property
    def threshold(self):
        """Spawn distance threshold: mean + 1*std of (1 - max_sim)."""
        return self.mean + self.std


# ─── TopKFold variants ───────────────────────────────────────────────────────

class TopKFold:
    """Fixed spawn threshold (Step 99 baseline)."""
    def __init__(self, d, lr=0.01, spawn_thresh=0.7):
        self.V      = torch.empty(0, d, device=DEVICE)
        self.labels = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.lr     = lr
        self.spawn_thresh = spawn_thresh
        self.d      = d
        self.n_spawned = 0

    def step(self, r, label):
        r = F.normalize(r, dim=0)
        if self.V.shape[0] == 0:
            max_sim = -1.0
        else:
            max_sim = (self.V @ r).max().item()

        if max_sim < self.spawn_thresh:
            self.V      = torch.cat([self.V, r.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([label], device=DEVICE)])
            self.n_spawned += 1
            return

        sims   = self.V @ r
        winner = sims.argmax().item()
        self.V[winner] = F.normalize(
            self.V[winner] + self.lr * (r - self.V[winner]), dim=0)

    def eval_batch(self, R, k):
        R     = F.normalize(R, dim=1)
        sims  = R @ self.V.T
        n     = len(R)
        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(n, n_cls, device=DEVICE)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0:
                continue
            class_sims = sims[:, mask]
            k_eff      = min(k, class_sims.shape[1])
            scores[:, c] = class_sims.topk(k_eff, dim=1).values.sum(dim=1)
        return scores.argmax(dim=1).cpu()


class AdaptiveTopKFold:
    """Adaptive spawn threshold: theta = mean(dist) + 1*std(dist) via Welford."""
    def __init__(self, d, lr=0.01, warmup_thresh=0.7, n_warmup=N_WARMUP):
        self.V      = torch.empty(0, d, device=DEVICE)
        self.labels = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.lr     = lr
        self.warmup_thresh = warmup_thresh
        self.n_warmup = n_warmup
        self.d      = d
        self.n_spawned = 0
        self.wf    = Welford()   # tracks (1 - max_sim) distribution

    def _spawn_threshold(self):
        """Current sim threshold: 1 - (mean_dist + 1*std_dist)."""
        if self.wf.n < self.n_warmup:
            return self.warmup_thresh
        # distance threshold in [0, 2]; sim threshold = 1 - dist_thresh
        return 1.0 - self.wf.threshold

    def step(self, r, label):
        r = F.normalize(r, dim=0)
        if self.V.shape[0] == 0:
            max_sim = -1.0
        else:
            max_sim = (self.V @ r).max().item()

        dist = 1.0 - max_sim
        self.wf.update(dist)
        thresh = self._spawn_threshold()

        if max_sim < thresh:
            self.V      = torch.cat([self.V, r.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([label], device=DEVICE)])
            self.n_spawned += 1
            return

        sims   = self.V @ r
        winner = sims.argmax().item()
        self.V[winner] = F.normalize(
            self.V[winner] + self.lr * (r - self.V[winner]), dim=0)

    def eval_batch(self, R, k):
        R     = F.normalize(R, dim=1)
        sims  = R @ self.V.T
        n     = len(R)
        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(n, n_cls, device=DEVICE)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0:
                continue
            class_sims = sims[:, mask]
            k_eff      = min(k, class_sims.shape[1])
            scores[:, c] = class_sims.topk(k_eff, dim=1).values.sum(dim=1)
        return scores.argmax(dim=1).cpu()

    @property
    def final_threshold(self):
        return self._spawn_threshold()


# ─── MNIST + embedding ────────────────────────────────────────────────────────

def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST(
        './data/mnist', train=True,  download=True)
    te = torchvision.datasets.MNIST(
        './data/mnist', train=False, download=True)
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


# ─── Run one model ────────────────────────────────────────────────────────────

def run(model, name, X_tr, y_tr, perms, test_embeds, y_te_t, n_tasks):
    acc = [[None] * n_tasks for _ in range(n_tasks)]
    for task_id in range(n_tasks):
        X_sub, y_sub = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS,
                                         seed=task_id * 1337)
        X_emb       = embed(X_sub, perms[task_id], P_GLOBAL)
        labels_list = y_sub.tolist()
        cb_before   = model.V.shape[0]

        for i in range(len(X_emb)):
            model.step(X_emb[i], labels_list[i])

        cb_after = model.V.shape[0]
        for eval_task in range(task_id + 1):
            preds = model.eval_batch(test_embeds[eval_task], K_VOTE)
            acc[eval_task][task_id] = (preds == y_te_t).float().mean().item()

        peek = acc[task_id][task_id]
        extra = ""
        if hasattr(model, 'final_threshold'):
            extra = f"  adaptive_thresh={model.final_threshold:.3f}"
        print(f"  [{name}] task={task_id} "
              f"cb={cb_before}->{cb_after} "
              f"peek={peek*100:.1f}%{extra}", flush=True)

    aa  = compute_aa(acc, n_tasks)
    fgt = compute_fgt(acc, n_tasks)
    cb  = model.V.shape[0]
    sp  = model.n_spawned
    return aa, fgt, cb, sp


# ─── Main ─────────────────────────────────────────────────────────────────────

P_GLOBAL = None


def main():
    global P_GLOBAL
    t0 = time.time()
    print(f"Step 291 -- Adaptive Spawn Threshold, P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, D_OUT={D_OUT}, DEVICE={DEVICE}", flush=True)
    print(f"Fixed thresh={FIXED_THRESH}, Adaptive: mean(dist)+1*std(dist), "
          f"warmup={N_WARMUP}", flush=True)
    print(f"k_vote={K_VOTE} (best from Step 99)", flush=True)
    print(flush=True)

    print("Loading MNIST...", flush=True)
    X_tr, y_tr, X_te, y_te = load_mnist()
    P_GLOBAL = make_projection()
    y_te_t   = torch.from_numpy(y_te).long()

    perms = [list(range(784))]
    for t in range(1, N_TASKS):
        perms.append(make_permutation(seed=t * 1000))

    print("Pre-embedding test sets...", flush=True)
    test_embeds = [embed(X_te, perms[t], P_GLOBAL) for t in range(N_TASKS)]
    print(flush=True)

    # --- Fixed threshold (Step 99 baseline) ---
    print("=== FIXED threshold (sp=0.7) ===", flush=True)
    fixed_model = TopKFold(D_OUT, lr=LR, spawn_thresh=FIXED_THRESH)
    aa_f, fgt_f, cb_f, sp_f = run(
        fixed_model, "fixed", X_tr, y_tr, perms, test_embeds, y_te_t, N_TASKS)
    print(flush=True)

    # --- Adaptive threshold ---
    print("=== ADAPTIVE threshold (mean+1std of dist) ===", flush=True)
    adapt_model = AdaptiveTopKFold(D_OUT, lr=LR, warmup_thresh=FIXED_THRESH,
                                   n_warmup=N_WARMUP)
    aa_a, fgt_a, cb_a, sp_a = run(
        adapt_model, "adapt", X_tr, y_tr, perms, test_embeds, y_te_t, N_TASKS)
    final_thresh = adapt_model.final_threshold
    print(flush=True)

    # ─── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=" * 70, flush=True)
    print("STEP 291 SUMMARY -- Adaptive vs Fixed Spawn Threshold, P-MNIST",
          flush=True)
    print("=" * 70, flush=True)
    print(f"\n{'':20} {'AA':>7} {'Forgetting':>11} {'CB':>6} {'Spawned':>8}",
          flush=True)
    print("-" * 56, flush=True)
    print(f"{'Fixed (sp=0.7)':20} {aa_f*100:>6.1f}% {fgt_f*100:>10.1f}pp "
          f"{cb_f:>6} {sp_f:>8}", flush=True)
    print(f"{'Adaptive (mean+1sd)':20} {aa_a*100:>6.1f}% {fgt_a*100:>10.1f}pp "
          f"{cb_a:>6} {sp_a:>8}", flush=True)
    delta = aa_a - aa_f
    print(f"\nAdaptive vs Fixed: {delta*100:+.1f}pp AA", flush=True)
    print(f"Adaptive final threshold: {final_thresh:.3f}", flush=True)

    print(flush=True)
    print("KILL CRITERION (Spec):", flush=True)
    if aa_a < aa_f - 0.05:
        print(f"  KILLED -- adaptive ({aa_a*100:.1f}%) < fixed ({aa_f*100:.1f}%) - 5pp",
              flush=True)
        print(f"  State distribution alone is insufficient as spawn criterion.",
              flush=True)
    else:
        print(f"  PASSES -- adaptive within 5pp or better of fixed.", flush=True)
        print(f"  Frozen frame (hardcoded spawn criterion) is collapsible.", flush=True)
        if fgt_a <= 0.05:
            print(f"  Forgetting: PASS ({fgt_a*100:.1f}pp)", flush=True)
            print(f"  -> Push to CIFAR-100.", flush=True)
        else:
            print(f"  Forgetting: FAIL ({fgt_a*100:.1f}pp > 5pp)", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
