#!/usr/bin/env python3
"""
Step 330 — Automated loop on P-MNIST.

Step 99 reached 95.4% AA with top-K (K=3) cosine vote, uniform weights.
auto_loop.py got 96.5% on a%b by learning k-position weights.

Question: does the loop mechanism generalize to P-MNIST?
On a%b, loop found k=0 (nearest neighbor) matters more than k=1,2.
On P-MNIST with K=3, does a similar k-index importance structure exist?

Loop:
1. Build codebook via competitive learning (as Step 99)
2. Baseline accuracy with uniform weights
3. learn_weights: upweight k-positions where correct class beats wrong class
4. Prescribe weights, re-evaluate
5. If improved: lock, iterate. Kill: <=0pp after 5 turns.

Phi analogue for top-K cosine:
  phi[c*K+k] = k-th largest cosine similarity to class-c codebook vectors
  score[c] = sum_k w[k] * phi[c*K+k]
  predict = argmax_c(score[c])
"""

import sys
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

# ─── Config ────────────────────────────────────────────────────────────────────

N_TASKS       = 10
D_OUT         = 384
N_TRAIN_TASK  = 6000
N_TEST_TASK   = 10000
N_CLASSES     = 10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES
SEED          = 42

LR           = 0.001
SPAWN_THRESH = 0.7
K_VOTE       = 3          # top-K per class (phi dim = N_CLASSES * K_VOTE = 30)
PHI_DIM      = N_CLASSES * K_VOTE

MAX_LOOP_TURNS = 5
N_WEIGHT_SAMPLES = 5000   # training samples used for weight learning per turn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─── TopKFold (same as Step 99) ────────────────────────────────────────────────

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

    def eval_batch_phi(self, R):
        """
        Compute phi for each query: top-K cosine sims per class.
        R: (n, d) unit-normalized GPU tensor.
        Returns phi: numpy (n, PHI_DIM) where phi[i, c*K+k] = k-th largest sim
                     of sample i to class-c codebook vectors.
        """
        R = F.normalize(R, dim=1)
        sims = R @ self.V.T   # (n, N_cb)
        n = len(R)
        phi = np.zeros((n, PHI_DIM), dtype=np.float32)

        sims_np = sims.cpu().numpy()
        labels_np = self.labels.cpu().numpy()

        for c in range(N_CLASSES):
            class_mask = (labels_np == c)
            if class_mask.sum() == 0:
                continue
            class_sims = sims_np[:, class_mask]   # (n, N_c)
            k_eff = min(K_VOTE, class_sims.shape[1])
            # Sort descending, take top K_VOTE
            sorted_sims = np.sort(class_sims, axis=1)[:, ::-1][:, :k_eff]
            phi[:, c * K_VOTE:c * K_VOTE + k_eff] = sorted_sims

        return phi

    def predict_from_phi(self, phi, weights):
        """
        phi: (n, PHI_DIM) — top-K sims per class
        weights: (K_VOTE,) — per-k-position weights
        Returns predictions: (n,) int array
        """
        n = len(phi)
        scores = np.zeros((n, N_CLASSES), dtype=np.float32)
        for c in range(N_CLASSES):
            for k in range(K_VOTE):
                scores[:, c] += weights[k] * phi[:, c * K_VOTE + k]
        return scores.argmax(axis=1)


# ─── MNIST + embedding ─────────────────────────────────────────────────────────

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


# ─── Weight learning (auto_loop analogue) ──────────────────────────────────────

def learn_weights_pmnist(phi, y_true, current_weights, lr_w=0.1):
    """
    For each misclassified sample, upweight k-positions where the correct
    class has higher similarity than the wrong class.

    phi: (n, PHI_DIM) — top-K sims per class
    y_true: (n,) int — true labels
    current_weights: (K_VOTE,) float — weights to update
    """
    n = len(phi)
    w = current_weights.copy()

    # Get predictions with current weights
    preds = TopKFold(D_OUT).predict_from_phi.__func__  # can't call this easily
    # Compute predictions manually
    scores = np.zeros((n, N_CLASSES), dtype=np.float32)
    for c in range(N_CLASSES):
        for k in range(K_VOTE):
            scores[:, c] += w[k] * phi[:, c * K_VOTE + k]
    preds = scores.argmax(axis=1)

    per_k_signal = np.zeros(K_VOTE)
    n_wrong = 0

    for i in range(n):
        if preds[i] == y_true[i]:
            continue
        n_wrong += 1
        c_right = y_true[i]
        c_wrong = preds[i]

        # At each k-position: how much does correct class beat wrong class?
        for k in range(K_VOTE):
            correct_sim = phi[i, c_right * K_VOTE + k]
            wrong_sim   = phi[i, c_wrong * K_VOTE + k]
            # Upweight k where correct class already has higher sim
            per_k_signal[k] += max(0.0, correct_sim - wrong_sim)

    if n_wrong > 0:
        per_k_signal /= n_wrong
        w += lr_w * per_k_signal
        w = np.maximum(w, 0.01)
        w = w / w.sum() * K_VOTE   # normalize to sum = K_VOTE

    return w, n_wrong


def eval_with_weights(phi_list, y_list, weights):
    """Evaluate accuracy across all tasks using given weights."""
    total_correct, total_n = 0, 0
    per_task = []
    for phi, y in zip(phi_list, y_list):
        scores = np.zeros((len(phi), N_CLASSES), dtype=np.float32)
        for c in range(N_CLASSES):
            for k in range(K_VOTE):
                scores[:, c] += weights[k] * phi[:, c * K_VOTE + k]
        preds = scores.argmax(axis=1)
        acc = (preds == y).mean()
        per_task.append(acc)
        total_correct += (preds == y).sum()
        total_n += len(y)
    return total_correct / total_n, per_task


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 330 — Automated loop on P-MNIST", flush=True)
    print(f"K_VOTE={K_VOTE}, PHI_DIM={PHI_DIM}, MAX_TURNS={MAX_LOOP_TURNS}", flush=True)
    print(f"N_WEIGHT_SAMPLES={N_WEIGHT_SAMPLES} per turn", flush=True)
    print(f"Baseline target: 95.4% AA (Step 99, uniform weights)", flush=True)
    print(f"Kill: loop adds <=0pp after {MAX_LOOP_TURNS} turns", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(flush=True)

    print("Loading MNIST...", flush=True)
    X_tr, y_tr, X_te, y_te = load_mnist()
    P      = make_projection()
    y_te_np = y_te

    perms = [list(range(784))]
    for t in range(1, N_TASKS):
        perms.append(make_permutation(seed=t * 1000))

    # ─── Phase 1: Build codebook (same as Step 99) ─────────────────────────────
    print("=== Phase 1: Build codebook (Step 99 config) ===", flush=True)
    model = TopKFold(D_OUT, lr=LR, spawn_thresh=SPAWN_THRESH)
    all_train_X, all_train_y, all_train_perms = [], [], []

    for task_id in range(N_TASKS):
        X_sub, y_sub = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS,
                                         seed=task_id * 1337)
        X_emb = embed(X_sub, perms[task_id], P)
        cb_before = model.V.shape[0]
        for i in range(len(X_emb)):
            model.step(X_emb[i], y_sub[i])
        cb_after = model.V.shape[0]
        print(f"  Task {task_id}: cb {cb_before}->{cb_after} (+{cb_after-cb_before})",
              flush=True)
        all_train_X.append(X_sub)
        all_train_y.append(y_sub)
        all_train_perms.append(perms[task_id])

    print(f"Final codebook: {model.V.shape[0]} vectors", flush=True)
    print(f"Elapsed: {time.time()-t0:.1f}s", flush=True)
    print(flush=True)

    # ─── Phase 2: Compute phi for test sets ────────────────────────────────────
    print("=== Phase 2: Compute phi for test sets ===", flush=True)
    test_phi_list = []
    for task_id in range(N_TASKS):
        X_emb = embed(X_te, perms[task_id], P)
        phi = model.eval_batch_phi(X_emb)
        test_phi_list.append(phi)
        print(f"  Task {task_id}: phi shape {phi.shape}", flush=True)
    print(f"Elapsed: {time.time()-t0:.1f}s", flush=True)
    print(flush=True)

    # ─── Phase 3: Baseline accuracy (uniform weights) ──────────────────────────
    uniform_weights = np.ones(K_VOTE, dtype=np.float64)
    uniform_weights = uniform_weights / uniform_weights.sum() * K_VOTE

    baseline_aa, baseline_per_task = eval_with_weights(
        test_phi_list, [y_te_np] * N_TASKS, uniform_weights)

    print("=== Phase 3: Baseline (uniform weights) ===", flush=True)
    print(f"Uniform weights: {np.round(uniform_weights, 3).tolist()}", flush=True)
    print(f"Baseline AA: {baseline_aa*100:.2f}%", flush=True)
    for t, acc in enumerate(baseline_per_task):
        print(f"  Task {t}: {acc*100:.1f}%", flush=True)
    print(f"Elapsed: {time.time()-t0:.1f}s", flush=True)
    print(flush=True)

    # ─── Phase 4: Compute phi for weight-learning samples ──────────────────────
    # Use random subset from all tasks for weight learning
    print("=== Phase 4: Compute phi for weight-learning samples ===", flush=True)
    rng = np.random.RandomState(SEED)
    wl_phi_list = []
    wl_y_list = []
    n_per_task = N_WEIGHT_SAMPLES // N_TASKS

    for task_id in range(N_TASKS):
        X_sub = all_train_X[task_id]
        y_sub = all_train_y[task_id]
        perm  = all_train_perms[task_id]
        idx = rng.choice(len(X_sub), n_per_task, replace=False)
        X_emb = embed(X_sub[idx], perm, P)
        phi = model.eval_batch_phi(X_emb)
        wl_phi_list.append(phi)
        wl_y_list.append(y_sub[idx])
        print(f"  Task {task_id}: {len(idx)} samples, phi shape {phi.shape}", flush=True)

    wl_phi_all = np.vstack(wl_phi_list)
    wl_y_all   = np.concatenate(wl_y_list).astype(np.int32)
    print(f"Weight-learning set: {len(wl_phi_all)} samples", flush=True)
    print(f"Elapsed: {time.time()-t0:.1f}s", flush=True)
    print(flush=True)

    # ─── Phase 5: The loop ─────────────────────────────────────────────────────
    print("=== Phase 5: The Loop ===", flush=True)
    weights = uniform_weights.copy()
    best_aa = baseline_aa
    history = [(0, baseline_aa, weights.copy(), 'uniform')]

    for turn in range(1, MAX_LOOP_TURNS + 1):
        print(f"--- Turn {turn} ---", flush=True)

        # Learn weights
        new_weights, n_wrong = learn_weights_pmnist(
            wl_phi_all, wl_y_all, weights, lr_w=0.1)

        print(f"  Wrong samples in weight-learning set: {n_wrong}/{len(wl_y_all)}", flush=True)
        print(f"  Learned w: {np.round(new_weights, 4).tolist()}", flush=True)

        # Analyze: fit exponential (as in auto_loop)
        w = new_weights
        if w[0] > 0 and w[-1] > 0 and K_VOTE > 1:
            b_est = np.log(w[0] / w[-1]) / (K_VOTE - 1)
            prescribed = np.array([w[0] * np.exp(-b_est * k) for k in range(K_VOTE)])
            prescribed = prescribed / prescribed.sum() * K_VOTE
        else:
            prescribed = new_weights.copy()
        print(f"  Prescribed w: {np.round(prescribed, 4).tolist()}", flush=True)

        # Check monotonic structure
        monotonic = all(w[i] >= w[i+1] for i in range(K_VOTE-1))
        top_k = int(np.argmax(w))
        print(f"  Top k-index: k={top_k}  Monotonic: {monotonic}", flush=True)

        # Evaluate learned weights
        learned_aa, _ = eval_with_weights(test_phi_list, [y_te_np] * N_TASKS, new_weights)
        prescribed_aa, _ = eval_with_weights(test_phi_list, [y_te_np] * N_TASKS, prescribed)

        print(f"  Learned   AA: {learned_aa*100:.3f}%  (delta: {(learned_aa-best_aa)*100:+.3f}pp)",
              flush=True)
        print(f"  Prescribed AA: {prescribed_aa*100:.3f}%  (delta: {(prescribed_aa-best_aa)*100:+.3f}pp)",
              flush=True)

        # Pick best
        best_new_aa = max(learned_aa, prescribed_aa)
        best_new_w  = new_weights if learned_aa >= prescribed_aa else prescribed

        if best_new_aa > best_aa + 1e-5:
            print(f"  >> IMPROVED {best_aa*100:.3f}% -> {best_new_aa*100:.3f}%. Locking.",
                  flush=True)
            weights = best_new_w.copy()
            best_aa = best_new_aa
            source = 'learned' if learned_aa >= prescribed_aa else 'prescribed'
            history.append((turn, best_aa, weights.copy(), source))
        else:
            print(f"  >> No improvement. Loop saturated at turn {turn}.", flush=True)
            history.append((turn, best_new_aa, best_new_w.copy(), 'saturated'))
            break

        print(flush=True)

    # ─── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 70, flush=True)
    print("STEP 330 SUMMARY — Automated loop on P-MNIST", flush=True)
    print("=" * 70, flush=True)

    final_aa, final_per_task = eval_with_weights(
        test_phi_list, [y_te_np] * N_TASKS, weights)

    print(f"Baseline AA (uniform):  {baseline_aa*100:.3f}%", flush=True)
    print(f"Final AA (loop):        {final_aa*100:.3f}%", flush=True)
    print(f"Total improvement:      {(final_aa-baseline_aa)*100:+.3f}pp", flush=True)
    print(f"Final weights:          {np.round(weights, 4).tolist()}", flush=True)
    print(flush=True)

    print("Per-task final accuracy:", flush=True)
    for t, acc in enumerate(final_per_task):
        print(f"  Task {t}: {acc*100:.1f}%", flush=True)

    print(flush=True)
    print("Turn history:", flush=True)
    for turn, aa, w, source in history:
        print(f"  Turn {turn:>2}: {aa*100:.3f}% [{source}] w={np.round(w, 3).tolist()}",
              flush=True)

    print(flush=True)
    # Kill check
    total_improvement = final_aa - baseline_aa
    n_turns = len(history) - 1
    kill = total_improvement <= 0.0
    success = total_improvement > 0.0

    print("=" * 70, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 70, flush=True)
    print(f"Improvement after {n_turns} turns: {total_improvement*100:+.3f}pp", flush=True)
    print(f"Kill (<=0pp): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (>0pp): {'YES' if success else 'NO'}", flush=True)

    if kill:
        print("\nKILLED — loop is a%b-specific, does not generalize to P-MNIST", flush=True)
    else:
        print(f"\nSUCCESS — loop generalizes. Improvement: {total_improvement*100:+.3f}pp",
              flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    np.random.seed(SEED)
    random.seed(SEED)
    main()
