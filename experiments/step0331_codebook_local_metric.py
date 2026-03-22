#!/usr/bin/env python3
"""
Step 331 — Self-discovered clustering + local metric on a%b.

Baselines:
  Uniform weights (Step 296):              86.8% LOO
  Global learned weights (Step 308):       91.2% LOO
  Per-b prescribed weights (Step 314):    ~91.2% LOO

Hypothesis: cluster phi vectors (the substrate's own representation),
then learn per-cluster weights. If phi space encodes b-grouping, we
rediscover it. If phi space finds FINER structure, we beat 91.2%.

Algorithm:
1. Build a%b dataset (400 entries, a,b in 1..20)
2. Compute phi (LOO, K=5) for each entry
3. k-means cluster in phi space: n_clusters in [5, 10, 15, 20]
4. Per-cluster weight learning (auto_loop style)
5. LOO accuracy with per-cluster weights
6. Measure R² of b-value from cluster assignment

Kill: per-cluster must beat global learned weights (91.2%).
"""

import sys
import numpy as np
import time
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX   = 20
K           = 5
SENTINEL    = TRAIN_MAX * 3
N_CLUSTER_VALS = [5, 10, 15, 20]
GLOBAL_BASELINE = 0.912   # Step 308 global learned weights

# ─── a%b dataset ───────────────────────────────────────────────────────────────

def build_dataset():
    A, B, Y = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b); Y.append(a % b)
    return np.array(A), np.array(B), np.array(Y)


# ─── Phi computation (from auto_loop.py) ──────────────────────────────────────

def compute_phi(query_a, query_b, A, B, Y, exclude_idx, K, max_class):
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
    same_b_mask = (B == query_b)
    for c in range(max_class):
        class_mask = (Y == c) & same_b_mask
        if exclude_idx is not None and exclude_idx < len(A) and class_mask[exclude_idx]:
            if Y[exclude_idx] == c:
                class_mask = class_mask.copy()
                class_mask[exclude_idx] = False
        idxs = np.where(class_mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A[idxs] - query_a).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(dists))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi


def compute_all_phi_loo(A, B, Y, K):
    """Compute LOO phi for all entries."""
    n = len(A)
    max_class = int(Y.max()) + 1
    dim = max_class * K
    all_phi = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        all_phi[i] = compute_phi(A[i], B[i], A, B, Y, i, K, max_class)
    return all_phi, max_class


# ─── LOO evaluation with weights ──────────────────────────────────────────────

def loo_with_weights(A, B, Y, all_phi, weights_per_entry, max_class, K):
    """
    LOO accuracy where each query uses its assigned weight vector.
    weights_per_entry: (n, K) array — weight vector for each entry.
    """
    n = len(A)
    correct = 0
    for i in range(n):
        phi_q = compute_phi(A[i], B[i], A, B, Y, i, K, max_class)
        w = weights_per_entry[i]
        w_expanded = np.tile(w, max_class)
        diffs = all_phi - phi_q
        dists = (diffs ** 2 * w_expanded).sum(axis=1)
        dists[i] = float('inf')
        best_j = np.argmin(dists)
        if Y[best_j] == Y[i]:
            correct += 1
    return correct / n


def loo_uniform(A, B, Y, all_phi, max_class, K):
    """LOO with uniform weights (baseline)."""
    n = len(A)
    uniform_w = np.ones(max_class * K)
    correct = 0
    for i in range(n):
        phi_q = compute_phi(A[i], B[i], A, B, Y, i, K, max_class)
        diffs = all_phi - phi_q
        dists = (diffs ** 2).sum(axis=1)
        dists[i] = float('inf')
        best_j = np.argmin(dists)
        if Y[best_j] == Y[i]:
            correct += 1
    return correct / n


# ─── Weight learning (adapted from auto_loop.py) ──────────────────────────────

def learn_weights_subset(A_sub, B_sub, Y_sub, A_full, B_full, Y_full,
                         all_phi_full, idx_sub, K, max_class,
                         lr_w=0.1, epochs=5):
    """
    Learn k-position weights using subset entries, evaluating against full codebook.
    idx_sub: indices into the full dataset that belong to this cluster.
    """
    w = np.ones(K, dtype=np.float64)

    # Phi for subset entries (LOO within subset? No — LOO within full set)
    # Use precomputed all_phi_full for cluster entries
    phi_sub = all_phi_full[idx_sub]  # (n_sub, max_class*K)
    y_sub   = Y_full[idx_sub]
    n_sub   = len(idx_sub)

    for epoch in range(epochs):
        order = np.random.permutation(n_sub)
        for rank in order:
            i_full = idx_sub[rank]
            phi_q = compute_phi(A_full[i_full], B_full[i_full],
                                A_full, B_full, Y_full, i_full, K, max_class)
            w_expanded = np.tile(w, max_class)
            diffs = all_phi_full - phi_q
            dists = (diffs ** 2 * w_expanded).sum(axis=1)
            dists[i_full] = float('inf')
            best_j = np.argmin(dists)

            if Y_full[best_j] != Y_full[i_full]:
                diff_sq = (phi_q - all_phi_full[best_j]) ** 2
                per_k_signal = np.zeros(K)
                for k in range(K):
                    indices = [c * K + k for c in range(max_class)]
                    per_k_signal[k] = diff_sq[indices].mean()
                w += lr_w * per_k_signal
                w = np.maximum(w, 0.01)

    w = w / w.sum() * K
    return w


# ─── R² measurement ───────────────────────────────────────────────────────────

def r2_b_from_clusters(B, cluster_assignments):
    """
    Compute R² of predicting b-value from cluster assignment.
    Treat cluster assignment as a categorical predictor:
    R² = 1 - SS_res / SS_tot
    where SS_res = variance of b within each cluster (summed),
          SS_tot = variance of b overall.
    """
    b_mean = B.mean()
    ss_tot = ((B - b_mean) ** 2).sum()
    ss_res = 0.0
    for c in np.unique(cluster_assignments):
        mask = cluster_assignments == c
        b_c = B[mask]
        ss_res += ((b_c - b_c.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def weight_diversity(cluster_weights):
    """Average pairwise cosine similarity between cluster weight vectors."""
    n = len(cluster_weights)
    if n < 2:
        return 1.0
    W = np.array(cluster_weights)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W_norm = W / np.maximum(norms, 1e-8)
    sims = W_norm @ W_norm.T
    # Upper triangle
    idxs = np.triu_indices(n, k=1)
    return float(sims[idxs].mean())


# ─── K-means (simple, no sklearn dependency) ──────────────────────────────────

def kmeans(X, n_clusters, n_init=5, max_iter=100, seed=42):
    """Simple k-means returning cluster assignments."""
    rng = np.random.RandomState(seed)
    best_inertia = float('inf')
    best_labels = None

    for _ in range(n_init):
        # Random init
        centers = X[rng.choice(len(X), n_clusters, replace=False)]

        for _ in range(max_iter):
            # Assign
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            # Update
            new_centers = np.zeros_like(centers)
            for c in range(n_clusters):
                mask = labels == c
                if mask.sum() > 0:
                    new_centers[c] = X[mask].mean(axis=0)
                else:
                    new_centers[c] = X[rng.randint(len(X))]
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        inertia = sum(
            ((X[labels == c] - centers[c]) ** 2).sum()
            for c in range(n_clusters)
            if (labels == c).sum() > 0
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    np.random.seed(42)

    print("Step 331 — Self-discovered clustering + local metric on a%b", flush=True)
    print(f"K={K}, n_clusters sweep: {N_CLUSTER_VALS}", flush=True)
    print(f"Baselines: uniform=86.8%, global_learned=91.2%, per-b=~91.2%", flush=True)
    print(f"Kill: per-cluster must beat global learned (91.2%)", flush=True)
    print(flush=True)

    A, B, Y = build_dataset()
    n = len(A)
    max_class = int(Y.max()) + 1
    print(f"Dataset: {n} entries, {max_class} classes", flush=True)
    print(flush=True)

    # ─── Compute all phi (LOO) ────────────────────────────────────────────────
    print("Computing phi (LOO)...", flush=True)
    all_phi, max_class = compute_all_phi_loo(A, B, Y, K)
    print(f"  phi shape: {all_phi.shape}  ({n} x {max_class}*{K}={max_class*K})", flush=True)
    print(f"  Elapsed: {time.time()-t0:.1f}s", flush=True)
    print(flush=True)

    # ─── Uniform baseline (should match ~86.8%) ───────────────────────────────
    print("Computing uniform baseline...", flush=True)
    uniform_acc = loo_uniform(A, B, Y, all_phi, max_class, K)
    print(f"  Uniform LOO: {uniform_acc*100:.1f}%  (expected: 86.8%)", flush=True)
    print(f"  Elapsed: {time.time()-t0:.1f}s", flush=True)
    print(flush=True)

    # ─── Global learned weights (quick run of auto_loop) ─────────────────────
    print("Computing global learned weights (5 epochs, lr=0.05)...", flush=True)
    # Reuse learn_weights_subset on ALL entries
    global_w = learn_weights_subset(A, B, Y, A, B, Y, all_phi,
                                    np.arange(n), K, max_class,
                                    lr_w=0.05, epochs=5)
    global_w_per_entry = np.tile(global_w, (n, 1))
    global_acc = loo_with_weights(A, B, Y, all_phi, global_w_per_entry, max_class, K)
    print(f"  Global learned w: {np.round(global_w, 3).tolist()}", flush=True)
    print(f"  Global LOO: {global_acc*100:.1f}%  (expected: ~91.2%)", flush=True)
    print(f"  Elapsed: {time.time()-t0:.1f}s", flush=True)
    print(flush=True)

    # ─── Per-cluster sweep ───────────────────────────────────────────────────
    results = {}
    print("=" * 65, flush=True)
    print("Per-cluster weight learning sweep", flush=True)
    print("=" * 65, flush=True)

    for n_clusters in N_CLUSTER_VALS:
        print(f"\n--- n_clusters={n_clusters} ---", flush=True)

        # Cluster in phi space
        labels = kmeans(all_phi, n_clusters, n_init=5, seed=42)
        cluster_sizes = [(labels == c).sum() for c in range(n_clusters)]
        print(f"  Cluster sizes: min={min(cluster_sizes)} max={max(cluster_sizes)} "
              f"mean={np.mean(cluster_sizes):.1f}", flush=True)

        # R² of b-value from cluster assignment
        r2 = r2_b_from_clusters(B, labels)
        print(f"  R²(b from cluster): {r2:.3f}", flush=True)

        # Per-cluster weight learning
        cluster_weights = []
        for c in range(n_clusters):
            idx_c = np.where(labels == c)[0]
            if len(idx_c) < 3:
                # Too few entries — use global weights
                cluster_weights.append(global_w.copy())
                continue
            w_c = learn_weights_subset(
                A[idx_c], B[idx_c], Y[idx_c],
                A, B, Y, all_phi, idx_c, K, max_class,
                lr_w=0.1, epochs=10
            )
            cluster_weights.append(w_c)

        # Weight diversity
        div = weight_diversity(cluster_weights)
        print(f"  Weight diversity (avg cosine sim): {div:.4f}  "
              f"(1.0=identical, lower=diverse)", flush=True)

        print(f"  Cluster weights:", flush=True)
        for c, w in enumerate(cluster_weights):
            size = cluster_sizes[c]
            print(f"    c={c:2d} (n={size:3d}): {np.round(w, 3).tolist()}", flush=True)

        # Assign each entry its cluster's weights
        weights_per_entry = np.array([cluster_weights[labels[i]] for i in range(n)])

        # LOO evaluation
        cluster_acc = loo_with_weights(A, B, Y, all_phi, weights_per_entry, max_class, K)
        delta_vs_global = cluster_acc - global_acc

        print(f"  Per-cluster LOO: {cluster_acc*100:.2f}%  "
              f"(delta vs global: {delta_vs_global*100:+.2f}pp)", flush=True)

        results[n_clusters] = {
            'acc': cluster_acc,
            'r2': r2,
            'diversity': div,
            'weights': cluster_weights,
        }

        print(f"  Elapsed: {time.time()-t0:.1f}s", flush=True)

    # ─── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 331 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"Uniform LOO:        {uniform_acc*100:.2f}%  (expected: 86.8%)", flush=True)
    print(f"Global learned LOO: {global_acc*100:.2f}%  (expected: 91.2%)", flush=True)
    print(flush=True)
    print(f"{'n_clusters':>12} {'LOO%':>8} {'delta':>8} {'R²(b)':>8} {'diversity':>10}",
          flush=True)
    print("-" * 55, flush=True)
    for k in N_CLUSTER_VALS:
        r = results[k]
        delta = (r['acc'] - global_acc) * 100
        print(f"{k:>12d} {r['acc']*100:>7.2f}% {delta:>+7.2f}pp {r['r2']:>8.3f} {r['diversity']:>10.4f}",
              flush=True)

    # Best result
    best_k = max(N_CLUSTER_VALS, key=lambda k: results[k]['acc'])
    best_acc = results[best_k]['acc']
    delta_vs_global = best_acc - global_acc
    kill = best_acc <= global_acc
    success = best_acc > global_acc

    print(flush=True)
    print(f"Best: n_clusters={best_k}, LOO={best_acc*100:.2f}%", flush=True)

    # Interpretation
    print(flush=True)
    print("Interpretation:", flush=True)
    for k in N_CLUSTER_VALS:
        r = results[k]
        if r['r2'] > 0.9:
            interp = "rediscovers b-groups (Stage 5 confirmed, no new discovery)"
        elif r['r2'] > 0.5:
            interp = "partial b-grouping (mixed)"
        else:
            interp = "finds DIFFERENT structure than b"
        print(f"  n_clusters={k}: R2={r['r2']:.3f} -> {interp}", flush=True)

    print(flush=True)
    print("=" * 65, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 65, flush=True)
    print(f"Global learned: {global_acc*100:.2f}%", flush=True)
    print(f"Best per-cluster: {best_acc*100:.2f}% (n_clusters={best_k})", flush=True)
    print(f"Delta: {delta_vs_global*100:+.2f}pp", flush=True)
    print(f"Kill (per-cluster <= global): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (per-cluster > global): {'YES' if success else 'NO'}", flush=True)

    if kill:
        print("\nKILLED — local metric gives no improvement over global", flush=True)
    else:
        print(f"\nSUCCESS — local metric beats global by {delta_vs_global*100:.2f}pp",
              flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
