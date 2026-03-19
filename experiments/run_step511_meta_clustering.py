#!/usr/bin/env python3
"""
Step 511 -- Hierarchical meta-clustering on chain centroids.

Protocol (R1-compliant -- NO labels used in clustering):
1. CIFAR 1-pass with dynamic growth (threshold=0.3) -> ~10K centroids
2. Run k-means on CENTROID POSITIONS at meta-k in [50, 100, 200]
3. Each image: nearest centroid -> meta-cluster assignment
4. Measure NMI and cluster purity vs true class

Baseline (Step 510 direct):
  direct k=100: NMI=0.188, purity=0.126
  direct k=1000: NMI=0.344, purity=0.587

Kill: meta NMI < 0.10 at k=100 = hierarchical clustering doesn't help.
"""
import time
import numpy as np

SPAWN_THRESHOLD = 0.3
META_K_VALUES = [50, 100, 200]


def encode_avgpool16(img):
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32) +
            0.587 * img[:, :, 1].astype(np.float32) +
            0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


def dynamic_growth_pass(X):
    """1-pass dynamic growth. Returns centroids and per-image centroid assignment."""
    centroids = None
    assignments = np.zeros(len(X), dtype=np.int32)
    for i in range(len(X)):
        x = encode_avgpool16(X[i])
        if centroids is None:
            centroids = x.reshape(1, -1).copy()
            assignments[i] = 0
        else:
            diffs = centroids - x
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            nearest = int(np.argmin(dists))
            if dists[nearest] > SPAWN_THRESHOLD:
                centroids = np.vstack([centroids, x.reshape(1, -1)])
                assignments[i] = len(centroids) - 1
            else:
                assignments[i] = nearest
    return centroids, assignments


def cluster_purity_and_nmi(labels_true, labels_pred, n_clusters):
    from sklearn.metrics import normalized_mutual_info_score
    purities = []
    for c in range(n_clusters):
        mask = labels_pred == c
        if mask.sum() == 0:
            continue
        counts = np.bincount(labels_true[mask], minlength=100)
        purities.append(counts.max() / mask.sum())
    purity = float(np.mean(purities)) if purities else 0.0
    n_pure = sum(p > 0.5 for p in purities)
    score_nmi = normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')
    return purity, score_nmi, n_pure


def run_meta_k(centroids, image_assignments, labels_true, meta_k):
    from sklearn.cluster import MiniBatchKMeans
    t0 = time.time()
    # Cluster centroid positions
    km = MiniBatchKMeans(n_clusters=meta_k, random_state=42, n_init=3,
                         max_iter=100, batch_size=512)
    km.fit(centroids)
    centroid_meta = km.labels_   # meta-cluster for each centroid
    # Map each image via its assigned centroid to meta-cluster
    meta_assignments = centroid_meta[image_assignments]
    purity, score_nmi, n_pure = cluster_purity_and_nmi(labels_true, meta_assignments, meta_k)
    elapsed = time.time() - t0
    print(f"  meta-k={meta_k:4d}: NMI={score_nmi:.4f}  purity={purity:.4f}  "
          f"n_pure>50%={n_pure:4d}/{meta_k}  {elapsed:.1f}s", flush=True)
    return purity, score_nmi, n_pure


def main():
    t_total = time.time()
    print("Step 511: Hierarchical meta-clustering on chain centroids", flush=True)
    print(f"spawn_threshold={SPAWN_THRESHOLD}  meta_k_values={META_K_VALUES}", flush=True)

    print("\nLoading CIFAR-100...", flush=True)
    import torchvision
    ds = torchvision.datasets.CIFAR100('C:/Users/Admin/cifar100_data', train=False, download=True)
    X = np.array(ds.data)
    y = np.array(ds.targets)
    print(f"  {len(X)} test images, {len(set(y))} classes", flush=True)

    print("\nPhase 1: Dynamic growth pass (1-pass CIFAR)...", flush=True)
    t0 = time.time()
    centroids, assignments = dynamic_growth_pass(X)
    n_centroids = len(centroids)
    print(f"  {n_centroids} centroids spawned  {time.time()-t0:.1f}s", flush=True)

    # Verify: how many unique centroids are assigned (should be all of them for threshold=0.3)
    unique_assigned = len(np.unique(assignments))
    print(f"  unique centroids assigned: {unique_assigned}", flush=True)

    print("\nMeta-clustering sweep...", flush=True)
    results = []
    for meta_k in META_K_VALUES:
        r = run_meta_k(centroids, assignments, y, meta_k)
        results.append((meta_k, r[0], r[1], r[2]))

    # Also compute direct k=100 for comparison (on raw features, same 10K images)
    print("\nDirect k=100 baseline (on raw image features)...", flush=True)
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import normalized_mutual_info_score
    t0 = time.time()
    feats = np.array([encode_avgpool16(X[i]) for i in range(len(X))], dtype=np.float32)
    km_direct = MiniBatchKMeans(n_clusters=100, random_state=42, n_init=3,
                                max_iter=100, batch_size=512)
    km_direct.fit(feats)
    direct_assignments = km_direct.labels_
    direct_nmi = normalized_mutual_info_score(y, direct_assignments, average_method='arithmetic')
    direct_purity_vals = []
    for c in range(100):
        mask = direct_assignments == c
        if mask.sum() == 0: continue
        counts = np.bincount(y[mask], minlength=100)
        direct_purity_vals.append(counts.max() / mask.sum())
    direct_purity = float(np.mean(direct_purity_vals))
    direct_npure = sum(p > 0.5 for p in direct_purity_vals)
    print(f"  direct k=100: NMI={direct_nmi:.4f}  purity={direct_purity:.4f}  "
          f"n_pure>50%={direct_npure}/100  {time.time()-t0:.1f}s", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("STEP 511 SUMMARY", flush=True)
    print(f"  Chain centroids: {n_centroids} (1-pass CIFAR, threshold={SPAWN_THRESHOLD})", flush=True)
    print(f"\n  {'Method':<30} {'NMI':>8}  {'purity':>8}  {'n_pure>50%':>12}", flush=True)
    print(f"  {'direct k=100 (baseline)':<30} {direct_nmi:>8.4f}  {direct_purity:>8.4f}  "
          f"{direct_npure:>6d}/100", flush=True)
    for meta_k, purity, score_nmi, n_pure in results:
        print(f"  {'meta k='+str(meta_k):<30} {score_nmi:>8.4f}  {purity:>8.4f}  "
              f"{n_pure:>6d}/{meta_k}", flush=True)

    best_meta_nmi = max(r[2] for r in results)
    best_meta_k = results[[r[2] for r in results].index(best_meta_nmi)][0]

    print(f"\nVERDICT:", flush=True)
    if best_meta_nmi > direct_nmi * 1.1:
        print(f"  Meta-clustering OUTPERFORMS direct k=100 (NMI {best_meta_nmi:.4f} vs {direct_nmi:.4f}).", flush=True)
        print(f"  Hierarchical approach adds value. Substrate geometry encodes class structure.", flush=True)
    elif best_meta_nmi >= direct_nmi * 0.9:
        print(f"  Meta-clustering MATCHES direct k=100 (NMI {best_meta_nmi:.4f} vs {direct_nmi:.4f}).", flush=True)
        print(f"  No benefit from hierarchy. Centroids are the images in disguise.", flush=True)
    else:
        print(f"  Meta-clustering UNDERPERFORMS direct k=100 (NMI {best_meta_nmi:.4f} vs {direct_nmi:.4f}).", flush=True)
        print(f"  KILL: hierarchical clustering doesn't help.", flush=True)

    print(f"\nNote: ARC navigation unaffected -- meta-clustering adds readout only.", flush=True)
    print(f"Total elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
