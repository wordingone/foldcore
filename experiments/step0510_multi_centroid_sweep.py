#!/usr/bin/env python3
"""
Step 510 -- Centroid count sweep on CIFAR-100.
Encoding: avgpool16+centered 256D (same as chain).

For k in [50, 100, 200, 500, 1000]:
  - Fit k-means on 10K CIFAR-100 test images
  - Cluster purity: fraction of majority class per centroid, averaged
  - NMI: normalized mutual information between centroid assignment and true class

Question: is there ANY centroid count where encoding separates classes?
If NMI peaks at some k -> encoding HAS class signal at that resolution.
If NMI flat near 0 -> encoding fundamentally can't separate classes.
"""
import time
import numpy as np

K_VALUES = [50, 100, 200, 500, 1000]


def encode_avgpool16(X):
    n = len(X)
    out = np.zeros((n, 256), dtype=np.float32)
    for i in range(n):
        img = X[i]
        gray = (0.299 * img[:, :, 0].astype(np.float32) +
                0.587 * img[:, :, 1].astype(np.float32) +
                0.114 * img[:, :, 2].astype(np.float32)) / 255.0
        arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
        out[i] = arr - arr.mean()
    return out


def cluster_purity(labels_true, labels_pred, n_clusters):
    purities = []
    for c in range(n_clusters):
        mask = labels_pred == c
        if mask.sum() == 0:
            continue
        counts = np.bincount(labels_true[mask], minlength=100)
        purities.append(counts.max() / mask.sum())
    return float(np.mean(purities)), purities


def nmi(labels_true, labels_pred):
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')


def run_k(feats, labels, k):
    from sklearn.cluster import MiniBatchKMeans
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3,
                         max_iter=100, batch_size=512)
    km.fit(feats)
    assigned = km.labels_
    purity_mean, purities = cluster_purity(labels, assigned, k)
    n_pure = sum(p > 0.5 for p in purities)
    score_nmi = nmi(labels, assigned)
    elapsed = time.time() - t0
    print(f"  k={k:5d}: purity={purity_mean:.4f}  NMI={score_nmi:.4f}  "
          f"n_pure>50%={n_pure:4d}/{k}  {elapsed:.1f}s", flush=True)
    return purity_mean, score_nmi, n_pure


def main():
    t_total = time.time()
    print("Step 510: Centroid count sweep on CIFAR-100 (avgpool16+centered)", flush=True)

    print("\nLoading CIFAR-100...", flush=True)
    import torchvision
    ds = torchvision.datasets.CIFAR100('./data/cifar100', train=False, download=True)
    X = np.array(ds.data)
    y = np.array(ds.targets)
    print(f"  {len(X)} test images, {len(set(y))} classes", flush=True)

    print("\nEncoding (avgpool16+centered)...", flush=True)
    t0 = time.time()
    feats = encode_avgpool16(X)
    print(f"  {feats.shape} in {time.time()-t0:.1f}s", flush=True)

    print("\nK-means sweep...", flush=True)
    results = []
    for k in K_VALUES:
        r = run_k(feats, y, k)
        results.append((k, r[0], r[1], r[2]))

    print(f"\n{'='*60}", flush=True)
    print("STEP 510 SUMMARY", flush=True)
    print(f"  {'k':>6}  {'purity':>8}  {'NMI':>8}  {'n_pure>50%':>12}", flush=True)
    for k, purity, score_nmi, n_pure in results:
        print(f"  {k:>6}  {purity:>8.4f}  {score_nmi:>8.4f}  {n_pure:>6d}/{k}", flush=True)

    nmis = [r[2] for r in results]
    best_k_idx = int(np.argmax(nmis))
    best_k, best_purity, best_nmi, best_npure = results[best_k_idx]

    print(f"\nVERDICT:", flush=True)
    if best_nmi < 0.05:
        print(f"  NMI flat near 0 at all k. Encoding fundamentally cannot separate CIFAR-100.", flush=True)
        print(f"  No centroid count recovers class signal. Learned repr is required.", flush=True)
    elif best_nmi < 0.15:
        print(f"  Weak class signal at k={best_k} (NMI={best_nmi:.4f}). Marginal separation.", flush=True)
        print(f"  Fixed encoding has trace class info but not enough for useful classification.", flush=True)
    else:
        print(f"  Class signal found at k={best_k} (NMI={best_nmi:.4f}, purity={best_purity:.4f}).", flush=True)
        print(f"  Encoding DOES separate classes at this resolution.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
