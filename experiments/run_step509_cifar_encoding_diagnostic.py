#!/usr/bin/env python3
"""
Step 509 -- CIFAR-100 encoding diagnostic.
Is avgpool16 the wall, or is CIFAR-100 unseparable at any fixed encoding?

3 encodings:
  1. avgpool16 + centered: 32x32 -> 16x16 -> 256D  (current chain encoding)
  2. avgpool8  + centered: 32x32 ->  8x8  ->  64D  (coarser)
  3. raw32     + centered: 32x32 full res -> 1024D  (full resolution)

For each:
  - Fit k-means n=100 and n=300
  - Cluster purity (dominant class per centroid)
  - Within-class L2 vs between-class L2 (1000 random pairs each)

Literature: SCAN (Van Gansbeke, ECCV 2020) ~50% unsupervised CIFAR-100 via learned repr.
"""
import time
import numpy as np

N_PAIR_SAMPLES = 1000


def encode_avgpool16(X):
    """32x32 RGB HWC uint8 -> 256D centered."""
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


def encode_avgpool8(X):
    """32x32 RGB HWC uint8 -> 64D centered."""
    n = len(X)
    out = np.zeros((n, 64), dtype=np.float32)
    for i in range(n):
        img = X[i]
        gray = (0.299 * img[:, :, 0].astype(np.float32) +
                0.587 * img[:, :, 1].astype(np.float32) +
                0.114 * img[:, :, 2].astype(np.float32)) / 255.0
        arr = gray.reshape(8, 4, 8, 4).mean(axis=(1, 3)).flatten()
        out[i] = arr - arr.mean()
    return out


def encode_raw32(X):
    """32x32 RGB HWC uint8 -> 1024D centered (gray flattened)."""
    n = len(X)
    out = np.zeros((n, 1024), dtype=np.float32)
    for i in range(n):
        img = X[i]
        gray = (0.299 * img[:, :, 0].astype(np.float32) +
                0.587 * img[:, :, 1].astype(np.float32) +
                0.114 * img[:, :, 2].astype(np.float32)) / 255.0
        arr = gray.flatten()
        out[i] = arr - arr.mean()
    return out


def l2_stats(feats, labels, n_samples=N_PAIR_SAMPLES, rng=None):
    """Sample within-class and between-class L2 distances."""
    if rng is None:
        rng = np.random.RandomState(0)
    n = len(feats)
    within = []
    between = []
    for _ in range(n_samples):
        i, j = rng.randint(0, n, size=2)
        d = np.sqrt(np.sum((feats[i] - feats[j]) ** 2))
        if labels[i] == labels[j]:
            within.append(d)
        else:
            between.append(d)
    return within, between


def cluster_purity(feats, labels, n_clusters):
    """Fit k-means, return mean purity across centroids."""
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3,
                         max_iter=100, batch_size=512)
    km.fit(feats)
    assignments = km.labels_
    purities = []
    for c in range(n_clusters):
        mask = assignments == c
        if mask.sum() == 0:
            continue
        cls_counts = np.bincount(labels[mask], minlength=100)
        purity = cls_counts.max() / mask.sum()
        purities.append(purity)
    return np.mean(purities), np.array(purities)


def run_encoding(name, feats, labels):
    print(f"\n  [{name}] dim={feats.shape[1]}", flush=True)
    rng = np.random.RandomState(0)

    # L2 stats (boost samples for better statistics)
    within, between = l2_stats(feats, labels, n_samples=2000, rng=rng)
    w_mean = np.mean(within) if within else 0
    b_mean = np.mean(between) if between else 0
    sep = b_mean / w_mean if w_mean > 0 else 0
    print(f"    L2 within={w_mean:.4f}  between={b_mean:.4f}  ratio={sep:.3f}x  "
          f"(n_within={len(within)} n_between={len(between)})", flush=True)

    for n_c in [100, 300]:
        t0 = time.time()
        p_mean, purities = cluster_purity(feats, labels, n_c)
        p_max = purities.max() if len(purities) > 0 else 0
        # How many centroids are >50% pure (single class dominant)?
        n_pure = (purities > 0.5).sum()
        print(f"    k={n_c}: purity_mean={p_mean:.3f}  purity_max={p_max:.3f}  "
              f"n_pure>50%={n_pure}/{n_c}  {time.time()-t0:.1f}s", flush=True)

    return w_mean, b_mean, sep


def main():
    t_total = time.time()
    print("Step 509: CIFAR-100 encoding diagnostic", flush=True)

    print("\nLoading CIFAR-100...", flush=True)
    import torchvision
    ds = torchvision.datasets.CIFAR100('C:/Users/Admin/cifar100_data', train=False, download=True)
    X = np.array(ds.data)   # [10000, 32, 32, 3] uint8
    y = np.array(ds.targets)
    print(f"  {len(X)} test images, {len(set(y))} classes", flush=True)

    print("\nComputing encodings...", flush=True)
    t0 = time.time()
    f16 = encode_avgpool16(X)
    print(f"  avgpool16: {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    f8 = encode_avgpool8(X)
    print(f"  avgpool8:  {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    f32 = encode_raw32(X)
    print(f"  raw32:     {time.time()-t0:.1f}s", flush=True)

    print("\nRunning diagnostics...", flush=True)
    r16 = run_encoding("avgpool16+centered 256D", f16, y)
    r8  = run_encoding("avgpool8+centered  64D",  f8,  y)
    r32 = run_encoding("raw32+centered    1024D", f32, y)

    print(f"\n{'='*60}", flush=True)
    print("STEP 509 SUMMARY", flush=True)
    print(f"  {'Encoding':<30} {'dim':>6}  {'within':>8}  {'between':>8}  {'ratio':>7}", flush=True)
    print(f"  {'avgpool16+centered':<30} {'256':>6}  {r16[0]:>8.4f}  {r16[1]:>8.4f}  {r16[2]:>7.3f}x", flush=True)
    print(f"  {'avgpool8+centered':<30} {'64':>6}  {r8[0]:>8.4f}  {r8[1]:>8.4f}  {r8[2]:>7.3f}x", flush=True)
    print(f"  {'raw32+centered':<30} {'1024':>6}  {r32[0]:>8.4f}  {r32[1]:>8.4f}  {r32[2]:>7.3f}x", flush=True)

    print(f"\nVERDICT:", flush=True)
    ratios = [r16[2], r8[2], r32[2]]
    if max(ratios) < 1.5:
        print("  ALL encodings: within ~= between. CIFAR-100 is unseparable at fixed encoding.", flush=True)
        print("  Mechanism (argmin) is NOT the bottleneck. Encoding is. Learned repr needed.", flush=True)
        print("  Literature baseline: SCAN ~50% uses learned features (ECCV 2020).", flush=True)
    else:
        best_i = int(np.argmax(ratios))
        names = ["avgpool16", "avgpool8", "raw32"]
        print(f"  Best encoding: {names[best_i]} (ratio={max(ratios):.3f}x)", flush=True)
        if r16[2] >= 1.5:
            print("  avgpool16 SEPARATES classes. Wall is the mechanism, not encoding.", flush=True)
        else:
            print(f"  avgpool16 does NOT separate (ratio={r16[2]:.3f}x). Current chain encoding is the wall.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
