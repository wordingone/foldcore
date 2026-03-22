#!/usr/bin/env python3
"""
Step 512 -- CIFAR threshold sweep.
Find the spawn threshold that gives best NMI on CIFAR-100 (avgpool16+centered).

threshold = [0.5, 1.0, 1.5, 2.0, 3.0]
For each: 1-pass 10K CIFAR images, measure centroid count + NMI + purity.

Context:
  CIFAR L2 mean=4.309, min=2.229 (Step 507)
  threshold=0.3 -> ~10K centroids (one per image), NMI unmeasured at this scale
  direct k=100 -> NMI=0.188 (Step 510 baseline)
  direct k=1000 -> NMI=0.344
"""
import time
import numpy as np

THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0]


def encode_avgpool16(img):
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32) +
            0.587 * img[:, :, 1].astype(np.float32) +
            0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


def dynamic_growth_pass(X, threshold):
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
            if dists[nearest] > threshold:
                centroids = np.vstack([centroids, x.reshape(1, -1)])
                assignments[i] = len(centroids) - 1
            else:
                assignments[i] = nearest
    return centroids, assignments


def measure(centroids, assignments, labels):
    from sklearn.metrics import normalized_mutual_info_score
    n_c = len(centroids)
    purities = []
    for c in range(n_c):
        mask = assignments == c
        if mask.sum() == 0:
            continue
        counts = np.bincount(labels[mask], minlength=100)
        purities.append(counts.max() / mask.sum())
    purity = float(np.mean(purities)) if purities else 0.0
    n_pure = sum(p > 0.5 for p in purities)
    score_nmi = normalized_mutual_info_score(labels, assignments, average_method='arithmetic')
    return purity, score_nmi, n_pure


def run_threshold(X, y, threshold):
    t0 = time.time()
    centroids, assignments = dynamic_growth_pass(X, threshold)
    n_c = len(centroids)
    purity, score_nmi, n_pure = measure(centroids, assignments, y)
    elapsed = time.time() - t0
    print(f"  threshold={threshold:.1f}: centroids={n_c:6d}  NMI={score_nmi:.4f}  "
          f"purity={purity:.4f}  n_pure>50%={n_pure:5d}/{n_c}  {elapsed:.1f}s", flush=True)
    return n_c, purity, score_nmi, n_pure


def main():
    t_total = time.time()
    print("Step 512: CIFAR-100 spawn threshold sweep (avgpool16+centered)", flush=True)
    print(f"thresholds={THRESHOLDS}  (CIFAR L2 mean~4.3, ARC L2 mean~0.5)", flush=True)

    print("\nLoading CIFAR-100...", flush=True)
    import torchvision
    ds = torchvision.datasets.CIFAR100('./data/cifar100', train=False, download=True)
    X = np.array(ds.data)
    y = np.array(ds.targets)
    print(f"  {len(X)} test images, {len(set(y))} classes", flush=True)

    print("\nThreshold sweep...", flush=True)
    results = []
    for th in THRESHOLDS:
        r = run_threshold(X, y, th)
        results.append((th, r[0], r[1], r[2], r[3]))

    print(f"\n{'='*60}", flush=True)
    print("STEP 512 SUMMARY", flush=True)
    print(f"  Baselines (Step 510): direct k=100 NMI=0.188 | direct k=1000 NMI=0.344", flush=True)
    print(f"\n  {'threshold':>10}  {'centroids':>10}  {'NMI':>8}  {'purity':>8}  {'n_pure>50%':>12}", flush=True)
    for th, n_c, purity, score_nmi, n_pure in results:
        print(f"  {th:>10.1f}  {n_c:>10d}  {score_nmi:>8.4f}  {purity:>8.4f}  "
              f"{n_pure:>6d}/{n_c}", flush=True)

    nmis = [r[3] for r in results]
    best_idx = int(np.argmax(nmis))
    best_th, best_nc, best_purity, best_nmi, best_npure = results[best_idx]

    print(f"\nVERDICT:", flush=True)
    print(f"  Best NMI at threshold={best_th}: NMI={best_nmi:.4f}  "
          f"centroids={best_nc}  purity={best_purity:.4f}", flush=True)
    if best_nmi > 0.25:
        print(f"  GOOD: threshold={best_th} gives strong class signal (NMI>{0.25:.2f}).", flush=True)
    elif best_nmi > 0.15:
        print(f"  MARGINAL: threshold={best_th} gives weak class signal.", flush=True)
    else:
        print(f"  FAIL: no threshold gives useful class signal.", flush=True)

    # Key chain implication
    print(f"\nChain implication:", flush=True)
    chain_thresh = [r for r in results if r[0] >= 1.5]
    if chain_thresh:
        ct = chain_thresh[0]
        print(f"  threshold={ct[0]}: {ct[1]} centroids, NMI={ct[3]:.4f}.", flush=True)
        print(f"  ARC L2 mean~0.5 -- at this threshold, ARC frames would all merge.", flush=True)
        print(f"  Domain-specific thresholds required for chain classification.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
