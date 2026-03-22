#!/usr/bin/env python3
"""
Step 526 -- LSH classification diagnostic (re-benchmark Step 432).

Encode 10K CIFAR-100 via avgpool16+centered (256D). Hash via LSH k=12.
Compute NMI(cell_ids, true_labels) and purity.

Compare to:
- Codebook NMI=0.42 at threshold=3.0 (Step 512)
- K-means k=100: NMI=0.188 (Step 510)
- K-means k=1000: NMI=0.344 (Step 510)

Prediction: NMI ~0.10 (random hyperplanes don't respect class boundaries).
Kill: NMI > 0.30.
"""
import time, logging
import numpy as np
from collections import defaultdict
logging.getLogger().setLevel(logging.WARNING)

K = 12
N_IMAGES = 10_000


def encode_cifar(img):
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32) +
            0.587 * img[:, :, 1].astype(np.float32) +
            0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


def main():
    t0 = time.time()
    print("Step 526: LSH classification diagnostic (NMI re-benchmark)", flush=True)
    print(f"k={K}  n_images={N_IMAGES}", flush=True)

    import torchvision
    from sklearn.metrics import normalized_mutual_info_score

    ds = torchvision.datasets.CIFAR100('./data/cifar100',
                                        train=False, download=True)
    X = np.array(ds.data[:N_IMAGES])
    y = np.array(ds.targets[:N_IMAGES])

    # Build LSH (same seed convention as navigation experiments)
    rng = np.random.RandomState(9999)   # seed=0+9999
    H = rng.randn(K, 256).astype(np.float32)
    powers = np.array([1 << i for i in range(K)], dtype=np.int64)

    print(f"\nEncoding {N_IMAGES} CIFAR images...", flush=True)
    cells = np.zeros(N_IMAGES, dtype=np.int32)
    for i in range(N_IMAGES):
        x = encode_cifar(X[i])
        bits = (H @ x > 0).astype(np.int64)
        cells[i] = int(np.dot(bits, powers))

    n_occupied = len(np.unique(cells))
    print(f"  Occupied cells: {n_occupied}/{2**K}", flush=True)

    # Cell size stats
    cell_counts = defaultdict(int)
    for c in cells:
        cell_counts[c] += 1
    counts = list(cell_counts.values())
    print(f"  Images/cell: min={min(counts)}  max={max(counts)}  "
          f"mean={np.mean(counts):.1f}  median={np.median(counts):.1f}", flush=True)

    # NMI
    nmi = normalized_mutual_info_score(y, cells)
    print(f"\n  NMI(cells, true_labels) = {nmi:.4f}", flush=True)

    # Purity: for each cell, fraction that is majority class
    cell_class_counts = defaultdict(lambda: defaultdict(int))
    for i in range(N_IMAGES):
        cell_class_counts[cells[i]][y[i]] += 1

    purity_vals = []
    majority_frac = []
    for cell, class_dict in cell_class_counts.items():
        total = sum(class_dict.values())
        majority = max(class_dict.values())
        purity_vals.append(majority / total)
        majority_frac.append(majority / total)
    weighted_purity = sum(max(d.values()) for d in cell_class_counts.values()) / N_IMAGES
    pure_cells = sum(1 for v in purity_vals if v == 1.0)
    majority_pure = sum(1 for v in purity_vals if v > 0.5)

    print(f"  Weighted purity: {weighted_purity:.4f}", flush=True)
    print(f"  Pure cells (100% same class): {pure_cells}/{n_occupied}", flush=True)
    print(f"  Majority cells (>50% same class): {majority_pure}/{n_occupied}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("STEP 526 SUMMARY", flush=True)
    print(f"  LSH k=12:         NMI={nmi:.4f}  purity={weighted_purity:.4f}", flush=True)
    print(f"  Codebook (S512):  NMI~0.42 at threshold=3.0 (2701 centroids)", flush=True)
    print(f"  K-means k=100:    NMI=0.188 (Step 510)", flush=True)
    print(f"  K-means k=1000:   NMI=0.344 (Step 510)", flush=True)
    print(f"  Occupied cells: {n_occupied}/4096  ~{N_IMAGES//n_occupied} imgs/cell", flush=True)

    print(f"\nVERDICT:", flush=True)
    if nmi > 0.30:
        print(f"  UNEXPECTED: NMI={nmi:.4f} > 0.30. LSH cells capture class structure.",
              flush=True)
        print(f"  Random projections are NOT entirely class-blind.", flush=True)
    elif nmi > 0.10:
        print(f"  PARTIAL: NMI={nmi:.4f} in 0.10-0.30 range. Some class signal.",
              flush=True)
        print(f"  LSH cells partially respect class boundaries via encoding structure.", flush=True)
    else:
        print(f"  CONFIRMED: NMI={nmi:.4f} ~0.10 or below. LSH cells are class-blind.",
              flush=True)
        print(f"  Random hyperplanes don't respect CIFAR class boundaries.", flush=True)
        print(f"  Confirms: encoding-is-the-bottleneck conclusion stands.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
