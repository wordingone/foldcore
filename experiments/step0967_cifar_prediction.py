"""Step 967: W_pred classification — prediction IS classification (capacity test).

FAMILY: Predictive coding (Rao & Ballard 1999 — prediction error minimizes to classify)

R3 HYPOTHESIS: W_pred @ [h, one_hot(label)] predicts enc for that class. At
test time: argmin_c ||enc - W_pred @ [h, one_hot(c)]|| = classification. Same
W_pred used for navigation. Genuinely one mechanism for all tasks.

CAPACITY TEST: Does linear W_pred on avgpool16 features classify CIFAR-100 at
>10% (10× chance)? Yes → chain can use same substrate for CIFAR. No → need deeper model.

Training: update W_pred with TRUE label each step. h carries trajectory context.
Input: [h (64D), one_hot(label) (100D)] = 164D → W_pred (256, 164) → pred enc (256D)

Kill: accuracy ≤ 2% (chance or below). Pass: >10%. Baseline: 1% random.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
N_CLASSES = 100
ETA_W = 0.01
TEST_SEEDS = list(range(1, 6))
CIFAR_STEPS = 5_000   # 5K images per seed (out of 10K test set)


def one_hot(c, n):
    v = np.zeros(n, dtype=np.float32)
    v[c] = 1.0
    return v


class CIFARPredictor967:
    """W_pred classifies via prediction error minimization."""

    def __init__(self, seed):
        rs = np.random.RandomState(seed + 10000)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        # W_pred predicts enc from [h, one_hot(label)]
        self.W_pred = np.zeros((ENC_DIM, H_DIM + N_CLASSES), dtype=np.float32)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._prev_h = None
        self._prev_label = None

    def _encode(self, img):
        enc_raw = _enc_frame(np.asarray(img, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        return enc

    def classify(self, img, true_label):
        """Process image, update W_pred with true label, return predicted label."""
        enc = self._encode(img)

        if self._prev_h is not None and self._prev_label is not None:
            pred_input = np.concatenate([self._prev_h, one_hot(self._prev_label, N_CLASSES)])
            pred = self.W_pred @ pred_input
            error = enc - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error *= (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W_pred += ETA_W * np.outer(error, pred_input)

        # Classify: for each candidate class c, compute ||enc - W_pred @ [h, one_hot(c)]||
        best_c = 0
        best_err = float('inf')
        for c in range(N_CLASSES):
            pred_input_c = np.concatenate([self.h, one_hot(c, N_CLASSES)])
            pred_c = self.W_pred @ pred_input_c
            err_c = float(np.linalg.norm(enc - pred_c))
            if err_c < best_err:
                best_err = err_c
                best_c = c

        self._prev_h = self.h.copy()
        self._prev_label = true_label  # train with TRUE label
        return best_c


def load_cifar():
    try:
        import torchvision, torchvision.transforms as T
        ds = torchvision.datasets.CIFAR100('B:/M/the-search/data', train=False,
                                            download=True, transform=T.ToTensor())
        imgs = np.array([np.array(ds[i][0]).transpose(1, 2, 0)
                         for i in range(len(ds))], dtype=np.float32)
        lbls = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
        return imgs, lbls
    except Exception as e:
        print(f"  CIFAR load failed: {e}"); return None, None


if __name__ == "__main__":
    import time
    print("=" * 70)
    print("STEP 967 — W_pred CLASSIFICATION (prediction IS classification)")
    print("=" * 70)
    t0 = time.time()
    print(f"W_pred: ({ENC_DIM}, {H_DIM+N_CLASSES})  CIFAR_STEPS={CIFAR_STEPS}  Seeds={TEST_SEEDS}")
    print(f"Baseline: chance=1.0%  Old codebook P-MNIST: 94.48%")
    print()

    imgs, lbls = load_cifar()
    if imgs is None:
        print("CIFAR load failed — abort")
        exit(1)

    results = []
    for seed in TEST_SEEDS:
        sub = CIFARPredictor967(seed=seed)
        rng = np.random.RandomState(seed * 1000)
        idx = rng.permutation(len(imgs))[:CIFAR_STEPS]
        correct = 0
        for i in idx:
            pred_label = sub.classify(imgs[i], int(lbls[i]))
            if pred_label == lbls[i]:
                correct += 1
        acc = correct / CIFAR_STEPS
        results.append(acc)
        print(f"  seed={seed}: accuracy={acc:.4f} ({correct}/{CIFAR_STEPS})")

    mean_acc = np.mean(results)
    print()
    print("=" * 70)
    verdict = ("PASS — >10%, W_pred has capacity" if mean_acc > 0.10
               else "FAIL — ≤10%, W_pred insufficient" if mean_acc > 0.02
               else "KILL — ≤2%, no signal above chance")
    print(f"  Mean accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    print(f"  VERDICT: {verdict}")
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("STEP 967 DONE")
