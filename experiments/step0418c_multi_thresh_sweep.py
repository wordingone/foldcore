#!/usr/bin/env python3
"""
Step 418c — ReadIsWriteClassifier spawn threshold sweep on P-MNIST.
Three thresholds: 0.9, 0.7, 0.5. Same tau=0.1, 1 task, 5K steps.
Eval with FROZEN codebook (no update, no spawn during eval).
"""

import time, math, random
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_OUT = 384
N_CLASSES = 10
SEED = 42


class ReadIsWriteClassifier:
    def __init__(self, d, tau=0.1, spawn_thresh=0.9, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.tau = tau
        self.spawn_thresh = spawn_thresh
        self.device = device
        self.n_spawns = 0

    def step(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)

        if self.V.shape[0] == 0:
            tgt = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([tgt], device=self.device)
            self.n_spawns += 1
            return tgt

        sims = self.V @ x
        weights = F.softmax(sims / self.tau, dim=0)
        output = weights @ self.V

        # Classify: vectorized
        n_cls = int(self.labels.max().item()) + 1
        cls_ids = torch.arange(n_cls, device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        scores = one_hot @ weights
        prediction = scores.argmax().item()

        # Update
        error = x - output
        self.V += torch.outer(weights, error)
        self.V = F.normalize(self.V, dim=1)

        # Spawn
        target = label if label is not None else prediction
        target_mask = (self.labels == target)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.spawn_thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([target], device=self.device)])
            self.n_spawns += 1

        return prediction

    def predict_frozen(self, x):
        """Classify only — no update, no spawn."""
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            return 0
        sims = self.V @ x
        weights = F.softmax(sims / self.tau, dim=0)
        n_cls = int(self.labels.max().item()) + 1
        cls_ids = torch.arange(n_cls, device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        scores = one_hot @ weights
        return scores.argmax().item()


def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST('./data/mnist', train=True, download=True)
    te = torchvision.datasets.MNIST('./data/mnist', train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_tr, tr.targets.numpy(), X_te, te.targets.numpy()


def make_projection(d_in=784, d_out=D_OUT, seed=12345):
    rng = np.random.RandomState(seed)
    P = rng.randn(d_out, d_in).astype(np.float32) / math.sqrt(d_in)
    return torch.from_numpy(P).to(DEVICE)


def make_permutation(seed):
    perm = list(range(784))
    random.Random(seed).shuffle(perm)
    return perm


def embed(X_flat_np, perm, P):
    X_t = torch.from_numpy(X_flat_np[:, perm]).to(DEVICE)
    return F.normalize(X_t @ P.T, dim=1)


def run_threshold(thresh, R_tr, y_tr, R_te, y_te, n_train=5000):
    sub = ReadIsWriteClassifier(d=D_OUT, tau=0.1, spawn_thresh=thresh)

    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(y_tr), n_train, replace=False)
    correct_train = 0

    t0 = time.time()
    for i, idx in enumerate(indices):
        pred = sub.step(R_tr[idx], label=int(y_tr[idx]))
        if pred == int(y_tr[idx]):
            correct_train += 1

    train_time = time.time() - t0
    train_acc = correct_train / n_train * 100
    cb_after_train = sub.V.shape[0]
    spawn_rate = sub.n_spawns / n_train * 100

    # Frozen eval
    t1 = time.time()
    correct = 0
    for i in range(len(R_te)):
        pred = sub.predict_frozen(R_te[i])
        if pred == int(y_te[i]):
            correct += 1
    eval_time = time.time() - t1
    test_acc = correct / len(R_te) * 100

    return {
        'thresh': thresh, 'test_acc': test_acc, 'train_acc': train_acc,
        'cb': cb_after_train, 'spawn_rate': spawn_rate,
        'train_time': train_time, 'eval_time': eval_time,
    }


def main():
    print(f"Step 418c: Spawn threshold sweep on P-MNIST")
    print(f"Device: {DEVICE}  Frozen eval.", flush=True)

    X_tr, y_tr, X_te, y_te = load_mnist()
    P = make_projection()
    perm = make_permutation(seed=SEED)
    R_tr = embed(X_tr, perm, P)
    R_te = embed(X_te, perm, P)

    results = []
    for thresh in [0.9, 0.7, 0.5]:
        print(f"\n--- thresh={thresh} ---", flush=True)
        r = run_threshold(thresh, R_tr, y_tr, R_te, y_te)
        results.append(r)
        print(f"  test={r['test_acc']:.2f}%  train={r['train_acc']:.1f}%  "
              f"cb={r['cb']}  spawn_rate={r['spawn_rate']:.0f}%  "
              f"train={r['train_time']:.1f}s  eval={r['eval_time']:.1f}s", flush=True)

    print(f"\n{'='*60}")
    print("STEP 418c SUMMARY")
    print(f"{'='*60}")
    print(f"{'thresh':<8} {'test_acc':>9} {'train_acc':>10} {'cb':>6} {'spawn%':>8}")
    print(f"{'-'*45}")
    for r in results:
        print(f"{r['thresh']:<8} {r['test_acc']:>8.2f}% {r['train_acc']:>9.1f}% {r['cb']:>6} {r['spawn_rate']:>7.0f}%")

    best = max(results, key=lambda r: r['test_acc'])
    print(f"\nBest: thresh={best['thresh']} -> test={best['test_acc']:.2f}%")
    if best['test_acc'] > 72.2:
        print(f"Tighter spawn IMPROVED generalization (+{best['test_acc']-72.2:.1f}pp)")
    else:
        print("Tighter spawn did NOT improve. Spawn selectivity is not the bottleneck.")


if __name__ == '__main__':
    main()
