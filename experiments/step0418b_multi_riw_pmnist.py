#!/usr/bin/env python3
"""
Step 418b — ReadIsWriteClassifier on P-MNIST.
1 task, 5K training steps. Evaluate on test set.
Chance=10%, baseline=91.2%. Kill <25%.
"""

import sys, time, math, random
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_OUT = 384
N_CLASSES = 10
SEED = 42


class ReadIsWriteClassifier:
    def __init__(self, d, tau=0.1, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.tau = tau
        self.device = device

    def step(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)

        if self.V.shape[0] == 0:
            tgt = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([tgt], device=self.device)
            return tgt

        sims = self.V @ x
        weights = F.softmax(sims / self.tau, dim=0)
        output = weights @ self.V

        # Classify: vectorized weighted vote
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
        if target_mask.sum() == 0 or sims[target_mask].max() < 0.9:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([target], device=self.device)])

        return prediction


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


def main():
    print(f"Step 418b: ReadIsWriteClassifier on P-MNIST")
    print(f"Device: {DEVICE}", flush=True)

    X_tr, y_tr, X_te, y_te = load_mnist()
    P = make_projection()
    perm = make_permutation(seed=SEED)

    R_tr = embed(X_tr, perm, P)
    R_te = embed(X_te, perm, P)

    sub = ReadIsWriteClassifier(d=D_OUT, tau=0.1)

    # Train: 5K steps
    n_train = 5000
    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(X_tr), n_train, replace=False)
    correct_train = 0

    t0 = time.time()
    for i, idx in enumerate(indices):
        x = R_tr[idx]
        label = int(y_tr[idx])
        pred = sub.step(x, label=label)
        if pred == label:
            correct_train += 1

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            train_acc = correct_train / (i + 1) * 100
            print(f"  [step {i+1:5d}]  cb={sub.V.shape[0]:5d}  train_acc={train_acc:.1f}%  {elapsed:.1f}s", flush=True)

    train_time = time.time() - t0
    train_acc = correct_train / n_train * 100
    print(f"\nTraining done: {train_time:.1f}s  cb={sub.V.shape[0]}  train_acc={train_acc:.1f}%")

    # Eval: full test set
    t1 = time.time()
    correct = 0
    for i in range(len(R_te)):
        x = R_te[i]
        label = int(y_te[i])
        pred = sub.step(x, label=None)  # no label during eval
        if pred == label:
            correct += 1

    eval_time = time.time() - t1
    test_acc = correct / len(R_te) * 100

    print(f"\n{'='*60}")
    print(f"STEP 418b RESULTS")
    print(f"{'='*60}")
    print(f"Test accuracy: {test_acc:.2f}%  (chance=10%, baseline=91.2%)")
    print(f"Codebook size: {sub.V.shape[0]}")
    print(f"Train time: {train_time:.1f}s  Eval time: {eval_time:.1f}s")
    print(f"Kill criterion: {'KILL (<25%)' if test_acc < 25 else 'PASS (>25%)'}")

    # Action distribution
    from collections import Counter
    label_counts = Counter(sub.labels.cpu().numpy().tolist())
    print(f"Label distribution: {dict(sorted(label_counts.items()))}")
    dom = max(label_counts.values()) / sum(label_counts.values()) * 100
    print(f"Dominant class: {dom:.0f}%")


if __name__ == '__main__':
    main()
