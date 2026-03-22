#!/usr/bin/env python3
"""
Step 423 — ReadIsWrite with V-derived tau on P-MNIST (10 tasks, CL).

R3 hypothesis: Can tau be V-derived (M rather than U) to reduce frozen frame?
tau = 1 / sqrt(codebook_size). Small codebook -> soft attention. Large -> sharp.
Base: same as 418d but tau=None (computed per-step). Ran March 18 2026 as inline exec.

Result: avg=91.03%, max_forget=0.68pp. Target >91%: BARELY MISS.
V-derived is 4th of 4 (worse than fixed tau=0.01 at 91.84%, baseline at 91.2%).
FINDING: V-derived tau introduces forgetting (0.68pp). Early entries learned under
tau=0.015 (small cb) are mis-recalled when tau drops to 0.005 (large cb). tau IS U.

Recovery: reconstructed from Bash exec command in ee4c4227:L3394.
"""

import time, math, random
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_OUT = 384
N_CLASSES = 10
N_TASKS = 10
N_TRAIN_TASK = 5000
SEED = 42


class ReadIsWriteClassifier:
    def __init__(self, d, tau=None, spawn_thresh=0.9, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.tau = tau  # None = V-derived
        self.spawn_thresh = spawn_thresh
        self.device = device

    def step(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            tgt = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([tgt], device=self.device)
            return tgt

        sims = self.V @ x
        tau = 1.0 / max(1.0, self.V.shape[0] ** 0.5); weights = F.softmax(sims / tau, dim=0)
        output = weights @ self.V

        n_cls = int(self.labels.max().item()) + 1
        cls_ids = torch.arange(n_cls, device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        scores = one_hot @ weights
        prediction = scores.argmax().item()

        error = x - output
        self.V += torch.outer(weights, error)
        self.V = F.normalize(self.V, dim=1)

        target = label if label is not None else prediction
        target_mask = (self.labels == target)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.spawn_thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([target], device=self.device)])
        return prediction

    def predict_frozen(self, x):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            return 0
        sims = self.V @ x
        tau = 1.0 / max(1.0, self.V.shape[0] ** 0.5); weights = F.softmax(sims / tau, dim=0)
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


def eval_frozen(sub, R_te, y_te):
    correct = 0
    for i in range(len(R_te)):
        pred = sub.predict_frozen(R_te[i])
        if pred == int(y_te[i]):
            correct += 1
    return correct / len(R_te) * 100


def main():
    print(f"Step 423: ReadIsWrite CL on P-MNIST (10 tasks)")
    print(f"Device: {DEVICE}  tau=V-derived  thresh=0.9  frozen eval", flush=True)

    X_tr, y_tr, X_te, y_te = load_mnist()
    P = make_projection()

    sub = ReadIsWriteClassifier(d=D_OUT, tau=None, spawn_thresh=0.9)

    perms = [make_permutation(seed=SEED + t) for t in range(N_TASKS)]

    task_test_data = []
    for t in range(N_TASKS):
        R_te_t = embed(X_te, perms[t], P)
        task_test_data.append(R_te_t)

    acc_matrix = np.zeros((N_TASKS, N_TASKS))

    t0 = time.time()
    rng = np.random.RandomState(SEED)

    for t in range(N_TASKS):
        perm = perms[t]
        R_tr_t = embed(X_tr, perm, P)
        indices = rng.choice(len(X_tr), N_TRAIN_TASK, replace=False)

        t_start = time.time()
        for idx in indices:
            sub.step(R_tr_t[idx], label=int(y_tr[idx]))
        train_time = time.time() - t_start

        for e in range(t + 1):
            acc_matrix[e][t] = eval_frozen(sub, task_test_data[e], y_te)

        current_avg = np.mean([acc_matrix[e][t] for e in range(t + 1)])
        print(f"  Task {t}: cb={sub.V.shape[0]:5d}  avg_acc={current_avg:.1f}%  "
              f"task_acc={acc_matrix[t][t]:.1f}%  train={train_time:.1f}s", flush=True)

    total_time = time.time() - t0

    final_accs = [acc_matrix[t][N_TASKS - 1] for t in range(N_TASKS)]
    peak_accs = [max(acc_matrix[t][t:]) for t in range(N_TASKS)]
    forgetting = [peak_accs[t] - final_accs[t] for t in range(N_TASKS)]

    avg_acc = np.mean(final_accs)
    max_forget = max(forgetting)
    avg_forget = np.mean(forgetting)

    print(f"\n{'='*60}")
    print("STEP 423 RESULTS — V-derived tau CL")
    print(f"{'='*60}")
    print(f"{'Task':<6} {'Final':>8} {'Peak':>8} {'Forget':>8}")
    print(f"{'-'*35}")
    for t in range(N_TASKS):
        print(f"{t:<6} {final_accs[t]:>7.1f}% {peak_accs[t]:>7.1f}% {forgetting[t]:>7.1f}pp")

    print(f"\nAverage accuracy: {avg_acc:.2f}%  (kill<89%: {'KILL' if avg_acc < 89 else 'PASS'})")
    print(f"Max forgetting: {max_forget:.2f}pp")
    print(f"Target >91%: {'PASS' if avg_acc > 91 else 'MISS'}")
    print(f"Codebook size: {sub.V.shape[0]}")
    print(f"Total time: {total_time:.0f}s")


if __name__ == '__main__':
    main()
