#!/usr/bin/env python3
"""
Step 418d — ReadIsWriteClassifier on full P-MNIST (10 tasks, continual learning).
tau=0.1, thresh=0.9, frozen eval. 5K steps per task.
Kill: avg <70% or forgetting >5pp.
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
    def __init__(self, d, tau=0.1, spawn_thresh=0.9, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.tau = tau
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
        weights = F.softmax(sims / self.tau, dim=0)
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


def eval_frozen(sub, R_te, y_te):
    correct = 0
    for i in range(len(R_te)):
        pred = sub.predict_frozen(R_te[i])
        if pred == int(y_te[i]):
            correct += 1
    return correct / len(R_te) * 100


def main():
    print(f"Step 418d: ReadIsWrite CL on P-MNIST (10 tasks)")
    print(f"Device: {DEVICE}  tau=0.1  thresh=0.9  frozen eval", flush=True)

    X_tr, y_tr, X_te, y_te = load_mnist()
    P = make_projection()

    sub = ReadIsWriteClassifier(d=D_OUT, tau=0.1, spawn_thresh=0.9)

    # Per-task permutations
    perms = [make_permutation(seed=SEED + t) for t in range(N_TASKS)]

    # Store per-task test data for forgetting measurement
    task_test_data = []
    for t in range(N_TASKS):
        R_te_t = embed(X_te, perms[t], P)
        task_test_data.append(R_te_t)

    # Accuracy after each task's training (for forgetting)
    acc_matrix = np.zeros((N_TASKS, N_TASKS))  # acc_matrix[eval_task][after_train_task]

    t0 = time.time()
    rng = np.random.RandomState(SEED)

    for t in range(N_TASKS):
        # Train on task t
        perm = perms[t]
        R_tr_t = embed(X_tr, perm, P)
        indices = rng.choice(len(X_tr), N_TRAIN_TASK, replace=False)

        t_start = time.time()
        for idx in indices:
            sub.step(R_tr_t[idx], label=int(y_tr[idx]))
        train_time = time.time() - t_start

        # Eval ALL tasks seen so far (frozen)
        for e in range(t + 1):
            acc_matrix[e][t] = eval_frozen(sub, task_test_data[e], y_te)

        current_avg = np.mean([acc_matrix[e][t] for e in range(t + 1)])
        print(f"  Task {t}: cb={sub.V.shape[0]:5d}  avg_acc={current_avg:.1f}%  "
              f"task_acc={acc_matrix[t][t]:.1f}%  train={train_time:.1f}s", flush=True)

    total_time = time.time() - t0

    # Final metrics
    final_accs = [acc_matrix[t][N_TASKS - 1] for t in range(N_TASKS)]
    peak_accs = [max(acc_matrix[t][t:]) for t in range(N_TASKS)]
    forgetting = [peak_accs[t] - final_accs[t] for t in range(N_TASKS)]

    avg_acc = np.mean(final_accs)
    max_forget = max(forgetting)
    avg_forget = np.mean(forgetting)

    print(f"\n{'='*60}")
    print("STEP 418d RESULTS — Continual Learning")
    print(f"{'='*60}")
    print(f"{'Task':<6} {'Final':>8} {'Peak':>8} {'Forget':>8}")
    print(f"{'-'*35}")
    for t in range(N_TASKS):
        print(f"{t:<6} {final_accs[t]:>7.1f}% {peak_accs[t]:>7.1f}% {forgetting[t]:>7.1f}pp")

    print(f"\nAverage accuracy: {avg_acc:.2f}%  (baseline=91.2%)")
    print(f"Max forgetting: {max_forget:.2f}pp  (baseline=0pp)")
    print(f"Avg forgetting: {avg_forget:.2f}pp")
    print(f"Codebook size: {sub.V.shape[0]}")
    print(f"Total time: {total_time:.0f}s")
    print(f"\nKill: avg<70%={'KILL' if avg_acc < 70 else 'PASS'}  forget>5pp={'KILL' if max_forget > 5 else 'PASS'}")


if __name__ == '__main__':
    main()
