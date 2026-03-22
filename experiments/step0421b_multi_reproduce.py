#!/usr/bin/env python3
"""
Step 421b — Reproduce ReadIsWrite tau=0.01 P-MNIST 10-task CL.
3 different seeds. Same config as Step 421.
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


class ReadIsWriteClassifier:
    def __init__(self, d, tau=0.01, spawn_thresh=0.9, device=DEVICE):
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
        if self.V.shape[0] == 0: return 0
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
        if sub.predict_frozen(R_te[i]) == int(y_te[i]): correct += 1
    return correct / len(R_te) * 100

def run_seed(seed, X_tr, y_tr, X_te, y_te, P):
    sub = ReadIsWriteClassifier(d=D_OUT, tau=0.01, spawn_thresh=0.9)
    perms = [make_permutation(seed=seed + t) for t in range(N_TASKS)]
    task_test = [embed(X_te, perms[t], P) for t in range(N_TASKS)]
    acc_matrix = np.zeros((N_TASKS, N_TASKS))
    rng = np.random.RandomState(seed)
    t0 = time.time()

    for t in range(N_TASKS):
        R_tr_t = embed(X_tr, perms[t], P)
        indices = rng.choice(len(X_tr), N_TRAIN_TASK, replace=False)
        for idx in indices:
            sub.step(R_tr_t[idx], label=int(y_tr[idx]))
        for e in range(t + 1):
            acc_matrix[e][t] = eval_frozen(sub, task_test[e], y_te)

    elapsed = time.time() - t0
    final = [acc_matrix[t][N_TASKS-1] for t in range(N_TASKS)]
    peak = [max(acc_matrix[t][t:]) for t in range(N_TASKS)]
    forget = [peak[t] - final[t] for t in range(N_TASKS)]
    return np.mean(final), max(forget), sub.V.shape[0], elapsed


def main():
    print(f"Step 421b: Reproduce tau=0.01 CL across 3 seeds")
    print(f"Device: {DEVICE}", flush=True)

    X_tr, y_tr, X_te, y_te = load_mnist()
    P = make_projection()

    seeds = [100, 200, 300]
    results = []

    for s in seeds:
        avg, mf, cb, elapsed = run_seed(s, X_tr, y_tr, X_te, y_te, P)
        results.append((s, avg, mf, cb, elapsed))
        print(f"  Seed {s}: avg={avg:.2f}%  max_forget={mf:.2f}pp  cb={cb}  {elapsed:.0f}s", flush=True)

    avgs = [r[1] for r in results]
    forgets = [r[2] for r in results]
    mean_acc = np.mean(avgs)
    std_acc = np.std(avgs)

    print(f"\n{'='*60}")
    print("STEP 421b REPRODUCTION RESULTS")
    print(f"{'='*60}")
    for s, avg, mf, cb, el in results:
        print(f"  Seed {s}: {avg:.2f}% avg, {mf:.2f}pp forget, cb={cb}")
    print(f"\nMean: {mean_acc:.2f}%  Std: {std_acc:.2f}pp")
    print(f"Max forgetting across seeds: {max(forgets):.2f}pp")
    print(f"Original (seed 42): 91.90%")

    kills = []
    if any(a < 89 for a in avgs): kills.append(f"seed < 89%")
    if std_acc > 2: kills.append(f"std={std_acc:.2f}>2pp")
    if kills:
        print(f"KILL: {', '.join(kills)}")
    else:
        print("PASS: all seeds > 89%, std < 2pp. Result is REPRODUCIBLE.")


if __name__ == '__main__':
    main()
