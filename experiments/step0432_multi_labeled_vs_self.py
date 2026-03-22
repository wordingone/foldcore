#!/usr/bin/env python3
"""
Step 432 — Labeled vs self-generated labels on P-MNIST.
A: External labels (label=y_tr). B: Self-labels (label=prediction).
Same process_novelty + softmax voting (tau=0.01). 10 tasks, frozen eval.
"""

import time, math, random
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_OUT = 384; N_CLASSES = 10; N_TASKS = 10; N_TRAIN = 5000; SEED = 42


class PNSoftmax:
    def __init__(self, d, tau=0.01, thresh=0.9, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.tau = tau; self.thresh = thresh; self.device = device

    def step(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            tgt = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([tgt], device=self.device)
            return tgt
        sims = self.V @ x
        weights = F.softmax(sims / self.tau, dim=0)
        n_cls = int(self.labels.max().item()) + 1
        cls_ids = torch.arange(n_cls, device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        scores = one_hot @ weights
        prediction = scores.argmax().item()
        # Attract (winner-take-all)
        target = label if label is not None else prediction
        target_mask = (self.labels == target)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([target], device=self.device)])
        else:
            ts = sims.clone(); ts[~target_mask] = -float('inf')
            w = ts.argmax().item()
            a = 1.0 - float(sims[w].item())
            self.V[w] = F.normalize(self.V[w] + a * (x - self.V[w]), dim=0)
        return prediction

    def predict_frozen(self, x):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0: return 0
        sims = self.V @ x
        weights = F.softmax(sims / self.tau, dim=0)
        n_cls = int(self.labels.max().item()) + 1
        cls_ids = torch.arange(n_cls, device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        return (one_hot @ weights).argmax().item()


def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST('./data/mnist', train=True, download=True)
    te = torchvision.datasets.MNIST('./data/mnist', train=False, download=True)
    return (tr.data.numpy().reshape(-1,784).astype(np.float32)/255.0, tr.targets.numpy(),
            te.data.numpy().reshape(-1,784).astype(np.float32)/255.0, te.targets.numpy())

def make_proj(seed=12345):
    rng = np.random.RandomState(seed)
    return torch.from_numpy(rng.randn(D_OUT, 784).astype(np.float32) / math.sqrt(784)).to(DEVICE)

def make_perm(seed):
    perm = list(range(784)); random.Random(seed).shuffle(perm); return perm

def embed(X, perm, P):
    return F.normalize(torch.from_numpy(X[:, perm]).to(DEVICE) @ P.T, dim=1)

def eval_frozen(sub, R_te, y_te):
    c = sum(1 for i in range(len(R_te)) if sub.predict_frozen(R_te[i]) == int(y_te[i]))
    return c / len(R_te) * 100


def run_mode(mode_name, use_external_labels, X_tr, y_tr, X_te, y_te, P):
    sub = PNSoftmax(d=D_OUT)
    perms = [make_perm(seed=SEED+t) for t in range(N_TASKS)]
    task_test = [embed(X_te, perms[t], P) for t in range(N_TASKS)]
    acc_matrix = np.zeros((N_TASKS, N_TASKS))
    rng = np.random.RandomState(SEED)
    t0 = time.time()

    for t in range(N_TASKS):
        R_tr_t = embed(X_tr, perms[t], P)
        indices = rng.choice(len(X_tr), N_TRAIN, replace=False)
        for idx in indices:
            if use_external_labels:
                sub.step(R_tr_t[idx], label=int(y_tr[idx]))
            else:
                sub.step(R_tr_t[idx], label=None)  # self-generated
        for e in range(t+1):
            acc_matrix[e][t] = eval_frozen(sub, task_test[e], y_te)
        avg = np.mean([acc_matrix[e][t] for e in range(t+1)])
        print(f"  {mode_name} Task {t}: cb={sub.V.shape[0]:5d}  avg={avg:.1f}%", flush=True)

    elapsed = time.time() - t0
    final = [acc_matrix[t][N_TASKS-1] for t in range(N_TASKS)]
    peak = [max(acc_matrix[t][t:]) for t in range(N_TASKS)]
    forget = [peak[t] - final[t] for t in range(N_TASKS)]
    return np.mean(final), max(forget), sub.V.shape[0], elapsed


def main():
    print(f"Step 432: Labeled vs self-generated labels")
    print(f"Device: {DEVICE}", flush=True)

    X_tr, y_tr, X_te, y_te = load_mnist()
    P = make_proj()

    print("\n--- Mode A: External labels ---", flush=True)
    avg_a, mf_a, cb_a, t_a = run_mode("A", True, X_tr, y_tr, X_te, y_te, P)

    print("\n--- Mode B: Self-generated labels ---", flush=True)
    avg_b, mf_b, cb_b, t_b = run_mode("B", False, X_tr, y_tr, X_te, y_te, P)

    print(f"\n{'='*60}")
    print("STEP 432 RESULTS")
    print(f"{'='*60}")
    print(f"Mode A (external labels): {avg_a:.2f}%  forget={mf_a:.2f}pp  cb={cb_a}  {t_a:.0f}s")
    print(f"Mode B (self-labels):     {avg_b:.2f}%  forget={mf_b:.2f}pp  cb={cb_b}  {t_b:.0f}s")
    print(f"Gap: {avg_a - avg_b:.2f}pp (expected ~2.6pp)")


if __name__ == '__main__':
    main()
