#!/usr/bin/env python3
"""
Step 425 — process_novelty with softmax voting (not top-K). Isolate 91.84% cause.
Winner-take-all attract. V-derived thresh. Only scoring changes.
P-MNIST 10 tasks, 5K/task, frozen eval, tau=0.01.
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


class ProcessNoveltySoftmaxVote:
    """process_novelty with softmax voting instead of top-K."""
    def __init__(self, d, tau=0.01, spawn_thresh=0.9, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.tau = tau
        self.spawn_thresh = spawn_thresh
        self.device = device

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        self.thresh = float(G.max(dim=1).values.median())

    def step(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            tgt = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([tgt], device=self.device)
            self.thresh = 0.7
            return tgt

        sims = self.V @ x

        # SOFTMAX VOTING (the change)
        weights = F.softmax(sims / self.tau, dim=0)
        n_cls = int(self.labels.max().item()) + 1
        cls_ids = torch.arange(n_cls, device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        scores = one_hot @ weights
        prediction = scores.argmax().item()

        # WINNER-TAKE-ALL attract (unchanged from process_novelty)
        target = label if label is not None else prediction
        target_mask = (self.labels == target)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.spawn_thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([target], device=self.device)])
            pass  # spawn_thresh is fixed, no Gram needed
        else:
            ts = sims.clone(); ts[~target_mask] = -float('inf')
            winner = ts.argmax().item()
            alpha = 1.0 - float(sims[winner].item())
            self.V[winner] = F.normalize(self.V[winner] + alpha * (x - self.V[winner]), dim=0)
            self._update_thresh()

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
    tr = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=True, download=True)
    te = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=False, download=True)
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


def main():
    print(f"Step 425: process_novelty + softmax voting (tau=0.01)")
    print(f"Device: {DEVICE}  Winner-take-all attract. Frozen eval.", flush=True)

    X_tr, y_tr, X_te, y_te = load_mnist()
    P = make_projection()
    sub = ProcessNoveltySoftmaxVote(d=D_OUT, tau=0.01, spawn_thresh=0.9)
    perms = [make_permutation(seed=42 + t) for t in range(N_TASKS)]
    task_test = [embed(X_te, perms[t], P) for t in range(N_TASKS)]
    acc_matrix = np.zeros((N_TASKS, N_TASKS))
    rng = np.random.RandomState(42)
    t0 = time.time()

    for t in range(N_TASKS):
        R_tr_t = embed(X_tr, perms[t], P)
        indices = rng.choice(len(X_tr), N_TRAIN_TASK, replace=False)
        for idx in indices:
            sub.step(R_tr_t[idx], label=int(y_tr[idx]))
        for e in range(t + 1):
            acc_matrix[e][t] = eval_frozen(sub, task_test[e], y_te)
        avg = np.mean([acc_matrix[e][t] for e in range(t + 1)])
        print(f"  Task {t}: cb={sub.V.shape[0]:5d}  avg={avg:.1f}%  task={acc_matrix[t][t]:.1f}%", flush=True)

    total_time = time.time() - t0
    final = [acc_matrix[t][N_TASKS-1] for t in range(N_TASKS)]
    peak = [max(acc_matrix[t][t:]) for t in range(N_TASKS)]
    forget = [peak[t] - final[t] for t in range(N_TASKS)]
    avg_acc = np.mean(final)
    max_forget = max(forget)

    print(f"\n{'='*60}")
    print("STEP 425 RESULTS")
    print(f"{'='*60}")
    for t in range(N_TASKS):
        print(f"  Task {t}: {final[t]:.1f}%  peak={peak[t]:.1f}%  forget={forget[t]:.1f}pp")
    print(f"\nAvg: {avg_acc:.2f}%  Max forget: {max_forget:.2f}pp  cb={sub.V.shape[0]}  {total_time:.0f}s")
    print(f"\nBaselines: process_novelty=91.2%/0pp  ReadIsWrite=91.84%/0.04pp")
    if avg_acc > 91.2:
        print(f"RESULT: softmax voting IMPROVES (+{avg_acc-91.2:.2f}pp). R2 not needed.")
    elif avg_acc > 90:
        print(f"RESULT: softmax voting neutral (~{avg_acc:.1f}%). Distributed update caused the improvement.")
    else:
        print(f"RESULT: softmax voting HURTS ({avg_acc:.1f}% < 91.2%). Don't mix.")


if __name__ == '__main__':
    main()
