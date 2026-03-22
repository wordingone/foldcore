#!/usr/bin/env python3
"""
Step 435 — EWC + Replay comparison on our exact P-MNIST protocol.
5K samples/task, 384D random projection, single-pass, 10 tasks.
2-layer MLP (384->256->10) with SGD.
Compare: experience replay (500 stored) vs EWC (Fisher diagonal, lambda=1000).
"""

import time, math, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_OUT = 384; N_CLASSES = 10; N_TASKS = 10; N_TRAIN = 5000; SEED = 42


class MLP(nn.Module):
    def __init__(self, d_in=384, d_hid=256, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_out)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


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

def evaluate(model, R_te, y_te):
    model.eval()
    with torch.no_grad():
        logits = model(R_te)
        preds = logits.argmax(dim=1).cpu().numpy()
    return (preds == y_te).mean() * 100


# ===== Experience Replay =====
def run_replay(X_tr, y_tr, X_te, y_te, P, buffer_size=500):
    model = MLP().to(DEVICE)
    opt = SGD(model.parameters(), lr=0.01)
    perms = [make_perm(seed=SEED+t) for t in range(N_TASKS)]
    task_test = [embed(X_te, perms[t], P) for t in range(N_TASKS)]
    y_te_np = y_te

    acc_matrix = np.zeros((N_TASKS, N_TASKS))
    rng = np.random.RandomState(SEED)
    buffer_x = []; buffer_y = []

    for t in range(N_TASKS):
        R_tr_t = embed(X_tr, perms[t], P)
        indices = rng.choice(len(X_tr), N_TRAIN, replace=False)
        X_task = R_tr_t[indices]
        Y_task = torch.from_numpy(y_tr[indices]).long().to(DEVICE)

        model.train()
        # Single pass through task data
        bs = 64
        for i in range(0, len(X_task), bs):
            xb = X_task[i:i+bs]; yb = Y_task[i:i+bs]
            # Mix with replay buffer
            if buffer_x:
                ridx = rng.choice(len(buffer_x), min(bs, len(buffer_x)), replace=False)
                rx = torch.stack([buffer_x[j] for j in ridx])
                ry = torch.tensor([buffer_y[j] for j in ridx], device=DEVICE)
                xb = torch.cat([xb, rx]); yb = torch.cat([yb, ry])
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()

        # Add to buffer (reservoir sampling)
        for i in range(len(X_task)):
            if len(buffer_x) < buffer_size:
                buffer_x.append(X_task[i]); buffer_y.append(int(Y_task[i]))
            else:
                j = rng.randint(0, t * N_TRAIN + i)
                if j < buffer_size:
                    buffer_x[j] = X_task[i]; buffer_y[j] = int(Y_task[i])

        for e in range(t+1):
            acc_matrix[e][t] = evaluate(model, task_test[e], y_te_np)

    final = [acc_matrix[t][N_TASKS-1] for t in range(N_TASKS)]
    peak = [max(acc_matrix[t][t:]) for t in range(N_TASKS)]
    forget = [peak[t] - final[t] for t in range(N_TASKS)]
    return np.mean(final), max(forget), np.mean(forget)


# ===== EWC =====
def run_ewc(X_tr, y_tr, X_te, y_te, P, lam=1000):
    model = MLP().to(DEVICE)
    opt = SGD(model.parameters(), lr=0.01)
    perms = [make_perm(seed=SEED+t) for t in range(N_TASKS)]
    task_test = [embed(X_te, perms[t], P) for t in range(N_TASKS)]
    y_te_np = y_te

    acc_matrix = np.zeros((N_TASKS, N_TASKS))
    rng = np.random.RandomState(SEED)
    fisher_params = []  # list of (fisher_diag, param_snapshot) per task

    for t in range(N_TASKS):
        R_tr_t = embed(X_tr, perms[t], P)
        indices = rng.choice(len(X_tr), N_TRAIN, replace=False)
        X_task = R_tr_t[indices]
        Y_task = torch.from_numpy(y_tr[indices]).long().to(DEVICE)

        model.train()
        bs = 64
        for i in range(0, len(X_task), bs):
            xb = X_task[i:i+bs]; yb = Y_task[i:i+bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            # EWC penalty
            ewc_loss = 0
            for fisher, old_params in fisher_params:
                for f, p_old, p in zip(fisher, old_params, model.parameters()):
                    ewc_loss += (f * (p - p_old) ** 2).sum()
            total_loss = loss + (lam / 2) * ewc_loss
            total_loss.backward(); opt.step()

        # Compute Fisher for this task
        model.eval()
        fisher = [torch.zeros_like(p) for p in model.parameters()]
        n_fisher = min(500, len(X_task))
        fidx = rng.choice(len(X_task), n_fisher, replace=False)
        for i in fidx:
            model.zero_grad()
            out = model(X_task[i:i+1])
            loss = F.cross_entropy(out, Y_task[i:i+1])
            loss.backward()
            for j, p in enumerate(model.parameters()):
                fisher[j] += p.grad.data ** 2 / n_fisher
        old_params = [p.data.clone() for p in model.parameters()]
        fisher_params.append((fisher, old_params))

        for e in range(t+1):
            acc_matrix[e][t] = evaluate(model, task_test[e], y_te_np)

    final = [acc_matrix[t][N_TASKS-1] for t in range(N_TASKS)]
    peak = [max(acc_matrix[t][t:]) for t in range(N_TASKS)]
    forget = [peak[t] - final[t] for t in range(N_TASKS)]
    return np.mean(final), max(forget), np.mean(forget)


def main():
    print(f"Step 435: EWC + Replay comparison on P-MNIST")
    print(f"Device: {DEVICE}", flush=True)

    X_tr, y_tr, X_te, y_te = load_mnist()
    P = make_proj()

    print("\n--- Experience Replay (500 buffer) ---", flush=True)
    t0 = time.time()
    avg_r, mf_r, af_r = run_replay(X_tr, y_tr, X_te, y_te, P)
    t_r = time.time() - t0
    print(f"  avg={avg_r:.2f}%  max_forget={mf_r:.2f}pp  avg_forget={af_r:.2f}pp  {t_r:.0f}s", flush=True)

    print("\n--- EWC (lambda=1000) ---", flush=True)
    t0 = time.time()
    avg_e, mf_e, af_e = run_ewc(X_tr, y_tr, X_te, y_te, P)
    t_e = time.time() - t0
    print(f"  avg={avg_e:.2f}%  max_forget={mf_e:.2f}pp  avg_forget={af_e:.2f}pp  {t_e:.0f}s", flush=True)

    print(f"\n{'='*60}")
    print("STEP 435 RESULTS")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Avg':>8} {'MaxForget':>10} {'AvgForget':>10}")
    print(f"{'-'*55}")
    print(f"{'Our substrate (softmax)':<25} {'94.48%':>8} {'0.00pp':>10} {'0.00pp':>10}")
    print(f"{'Our substrate (baseline)':<25} {'91.20%':>8} {'0.00pp':>10} {'0.00pp':>10}")
    print(f"{'Experience Replay':<25} {f'{avg_r:.2f}%':>8} {f'{mf_r:.2f}pp':>10} {f'{af_r:.2f}pp':>10}")
    print(f"{'EWC (lambda=1000)':<25} {f'{avg_e:.2f}%':>8} {f'{mf_e:.2f}pp':>10} {f'{af_e:.2f}pp':>10}")


if __name__ == '__main__':
    main()
