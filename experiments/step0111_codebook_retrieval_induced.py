#!/usr/bin/env python3
"""
Step 111 -- Retrieval-Induced Learning. P-MNIST 10-task, raw pixels.
Spec.

Hypothesis: retrieval IS the adaptation signal. Top-k neighbors that
vote on classification also receive contrastive update (pull same-class,
push different-class). Retrieval and learning are one operation.

Sweep lr={0.001, 0.01, 0.1}. Baseline: k-NN (lr=0) = 95.4% AA (Step 110).
Kill: retrieval-induced <= 95.4%.
"""

import random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

N_TASKS       = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_TRAIN_TASK  = 6000
N_CLASSES     = 10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES
SEED          = 42
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
K             = 5
LR_VALS       = [0.0, 0.001, 0.01, 0.1]   # 0.0 = k-NN baseline


def process(V, labels, r, label=None, k=K, lr=0.0):
    r = F.normalize(r, dim=0)

    # Predict (always)
    if V.shape[0] == 0:
        use_label = label if label is not None else 0
        V = r.unsqueeze(0)
        labels = torch.tensor([use_label], device=DEVICE)
        return 0, V, labels

    sims   = V @ r
    n_cls  = int(labels.max().item()) + 1
    scores = torch.zeros(n_cls, device=DEVICE)
    for c in range(n_cls):
        cs = sims[labels == c]
        if cs.shape[0] > 0:
            scores[c] = cs.topk(min(k, cs.shape[0])).values.sum()
    prediction = int(scores.argmax().item())

    # Store input (always)
    use_label = label if label is not None else prediction
    V      = torch.cat([V, r.unsqueeze(0)])
    labels = torch.cat([labels, torch.tensor([use_label], device=DEVICE)])

    # Retrieval-induced contrastive update (training only)
    if label is not None and lr > 0.0:
        top_k_idx = sims.topk(min(k, sims.shape[0])).indices
        for idx in top_k_idx:
            if labels[idx].item() == label:
                V[idx] = F.normalize(V[idx] + lr * (r - V[idx]), dim=0)
            else:
                V[idx] = F.normalize(V[idx] - lr * (r - V[idx]), dim=0)

    return prediction, V, labels


def eval_topk_batch(V, labels, R_te, k=K):
    sims   = R_te @ V.T                          # (n, N)
    n      = R_te.shape[0]
    n_cls  = int(labels.max().item()) + 1
    scores = torch.zeros(n, n_cls, device=DEVICE)
    for c in range(n_cls):
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        cs   = sims[:, mask]
        keff = min(k, cs.shape[1])
        scores[:, c] = cs.topk(keff, dim=1).values.sum(dim=1)
    return scores.argmax(dim=1).cpu()


def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=True,  download=True)
    te = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_tr, tr.targets.numpy(), X_te, te.targets.numpy()


def make_permutation(seed):
    perm = list(range(784))
    random.Random(seed).shuffle(perm)
    return perm


def embed_raw(X_np, perm):
    X_t = torch.from_numpy(X_np[:, perm]).to(DEVICE)
    return F.normalize(X_t, dim=1)


def stratified_sample(X, y, n_per_class, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(N_CLASSES):
        chosen = rng.choice(np.where(y == c)[0], n_per_class, replace=False)
        idx.extend(chosen.tolist())
    rng.shuffle(idx)
    return X[idx], y[idx]


def compute_aa(mat, n):
    vals = [mat[t][n-1] for t in range(n) if mat[t][n-1] is not None]
    return sum(vals) / len(vals) if vals else 0.0

def compute_fgt(mat, n):
    vals = []
    for t in range(n - 1):
        if mat[t][t] is not None and mat[t][n-1] is not None:
            vals.append(max(0.0, mat[t][t] - mat[t][n-1]))
    return sum(vals) / len(vals) if vals else 0.0


def run_lr(lr, tasks_train, tasks_test, perms, test_embeds, y_te_t):
    n      = N_TASKS
    mat    = [[None]*n for _ in range(n)]
    V      = torch.empty(0, 784, device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)

    for task_id in range(n):
        t_train = time.time()
        X_tr, y_tr = tasks_train[task_id]
        R_tr = embed_raw(X_tr, perms[task_id])
        for i in range(len(y_tr)):
            _, V, labels = process(V, labels, R_tr[i], label=int(y_tr[i]), lr=lr)
        train_s = time.time() - t_train

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            R_te = test_embeds[eval_task]
            preds = eval_topk_batch(V, labels, R_te)
            acc   = (preds == y_te_t).float().mean().item()
            mat[eval_task][task_id] = acc
        eval_s = time.time() - t_eval

        aa_k5 = sum(mat[t][task_id] for t in range(task_id + 1)) / (task_id + 1)
        print(f"    T{task_id}: train={train_s:.1f}s eval={eval_s:.1f}s "
              f"cb={V.shape[0]} aa={aa_k5*100:.1f}%", flush=True)

    return compute_aa(mat, n), compute_fgt(mat, n)


def main():
    t0 = time.time()
    print(f"Step 111 -- Retrieval-Induced Learning, P-MNIST raw pixels", flush=True)
    print(f"N_TASKS={N_TASKS}, k={K}, DEVICE={DEVICE}", flush=True)
    print(f"Baseline (Step 110 always-spawn raw): k=5→95.4%", flush=True)
    print(f"Kill: retrieval-induced AA <= 95.4%", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_tr_all, y_tr_all, X_te_all, y_te_all = load_mnist()
    perms = [make_permutation(seed=t * 100) for t in range(N_TASKS)]

    tasks_train, tasks_test = [], []
    for t in range(N_TASKS):
        Xtr, ytr = stratified_sample(X_tr_all, y_tr_all, TRAIN_PER_CLS, seed=t * 7)
        tasks_train.append((Xtr, ytr))
        tasks_test.append((X_te_all, y_te_all))

    test_embeds = [embed_raw(tasks_test[t][0], perms[t]) for t in range(N_TASKS)]
    y_te_t = torch.from_numpy(y_te_all)

    results = {}
    for lr in LR_VALS:
        label = f"lr={lr}"
        print(f"\n{'='*60}", flush=True)
        print(f"Run: {label}", flush=True)
        print(f"{'='*60}", flush=True)
        aa, fgt = run_lr(lr, tasks_train, tasks_test, perms, test_embeds, y_te_t)
        results[label] = (aa, fgt)

    elapsed = time.time() - t0

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 111 FINAL -- Retrieval-Induced Learning", flush=True)
    print(f"Baseline (Step 110 k-NN): AA=95.4%  fgt=0.0pp", flush=True)
    print(f"{'='*72}", flush=True)
    baseline = 0.954
    for name, (aa, fgt) in results.items():
        delta = aa - baseline
        verdict = "PASSES" if aa > baseline else "DISPROVED"
        print(f"  {name:<12} AA={aa*100:.1f}%  fgt={fgt*100:.1f}pp  "
              f"delta={delta*100:+.1f}pp  [{verdict}]", flush=True)

    best_aa = max(aa for aa, _ in results.values())
    best_name = max(results, key=lambda k: results[k][0])
    overall = "PASSES" if best_aa > baseline else "DISPROVED"
    print(f"\n  OVERALL: {overall} (best: {best_name} → {best_aa*100:.1f}%)", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
