#!/usr/bin/env python3
"""
Step 119 -- Constitutional audit, ResNet-18 features, CIFAR-100, 10 tasks.
[researcher] mail 1284.

Test A: Always-spawn k=5 (baseline)
Test B: Self-supervised eval (S1 reconfirmation, label=None during eval)
Test C: Always-spawn vs threshold-spawn (sp=0.95) on ResNet features
Test D: Fixed k sweep k={1,3,5,10,20} with always-spawn

ResNet-18 features: C:/Users/Admin/cifar100_resnet18_features.npz
"""

import sys, time
import numpy as np
import torch
import torch.nn.functional as F

N_TASKS      = 10
CLASSES_TASK = 10
SEED         = 42
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
D_FEAT       = 512
SP_THRESH    = 0.95
RESNET_CACHE = '/mnt/c/Users/Admin/cifar100_resnet18_features.npz'


def eval_topk_batch(V, labels, R_te, k=5):
    sims   = R_te @ V.T
    n      = R_te.shape[0]
    n_cls  = int(labels.max().item()) + 1
    scores = torch.zeros(n, n_cls, device=DEVICE)
    for c in range(n_cls):
        mask = (labels == c)
        if mask.sum() == 0: continue
        cs   = sims[:, mask]
        keff = min(k, cs.shape[1])
        scores[:, c] = cs.topk(keff, dim=1).values.sum(dim=1)
    return scores.argmax(dim=1).cpu()


def load_resnet_features():
    data = np.load(RESNET_CACHE)
    X_tr = torch.from_numpy(data['X_train'].astype(np.float32)).to(DEVICE)
    y_tr = torch.from_numpy(data['y_train'].astype(np.int64))
    X_te = torch.from_numpy(data['X_test'].astype(np.float32)).to(DEVICE)
    y_te = torch.from_numpy(data['y_test'].astype(np.int64))
    return X_tr, y_tr, X_te, y_te


def make_splits(X_tr, y_tr, X_te, y_te):
    splits = []
    for t in range(N_TASKS):
        c0, c1 = t*CLASSES_TASK, (t+1)*CLASSES_TASK
        mtr = torch.isin(y_tr, torch.arange(c0, c1))
        mte = torch.isin(y_te, torch.arange(c0, c1))
        splits.append((X_tr[mtr], y_tr[mtr], X_te[mte], y_te[mte]))
    return splits


def compute_aa(mat, n):
    vals = [mat[t][n-1] for t in range(n) if mat[t][n-1] is not None]
    return sum(vals)/len(vals) if vals else 0.0

def compute_fgt(mat, n):
    vals = []
    for t in range(n-1):
        if mat[t][t] is not None and mat[t][n-1] is not None:
            vals.append(max(0.0, mat[t][t]-mat[t][n-1]))
    return sum(vals)/len(vals) if vals else 0.0


def run_always_spawn(splits, k=5, sp_threshold=None, self_supervised_eval=False):
    """
    Core runner. sp_threshold=None → always-spawn train.
    self_supervised_eval=True → batch-predict then append test vectors using predicted labels.
    """
    n   = N_TASKS
    mat = [[None]*n for _ in range(n)]
    V      = torch.empty(0, D_FEAT, device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)
    eval_spawns = 0

    for tid in range(n):
        X_tr, y_tr, _, _ = splits[tid]
        R_tr = F.normalize(X_tr, dim=1)

        if sp_threshold is None:
            # Always-spawn: store all
            V      = torch.cat([V, R_tr])
            labels = torch.cat([labels, y_tr.to(DEVICE)])
        else:
            # Threshold-spawn: only spawn if max_sim < threshold
            if V.shape[0] == 0:
                V      = torch.cat([V, R_tr])
                labels = torch.cat([labels, y_tr.to(DEVICE)])
            else:
                sims_max = (R_tr @ V.T).max(dim=1).values  # (N_tr,)
                mask = sims_max < sp_threshold
                mask_cpu = mask.cpu()
                if mask.sum() > 0:
                    V      = torch.cat([V, R_tr[mask]])
                    labels = torch.cat([labels, y_tr[mask_cpu].to(DEVICE)])

        for et in range(tid+1):
            _, _, X_te, y_te = splits[et]
            R_te = F.normalize(X_te, dim=1)
            preds = eval_topk_batch(V, labels, R_te, k)
            acc   = (preds == y_te).float().mean().item()
            mat[et][tid] = acc

            if self_supervised_eval:
                # Append test vectors using predicted labels (batch)
                V      = torch.cat([V, R_te])
                labels = torch.cat([labels, preds.to(DEVICE)])
                eval_spawns += R_te.shape[0]

        aa = sum(mat[t][tid] for t in range(tid+1))/(tid+1)
        print(f"    T{tid}: cb={V.shape[0]} aa={aa*100:.1f}%", flush=True)

    return compute_aa(mat, n), compute_fgt(mat, n), int(V.shape[0]), eval_spawns


def main():
    t0 = time.time()
    print(f"Step 119 -- Constitutional Audit, ResNet-18 CIFAR-100", flush=True)
    print(f"N_TASKS={N_TASKS}, DEVICE={DEVICE}", flush=True)
    print(f"Reference: 2-task sp=0.95 k=5 → 38.3% AA, 11.6pp fgt (Step 100b/107)", flush=True)

    torch.manual_seed(SEED); np.random.seed(SEED)
    print("Loading ResNet features...", flush=True)
    X_tr, y_tr, X_te, y_te = load_resnet_features()
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}", flush=True)
    splits = make_splits(X_tr, y_tr, X_te, y_te)

    results = {}

    print(f"\n{'='*60}\nTest A: Always-spawn k=5 (baseline)\n{'='*60}", flush=True)
    aa, fgt, cb, _ = run_always_spawn(splits, k=5)
    results['A_always_k5'] = (aa, fgt, cb)

    print(f"\n{'='*60}\nTest B: Self-supervised eval (S1 reconfirmation)\n{'='*60}", flush=True)
    aa, fgt, cb, esp = run_always_spawn(splits, k=5, self_supervised_eval=True)
    results['B_s1_eval'] = (aa, fgt, cb)
    b_eval_spawns = esp

    print(f"\n{'='*60}\nTest C: Threshold-spawn sp=0.95 (vs always-spawn)\n{'='*60}", flush=True)
    aa, fgt, cb, _ = run_always_spawn(splits, k=5, sp_threshold=SP_THRESH)
    results['C_thresh_0.95'] = (aa, fgt, cb)

    print(f"\n{'='*60}\nTest D: k sweep (always-spawn)\n{'='*60}", flush=True)
    for k in [1, 3, 10, 20]:
        print(f"  --- k={k} ---", flush=True)
        aa, fgt, cb, _ = run_always_spawn(splits, k=k)
        results[f'D_k{k}'] = (aa, fgt, cb)

    elapsed = time.time() - t0
    a_aa = results['A_always_k5'][0]

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 119 FINAL -- Constitutional Audit, ResNet-18 CIFAR-100", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"{'Test':<20} {'AA':>7}  {'fgt':>7}  {'CB':>8}  Notes", flush=True)
    print(f"{'-'*72}", flush=True)
    for name, (aa, fgt, cb) in results.items():
        note = ''
        if name == 'B_s1_eval':
            delta = aa - a_aa
            note = f'delta_vs_A={delta*100:+.2f}pp, eval_spawns={b_eval_spawns}'
        elif name == 'C_thresh_0.95':
            delta = aa - a_aa
            note = f'delta_vs_always={delta*100:+.2f}pp'
        elif name.startswith('D_'):
            delta = aa - a_aa
            note = f'delta_vs_k5={delta*100:+.2f}pp'
        print(f"  {name:<18} {aa*100:>6.1f}%  {fgt*100:>6.1f}pp  {cb:>8}  {note}", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
