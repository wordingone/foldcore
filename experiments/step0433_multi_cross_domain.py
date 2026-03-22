#!/usr/bin/env python3
"""
Step 433 — Cross-domain survival. P-MNIST(5) -> LS20(10K) -> P-MNIST(5).
Single codebook, never reset. Measure contamination and accumulation.
"""

import time, math, random, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_OUT = 384; D_LS20 = 256; N_CLASSES = 10; N_TRAIN = 5000; SEED = 42


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

    def step_nav(self, x, n_actions):
        """Navigation mode: argmin for exploration, no labels."""
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([0], device=self.device)
            return 0
        sims = self.V @ x
        n_cls = int(self.labels.max().item()) + 1
        cls_ids = torch.arange(max(n_cls, n_actions), device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        masked = sims.unsqueeze(0) * one_hot + (1 - one_hot) * (-1e9)
        topk_k = min(3, masked.shape[1])
        scores = masked.topk(topk_k, dim=1).values.sum(dim=1)
        prediction = scores[:n_actions].argmin().item()
        target_mask = (self.labels == prediction)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([prediction], device=self.device)])
        else:
            ts = sims.clone(); ts[~target_mask] = -float('inf')
            w = ts.argmax().item()
            a = 1.0 - float(sims[w].item())
            self.V[w] = F.normalize(self.V[w] + a * (x - self.V[w]), dim=0)
        return prediction


def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST('./data/mnist', train=True, download=True)
    te = torchvision.datasets.MNIST('./data/mnist', train=False, download=True)
    return (tr.data.numpy().reshape(-1,784).astype(np.float32)/255.0, tr.targets.numpy(),
            te.data.numpy().reshape(-1,784).astype(np.float32)/255.0, te.targets.numpy())

def make_proj(d_out, seed=12345):
    rng = np.random.RandomState(seed)
    return torch.from_numpy(rng.randn(d_out, 784).astype(np.float32) / math.sqrt(784)).to(DEVICE)

def make_perm(seed):
    perm = list(range(784)); random.Random(seed).shuffle(perm); return perm

def embed_mnist(X, perm, P):
    return F.normalize(torch.from_numpy(X[:, perm]).to(DEVICE) @ P.T, dim=1)

def eval_frozen(sub, R_te, y_te):
    c = sum(1 for i in range(len(R_te)) if sub.predict_frozen(R_te[i]) == int(y_te[i]))
    return c / len(R_te) * 100

def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()

def centered_enc_nav(pooled, sub):
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    # Note: V may have mixed dimensionalities from P-MNIST (384) and LS20 (256)
    # Can't subtract V.mean() if dims don't match. Skip centering for nav.
    return t_unit


def main():
    from arcengine import GameState
    import arc_agi

    print(f"Step 433: Cross-domain survival")
    print(f"Device: {DEVICE}", flush=True)

    X_tr, y_tr, X_te, y_te = load_mnist()
    # Use D_LS20=256 for everything so codebook is consistent
    P = make_proj(d_out=D_LS20, seed=12345)

    sub = PNSoftmax(d=D_LS20, tau=0.01, thresh=0.9)

    perms = [make_perm(seed=SEED+t) for t in range(10)]
    task_test = [embed_mnist(X_te, perms[t], P) for t in range(10)]
    rng = np.random.RandomState(SEED)

    # ===== PHASE 1: P-MNIST tasks 0-4 =====
    print("\n--- Phase 1: P-MNIST tasks 0-4 ---", flush=True)
    acc_after_phase1 = {}
    for t in range(5):
        R_tr_t = embed_mnist(X_tr, perms[t], P)
        indices = rng.choice(len(X_tr), N_TRAIN, replace=False)
        for idx in indices:
            sub.step(R_tr_t[idx], label=int(y_tr[idx]))
        acc = eval_frozen(sub, task_test[t], y_te)
        acc_after_phase1[t] = acc
        print(f"  Task {t}: cb={sub.V.shape[0]:5d}  acc={acc:.1f}%", flush=True)

    cb_after_mnist1 = sub.V.shape[0]
    print(f"  cb after Phase 1: {cb_after_mnist1}", flush=True)

    # Measure all 5 tasks accuracy before LS20
    pre_ls20_accs = {}
    for t in range(5):
        pre_ls20_accs[t] = eval_frozen(sub, task_test[t], y_te)

    # ===== PHASE 2: LS20 (10K steps, no reset) =====
    print("\n--- Phase 2: LS20 (10K steps, no codebook reset) ---", flush=True)
    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: ls20 not found"); return

    env = arc.make(ls20.game_id); obs = env.reset()
    na = len(env.action_space)
    ts = go = lvls = 0; unique = set(); t0 = time.time()

    while ts < 10000:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset()
            if obs is None: break
            continue
        if obs.state == GameState.WIN:
            print(f"  WIN at step {ts}!", flush=True); break
        pooled = avgpool16(obs.frame); unique.add(hash(pooled.tobytes()))
        x = centered_enc_nav(pooled, sub)
        idx = sub.step_nav(x, n_actions=na)
        action = env.action_space[idx % na]; data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0]); cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}
        obs = env.step(action, data=data); ts += 1
        if obs is None: break

    cb_after_ls20 = sub.V.shape[0]
    ls20_elapsed = time.time() - t0
    print(f"  LS20: unique={len(unique)}  lvls={lvls}  go={go}  cb={cb_after_ls20}  {ls20_elapsed:.0f}s", flush=True)

    # Measure contamination: tasks 0-4 accuracy AFTER LS20
    post_ls20_accs = {}
    for t in range(5):
        post_ls20_accs[t] = eval_frozen(sub, task_test[t], y_te)

    # ===== PHASE 3: P-MNIST tasks 5-9 =====
    print("\n--- Phase 3: P-MNIST tasks 5-9 (no codebook reset) ---", flush=True)
    for t in range(5, 10):
        R_tr_t = embed_mnist(X_tr, perms[t], P)
        indices = rng.choice(len(X_tr), N_TRAIN, replace=False)
        for idx in indices:
            sub.step(R_tr_t[idx], label=int(y_tr[idx]))
        acc = eval_frozen(sub, task_test[t], y_te)
        print(f"  Task {t}: cb={sub.V.shape[0]:5d}  acc={acc:.1f}%", flush=True)

    cb_final = sub.V.shape[0]

    # Final accuracy on all 10 tasks
    final_accs = {}
    for t in range(10):
        final_accs[t] = eval_frozen(sub, task_test[t], y_te)

    # ===== RESULTS =====
    print(f"\n{'='*60}")
    print("STEP 433 — CROSS-DOMAIN SURVIVAL")
    print(f"{'='*60}")

    print(f"\nCodebook sizes: after_P1={cb_after_mnist1}  after_LS20={cb_after_ls20}  final={cb_final}")
    print(f"LS20 added {cb_after_ls20 - cb_after_mnist1} entries. Phase 3 added {cb_final - cb_after_ls20}.")

    print(f"\nContamination (tasks 0-4 accuracy):")
    print(f"{'Task':<6} {'Before LS20':>12} {'After LS20':>12} {'Degradation':>12}")
    max_degrade = 0
    for t in range(5):
        deg = pre_ls20_accs[t] - post_ls20_accs[t]
        max_degrade = max(max_degrade, deg)
        print(f"{t:<6} {pre_ls20_accs[t]:>11.1f}% {post_ls20_accs[t]:>11.1f}% {deg:>11.1f}pp")

    print(f"\nFinal accuracy (all 10 tasks):")
    for t in range(10):
        phase = "P1" if t < 5 else "P3"
        print(f"  Task {t} ({phase}): {final_accs[t]:.1f}%")

    avg_p1 = np.mean([final_accs[t] for t in range(5)])
    avg_p3 = np.mean([final_accs[t] for t in range(5, 10)])
    print(f"\nAvg tasks 0-4: {avg_p1:.1f}%  Avg tasks 5-9: {avg_p3:.1f}%")
    print(f"Max contamination: {max_degrade:.1f}pp")
    print(f"LS20 unique states: {len(unique)}")

    if max_degrade > 5:
        print(f"\nKILL: contamination {max_degrade:.1f}pp > 5pp")
    else:
        print(f"\nPASS: contamination {max_degrade:.1f}pp <= 5pp. Cross-domain survival confirmed.")


if __name__ == '__main__':
    main()
