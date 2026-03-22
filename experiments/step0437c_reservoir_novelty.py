#!/usr/bin/env python3
"""
Step 437c — Reservoir with novelty detection (delta_h threshold + perturbation).
Spawn analogue: ||delta_h|| < median → perturb W.
"""

import time, math, random, logging
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cpu'
SR_TARGET = 0.95


class ReservoirNovelty:
    def __init__(self, d=256, input_dim=384, n_actions=10):
        self.W = torch.randn(d, d) * (1.0 / d**0.5)
        self.U = torch.randn(d, input_dim) * (1.0 / input_dim**0.5)
        self.h = torch.zeros(d)
        self.n_actions = n_actions
        self.hebb_lr = 0.001
        self.delta_history = []
        self.n_perturb = 0
        self.last_delta = 0.0
        self._normalize_sr()

    def _normalize_sr(self):
        sv = torch.linalg.svdvals(self.W)
        sr = sv[0].item()
        if sr > SR_TARGET:
            self.W *= SR_TARGET / sr

    def step(self, x, label=None):
        x = x.float()
        if x.dim() > 1: x = x.flatten()
        h_new = torch.tanh(self.W @ self.h + self.U @ x)
        scores = h_new[:self.n_actions]
        action = scores.argmax().item()

        delta_h = h_new - self.h
        self.last_delta = delta_h.norm().item()

        # Hebbian update
        self.W += self.hebb_lr * torch.outer(delta_h, self.h)
        self._normalize_sr()

        # Novelty detection: perturb if trajectory converged
        self.delta_history.append(self.last_delta)
        if len(self.delta_history) > 100:
            self.delta_history.pop(0)
        if len(self.delta_history) >= 10:
            thresh = sorted(self.delta_history)[len(self.delta_history) // 2]
            if self.last_delta < thresh:
                self.W += torch.randn_like(self.W) * 0.01
                self._normalize_sr()
                self.n_perturb += 1

        self.h = h_new.detach()
        return action


def test_pmnist():
    import torchvision
    print("\n--- Test A: P-MNIST 1-task (5K samples) ---", flush=True)

    tr = torchvision.datasets.MNIST('./data/mnist', train=True, download=True)
    te = torchvision.datasets.MNIST('./data/mnist', train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_tr = tr.targets.numpy()
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_te = te.targets.numpy()

    rng = np.random.RandomState(42)
    P = torch.from_numpy(rng.randn(384, 784).astype(np.float32) / math.sqrt(784))
    perm = list(range(784)); random.Random(42).shuffle(perm)
    def embed(X): return F.normalize(torch.from_numpy(X[:, perm]) @ P.T, dim=1)

    R_tr = embed(X_tr); R_te = embed(X_te)
    res = ReservoirNovelty(d=256, input_dim=384, n_actions=10)
    indices = rng.choice(len(X_tr), 5000, replace=False)

    t0 = time.time(); correct = 0
    for i, idx in enumerate(indices):
        action = res.step(R_tr[idx])
        if action == int(y_tr[idx]): correct += 1
        if (i+1) % 1000 == 0:
            perturb_pct = res.n_perturb / (i+1) * 100
            n_unique = len(set([res.step(R_tr[indices[j]]) for j in range(max(0,i-9), i+1)]))
            print(f"  [step {i+1}]  acc={correct/(i+1)*100:.1f}%  delta={res.last_delta:.4f}"
                  f"  perturb={perturb_pct:.0f}%  W={res.W.norm():.1f}", flush=True)

    train_acc = correct / 5000 * 100
    correct = 0
    for i in range(min(2000, len(R_te))):  # eval subset for speed
        if res.step(R_te[i]) == int(y_te[i]): correct += 1
    test_acc = correct / min(2000, len(R_te)) * 100

    perturb_pct = res.n_perturb / 5000 * 100
    print(f"\n  Train: {train_acc:.1f}%  Test: {test_acc:.1f}%  perturb={perturb_pct:.0f}%  time={time.time()-t0:.1f}s")
    return test_acc, perturb_pct


def test_ls20():
    from arcengine import GameState
    import arc_agi
    print("\n--- Test B: LS20 navigation (30K steps) ---", flush=True)

    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return None

    res = ReservoirNovelty(d=256, input_dim=256, n_actions=4)
    env = arc.make(ls20.game_id); obs = env.reset()
    na = len(env.action_space)
    ts = go = lvls = 0; unique = set(); action_counts = [0]*na; t0 = time.time()

    def avgpool16(frame):
        arr = np.array(frame[0], dtype=np.float32) / 15.0
        return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()

    while ts < 30000:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: print(f"  WIN at step {ts}!", flush=True); break
        pooled = avgpool16(obs.frame); unique.add(hash(pooled.tobytes()))
        x = F.normalize(torch.from_numpy(pooled.astype(np.float32)), dim=0)
        idx = res.step(x); action_counts[idx % na] += 1
        action = env.action_space[idx % na]; data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0]); cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}
        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed; print(f"  LEVEL {lvls} at step {ts}", flush=True)
        if ts % 5000 == 0:
            total = sum(action_counts); dom = max(action_counts)/total*100
            perturb_pct = res.n_perturb / ts * 100
            print(f"  [step {ts}]  unique={len(unique)}  lvls={lvls}  go={go}  dom={dom:.0f}%"
                  f"  perturb={perturb_pct:.0f}%  delta={res.last_delta:.4f}", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts); dom = max(action_counts)/total*100
    perturb_pct = res.n_perturb / ts * 100
    print(f"\n  unique={len(unique)}  levels={lvls}  go={go}  dom={dom:.0f}%  perturb={perturb_pct:.0f}%  acts={action_counts}  {elapsed:.0f}s")
    return {'unique': len(unique), 'levels': lvls, 'dom': dom, 'perturb': perturb_pct}


def main():
    print(f"Step 437c: Reservoir + novelty detection (perturbation when delta_h < median)", flush=True)
    acc, pp = test_pmnist()
    nav = test_ls20()
    print(f"\n{'='*60}")
    print("STEP 437c RESULTS")
    print(f"{'='*60}")
    print(f"P-MNIST: {acc:.1f}% (chance=10%, gate=25%) {'PASS' if acc > 25 else 'FAIL'}  perturb={pp:.0f}%")
    if nav:
        print(f"LS20: unique={nav['unique']}  levels={nav['levels']}  dom={nav['dom']:.0f}%  perturb={nav['perturb']:.0f}%")

if __name__ == '__main__':
    main()
