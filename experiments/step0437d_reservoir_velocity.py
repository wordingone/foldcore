#!/usr/bin/env python3
"""
Step 437d — Reservoir with velocity readout. action = delta_h[:n].abs().argmax().
Same as 437b (spectral control, no perturbation) but different readout.
"""

import time, math, random, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cpu'
SR_TARGET = 0.95


class ReservoirVelocity:
    def __init__(self, d=256, input_dim=384, n_actions=10):
        self.W = torch.randn(d, d) * (1.0 / d**0.5)
        self.U = torch.randn(d, input_dim) * (1.0 / input_dim**0.5)
        self.h = torch.zeros(d)
        self.n_actions = n_actions
        self.hebb_lr = 0.001
        self._normalize_sr()

    def _normalize_sr(self):
        sv = torch.linalg.svdvals(self.W)
        if sv[0].item() > SR_TARGET:
            self.W *= SR_TARGET / sv[0].item()

    def step(self, x, label=None):
        x = x.float()
        if x.dim() > 1: x = x.flatten()
        h_new = torch.tanh(self.W @ self.h + self.U @ x)
        # VELOCITY READOUT: dimension of maximum surprise
        delta_h = h_new - self.h
        action = delta_h[:self.n_actions].abs().argmax().item()
        # Hebbian
        self.W += self.hebb_lr * torch.outer(delta_h, self.h)
        self._normalize_sr()
        self.h = h_new.detach()
        return action


def test_pmnist():
    import torchvision
    print("\n--- Test A: P-MNIST 1-task (5K) ---", flush=True)
    tr = torchvision.datasets.MNIST('./data/mnist', train=True, download=True)
    te = torchvision.datasets.MNIST('./data/mnist', train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1,784).astype(np.float32)/255.0; y_tr = tr.targets.numpy()
    X_te = te.data.numpy().reshape(-1,784).astype(np.float32)/255.0; y_te = te.targets.numpy()
    rng = np.random.RandomState(42)
    P = torch.from_numpy(rng.randn(384,784).astype(np.float32)/math.sqrt(784))
    perm = list(range(784)); random.Random(42).shuffle(perm)
    def embed(X): return F.normalize(torch.from_numpy(X[:,perm]) @ P.T, dim=1)
    R_tr = embed(X_tr); R_te = embed(X_te)
    res = ReservoirVelocity(d=256, input_dim=384, n_actions=10)
    indices = rng.choice(len(X_tr), 5000, replace=False)
    t0 = time.time(); correct = 0; action_hist = []
    for i, idx in enumerate(indices):
        a = res.step(R_tr[idx]); action_hist.append(a)
        if a == int(y_tr[idx]): correct += 1
        if (i+1) % 1000 == 0:
            n_unique = len(set(action_hist[-100:]))
            print(f"  [step {i+1}]  acc={correct/(i+1)*100:.1f}%  W={res.W.norm():.1f}  h={res.h.norm():.2f}  unique_act={n_unique}", flush=True)
    train_acc = correct/5000*100
    correct = 0
    for i in range(min(2000, len(R_te))):
        if res.step(R_te[i]) == int(y_te[i]): correct += 1
    test_acc = correct/min(2000,len(R_te))*100
    print(f"\n  Train: {train_acc:.1f}%  Test: {test_acc:.1f}%  {time.time()-t0:.1f}s")
    return test_acc


def test_ls20():
    from arcengine import GameState; import arc_agi
    print("\n--- Test B: LS20 (30K) ---", flush=True)
    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return None
    res = ReservoirVelocity(d=256, input_dim=256, n_actions=4)
    env = arc.make(ls20.game_id); obs = env.reset(); na = len(env.action_space)
    ts=go=lvls=0; unique=set(); action_counts=[0]*na; t0=time.time()
    def avgpool16(f):
        a=np.array(f[0],dtype=np.float32)/15.0; return a.reshape(16,4,16,4).mean(axis=(1,3)).flatten()
    while ts < 30000:
        if obs is None: obs=env.reset(); continue
        if obs.state == GameState.GAME_OVER: go+=1; obs=env.reset(); continue
        if obs.state == GameState.WIN: print(f"  WIN at step {ts}!",flush=True); break
        pooled=avgpool16(obs.frame); unique.add(hash(pooled.tobytes()))
        x=F.normalize(torch.from_numpy(pooled.astype(np.float32)),dim=0)
        idx=res.step(x); action_counts[idx%na]+=1
        action=env.action_space[idx%na]; data={}
        if action.is_complex():
            arr=np.array(obs.frame[0]); cy,cx=divmod(int(np.argmax(arr)),64); data={"x":cx,"y":cy}
        obs_before=obs.levels_completed
        obs=env.step(action,data=data); ts+=1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls=obs.levels_completed; print(f"  LEVEL {lvls} at step {ts}",flush=True)
        if ts%5000==0:
            total=sum(action_counts); dom=max(action_counts)/total*100
            print(f"  [step {ts}]  unique={len(unique)}  lvls={lvls}  go={go}  dom={dom:.0f}%  W={res.W.norm():.1f}  h={res.h.norm():.2f}",flush=True)
    elapsed=time.time()-t0; total=sum(action_counts); dom=max(action_counts)/total*100
    print(f"\n  unique={len(unique)}  levels={lvls}  go={go}  dom={dom:.0f}%  acts={action_counts}  {elapsed:.0f}s")
    return {'unique':len(unique),'levels':lvls,'dom':dom}


def main():
    print(f"Step 437d: Reservoir + velocity readout (delta_h[:n].abs().argmax())",flush=True)
    acc = test_pmnist()
    nav = test_ls20()
    print(f"\n{'='*60}")
    print("STEP 437d RESULTS")
    print(f"{'='*60}")
    if acc is not None: print(f"P-MNIST: {acc:.1f}% {'PASS' if acc>25 else 'FAIL'}")
    if nav: print(f"LS20: unique={nav['unique']}  levels={nav['levels']}  dom={nav['dom']:.0f}%")

if __name__ == '__main__':
    main()
