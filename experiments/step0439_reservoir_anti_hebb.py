#!/usr/bin/env python3
"""
Step 439 — Reservoir with anti-Hebbian decorrelation.
Hebb + anti-Hebb (0.1x) + spectral control + velocity readout.
d=256, LS20 only, 30K steps.
"""

import time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cpu'
SR_TARGET = 0.95


class ReservoirAntiHebb:
    def __init__(self, d=256, input_dim=256, n_actions=4):
        self.d = d
        self.W = torch.randn(d, d) * (1.0 / d**0.5)
        self.U = torch.randn(d, input_dim) * (0.1 / input_dim**0.5)
        self.h = torch.zeros(d)
        self.n_actions = n_actions
        self.hebb_lr = 0.001
        self.h_history = []
        self.last_rank = 0
        self._normalize_sr()

    def _normalize_sr(self):
        sv = torch.linalg.svdvals(self.W)
        if sv[0].item() > SR_TARGET:
            self.W *= SR_TARGET / sv[0].item()

    def step(self, x, label=None):
        x = x.float().flatten()
        h_new = torch.tanh(self.W @ self.h + self.U @ x)
        delta_h = h_new - self.h
        action = delta_h[:self.n_actions].abs().argmax().item()

        # Hebbian
        self.W += self.hebb_lr * torch.outer(delta_h, self.h)
        # Anti-Hebbian decorrelation
        self.W -= self.hebb_lr * 0.1 * torch.outer(self.h, self.h)
        # Spectral control
        self._normalize_sr()

        self.h = h_new.detach()
        self.h_history.append(self.h.clone())
        if len(self.h_history) > 200:
            self.h_history = self.h_history[-200:]
        return action

    def compute_rank(self):
        if len(self.h_history) < 100:
            return 0
        H = torch.stack(self.h_history[-100:])
        return torch.linalg.matrix_rank(H, tol=0.1).item()


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def main():
    from arcengine import GameState
    import arc_agi

    print(f"Step 439: Reservoir + anti-Hebbian decorrelation on LS20", flush=True)
    print(f"d=256, velocity readout, spectral control, anti-Hebb 0.1x", flush=True)

    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return

    res = ReservoirAntiHebb(d=256, input_dim=256, n_actions=4)
    env = arc.make(ls20.game_id); obs = env.reset()
    na = len(env.action_space)
    ts = go = lvls = 0; unique = set(); action_counts = [0]*na; t0 = time.time()

    while ts < 30000:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: print(f"WIN at step {ts}!", flush=True); break
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
            lvls = obs.levels_completed; print(f"LEVEL {lvls} at step {ts}", flush=True)
        if ts % 5000 == 0:
            rank = res.compute_rank()
            res.last_rank = rank
            total = sum(action_counts); dom = max(action_counts)/total*100
            print(f"  [step {ts}]  rank={rank}  unique={len(unique)}  lvls={lvls}  go={go}"
                  f"  dom={dom:.0f}%  W={res.W.norm():.1f}  h={res.h.norm():.2f}  {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts); dom = max(action_counts)/total*100
    rank = res.compute_rank()
    print(f"\n{'='*60}")
    print("STEP 439 RESULTS")
    print(f"{'='*60}")
    print(f"rank={rank}  unique={len(unique)}  levels={lvls}  dom={dom:.0f}%")
    print(f"acts={action_counts}  W={res.W.norm():.1f}  {elapsed:.0f}s")
    if rank <= 2:
        print("KILL: rank still 1-2. Anti-Hebb too weak.")
    elif rank > 50 and lvls == 0:
        print(f"HIGH-D trajectory (rank={rank}) but 0 levels.")
    elif len(unique) > 2000 and dom < 35:
        print(f"PROMISING: unique={len(unique)}, dom={dom:.0f}%, rank={rank}")

if __name__ == '__main__':
    main()
