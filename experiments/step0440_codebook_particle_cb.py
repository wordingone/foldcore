#!/usr/bin/env python3
"""
Step 440 — Interacting particle codebook. Entries have position + velocity.
Drift between observations. LS20, 30K steps.
"""

import time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ParticleCodebook:
    def __init__(self, d=256, n_actions=4, device=DEVICE):
        self.V = F.normalize(torch.randn(n_actions, d, device=device), dim=1)
        self.D = torch.zeros_like(self.V)
        self.n_actions = n_actions
        self.drift_lr = 0.001
        self.device = device

    def step(self, x, label=None):
        x = F.normalize(x.to(self.device).float().flatten().unsqueeze(0), dim=1)

        # Drift
        self.V = self.V + self.drift_lr * self.D
        self.V = F.normalize(self.V, dim=1)

        # Match
        sims = (self.V @ x.T).squeeze()
        winner = sims.argmax().item()
        action = winner % self.n_actions

        # Attract
        sim = sims[winner].item()
        lr = max(0.0, 1.0 - sim)
        self.V[winner] += lr * (x.squeeze() - self.V[winner])
        self.V[winner] = F.normalize(self.V[winner], dim=0)

        # Velocity update
        self.D[winner] = 0.9 * self.D[winner] + 0.1 * (x.squeeze() - self.V[winner])

        # Spawn
        if sims.max().item() < 0.95 and self.V.shape[0] < 10000:
            new_v = x.squeeze().clone().unsqueeze(0)
            new_d = torch.zeros(1, self.V.shape[1], device=self.device)
            self.V = torch.cat([self.V, new_v])
            self.D = torch.cat([self.D, new_d])

        return action


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()

def centered_enc(pooled, sub):
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if sub.V.shape[0] > 2:
        t_unit = t_unit - sub.V.mean(dim=0).cpu()
    return t_unit


def main():
    from arcengine import GameState
    import arc_agi

    print(f"Step 440: Particle codebook (entries with velocity) on LS20", flush=True)
    print(f"Device: {DEVICE}  30K steps", flush=True)

    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return

    sub = ParticleCodebook(d=256, n_actions=4)
    env = arc.make(ls20.game_id); obs = env.reset()
    na = len(env.action_space)
    ts = go = lvls = 0; unique = set(); action_counts = [0]*na; t0 = time.time()

    while ts < 30000:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: print(f"WIN at step {ts}!", flush=True); break
        pooled = avgpool16(obs.frame); unique.add(hash(pooled.tobytes()))
        x = centered_enc(pooled, sub)
        idx = sub.step(x); action_counts[idx % na] += 1
        action = env.action_space[idx % na]; data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0]); cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}
        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed; print(f"LEVEL {lvls} at step {ts}  cb={sub.V.shape[0]}", flush=True)
        if ts % 5000 == 0:
            total = sum(action_counts); dom = max(action_counts)/total*100
            avg_d = sub.D.norm(dim=1).mean().item()
            print(f"  [step {ts}]  cb={sub.V.shape[0]:5d}  unique={len(unique):5d}  lvls={lvls}"
                  f"  go={go}  dom={dom:.0f}%  avg_D={avg_d:.4f}  {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts); dom = max(action_counts)/total*100
    avg_d = sub.D.norm(dim=1).mean().item()
    print(f"\n{'='*60}")
    print("STEP 440 RESULTS")
    print(f"{'='*60}")
    print(f"cb={sub.V.shape[0]}  unique={len(unique)}  levels={lvls}  dom={dom:.0f}%")
    print(f"avg_D={avg_d:.4f}  acts={action_counts}  {elapsed:.0f}s")
    if avg_d < 0.001:
        print("velocities -> 0: drift has no effect")
    elif len(unique) > 3000 and avg_d > 0:
        print("PROMISING: dynamics contributing + exploring")

if __name__ == '__main__':
    main()
