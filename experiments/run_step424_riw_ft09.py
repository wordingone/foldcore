#!/usr/bin/env python3
"""
Step 424 — ReadIsWrite hybrid (tau=0.01 + argmin + error spawn) on FT09 click-space.
69 classes (64 click + 5 simple). 10K steps. Baseline: Level 1 at step 82.
"""

import sys, time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256

CLICK_GRID = [(gx*8+4, gy*8+4) for gy in range(8) for gx in range(8)]
N_CLICK = 64


class RIWHybrid:
    def __init__(self, d, tau=0.01, k=3, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.tau = tau; self.k = k; self.d = d; self.device = device
        self.recent_errors = []; self.n_spawns = 0; self.last_err = 0.0

    def step(self, x, n_actions):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] < n_actions:
            label = self.V.shape[0]
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([label], device=self.device)])
            self.n_spawns += 1
            return label

        sims = self.V @ x
        weights = F.softmax(sims / self.tau, dim=0)
        output = weights @ self.V
        error = x - output
        self.V += torch.outer(weights, error)
        self.V = F.normalize(self.V, dim=1)

        n_cls = int(self.labels.max().item()) + 1
        cls_ids = torch.arange(n_cls, device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        masked = sims.unsqueeze(0) * one_hot + (1 - one_hot) * (-1e9)
        topk_k = min(self.k, masked.shape[1])
        scores = masked.topk(topk_k, dim=1).values.sum(dim=1)
        prediction = scores.argmin().item()

        err_norm = error.norm().item()
        self.last_err = err_norm
        self.recent_errors.append(err_norm)
        if len(self.recent_errors) > 1000:
            self.recent_errors = self.recent_errors[-1000:]
        if len(self.recent_errors) < 10 or err_norm > sorted(self.recent_errors)[len(self.recent_errors)//2]:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([prediction], device=self.device)])
            self.n_spawns += 1
        return prediction


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()

def centered_enc(pooled, sub):
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if sub.V.shape[0] > 2:
        t_unit = t_unit - sub.V.mean(dim=0).cpu()
    return t_unit

def cls_to_action(cls_id, action_space):
    if cls_id < N_CLICK:
        cx, cy = CLICK_GRID[cls_id]
        action6 = next(a for a in action_space if a.is_complex())
        return action6, {"x": cx, "y": cy}
    else:
        simple = [a for a in action_space if not a.is_complex()]
        return simple[(cls_id - N_CLICK) % len(simple)], {}


def main():
    from arcengine import GameState
    import arc_agi

    n_cls = N_CLICK + 5  # 69

    print(f"Step 424: RIW Hybrid on FT09 click-space ({n_cls} classes)")
    print(f"Device: {DEVICE}  tau=0.01  k=3  10K steps", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ft09 = next((g for g in games if 'ft09' in g.game_id.lower()), None)
    if not ft09:
        print("SKIP: ft09 not found"); return

    sub = RIWHybrid(d=D_ENC, tau=0.01, k=3)
    env = arc.make(ft09.game_id)
    obs = env.reset()

    ts = go = lvls = 0
    unique = set()
    t0 = time.time()

    while ts < 10000:
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset()
            if obs is None: break
            continue
        if obs.state == GameState.WIN:
            print(f"WIN at step {ts}!", flush=True); break

        pooled = avgpool16(obs.frame)
        unique.add(hash(pooled.tobytes()))
        x = centered_enc(pooled, sub)

        cls_used = sub.step(x, n_actions=n_cls)
        action, data = cls_to_action(cls_used, env.action_space)

        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            print(f"LEVEL {lvls} at step {ts}  cb={sub.V.shape[0]}", flush=True)

        if ts % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  [step {ts:6d}]  cb={sub.V.shape[0]:5d}  unique={len(unique):5d}"
                  f"  lvls={lvls}  go={go}  {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print("STEP 424 RESULTS")
    print(f"{'='*60}")
    print(f"unique={len(unique)}  levels={lvls}  go={go}  cb={sub.V.shape[0]}  elapsed={elapsed:.0f}s")
    if lvls > 0:
        print(f"Level 1 at step {ts} (baseline=82)")
    else:
        print("No Level 1 (baseline=82)")


if __name__ == '__main__':
    main()
