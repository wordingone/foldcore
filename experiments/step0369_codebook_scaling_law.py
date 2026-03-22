#!/usr/bin/env python3
"""
Step 369 -- Scaling law: LS20 with 2 actions only (ACTION1+ACTION2).

Prediction: 2 actions → ~42 steps per level (vs ~84-113 with 4 actions).
3 trials x 10K steps.
Script: scripts/run_step369_scaling_law.py
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256


class CompressedFold:
    def __init__(self, d, k=3, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7
        self.k, self.d, self.device = k, d, device

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels, torch.tensor([label], device=self.device)])
        self._update_thresh()

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        ss = min(500, n)
        idx = torch.randperm(n, device=self.device)[:ss]
        sims = self.V[idx] @ self.V.T
        topk = sims.topk(min(2, n), dim=1).values
        self.thresh = float((topk[:, 1] if topk.shape[1] >= 2 else topk[:, 0]).median())

    def process_novelty(self, x, n_cls):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([0], device=self.device)
            return 0
        sims = self.V @ x
        ac = int(self.labels.max().item()) + 1
        scores = torch.zeros(max(ac, n_cls), device=self.device)
        for c in range(ac):
            m = (self.labels == c)
            if m.sum() == 0: continue
            cs = sims[m]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()
        pred = scores[:n_cls].argmin().item()
        tm = (self.labels == pred)
        if tm.sum() == 0 or sims[tm].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([pred], device=self.device)])
            self._update_thresh()
        else:
            ts = sims.clone(); ts[~tm] = -float('inf')
            w = ts.argmax().item()
            a = 1.0 - float(sims[w].item())
            self.V[w] = F.normalize(self.V[w] + a * (x - self.V[w]), dim=0)
        return pred


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()

def centered_enc(pooled, fold):
    t = F.normalize(torch.from_numpy(pooled.astype(np.float32)), dim=0)
    if fold.V.shape[0] > 2:
        t = t - fold.V.mean(dim=0).cpu()
    return t


def run_trial(arc, game_id, n_actions, max_steps=10000):
    from arcengine import GameState
    fold = CompressedFold(d=D_ENC, k=3)
    env = arc.make(game_id)
    obs = env.reset()
    action_space = env.action_space[:n_actions]  # restrict to first N
    n_cls = n_actions

    total_steps = 0; go = 0; levels = 0; seeded = False
    lvl_steps = []; lvl_start = 0; action_counts = {}

    while total_steps < max_steps and go < 200:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); lvl_start = total_steps
            if obs is None: break; continue
        if obs.state == GameState.WIN: break

        enc = centered_enc(avgpool16(obs.frame), fold)

        if not seeded and fold.V.shape[0] < n_cls:
            i = fold.V.shape[0]; fold._force_add(enc, label=i)
            action = action_space[i]
            obs = env.step(action); total_steps += 1
            action_counts[action.name] = action_counts.get(action.name, 0) + 1
            if fold.V.shape[0] >= n_cls: seeded = True
            continue
        if not seeded: seeded = True

        cls = fold.process_novelty(enc, n_cls=n_cls)
        action = action_space[cls % n_cls]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1

        ol = obs.levels_completed
        obs = env.step(action); total_steps += 1
        if obs is None: break
        if obs.levels_completed > ol:
            levels = obs.levels_completed
            s = total_steps - lvl_start; lvl_steps.append(s); lvl_start = total_steps

    return {'levels': levels, 'steps': total_steps, 'go': go,
            'cb': fold.V.shape[0], 'lvl_steps': lvl_steps,
            'action_counts': action_counts}


def main():
    t0 = time.time()
    print("Step 369 -- Scaling law: 2-action vs 4-action on LS20", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(flush=True)

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    for n_act in [2, 4]:
        print(f"=== {n_act} actions, 3 trials x 10K steps ===", flush=True)
        results = []
        for i in range(3):
            t1 = time.time()
            r = run_trial(arc, ls20.game_id, n_actions=n_act, max_steps=10000)
            dt = time.time() - t1
            results.append(r)
            first = r['lvl_steps'][0] if r['lvl_steps'] else 'none'
            print(f"  Trial {i+1}: levels={r['levels']}  first={first}"
                  f"  cb={r['cb']}  go={r['go']}  acts={r['action_counts']}  {dt:.1f}s", flush=True)
        n_success = sum(1 for r in results if r['levels'] > 0)
        all_firsts = [r['lvl_steps'][0] for r in results if r['lvl_steps']]
        avg_first = np.mean(all_firsts) if all_firsts else float('nan')
        print(f"  P(level in 10K) = {n_success}/3"
              f"  avg_first_level_steps = {avg_first:.0f}" if all_firsts else
              f"  P(level in 10K) = {n_success}/3  avg_first_level_steps = N/A", flush=True)
        print(flush=True)

    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
