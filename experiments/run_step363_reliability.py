#!/usr/bin/env python3
"""
Step 363 -- Reliability + multi-level: LS20 5x50K + FT09 1x50K.

A. LS20: 5 trials x 50K steps. Report P(level 1 in 50K).
B. FT09: 1 trial x 50K steps (click-space). Report multi-level count.

Same configs as Steps 358/361. Sampled thresh, 16x16, pure argmin.
Script: scripts/run_step363_reliability.py
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC  = 256

CLICK_GRID = []
for gy in range(8):
    for gx in range(8):
        CLICK_GRID.append((gx * 8 + 4, gy * 8 + 4))
N_CLICK = 64


class CompressedFold:
    def __init__(self, d, k=3, device=DEVICE):
        self.V      = torch.zeros(0, d, device=device)
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

    def process_novelty(self, x, n_cls, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            sl = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([sl], device=self.device)
            return sl
        sims = self.V @ x
        ac = int(self.labels.max().item()) + 1
        scores = torch.zeros(max(ac, n_cls), device=self.device)
        for c in range(ac):
            m = (self.labels == c)
            if m.sum() == 0: continue
            cs = sims[m]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()
        pred = scores[:n_cls].argmin().item()
        sl = label if label is not None else pred
        tm = (self.labels == pred)
        if tm.sum() == 0 or sims[tm].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([sl], device=self.device)])
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


def run_ls20(arc, game_id, max_steps=50000):
    from arcengine import GameState
    fold = CompressedFold(d=D_ENC, k=3)
    env = arc.make(game_id)
    obs = env.reset()
    n_acts = len(env.action_space)
    total_steps = 0; levels = 0; go = 0; seeded = False
    lvl_steps = []; lvl_start = 0
    while total_steps < max_steps and go < 500:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); lvl_start = total_steps
            if obs is None: break; continue
        if obs.state == GameState.WIN: break
        enc = centered_enc(avgpool16(obs.frame), fold)
        if not seeded and fold.V.shape[0] < n_acts:
            i = fold.V.shape[0]; fold._force_add(enc, label=i)
            action = env.action_space[i]
            obs = env.step(action); total_steps += 1
            if fold.V.shape[0] >= n_acts: seeded = True
            continue
        if not seeded: seeded = True
        cls = fold.process_novelty(enc, n_cls=n_acts)
        ol = obs.levels_completed
        obs = env.step(env.action_space[cls % n_acts]); total_steps += 1
        if obs and obs.levels_completed > ol:
            levels = obs.levels_completed
            lvl_steps.append(total_steps - lvl_start); lvl_start = total_steps
    return {'levels': levels, 'steps': total_steps, 'go': go,
            'cb': fold.V.shape[0], 'lvl_steps': lvl_steps}


def run_ft09(arc, game_id, max_steps=50000):
    from arcengine import GameState
    n_cls = N_CLICK + 5  # 64 click + 5 simple
    fold = CompressedFold(d=D_ENC, k=3)
    env = arc.make(game_id)
    obs = env.reset()
    action_space = env.action_space
    action6 = next(a for a in action_space if a.is_complex())
    simple_actions = [a for a in action_space if not a.is_complex()]
    total_steps = 0; levels = 0; go = 0; seeded = False
    lvl_steps = []; lvl_start = 0
    while total_steps < max_steps and go < 500:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); lvl_start = total_steps
            if obs is None: break; continue
        if obs.state == GameState.WIN:
            print(f"    [FT09] WIN at step {total_steps}! levels={obs.levels_completed}", flush=True)
            break
        enc = centered_enc(avgpool16(obs.frame), fold)
        if not seeded and fold.V.shape[0] < n_cls:
            i = fold.V.shape[0]; fold._force_add(enc, label=i)
            if i < N_CLICK:
                cx, cy = CLICK_GRID[i]
                action, data = action6, {"x": cx, "y": cy}
            else:
                action, data = simple_actions[(i - N_CLICK) % len(simple_actions)], {}
            obs = env.step(action, data=data); total_steps += 1
            if fold.V.shape[0] >= n_cls: seeded = True
            continue
        if not seeded: seeded = True
        cls = fold.process_novelty(enc, n_cls=n_cls)
        ol = obs.levels_completed
        if cls < N_CLICK:
            cx, cy = CLICK_GRID[cls]
            action, data = action6, {"x": cx, "y": cy}
        else:
            action, data = simple_actions[(cls - N_CLICK) % len(simple_actions)], {}
        obs = env.step(action, data=data); total_steps += 1
        if obs and obs.levels_completed > ol:
            levels = obs.levels_completed
            s = total_steps - lvl_start; lvl_steps.append(s); lvl_start = total_steps
            print(f"    [FT09] LEVEL {levels} at step {total_steps} ({s} steps)", flush=True)
    return {'levels': levels, 'steps': total_steps, 'go': go,
            'cb': fold.V.shape[0], 'lvl_steps': lvl_steps}


def main():
    t0 = time.time()
    print("Step 363 -- Reliability trials: LS20 5x50K + FT09 1x50K", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(flush=True)

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())
    ft09 = next(g for g in games if 'ft09' in g.game_id.lower())

    # A. LS20: 5 trials
    print("=== A. LS20: 5 trials x 50K steps ===", flush=True)
    ls20_results = []
    for trial in range(5):
        t1 = time.time()
        r = run_ls20(arc, ls20.game_id, max_steps=50000)
        dt = time.time() - t1
        ls20_results.append(r)
        first_lvl = r['lvl_steps'][0] if r['lvl_steps'] else 'none'
        print(f"  Trial {trial+1}: levels={r['levels']}  first_level_at={first_lvl}"
              f"  cb={r['cb']}  go={r['go']}  {dt:.1f}s", flush=True)

    n_success = sum(1 for r in ls20_results if r['levels'] > 0)
    print(f"\n  LS20 P(level 1 in 50K) = {n_success}/5 = {n_success/5:.0%}", flush=True)
    print(flush=True)

    # B. FT09: 1 trial x 50K
    print("=== B. FT09: 1 trial x 50K steps (click-space) ===", flush=True)
    t1 = time.time()
    ft09_r = run_ft09(arc, ft09.game_id, max_steps=50000)
    dt = time.time() - t1
    print(f"  FT09: levels={ft09_r['levels']}  cb={ft09_r['cb']}"
          f"  go={ft09_r['go']}  {dt:.1f}s", flush=True)
    print(f"  steps_per_level: {ft09_r['lvl_steps']}", flush=True)

    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 363 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"LS20: {n_success}/5 trials completed level 1", flush=True)
    for i, r in enumerate(ls20_results):
        print(f"  trial {i+1}: levels={r['levels']}  lvl_steps={r['lvl_steps']}", flush=True)
    print(flush=True)
    print(f"FT09: {ft09_r['levels']} levels in 50K steps", flush=True)
    print(f"  lvl_steps: {ft09_r['lvl_steps']}", flush=True)
    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
