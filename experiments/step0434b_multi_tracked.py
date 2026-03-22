#!/usr/bin/env python3
"""
Step 434b — Random vs substrate with step-to-level tracking.
10 seeds random + 10 seeds substrate, 50K each. Record exact step of Level 1.
"""

import time, logging, random as pyrandom, math
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256


class ProcessNovelty:
    def __init__(self, d, k=3, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7; self.k = k; self.d = d; self.device = device
    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels, torch.tensor([label], device=self.device)])
    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        ss = min(500, n)
        idx = torch.randperm(n, device=self.device)[:ss]
        s = self.V[idx] @ self.V.T
        t = s.topk(min(2, n), dim=1).values
        self.thresh = float((t[:, 1] if t.shape[1] >= 2 else t[:, 0]).median())
    def process_novelty(self, x, n_cls, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            sl = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([sl], device=self.device)
            return sl
        sims = self.V @ x
        actual_cls = int(self.labels.max().item()) + 1
        cls_ids = torch.arange(max(actual_cls, n_cls), device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        masked = sims.unsqueeze(0) * one_hot + (1 - one_hot) * (-1e9)
        topk_k = min(self.k, masked.shape[1])
        scores = masked.topk(topk_k, dim=1).values.sum(dim=1)
        prediction = scores[:n_cls].argmin().item()
        sl = label if label is not None else prediction
        target_mask = (self.labels == prediction)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([sl], device=self.device)])
            if self.V.shape[0] % 100 == 0: self._update_thresh()
        else:
            ts = sims.clone(); ts[~target_mask] = -float('inf')
            w = ts.argmax().item()
            a = 1.0 - float(sims[w].item())
            self.V[w] = F.normalize(self.V[w] + a * (x - self.V[w]), dim=0)
        return prediction


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()

def centered_enc(pooled, fold):
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if fold.V.shape[0] > 2:
        t_unit = t_unit - fold.V.mean(dim=0).cpu()
    return t_unit


def run_random(seed, arc, ls20, max_steps=50000):
    from arcengine import GameState
    env = arc.make(ls20.game_id); obs = env.reset()
    na = len(env.action_space); rng = pyrandom.Random(seed)
    ts = go = 0; level_step = None
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        idx = rng.randint(0, na-1); action = env.action_space[idx]; data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0]); cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}
        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts += 1
        if obs is not None and obs.levels_completed > obs_before and level_step is None:
            level_step = ts
    return level_step


def run_substrate(seed, arc, ls20, max_steps=50000):
    from arcengine import GameState
    fold = ProcessNovelty(d=D_ENC, k=3)
    env = arc.make(ls20.game_id); obs = env.reset()
    na = len(env.action_space); np.random.seed(seed)
    ts = go = 0; seeded = False; level_step = None
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); seeded = False; continue
        if obs.state == GameState.WIN: break
        pooled = avgpool16(obs.frame); x = centered_enc(pooled, fold)
        if not seeded and fold.V.shape[0] < na:
            i = fold.V.shape[0]; fold._force_add(x, label=i)
            action = env.action_space[i % na]; data = {}
            if action.is_complex():
                arr = np.array(obs.frame[0]); cy, cx = divmod(int(np.argmax(arr)), 64)
                data = {"x": cx, "y": cy}
            obs_before = obs.levels_completed
            obs = env.step(action, data=data); ts += 1
            if fold.V.shape[0] >= na: seeded = True; fold._update_thresh()
            if obs is not None and obs.levels_completed > obs_before and level_step is None:
                level_step = ts
            continue
        if not seeded: seeded = True
        cls_used = fold.process_novelty(x, n_cls=na, label=None)
        action = env.action_space[cls_used % na]; data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0]); cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}
        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts += 1
        if obs is not None and obs.levels_completed > obs_before and level_step is None:
            level_step = ts
    return level_step


def main():
    import arc_agi
    print(f"Step 434b: Random vs substrate with step tracking. 10 seeds each, 50K.", flush=True)
    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return

    print("\n--- Random walk ---", flush=True)
    t0 = time.time()
    rand_results = []
    for seed in range(10):
        step = run_random(seed, arc, ls20)
        rand_results.append(step)
        print(f"  Seed {seed}: {'step '+str(step) if step else 'none at 50K'}", flush=True)
    t_rand = time.time() - t0

    print(f"\n--- Substrate (process_novelty) ---", flush=True)
    t0 = time.time()
    sub_results = []
    for seed in range(10):
        step = run_substrate(seed, arc, ls20)
        sub_results.append(step)
        print(f"  Seed {seed}: {'step '+str(step) if step else 'none at 50K'}", flush=True)
    t_sub = time.time() - t0

    print(f"\n{'='*60}")
    print("STEP 434b — STEP-TO-LEVEL COMPARISON")
    print(f"{'='*60}")

    rand_nav = [s for s in rand_results if s is not None]
    sub_nav = [s for s in sub_results if s is not None]

    print(f"Random:    {len(rand_nav)}/10 navigated. Steps: {rand_nav}")
    print(f"Substrate: {len(sub_nav)}/10 navigated. Steps: {sub_nav}")

    if rand_nav:
        print(f"Random mean: {np.mean(rand_nav):.0f}  median: {np.median(rand_nav):.0f}")
    if sub_nav:
        print(f"Substrate mean: {np.mean(sub_nav):.0f}  median: {np.median(sub_nav):.0f}")

    print(f"\nRandom: {t_rand:.0f}s  Substrate: {t_sub:.0f}s")


if __name__ == '__main__':
    main()
