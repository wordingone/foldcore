#!/usr/bin/env python3
"""
Step 440b — process_novelty + entry drift/velocity.
Exact baseline spawning (V-derived Gram thresh, argmin, class-restricted).
Add: D velocity per entry, drift before match, momentum on attract.
3 seeds, 30K, LS20. Compare to 434b baseline (median 19K, 60%).
"""

import time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256
DRIFT_LR = 0.001


class ProcessNoveltyDrift:
    def __init__(self, d, k=3, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.D = torch.zeros(0, d, device=device)  # velocities
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7; self.k = k; self.d = d; self.device = device

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V = torch.cat([self.V, x_n.unsqueeze(0)])
        self.D = torch.cat([self.D, torch.zeros(1, self.d, device=self.device)])
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
            self.D = torch.zeros(1, self.d, device=self.device)
            self.labels = torch.tensor([sl], device=self.device)
            return sl

        # DRIFT: entries move before matching
        self.V = self.V + DRIFT_LR * self.D
        self.V = F.normalize(self.V, dim=1)

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
            self.D = torch.cat([self.D, torch.zeros(1, self.d, device=self.device)])
            self.labels = torch.cat([self.labels, torch.tensor([sl], device=self.device)])
            if self.V.shape[0] % 100 == 0: self._update_thresh()
        else:
            ts = sims.clone(); ts[~target_mask] = -float('inf')
            winner = ts.argmax().item()
            alpha = 1.0 - float(sims[winner].item())
            attract_dir = x - self.V[winner]
            self.V[winner] = F.normalize(self.V[winner] + alpha * attract_dir, dim=0)
            # MOMENTUM: velocity tracks attract direction
            self.D[winner] = 0.9 * self.D[winner] + 0.1 * attract_dir

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


def run_seed(seed, arc, ls20, max_steps=30000):
    from arcengine import GameState
    fold = ProcessNoveltyDrift(d=D_ENC, k=3)
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
            obs = env.step(action, data=data); ts += 1
            if fold.V.shape[0] >= na: seeded = True; fold._update_thresh()
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
    print(f"Step 440b: process_novelty + drift on LS20. 3 seeds, 30K.", flush=True)
    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return

    results = []
    for seed in [42, 100, 200]:
        t0 = time.time()
        step = run_seed(seed, arc, ls20)
        elapsed = time.time() - t0
        results.append(step)
        print(f"  Seed {seed}: {'step '+str(step) if step else 'none at 30K'}  {elapsed:.0f}s", flush=True)

    nav = [s for s in results if s is not None]
    print(f"\n{'='*60}")
    print("STEP 440b RESULTS")
    print(f"{'='*60}")
    print(f"Navigated: {len(nav)}/3. Steps: {nav}")
    if nav: print(f"Mean: {np.mean(nav):.0f}  Median: {np.median(nav):.0f}")
    print(f"Baseline (434b): median 19K, 60% at 50K")
    if not nav:
        print("Drift may have HURT navigation (0/3 at 30K vs baseline 60% at 50K)")

if __name__ == '__main__':
    main()
