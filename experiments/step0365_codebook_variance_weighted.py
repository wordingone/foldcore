#!/usr/bin/env python3
"""
Step 365 -- Variance-weighted encoding on LS20.

Weight cells by codebook variance: high-variance cells (sprite) amplified,
low-variance cells (timer, background) suppressed.
Recompute weights every 100 steps.

365a: single-state + variance weighting (2K steps)
Script: scripts/run_step365_variance_weighted.py
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC  = 256


class VarianceWeightedFold:
    def __init__(self, d, k=3, device=DEVICE):
        self.V      = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7
        self.k, self.d, self.device = k, d, device
        self.weights = torch.ones(d, device=device)  # start uniform
        self.step_count = 0

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels, torch.tensor([label], device=self.device)])
        self._update_thresh()

    def _update_weights(self):
        if self.V.shape[0] < 10: return
        cell_var = self.V.var(dim=0)
        mx = cell_var.max()
        if mx > 0:
            self.weights = cell_var / mx
        else:
            self.weights = torch.ones(self.d, device=self.device)

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        ss = min(500, n)
        idx = torch.randperm(n, device=self.device)[:ss]
        wV = self.V[idx] * self.weights.unsqueeze(0)
        wV_all = self.V * self.weights.unsqueeze(0)
        sims = F.normalize(wV, dim=1) @ F.normalize(wV_all, dim=1).T
        topk = sims.topk(min(2, n), dim=1).values
        self.thresh = float((topk[:, 1] if topk.shape[1] >= 2 else topk[:, 0]).median())

    def process_novelty(self, x, n_cls, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        self.step_count += 1

        # Recompute weights every 100 steps
        if self.step_count % 100 == 0:
            self._update_weights()

        if self.V.shape[0] == 0:
            sl = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([sl], device=self.device)
            return sl, 0.0

        # Weighted similarity
        wx = F.normalize(x * self.weights, dim=0)
        wV = F.normalize(self.V * self.weights.unsqueeze(0), dim=1)
        sims = wV @ wx

        ac = int(self.labels.max().item()) + 1
        scores = torch.zeros(max(ac, n_cls), device=self.device)
        for c in range(ac):
            m = (self.labels == c)
            if m.sum() == 0: continue
            cs = sims[m]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()

        pred = scores[:n_cls].argmin().item()
        sl = label if label is not None else pred

        # Spawn/attract using weighted sims
        tm = (self.labels == pred)
        nearest_sim = float(sims.max().item())
        if tm.sum() == 0 or sims[tm].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([sl], device=self.device)])
            self._update_thresh()
        else:
            ts = sims.clone(); ts[~tm] = -float('inf')
            w = ts.argmax().item()
            # Use unweighted V for attract (preserve raw representation)
            raw_sim = float((self.V[w] @ x).item())
            a = 1.0 - raw_sim
            self.V[w] = F.normalize(self.V[w] + a * (x - self.V[w]), dim=0)
        return pred, nearest_sim


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(pooled, fold):
    t = F.normalize(torch.from_numpy(pooled.astype(np.float32)), dim=0)
    if fold.V.shape[0] > 2:
        t = t - fold.V.mean(dim=0).cpu()
    return t


def main():
    t0 = time.time()
    print("Step 365 -- Variance-weighted encoding on LS20 (2K steps)", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("Weights = per-cell codebook variance. Recompute every 100 steps.", flush=True)
    print(flush=True)

    import arc_agi
    from arcengine import GameState

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    print(f"Running: {ls20.title} ({ls20.game_id})", flush=True)
    print(f"max_steps=2000  k=3", flush=True)
    print(flush=True)

    fold = VarianceWeightedFold(d=D_ENC, k=3)
    env  = arc.make(ls20.game_id)
    obs  = env.reset()
    n_acts = len(env.action_space)

    total_steps     = 0
    game_over_count = 0
    total_levels    = 0
    unique_states   = set()
    action_counts   = {}
    seeded          = False

    max_steps = 2000

    while total_steps < max_steps and game_over_count < 50:
        if obs is None or obs.state == GameState.GAME_OVER:
            game_over_count += 1
            obs = env.reset()
            if obs is None: break
            continue

        if obs.state == GameState.WIN:
            print(f"    WIN at step {total_steps}!", flush=True)
            break

        curr_pooled = avgpool16(obs.frame)
        enc = centered_enc(curr_pooled, fold)
        unique_states.add(hash(curr_pooled.tobytes()))

        if not seeded and fold.V.shape[0] < n_acts:
            i = fold.V.shape[0]
            fold._force_add(enc, label=i)
            action = env.action_space[i]
            obs = env.step(action)
            total_steps += 1
            action_counts[action.name] = action_counts.get(action.name, 0) + 1
            if fold.V.shape[0] >= n_acts:
                seeded = True
                print(f"    [seed done, step {total_steps}]"
                      f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
            continue

        if not seeded: seeded = True

        cls, nsim = fold.process_novelty(enc, n_cls=n_acts)
        action = env.action_space[cls % n_acts]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1

        ol = obs.levels_completed
        obs = env.step(action)
        total_steps += 1
        if obs is None: break

        if obs.levels_completed > ol:
            total_levels = obs.levels_completed
            print(f"    LEVEL {total_levels} at step {total_steps}"
                  f"  cb={fold.V.shape[0]}", flush=True)

        if total_steps % 500 == 0:
            # Show weight stats
            w = fold.weights.cpu()
            top5 = torch.argsort(w, descending=True)[:5]
            bot5 = torch.argsort(w)[:5]
            print(f"    [step {total_steps:5d}] cb={fold.V.shape[0]}"
                  f"  unique={len(unique_states)}  go={game_over_count}"
                  f"  thresh={fold.thresh:.4f}", flush=True)
            print(f"      top-5 cells: {[(int(i), f'{w[i]:.3f}') for i in top5]}", flush=True)
            print(f"      bot-5 cells: {[(int(i), f'{w[i]:.3f}') for i in bot5]}", flush=True)
            print(f"      actions: {action_counts}", flush=True)

    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 365 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"levels={total_levels}  steps={total_steps}  go={game_over_count}", flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
    print(f"unique_states={len(unique_states)}", flush=True)
    print(f"action_counts: {action_counts}", flush=True)
    print(flush=True)

    # Weight analysis
    w = fold.weights.cpu().numpy()
    print(f"Weight stats: min={w.min():.4f} max={w.max():.4f} mean={w.mean():.4f}", flush=True)
    n_zero = (w < 0.01).sum()
    n_high = (w > 0.5).sum()
    print(f"  near-zero (<0.01): {n_zero}/256 cells (suppressed)", flush=True)
    print(f"  high (>0.5): {n_high}/256 cells (amplified)", flush=True)

    # Show as 16x16 grid
    w_grid = w.reshape(16, 16)
    print("\n  Weight grid (16x16):", flush=True)
    for row in w_grid:
        print("    " + " ".join(f"{v:.2f}" for v in row), flush=True)

    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
