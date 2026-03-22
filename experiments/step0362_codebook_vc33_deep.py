#!/usr/bin/env python3
"""
Step 362 -- VC33 deep: 50K steps, click-space, analyze game-over patterns.

Track frame diffs at game-over vs mid-life.
Try 8x8 click grid (64 regions). 50K steps.
Script: scripts/run_step362_vc33_deep.py
"""

import time
import logging
import random as rng
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
        self.k      = k
        self.d      = d
        self.device = device

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        sample_size = min(500, n)
        idx = torch.randperm(n, device=self.device)[:sample_size]
        sample = self.V[idx]
        sims = sample @ self.V.T
        topk_vals = sims.topk(min(2, n), dim=1).values
        if topk_vals.shape[1] >= 2:
            nearest = topk_vals[:, 1]
        else:
            nearest = topk_vals[:, 0]
        self.thresh = float(nearest.median())

    def process_novelty(self, x, n_cls, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            spawn_label = label if label is not None else 0
            self.V      = x.unsqueeze(0)
            self.labels = torch.tensor([spawn_label], device=self.device)
            return spawn_label
        sims = self.V @ x
        actual_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(max(actual_cls, n_cls), device=self.device)
        for c in range(actual_cls):
            mask = (self.labels == c)
            if mask.sum() == 0: continue
            cs = sims[mask]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()

        prediction  = scores[:n_cls].argmin().item()
        spawn_label = label if label is not None else prediction
        target_mask = (self.labels == prediction)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V      = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([spawn_label], device=self.device)])
            self._update_thresh()
        else:
            target_sims = sims.clone()
            target_sims[~target_mask] = -float('inf')
            winner = target_sims.argmax().item()
            alpha  = 1.0 - float(sims[winner].item())
            self.V[winner] = F.normalize(
                self.V[winner] + alpha * (x - self.V[winner]), dim=0)
        return prediction


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(pooled, fold):
    t      = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if fold.V.shape[0] > 2:
        mean_V = fold.V.mean(dim=0).cpu()
        t_unit = t_unit - mean_V
    return t_unit


def main():
    t0 = time.time()
    print("Step 362 -- VC33 deep: 50K steps + game-over analysis", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(flush=True)

    import arc_agi
    from arcengine import GameState

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    vc33 = next(g for g in games if 'vc33' in g.game_id.lower())

    print(f"Running: {vc33.title} ({vc33.game_id})", flush=True)
    print(f"max_steps=50000  click_grid=8x8 (64 regions)  k=3", flush=True)
    print(flush=True)

    fold = CompressedFold(d=D_ENC, k=3)
    env  = arc.make(vc33.game_id)
    obs  = env.reset()
    action_space = env.action_space

    total_steps     = 0
    game_over_count = 0
    total_levels    = 0
    unique_states   = set()
    cls_counts      = {}
    life_steps      = 0
    steps_per_lvl   = []
    win             = False

    # Game-over analysis
    life_lengths    = []
    go_frames       = []  # last frame before game-over (first 10)
    start_frames    = []  # first frame of life (first 10)
    mid_frames      = []  # mid-life frame (first 10)

    life_start_pooled = None

    max_steps = 50000
    max_resets = 1500

    while total_steps < max_steps and game_over_count < max_resets:
        if obs is None:
            game_over_count += 1
            obs = env.reset()
            if obs is None: break
            life_steps = 0
            life_start_pooled = avgpool16(obs.frame)
            if len(start_frames) < 10:
                start_frames.append(life_start_pooled.copy())
            continue

        if obs.state == GameState.GAME_OVER:
            life_lengths.append(life_steps)
            if len(go_frames) < 10:
                go_frames.append(avgpool16(obs.frame))
            game_over_count += 1
            obs = env.reset()
            if obs is None: break
            life_steps = 0
            life_start_pooled = avgpool16(obs.frame)
            if len(start_frames) < 10:
                start_frames.append(life_start_pooled.copy())
            continue

        if obs.state == GameState.WIN:
            win = True
            print(f"    WIN at step {total_steps}! levels={obs.levels_completed}", flush=True)
            break

        curr_pooled = avgpool16(obs.frame)
        enc         = centered_enc(curr_pooled, fold)
        unique_states.add(hash(curr_pooled.tobytes()))

        # Capture mid-life frame
        if life_steps == 25 and len(mid_frames) < 10:
            mid_frames.append(curr_pooled.copy())

        cls_used = fold.process_novelty(enc, n_cls=N_CLICK, label=None)
        cls_counts[cls_used] = cls_counts.get(cls_used, 0) + 1

        cx, cy = CLICK_GRID[cls_used % N_CLICK]
        obs_levels_before = obs.levels_completed
        obs = env.step(action_space[0], data={"x": cx, "y": cy})
        total_steps += 1
        life_steps  += 1
        if obs is None: break

        if obs.levels_completed > obs_levels_before:
            total_levels = obs.levels_completed
            steps_per_lvl.append(total_steps)
            print(f"    LEVEL {obs.levels_completed} at step {total_steps}"
                  f"  cb={fold.V.shape[0]}", flush=True)

        if obs.state == GameState.WIN:
            win = True
            break

        if total_steps % 10000 == 0:
            print(f"    [step {total_steps:6d}] cb={fold.V.shape[0]}"
                  f"  unique={len(unique_states)}  levels={total_levels}"
                  f"  go={game_over_count}", flush=True)

    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 362 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"win={win}  levels={total_levels}  steps={total_steps}", flush=True)
    print(f"game_overs={game_over_count}", flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
    print(f"unique_states={len(unique_states)}", flush=True)
    print(f"steps_per_level: {steps_per_lvl}", flush=True)
    print(flush=True)

    # Life length analysis
    if life_lengths:
        ll = np.array(life_lengths)
        print(f"Life lengths: min={ll.min()} max={ll.max()} mean={ll.mean():.1f}"
              f" median={np.median(ll):.0f}", flush=True)
        # Distribution
        from collections import Counter
        length_dist = Counter(ll)
        for length in sorted(length_dist.keys())[:10]:
            print(f"  length={length}: {length_dist[length]} lives", flush=True)
    print(flush=True)

    # Frame analysis: start vs mid vs game-over
    if start_frames and go_frames:
        print("Frame analysis (16x16, first 10 lives):", flush=True)
        for i in range(min(len(start_frames), len(go_frames))):
            diff = np.abs(go_frames[i] - start_frames[i])
            n = (diff > 0.01).sum()
            mx = diff.max()
            print(f"  life {i+1}: start->GO diff: max={mx:.4f} cells_changed={n}/256", flush=True)
        if mid_frames:
            print(flush=True)
            for i in range(min(len(mid_frames), len(go_frames))):
                diff = np.abs(go_frames[i] - mid_frames[i])
                n = (diff > 0.01).sum()
                mx = diff.max()
                print(f"  life {i+1}: mid->GO diff: max={mx:.4f} cells_changed={n}/256", flush=True)
    print(flush=True)

    # Top click regions
    top_regions = sorted(cls_counts.items(), key=lambda x: -x[1])[:10]
    print("Top 10 click regions:", flush=True)
    for cls_id, count in top_regions:
        cx, cy = CLICK_GRID[cls_id]
        print(f"  cls {cls_id:2d} (click {cx:2d},{cy:2d}): {count}", flush=True)

    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
