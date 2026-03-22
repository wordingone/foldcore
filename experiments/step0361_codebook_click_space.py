#!/usr/bin/env python3
"""
Step 361 -- Click-space exploration for FT09 and VC33.

Discretize 64x64 into 8x8 regions = 64 click positions.
FT09: 70 classes (64 click regions + ACTION1-5 + ACTION6-center).
VC33: 64 classes (click regions only).
Pure argmin, 16x16 encoding, 10000 steps per game.
Script: scripts/run_step361_click_space.py
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC  = 256

# 8x8 grid of click positions (center of each region)
CLICK_GRID = []
for gy in range(8):
    for gx in range(8):
        cx = gx * 8 + 4
        cy = gy * 8 + 4
        CLICK_GRID.append((cx, cy))
N_CLICK = len(CLICK_GRID)  # 64


class CompressedFold:
    def __init__(self, d, k=3, device=DEVICE):
        self.V      = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7
        self.k      = k
        self.d      = d
        self.device = device

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V      = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels,
                                  torch.tensor([label], device=self.device)])
        self._update_thresh()

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


def cls_to_action(cls_id, action_space, game_type):
    """Map class ID to (action, data)."""
    if game_type == 'vc33':
        # VC33: 64 click regions only, one action (ACTION6)
        cx, cy = CLICK_GRID[cls_id % N_CLICK]
        return action_space[0], {"x": cx, "y": cy}
    else:
        # FT09: classes 0-63 = click regions (ACTION6), 64-68 = ACTION1-5
        if cls_id < N_CLICK:
            cx, cy = CLICK_GRID[cls_id]
            # ACTION6 is the complex one (index 5)
            action6 = next(a for a in action_space if a.is_complex())
            return action6, {"x": cx, "y": cy}
        else:
            # Simple actions (ACTION1-5 = indices 0-4)
            simple_idx = cls_id - N_CLICK
            simple_actions = [a for a in action_space if not a.is_complex()]
            return simple_actions[simple_idx % len(simple_actions)], {}


def run_click_game(arc, game_id, title, game_type, max_steps=10000,
                   max_resets=200, k=3):
    from arcengine import GameState

    if game_type == 'vc33':
        n_cls = N_CLICK  # 64
    else:
        n_cls = N_CLICK + 5  # 64 + 5 = 69

    fold = CompressedFold(d=D_ENC, k=k)
    env  = arc.make(game_id)
    obs  = env.reset()

    total_steps     = 0
    total_resets    = 0
    total_levels    = 0
    game_over_count = 0
    win             = False
    unique_states   = set()
    cls_counts      = {}
    frame_changes   = 0
    steps_per_lvl   = []
    lvl_step_start  = 0

    prev_pooled = None

    while total_steps < max_steps and total_resets < max_resets:
        if obs is None:
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            prev_pooled = None
            lvl_step_start = total_steps
            continue

        if obs.state == GameState.GAME_OVER:
            game_over_count += 1
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            prev_pooled = None
            lvl_step_start = total_steps
            continue

        if obs.state == GameState.WIN:
            win = True
            print(f"    [{title}] WIN at step {total_steps}!", flush=True)
            break

        action_space = env.action_space
        curr_pooled  = avgpool16(obs.frame)
        enc          = centered_enc(curr_pooled, fold)
        unique_states.add(hash(curr_pooled.tobytes()))

        if prev_pooled is not None:
            diff = np.abs(curr_pooled - prev_pooled).max()
            if diff > 0.01:
                frame_changes += 1

        cls_used = fold.process_novelty(enc, n_cls=n_cls, label=None)
        cls_counts[cls_used] = cls_counts.get(cls_used, 0) + 1

        action, data = cls_to_action(cls_used, action_space, game_type)

        obs_levels_before = obs.levels_completed
        prev_pooled = curr_pooled
        obs = env.step(action, data=data)
        total_steps += 1
        if obs is None: break

        if obs.levels_completed > obs_levels_before:
            total_levels = obs.levels_completed
            steps_this   = total_steps - lvl_step_start
            steps_per_lvl.append(steps_this)
            lvl_step_start = total_steps
            print(f"    [{title}] LEVEL {obs.levels_completed} at step {total_steps}"
                  f" ({steps_this} steps)  cb={fold.V.shape[0]}", flush=True)

        if obs.state == GameState.WIN:
            win = True
            break

        if total_steps % 2000 == 0:
            print(f"    [{title}] step {total_steps:5d}"
                  f"  cb={fold.V.shape[0]}  unique={len(unique_states)}"
                  f"  levels={total_levels}  go={game_over_count}"
                  f"  frame_chg={frame_changes}", flush=True)

    # Top click regions
    top_regions = sorted(cls_counts.items(), key=lambda x: -x[1])[:10]

    return {
        'title': title, 'game_id': game_id, 'game_type': game_type,
        'win': win, 'levels': total_levels, 'steps': total_steps,
        'resets': total_resets, 'game_over': game_over_count,
        'cb_final': fold.V.shape[0], 'thresh_final': fold.thresh,
        'unique_states': len(unique_states),
        'frame_changes': frame_changes,
        'n_cls': n_cls,
        'top_regions': top_regions,
        'steps_per_level': steps_per_lvl,
        'n_classes_used': len(cls_counts),
    }


def main():
    t0 = time.time()
    print("Step 361 -- click-space exploration for FT09 and VC33", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Click grid: 8x8 = {N_CLICK} regions", flush=True)
    print(flush=True)

    import arc_agi
    arc   = arc_agi.Arcade()
    games = arc.get_environments()

    ft09 = next(g for g in games if 'ft09' in g.game_id.lower())
    vc33 = next(g for g in games if 'vc33' in g.game_id.lower())

    results = []
    for g, gtype in [(ft09, 'ft09'), (vc33, 'vc33')]:
        print(f"--- Running: {g.title} ({g.game_id}) [{gtype}] ---", flush=True)
        r = run_click_game(arc, g.game_id, g.title, gtype,
                           max_steps=10000, max_resets=200, k=3)
        results.append(r)
        print(f"    Done: levels={r['levels']} cb={r['cb_final']}"
              f" unique={r['unique_states']} frame_chg={r['frame_changes']}", flush=True)
        print(flush=True)

    elapsed = time.time() - t0

    print("=" * 60, flush=True)
    print("STEP 361 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        print(flush=True)
        print(f"  {r['title']} ({r['game_type']}):", flush=True)
        print(f"    win={r['win']}  levels={r['levels']}  steps={r['steps']}", flush=True)
        print(f"    game_overs={r['game_over']}", flush=True)
        print(f"    cb_final={r['cb_final']}  thresh={r['thresh_final']:.4f}", flush=True)
        print(f"    unique_states={r['unique_states']}", flush=True)
        print(f"    frame_changes={r['frame_changes']}/{r['steps']}", flush=True)
        print(f"    n_classes={r['n_cls']}  classes_used={r['n_classes_used']}", flush=True)
        print(f"    steps_per_level: {r['steps_per_level']}", flush=True)
        print(f"    top 10 click regions (cls_id, count):", flush=True)
        for cls_id, count in r['top_regions']:
            if cls_id < N_CLICK:
                cx, cy = CLICK_GRID[cls_id]
                print(f"      cls {cls_id:2d} (click {cx:2d},{cy:2d}): {count}", flush=True)
            else:
                print(f"      cls {cls_id:2d} (ACTION{cls_id-N_CLICK+1}): {count}", flush=True)

    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
