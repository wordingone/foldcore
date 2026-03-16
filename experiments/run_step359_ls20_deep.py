#!/usr/bin/env python3
"""
Step 359 -- LS20 deep: 50000 steps, pure argmin, 16x16, sampled thresh.

How many levels can pure novelty clear with sampled thresh?
Script: scripts/run_step359_ls20_deep.py
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC  = 256


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

    def process_novelty(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            spawn_label = label if label is not None else 0
            self.V      = x.unsqueeze(0)
            self.labels = torch.tensor([spawn_label], device=self.device)
            return spawn_label
        sims = self.V @ x
        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(n_cls, device=self.device)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0: continue
            cs = sims[mask]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()

        prediction  = scores.argmin().item()
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


def run_game(arc, game_id, title, max_steps=10000, max_resets=200, k=3):
    from arcengine import GameState

    fold = CompressedFold(d=D_ENC, k=k)
    env  = arc.make(game_id)
    obs  = env.reset()

    total_steps     = 0
    total_resets    = 0
    total_levels    = 0
    game_over_count = 0
    action_counts   = {}
    unique_states   = set()
    win             = False
    seeded          = False
    steps_per_lvl   = []
    lvl_step_start  = 0
    frame_changes   = 0  # how many steps had frame change > 0.01

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
            print(f"    [{title}] WIN at step {total_steps}! levels={obs.levels_completed}", flush=True)
            break

        action_space = env.action_space
        n_acts       = len(action_space)

        curr_pooled = avgpool16(obs.frame)
        enc         = centered_enc(curr_pooled, fold)
        unique_states.add(hash(curr_pooled.tobytes()))

        # Track frame changes
        if prev_pooled is not None:
            diff = np.abs(curr_pooled - prev_pooled).max()
            if diff > 0.01:
                frame_changes += 1

        if not seeded and fold.V.shape[0] < n_acts:
            i = fold.V.shape[0]
            fold._force_add(enc, label=i)
            action = action_space[i % n_acts]
            data = {}
            if action.is_complex():
                arr = np.array(obs.frame[0])
                cy, cx = divmod(int(np.argmax(arr)), 64)
                data = {"x": cx, "y": cy}
            prev_pooled = curr_pooled
            obs = env.step(action, data=data)
            total_steps += 1
            action_counts[action.name] = action_counts.get(action.name, 0) + 1
            if fold.V.shape[0] >= n_acts:
                seeded = True
            continue

        if not seeded:
            seeded = True

        cls_used = fold.process_novelty(enc, label=None)
        action = action_space[cls_used % n_acts]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1

        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}

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
                  f" (this_life={steps_this})  cb={fold.V.shape[0]}", flush=True)

        if obs.state == GameState.WIN:
            win = True
            print(f"    [{title}] WIN at step {total_steps}! levels={obs.levels_completed}", flush=True)
            break

    cls_dist = {}
    if fold.V.shape[0] > 0:
        for lbl in fold.labels.cpu().numpy():
            k_ = int(lbl)
            cls_dist[k_] = cls_dist.get(k_, 0) + 1

    return {
        'title': title, 'game_id': game_id,
        'win': win, 'levels': total_levels, 'steps': total_steps,
        'resets': total_resets, 'game_over': game_over_count,
        'steps_per_level': steps_per_lvl,
        'cb_final': fold.V.shape[0], 'thresh_final': fold.thresh,
        'cls_dist': dict(sorted(cls_dist.items())),
        'action_counts': action_counts, 'unique_states': len(unique_states),
        'n_acts': n_acts, 'frame_changes': frame_changes,
        'frame_change_rate': frame_changes / max(total_steps - 1, 1),
    }


def main():
    t0 = time.time()
    print("Step 359 -- LS20 deep: 50000 steps, pure argmin, sampled thresh", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(flush=True)

    import arc_agi
    arc   = arc_agi.Arcade()
    games = arc.get_environments()
    ls20  = next(g for g in games if 'ls20' in g.game_id.lower())

    print(f"Running: {ls20.title} ({ls20.game_id})", flush=True)
    print(f"max_steps=50000  k=3", flush=True)
    print(flush=True)

    r = run_game(arc, ls20.game_id, ls20.title, max_steps=50000, max_resets=500, k=3)

    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 359 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"win={r['win']}  levels={r['levels']}  steps={r['steps']}", flush=True)
    print(f"game_overs={r['game_over']}  n_actions={r['n_acts']}", flush=True)
    print(f"cb_final={r['cb_final']}  thresh={r['thresh_final']:.4f}", flush=True)
    print(f"unique_states={r['unique_states']}", flush=True)
    print(f"frame_changes={r['frame_changes']}/{r['steps']}"
          f" ({r['frame_change_rate']:.1%})", flush=True)
    print(f"steps_per_level: {r['steps_per_level']}", flush=True)
    print(f"action_counts: {r['action_counts']}", flush=True)
    print(f"cls_dist: {r['cls_dist']}", flush=True)
    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
