#!/usr/bin/env python3
"""
Step 353 -- ARC-AGI-3: pure novelty-seeking (argmin), persistent codebook, 50000 steps.

No exploit phase. argmin for every step across all lives.
Persistent codebook across GAME_OVER resets.

Hypothesis: argmin + persistent codebook = progressive exploration.
Each life starts from same position; argmin pushes beyond known territory.

16x16 encoding (256 dims). 50000 steps on LS20.
Script: scripts/run_step353_pure_novelty.py
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC  = 256   # 16x16


# ==============================================================================
# CompressedFold (novelty mode only)
# ==============================================================================

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
        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        self.thresh = float(G.max(dim=1).values.median())

    def process_novelty(self, x, label=None):
        """Always argmin (least familiar action)."""
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

        prediction  = scores.argmin().item()   # novelty: least familiar
        spawn_label = label if label is not None else prediction
        target_mask = (self.labels == prediction)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V      = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([spawn_label], device=self.device)])
        else:
            target_sims = sims.clone()
            target_sims[~target_mask] = -float('inf')
            winner = target_sims.argmax().item()
            alpha  = 1.0 - float(sims[winner].item())
            self.V[winner] = F.normalize(
                self.V[winner] + alpha * (x - self.V[winner]), dim=0)
        self._update_thresh()
        return prediction


# ==============================================================================
# Encoding
# ==============================================================================

def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()  # (256,)


def centered_enc(pooled, fold):
    t      = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if fold.V.shape[0] > 2:
        mean_V = fold.V.mean(dim=0).cpu()
        t_unit = t_unit - mean_V
    return t_unit


# ==============================================================================
# Game runner
# ==============================================================================

def run_game(arc, game_id, max_steps=50000, max_resets=500,
             k=3, verbose=True):
    from arcengine import GameState

    fold = CompressedFold(d=D_ENC, k=k)
    env  = arc.make(game_id)
    obs  = env.reset()

    total_steps      = 0
    total_resets     = 0
    total_levels     = 0
    game_over_count  = 0
    steps_per_lvl    = []
    cb_snapshots     = []
    action_counts    = {}
    unique_states    = set()
    lvl_step_start   = 0
    win              = False
    seeded           = False

    life_unique_start = 0
    life_new_states   = []

    while total_steps < max_steps and total_resets < max_resets:
        if obs is None:
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            lvl_step_start = total_steps
            continue

        if obs.state == GameState.GAME_OVER:
            game_over_count += 1
            life_new_states.append(len(unique_states) - life_unique_start)
            life_unique_start = len(unique_states)
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            lvl_step_start = total_steps
            continue

        if obs.state == GameState.WIN:
            win = True
            if verbose:
                print(f"    [step {total_steps}] WIN! levels={obs.levels_completed}"
                      f"  cb={fold.V.shape[0]}", flush=True)
            break

        action_space = env.action_space
        n_acts       = len(action_space)

        curr_pooled = avgpool16(obs.frame)
        enc         = centered_enc(curr_pooled, fold)

        state_hash = hash(curr_pooled.tobytes())
        unique_states.add(state_hash)

        # Force-seed: one entry per action class
        if not seeded and fold.V.shape[0] < n_acts:
            i = fold.V.shape[0]
            fold._force_add(enc, label=i)
            action   = action_space[i % n_acts]
            cls_used = i
            data = {}
            if action.is_complex():
                arr = np.array(obs.frame[0])
                cy, cx = divmod(int(np.argmax(arr)), 64)
                data = {"x": cx, "y": cy}
            obs_levels_before = obs.levels_completed
            obs = env.step(action, data=data)
            total_steps += 1
            action_counts[action.name] = action_counts.get(action.name, 0) + 1
            if fold.V.shape[0] >= n_acts:
                seeded = True
                life_unique_start = len(unique_states)
                if verbose:
                    print(f"    [seed done, step {total_steps}]"
                          f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
            if obs is not None and obs.levels_completed > obs_levels_before:
                total_levels = obs.levels_completed
            if obs is not None and obs.state == GameState.WIN:
                win = True
            continue

        if not seeded:
            seeded = True

        # Always novelty
        cls_used = fold.process_novelty(enc, label=None)
        action   = action_space[cls_used % n_acts]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1

        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}

        obs_levels_before = obs.levels_completed
        obs = env.step(action, data=data)
        total_steps += 1

        if obs is None: break

        # Snapshot every 5000 steps
        if total_steps % 5000 == 0:
            n = fold.V.shape[0]
            mean_norm = float(fold.V.mean(dim=0).norm().item()) if n > 0 else 0.0
            cb_snapshots.append((total_steps, n, fold.thresh, mean_norm, len(unique_states)))
            if verbose:
                # new states in last 5000
                recent_new = sum(life_new_states[-38:]) if len(life_new_states) >= 38 else sum(life_new_states)
                print(f"    [step {total_steps:6d}]"
                      f"  cb={n:5d}  thresh={fold.thresh:.4f}"
                      f"  unique={len(unique_states):5d}"
                      f"  levels={total_levels}  go={game_over_count}", flush=True)

        if obs.levels_completed > obs_levels_before:
            total_levels = obs.levels_completed
            steps_this   = total_steps - lvl_step_start
            steps_per_lvl.append(steps_this)
            lvl_step_start = total_steps
            if verbose:
                print(f"    [step {total_steps}] LEVEL {obs.levels_completed}"
                      f"  steps_this={steps_this}"
                      f"  cb={fold.V.shape[0]}", flush=True)

        if obs.state == GameState.WIN:
            win = True
            break

    cls_dist = {}
    if fold.V.shape[0] > 0:
        for lbl in fold.labels.cpu().numpy():
            k_ = int(lbl)
            cls_dist[k_] = cls_dist.get(k_, 0) + 1

    return {
        'win':             win,
        'levels':          total_levels,
        'steps':           total_steps,
        'resets':          total_resets,
        'game_over':       game_over_count,
        'steps_per_level': steps_per_lvl,
        'cb_final':        fold.V.shape[0],
        'thresh_final':    fold.thresh,
        'cb_snapshots':    cb_snapshots,
        'cls_dist':        dict(sorted(cls_dist.items())),
        'action_counts':   action_counts,
        'unique_states':   len(unique_states),
        'life_new_states': life_new_states,
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    t0 = time.time()
    print("Step 353 -- ARC-AGI-3: pure novelty, persistent codebook, 50000 steps", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("All steps: argmin (novelty). Codebook persists across lives.", flush=True)
    print(flush=True)

    import arc_agi
    arc   = arc_agi.Arcade()
    games = arc.get_environments()
    ls20  = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if ls20 is None:
        print("ERROR: LS20 not found", flush=True)
        return

    print(f"Running: {ls20.title} ({ls20.game_id})", flush=True)
    print(f"max_steps=50000  k=3", flush=True)
    print(flush=True)

    r = run_game(arc, ls20.game_id,
                 max_steps=50000, max_resets=500,
                 k=3, verbose=True)

    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 353 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"win={r['win']}  levels={r['levels']}  steps={r['steps']}", flush=True)
    print(f"resets={r['resets']}  game_over={r['game_over']}", flush=True)
    print(f"cb_final={r['cb_final']}  thresh={r['thresh_final']:.4f}", flush=True)
    print(f"unique_states={r['unique_states']}", flush=True)
    print(flush=True)
    print(f"steps_per_level: {r['steps_per_level']}", flush=True)
    print(f"action_counts: {r['action_counts']}", flush=True)
    print(f"cls_dist: {r['cls_dist']}", flush=True)
    print(flush=True)

    print("Codebook over time:", flush=True)
    for s, cb, th, mn, uniq in r['cb_snapshots']:
        print(f"  step {s:6d}:  cb={cb:6d}  thresh={th:.4f}"
              f"  mean_norm={mn:.4f}  unique={uniq:5d}", flush=True)

    print(flush=True)
    # Frontier per life (first 50 + last 20)
    lns = r['life_new_states']
    print(f"New states per life (first 30):", flush=True)
    for i, n in enumerate(lns[:30]):
        print(f"  life {i+1:3d}: +{n:4d}", flush=True)
    if len(lns) > 50:
        print(f"  ...", flush=True)
        print(f"New states per life (last 20):", flush=True)
        for i, n in enumerate(lns[-20:]):
            print(f"  life {len(lns)-19+i:3d}: +{n:4d}", flush=True)
    elif len(lns) > 30:
        for i, n in enumerate(lns[30:]):
            print(f"  life {31+i:3d}: +{n:4d}", flush=True)
    print(f"Total lives: {len(lns)}", flush=True)

    # Convergence check
    if len(lns) >= 20:
        avg_last20 = sum(lns[-20:]) / 20
        print(f"Avg new states/life (last 20): {avg_last20:.1f}", flush=True)

    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
