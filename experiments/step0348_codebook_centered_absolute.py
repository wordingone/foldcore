#!/usr/bin/env python3
"""
Step 348 -- ARC-AGI-3: centered absolute frame encoding on LS20.

ENCODING: absolute frame (not diff), centered by codebook mean.
  frame_8x8  = avgpool(frame, 8) / 15.0   # (64,) in [0,1]
  if V.shape[0] > 2:
      x = F.normalize(frame_8x8 - mean_V, dim=0)
  else:
      x = F.normalize(frame_8x8, dim=0)

LABEL: action taken from this state — always.
  Bootstrap (step < 200): action = random, process(state, label=action_idx)
  Exploitation: 90% greedy (label=None, action=prediction), 10% random (label=action_idx)

GAME_OVER: no stamp (don't reinforce fatal state→action pairs).
LEVEL COMPLETE: trust codebook (no extra stamp).

5000 steps on LS20. Report codebook size, levels, action distribution.
Script: scripts/run_step348_centered_absolute.py

FIX (348b): force-seed N_ACTS entries (one per action class) using _force_add()
before bootstrap begins. After N_ACTS entries: centering engages. Bootstrap
then builds diverse state→action pairs in the correctly-centered space.
"""

import time
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC  = 64


# ==============================================================================
# CompressedFold — same as step 341 (state-derived thresh, Stage 2/3/7)
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
        """Bypass threshold — directly append entry (used for seeding)."""
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

    def process(self, x, label=None):
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
        prediction  = scores.argmax().item()
        attract_target = prediction
        spawn_label    = label if label is not None else prediction
        target_mask = (self.labels == attract_target)
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

def avgpool8(frame):
    """avgpool 8x8 -> 64-dim float32, normalized to [0,1]."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(8, 8, 8, 8).mean(axis=(2, 3)).flatten()  # (64,)

def centered_frame(pooled, fold):
    """
    Center absolute frame in unit-norm space.
    Step 1: normalize pooled to unit sphere.
    Step 2: subtract codebook mean (also in unit-norm space).
    process() will normalize the result again.
    """
    t     = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)  # unit sphere first
    if fold.V.shape[0] > 2:
        mean_V = fold.V.mean(dim=0).cpu()  # mean of unit vectors
        t_unit = t_unit - mean_V           # center in unit-norm space
    return t_unit  # (64,) — process() will normalize again


# ==============================================================================
# Game runner
# ==============================================================================

def run_game(arc, game_id, max_steps=5000, max_resets=50,
             bootstrap_steps=200, epsilon=0.10, k=3, verbose=True):
    from arcengine import GameState

    fold = CompressedFold(d=D_ENC, k=k)
    env  = arc.make(game_id)
    obs  = env.reset()

    total_steps     = 0
    total_resets    = 0
    total_levels    = 0
    game_over_count = 0
    steps_per_lvl   = []
    cb_snapshots    = []   # (step, cb_size, thresh, mean_norm)
    action_counts   = {}
    lvl_step_start  = 0
    win             = False

    prev_action  = None
    prev_levels  = 0
    seeded       = False   # force-seeded one entry per action class yet?

    while total_steps < max_steps and total_resets < max_resets:
        if obs is None:
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            prev_action = None
            prev_levels = 0
            lvl_step_start = total_steps
            continue

        if obs.state == GameState.GAME_OVER:
            game_over_count += 1
            # No stamp — don't reinforce fatal state→action
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            prev_action = None
            prev_levels = 0
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

        # Encode: centered absolute frame
        pooled = avgpool8(obs.frame)
        enc    = centered_frame(pooled, fold)

        # Force-seed: one entry per action class before bootstrap starts
        if not seeded and fold.V.shape[0] < n_acts:
            i = fold.V.shape[0]
            fold._force_add(enc, label=i)
            action   = action_space[i % n_acts]
            cls_used = i
            # Execute seed action
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
                if verbose:
                    print(f"    [seed done, step {total_steps}]"
                          f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
            if obs is not None and obs.levels_completed > obs_levels_before:
                total_levels = obs.levels_completed
                if verbose:
                    print(f"    [step {total_steps}] LEVEL {obs.levels_completed}"
                          f"  (during seed)", flush=True)
            if obs is not None and obs.state == GameState.WIN:
                win = True
            continue

        if not seeded:
            seeded = True

        # Action selection + codebook update
        if total_steps < bootstrap_steps:
            # Pure random exploration — stamp state→action
            action   = random.choice(action_space)
            cls_used = action_space.index(action) if action in action_space else 0
            fold.process(enc, label=cls_used)
        else:
            # Epsilon-greedy exploitation
            if random.random() < epsilon:
                action   = random.choice(action_space)
                cls_used = action_space.index(action) if action in action_space else 0
                fold.process(enc, label=cls_used)
            else:
                cls_used = fold.process(enc, label=None)
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

        # Snapshot every 250 steps
        if total_steps % 250 == 0:
            n = fold.V.shape[0]
            mean_norm = float(fold.V.mean(dim=0).norm().item()) if n > 0 else 0.0
            cb_snapshots.append((total_steps, n, fold.thresh, mean_norm))
            if verbose:
                phase = "bootstrap" if total_steps <= bootstrap_steps else "exploit"
                print(f"    [step {total_steps:4d}] {phase:9s}"
                      f"  cb={n:3d}  thresh={fold.thresh:.4f}"
                      f"  mean_norm={mean_norm:.4f}"
                      f"  levels={total_levels}  go={game_over_count}", flush=True)

        # Level completion — trust existing codebook, no extra stamp
        if obs.levels_completed > obs_levels_before:
            total_levels = obs.levels_completed
            steps_this   = total_steps - lvl_step_start
            steps_per_lvl.append(steps_this)
            lvl_step_start = total_steps
            if verbose:
                print(f"    [step {total_steps}] LEVEL {obs.levels_completed}"
                      f"  steps_this={steps_this}"
                      f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)

        if obs.state == GameState.WIN:
            win = True
            break

        prev_levels = obs.levels_completed

    cls_dist = {}
    if fold.V.shape[0] > 0:
        for lbl in fold.labels.cpu().numpy():
            k_ = int(lbl)
            cls_dist[k_] = cls_dist.get(k_, 0) + 1

    return {
        'win':           win,
        'levels':        total_levels,
        'steps':         total_steps,
        'resets':        total_resets,
        'game_over':     game_over_count,
        'steps_per_level': steps_per_lvl,
        'cb_final':      fold.V.shape[0],
        'thresh_final':  fold.thresh,
        'cb_snapshots':  cb_snapshots,
        'cls_dist':      dict(sorted(cls_dist.items())),
        'action_counts': action_counts,
        'fold':          fold,
    }


# ==============================================================================
# Centering analysis: cosine sim per action in centered absolute space
# ==============================================================================

def analyze_separation(arc, game_id, fold):
    """
    From same start state: check centered-frame cos-sim per action.
    How well does centering separate action classes?
    """
    if fold.V.shape[0] < 3:
        print("  [analysis] codebook too small", flush=True)
        return

    mean_V = fold.V.mean(dim=0)
    print(f"  codebook mean norm: {mean_V.norm():.4f}", flush=True)

    # Sample 10 frames, compute centered version, check nearest codebook class
    env  = arc.make(game_id)
    obs  = env.reset()
    action_space = env.action_space
    n_acts = len(action_space)

    correct_preds = {a.name: 0 for a in action_space}
    total_preds   = {a.name: 0 for a in action_space}

    print(f"  Nearest-neighbor class per action (10 trials, from start state):", flush=True)
    for act_idx, act in enumerate(action_space):
        preds = []
        for trial in range(10):
            env2  = arc.make(game_id)
            obs2  = env2.reset()
            p     = avgpool8(obs2.frame)
            t     = torch.from_numpy(p.astype(np.float32))
            cx    = t - mean_V.cpu()
            cx_n  = F.normalize(cx, dim=0)
            sims  = (fold.V.cpu() @ cx_n)
            pred  = int(sims.argmax().item())
            pred_cls = int(fold.labels[pred].item())
            preds.append(pred_cls)
        counts = {}
        for c in preds: counts[c] = counts.get(c, 0) + 1
        print(f"    {act.name} (class {act_idx}): nearest={counts}", flush=True)


# ==============================================================================
# Main
# ==============================================================================

def main():
    t0 = time.time()
    print("Step 348c -- ARC-AGI-3: abs frame, unit-norm centering + force-seed", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("Encoding: abs_frame - codebook_mean, label = action taken", flush=True)
    print("Bootstrap: 200 steps random. Exploitation: 10% epsilon.", flush=True)
    print("GAME_OVER: no stamp. LEVEL: trust codebook.", flush=True)
    print(flush=True)

    import arc_agi
    arc   = arc_agi.Arcade()
    games = arc.get_environments()
    ls20  = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if ls20 is None:
        print("ERROR: LS20 not found", flush=True)
        return

    print(f"Running: {ls20.title} ({ls20.game_id})", flush=True)
    print(f"max_steps=5000  bootstrap=200  epsilon=0.10  k=3", flush=True)
    print(flush=True)

    r = run_game(arc, ls20.game_id,
                 max_steps=5000, max_resets=50,
                 bootstrap_steps=200, epsilon=0.10, k=3, verbose=True)

    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 348c SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"win={r['win']}  levels={r['levels']}  steps={r['steps']}", flush=True)
    print(f"resets={r['resets']}  game_over={r['game_over']}", flush=True)
    print(f"cb_final={r['cb_final']}  thresh={r['thresh_final']:.4f}", flush=True)
    print(flush=True)
    print(f"steps_per_level: {r['steps_per_level']}", flush=True)
    print(f"action_counts: {r['action_counts']}", flush=True)
    print(f"cls_dist: {r['cls_dist']}", flush=True)
    print(flush=True)

    print("Codebook over time:", flush=True)
    for s, cb, th, mn in r['cb_snapshots']:
        phase = "bootstrap" if s <= 200 else "exploit"
        print(f"  step {s:4d} [{phase:9s}]:  cb={cb:3d}  thresh={th:.4f}"
              f"  mean_norm={mn:.4f}", flush=True)

    print(flush=True)
    print("Separation analysis (centered absolute):", flush=True)
    analyze_separation(arc, ls20.game_id, r['fold'])

    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
