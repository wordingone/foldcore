#!/usr/bin/env python3
"""
Step 345 -- ARC-AGI-3: pure exploration + negative signal from GAME_OVER.

Two changes from Step 344:
  1. PURE EXPLORATION FIRST: steps 0-499 — action=random, but process(diff, label=action)
     builds the codebook. After step 500: process() selects actions (epsilon=0.10).
  2. GAME_OVER NEGATIVE SIGNAL: on GAME_OVER, stamp prev_diff with class = prev_action + N_ACTS.
     When process() predicts class >= N_ACTS: pick a random DIFFERENT action (avoid the bad one).

Good classes:  0 .. N_ACTS-1    (action_space[cls])
Bad classes:   N_ACTS .. 2*N_ACTS-1  (avoid action_space[cls - N_ACTS])

LS20 only. 5000 steps max.
Script: scripts/run_step345_explore_negative.py
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
# CompressedFold -- all 7 stages (unchanged from step 344)
# ==============================================================================

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
        prediction = scores.argmax().item()
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
            alpha = 1.0 - float(sims[winner].item())
            self.V[winner] = F.normalize(
                self.V[winner] + alpha * (x - self.V[winner]), dim=0)
        self._update_thresh()
        return prediction


# ==============================================================================
# Encoding
# ==============================================================================

def avgpool8(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(8, 8, 8, 8).mean(axis=(2, 3)).flatten()  # (64,)

def diff_encode(pooled_curr, pooled_prev):
    diff = pooled_curr - pooled_prev
    return torch.from_numpy(diff.astype(np.float32))  # (64,)

def is_zero(t, eps=1e-6):
    return float(t.norm()) < eps


# ==============================================================================
# Game runner
# ==============================================================================

def run_game(arc, game_id, max_steps=5000, max_resets=50,
             explore_steps=500, epsilon=0.10, k=3, verbose=True):
    """
    Pure exploration for first `explore_steps` steps, then epsilon-greedy.
    GAME_OVER stamps negative class (prev_action + N_ACTS).
    Returns metrics dict.
    """
    from arcengine import GameState

    fold = CompressedFold(d=D_ENC, k=k)
    env  = arc.make(game_id)
    obs  = env.reset()

    total_steps      = 0
    total_resets     = 0
    total_levels     = 0
    game_over_count  = 0
    steps_per_lvl    = []
    cb_snapshots     = []   # (step, cb_size, thresh)
    action_counts    = {}   # action.name -> count (all actions)
    fatal_counts     = {}   # action.name -> count (fatal = preceded GAME_OVER)
    bad_pred_log     = []   # (step, cls_pred) for cls_pred >= N_ACTS
    lvl_step_start   = 0
    win              = False

    # "previous step" state for reward stamping
    prev_pooled  = None
    prev_diff    = None
    prev_cls     = None
    prev_action  = None
    prev_levels  = 0

    while total_steps < max_steps and total_resets < max_resets:
        if obs is None:
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            prev_pooled = prev_diff = prev_cls = prev_action = None
            prev_levels = 0
            lvl_step_start = total_steps
            continue

        if obs.state == GameState.GAME_OVER:
            game_over_count += 1
            # Negative stamp: mark the action that caused GAME_OVER as bad
            if prev_diff is not None and not is_zero(prev_diff) and prev_cls is not None:
                n_acts = len(env.action_space)
                bad_label = prev_cls + n_acts
                fold.process(prev_diff, label=bad_label)
                if prev_action is not None:
                    fatal_counts[prev_action.name] = fatal_counts.get(prev_action.name, 0) + 1
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            prev_pooled = prev_diff = prev_cls = prev_action = None
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

        # Compute diff encoding
        curr_pooled = avgpool8(obs.frame)
        if prev_pooled is None:
            enc = torch.from_numpy(curr_pooled.astype(np.float32))
        else:
            enc = diff_encode(curr_pooled, prev_pooled)

        # Action selection
        if is_zero(enc):
            # Zero diff: take random action, stamp nothing (undefined direction)
            action   = random.choice(action_space)
            cls_used = action_space.index(action) if action in action_space else 0
        elif total_steps < explore_steps:
            # Pure exploration: random action, stamp it as label
            action   = random.choice(action_space)
            cls_used = action_space.index(action) if action in action_space else 0
            fold.process(enc, label=cls_used)
        else:
            # Exploitation with epsilon-greedy
            if random.random() < epsilon:
                action   = random.choice(action_space)
                cls_used = action_space.index(action) if action in action_space else 0
                fold.process(enc, label=cls_used)
            else:
                cls_pred = fold.process(enc, label=None)
                if cls_pred >= n_acts:
                    # Bad class predicted: avoid the corresponding action
                    bad_action = action_space[(cls_pred - n_acts) % n_acts]
                    choices    = [a for a in action_space if a != bad_action]
                    action     = random.choice(choices) if choices else random.choice(action_space)
                    cls_used   = action_space.index(action) if action in action_space else 0
                    bad_pred_log.append((total_steps, cls_pred))
                    if verbose and len(bad_pred_log) <= 10:
                        print(f"    [step {total_steps}] bad pred cls={cls_pred}"
                              f"  avoiding={bad_action.name}"
                              f"  chose={action.name}", flush=True)
                else:
                    cls_used = cls_pred % n_acts
                    action   = action_space[cls_used]

        action_counts[action.name] = action_counts.get(action.name, 0) + 1

        # Execute action
        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}

        enc_this  = enc
        cls_this  = cls_used
        obs_levels_before = obs.levels_completed

        obs = env.step(action, data=data)
        total_steps += 1

        if obs is None: break

        # Codebook snapshot every 200 steps
        if total_steps % 200 == 0:
            cb_snapshots.append((total_steps, fold.V.shape[0], fold.thresh))
            if verbose:
                phase = "explore" if total_steps < explore_steps else "exploit"
                print(f"    [step {total_steps}] {phase}"
                      f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}"
                      f"  levels={total_levels}  go={game_over_count}", flush=True)

        # Positive stamp on level completion
        if obs.levels_completed > obs_levels_before:
            total_levels = obs.levels_completed
            steps_this   = total_steps - lvl_step_start
            steps_per_lvl.append(steps_this)
            lvl_step_start = total_steps
            if verbose:
                print(f"    [step {total_steps}] LEVEL {obs.levels_completed}"
                      f"  steps_this_lvl={steps_this}"
                      f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
            # Positive stamp: reinforce the action that completed the level
            if not is_zero(enc_this):
                fold.process(enc_this, label=cls_this)

        if obs.state == GameState.WIN:
            win = True
            if verbose:
                print(f"    [step {total_steps}] WIN! levels={obs.levels_completed}"
                      f"  cb={fold.V.shape[0]}", flush=True)
            break

        prev_pooled = curr_pooled
        prev_diff   = enc_this
        prev_cls    = cls_this
        prev_action = action
        prev_levels = obs.levels_completed

    # Summarize class distribution
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
        'fatal_counts':  fatal_counts,
        'bad_pred_count': len(bad_pred_log),
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    t0 = time.time()
    print("Step 345 -- ARC-AGI-3: pure exploration + negative signal", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("Phase 1 (steps 0-499): random actions, process(diff, label=action) builds codebook", flush=True)
    print("Phase 2 (step 500+):   process() selects (epsilon=0.10), bad class -> avoid action", flush=True)
    print("GAME_OVER: stamp (prev_diff, prev_action + N_ACTS) as negative class", flush=True)
    print(flush=True)

    import arc_agi
    arc   = arc_agi.Arcade()
    games = arc.get_environments()

    # LS20 only
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if ls20 is None:
        print("ERROR: LS20 not found in environment list", flush=True)
        for g in games:
            print(f"  {g.game_id}: {g.title}", flush=True)
        return

    print(f"Running: {ls20.title} ({ls20.game_id})", flush=True)
    print(f"max_steps=5000  max_resets=50  explore_steps=500  epsilon=0.10", flush=True)
    print(flush=True)

    r = run_game(arc, ls20.game_id,
                 max_steps=5000, max_resets=50,
                 explore_steps=500, epsilon=0.10, k=3, verbose=True)

    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 345 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"win={r['win']}  levels={r['levels']}  steps={r['steps']}", flush=True)
    print(f"resets={r['resets']}  game_over={r['game_over']}", flush=True)
    print(f"cb_final={r['cb_final']}  thresh={r['thresh_final']:.4f}", flush=True)
    print(flush=True)
    print(f"steps_per_level: {r['steps_per_level']}", flush=True)
    print(flush=True)
    print(f"action_counts (all steps): {r['action_counts']}", flush=True)
    print(f"fatal_counts (preceded GAME_OVER): {r['fatal_counts']}", flush=True)
    print(f"bad_pred_count (cls>=N_ACTS predicted): {r['bad_pred_count']}", flush=True)
    print(flush=True)
    print(f"cls_dist (good=0..N-1, bad=N..2N-1): {r['cls_dist']}", flush=True)
    print(flush=True)
    if r['cb_snapshots']:
        print("Codebook over time:", flush=True)
        for s, cb, th in r['cb_snapshots']:
            phase = "explore" if s <= 500 else "exploit"
            print(f"  step {s:4d} [{phase}]:  cb={cb:3d}  thresh={th:.4f}", flush=True)
    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)

    # Analysis
    print(flush=True)
    print("Analysis:", flush=True)
    n_acts = len(arc.make(ls20.game_id).action_space)
    good_cls = {k: v for k, v in r['cls_dist'].items() if k < n_acts}
    bad_cls  = {k: v for k, v in r['cls_dist'].items() if k >= n_acts}
    print(f"  N_ACTS={n_acts}", flush=True)
    print(f"  Good class entries in codebook: {good_cls}", flush=True)
    print(f"  Bad class entries in codebook:  {bad_cls}", flush=True)
    if r['bad_pred_count'] > 0:
        print(f"  process() predicted bad class {r['bad_pred_count']} times (avoidance fired)", flush=True)
    else:
        print("  process() never predicted a bad class", flush=True)
    if r['levels'] == 0:
        print(flush=True)
        print("  No levels completed. Possible causes:", flush=True)
        print("  - Exploration phase too short / codebook not diverse enough", flush=True)
        print("  - Positive reward loop never fired -> bad classes dominate", flush=True)
        print("  - LS20 requires specific sequence not reached randomly in 500 steps", flush=True)


if __name__ == '__main__':
    main()
