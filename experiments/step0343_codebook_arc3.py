#!/usr/bin/env python3
"""
Step 343 — ARC-AGI-3: process() with proper encoding + reward signal.

Changes from Step 342:
  Encoding:  avg_pool2d(frame, 8x8) -> 8x8 -> flatten -> 64-dim (not 4096-dim flat)
  Reward:    when levels_completed increases: process(prev_state, label=prev_cls)
  Bootstrap: first len(action_space) steps spawn one entry per action class
  Epsilon:   20% random actions (decaying at 0.99/step) for exploration

process() IS the agent. No modules beyond encoding.
Script: scripts/run_step343_arc3.py
"""

import time
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 64   # 8x8 avg pool


# ══════════════════════════════════════════════════════════════════════════════
# CompressedFold — Step 342 (all 7 stages)
# ══════════════════════════════════════════════════════════════════════════════

class CompressedFold:
    """Step 342: S2+S3+S7. target=prediction always. alpha=1-sim. thresh=state."""
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
        x = F.normalize(x.to(self.device), dim=0)
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


# ══════════════════════════════════════════════════════════════════════════════
# Encoding: avg pool 8x8 -> 64-dim
# ══════════════════════════════════════════════════════════════════════════════

def encode_frame(frame):
    """
    avg_pool2d(frame, kernel=8) -> 8x8 -> flatten -> 64-dim float32 tensor.
    frame[0] is (64, 64) numpy array, values 0-15.
    """
    arr = np.array(frame[0], dtype=np.float32) / 15.0  # normalize 0-15 -> 0-1
    # Reshape to (8, 8, 8, 8) and average over the 8x8 sub-blocks
    pooled = arr.reshape(8, 8, 8, 8).mean(axis=(2, 3))  # (8, 8)
    return torch.from_numpy(pooled.flatten())            # (64,)


# ══════════════════════════════════════════════════════════════════════════════
# Game runner
# ══════════════════════════════════════════════════════════════════════════════

def run_game(arc, game_id, max_steps=1000, max_resets=20, epsilon_start=0.2,
             epsilon_decay=0.99, k=3, verbose=True):
    """
    Run one game. Returns result dict with metrics.
    Encoding: 8x8 avg pool -> 64-dim.
    Reward: process(prev_x, label=prev_cls) when level completes.
    Bootstrap: first len(action_space) steps spawn one entry per action.
    Epsilon: decaying random exploration.
    """
    from arcengine import GameState

    fold = CompressedFold(d=D_ENC, k=k)
    env  = arc.make(game_id)
    obs  = env.reset()

    total_steps   = 0
    total_resets  = 0
    total_levels  = 0
    steps_per_lvl = []   # steps used per completed level
    cb_growth     = []   # (step, cb_size) snapshots
    epsilon       = epsilon_start

    # Bootstrap state
    bootstrapped = False

    # Previous step state (for reward stamping)
    prev_x   = None
    prev_cls = None
    lvl_step_start = 0  # step count at start of current level

    win = False

    while total_steps < max_steps and total_resets < max_resets:
        if obs is None or obs.state in (GameState.GAME_OVER,):
            total_resets += 1
            if total_resets >= max_resets:
                break
            obs = env.reset()
            if obs is None: break
            lvl_step_start = total_steps
            prev_x = prev_cls = None
            continue

        if obs.state == GameState.WIN:
            win = True
            break

        x            = encode_frame(obs.frame)
        action_space = env.action_space  # list of GameAction

        # Bootstrap: force-spawn one entry per action class on first N steps
        if not bootstrapped and fold.V.shape[0] < len(action_space):
            i = fold.V.shape[0]  # next class to spawn
            fold.process(x, label=i)
            action = action_space[i % len(action_space)]
            cls_used = i
        else:
            bootstrapped = True
            # Epsilon-greedy: random or process()
            if random.random() < epsilon:
                action = random.choice(action_space)
                cls_used = action_space.index(action) if action in action_space else 0
                fold.process(x, label=cls_used)  # still update codebook
            else:
                cls_used = fold.process(x, label=None)
                action   = action_space[cls_used % len(action_space)]
            epsilon *= epsilon_decay

        # Execute action
        data = {}
        if action.is_complex():
            # ACTION6: click at argmax of frame
            arr  = np.array(obs.frame[0])
            idx  = int(np.argmax(arr))
            cy, cx = divmod(idx, 64)
            data = {"x": cx, "y": cy}

        prev_x_step   = x
        prev_cls_step = cls_used
        prev_levels   = obs.levels_completed

        obs = env.step(action, data=data)
        total_steps += 1

        if obs is None: break

        # Record codebook growth every 50 steps
        if total_steps % 50 == 0:
            cb_growth.append((total_steps, fold.V.shape[0]))

        # Reward stamp: level completed
        if obs.levels_completed > prev_levels:
            total_levels = obs.levels_completed
            steps_this_lvl = total_steps - lvl_step_start
            steps_per_lvl.append(steps_this_lvl)
            lvl_step_start = total_steps
            if verbose:
                print(f"    [step {total_steps}] Level {obs.levels_completed} completed"
                      f"  steps_this_lvl={steps_this_lvl}"
                      f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
            # Reward: stamp previous (state, action_class) as positive
            if prev_x_step is not None:
                fold.process(prev_x_step, label=prev_cls_step)

        if obs.state == GameState.WIN:
            win = True
            if verbose:
                print(f"    [step {total_steps}] WIN! levels={obs.levels_completed}", flush=True)
            break

        prev_x   = prev_x_step
        prev_cls = prev_cls_step

    # Class distribution
    cls_dist = {}
    if fold.V.shape[0] > 0:
        for lbl in fold.labels.cpu().numpy():
            k_ = int(lbl)
            cls_dist[k_] = cls_dist.get(k_, 0) + 1

    return {
        'win': win,
        'levels': total_levels,
        'steps': total_steps,
        'resets': total_resets,
        'steps_per_level': steps_per_lvl,
        'cb_final': fold.V.shape[0],
        'thresh_final': fold.thresh,
        'cb_growth': cb_growth,
        'cls_dist': dict(sorted(cls_dist.items())),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("Step 343 -- ARC-AGI-3: process() with 8x8 avg-pool encoding", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("Encoding: avg_pool(8x8) -> 64-dim. Reward: level_completed -> label stamp.", flush=True)
    print("Bootstrap: one spawn per action class. Epsilon=0.2 decaying.", flush=True)
    print(flush=True)

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    print(f"Games ({len(games)}):", flush=True)
    for g in games:
        print(f"  {g.game_id}: {g.title}", flush=True)
    print(flush=True)

    # Verify encoding
    env_test = arc.make(games[0].game_id)
    obs_test  = env_test.reset()
    x_test    = encode_frame(obs_test.frame)
    print(f"Encoding check:", flush=True)
    print(f"  frame[0] shape: {np.array(obs_test.frame[0]).shape}", flush=True)
    print(f"  encoded shape:  {x_test.shape}  (target: 64)", flush=True)
    print(f"  encoded range:  [{x_test.min():.3f}, {x_test.max():.3f}]", flush=True)
    print(f"  L2 norm:        {x_test.norm():.3f}", flush=True)
    # Cosine sim between two frames
    obs_test2 = env_test.step(env_test.action_space[0])
    if obs_test2 is not None:
        x_test2 = encode_frame(obs_test2.frame)
        cos_sim = float(F.cosine_similarity(
            x_test.unsqueeze(0), x_test2.unsqueeze(0)))
        print(f"  cos_sim(frame0, frame1): {cos_sim:.4f}", flush=True)
    print(flush=True)

    results = {}
    MAX_STEPS  = 2000
    MAX_RESETS = 30

    for g in games:
        game_id = g.game_id
        title   = g.title
        print("=" * 60, flush=True)
        print(f"Game: {title} ({game_id})", flush=True)
        print(f"  max_steps={MAX_STEPS}  max_resets={MAX_RESETS}", flush=True)
        t1 = time.time()

        r = run_game(arc, game_id,
                     max_steps=MAX_STEPS, max_resets=MAX_RESETS,
                     epsilon_start=0.2, epsilon_decay=0.99, k=3, verbose=True)

        elapsed_g = time.time() - t1
        print(f"  win={r['win']}  levels={r['levels']}  steps={r['steps']}"
              f"  resets={r['resets']}", flush=True)
        print(f"  cb_final={r['cb_final']}  thresh={r['thresh_final']:.4f}", flush=True)
        print(f"  steps_per_level: {r['steps_per_level']}", flush=True)
        print(f"  cls_dist: {r['cls_dist']}", flush=True)
        if r['cb_growth']:
            snap_str = '  '.join(f"s{s}:{c}" for s, c in r['cb_growth'][:6])
            print(f"  cb_growth: {snap_str}", flush=True)
        print(f"  elapsed: {elapsed_g:.1f}s", flush=True)

        results[title] = r
        print(flush=True)

    elapsed = time.time() - t0

    print("=" * 60, flush=True)
    print("STEP 343 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Encoding: avg_pool(8x8) -> 64-dim", flush=True)
    print(f"Reward: level_completed stamp", flush=True)
    print(f"Bootstrap: {len(games[0].game_id)}-step seed (approx)", flush=True)
    print(flush=True)

    total_wins   = 0
    total_levels = 0
    for title, r in results.items():
        win_str = "WIN" if r['win'] else "---"
        spl = f"{np.mean(r['steps_per_level']):.0f}/lvl" if r['steps_per_level'] else "no lvls"
        print(f"  {title}: [{win_str}]  levels={r['levels']}  steps={r['steps']}"
              f"  {spl}  cb={r['cb_final']}", flush=True)
        if r['win']: total_wins += 1
        total_levels += r['levels']

    print(flush=True)
    print(f"Games won: {total_wins}/{len(results)}", flush=True)
    print(f"Total levels completed: {total_levels}", flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)

    if total_levels == 0:
        print(flush=True)
        print("Failure analysis:", flush=True)
        print("  If cb still small: avg-pool may still produce high cosine sim.", flush=True)
        print("  If cb grows but no levels: epsilon exploration insufficient.", flush=True)
        print("  Next step: check cosine sim distribution, tune thresh init or epsilon.", flush=True)


if __name__ == '__main__':
    main()
