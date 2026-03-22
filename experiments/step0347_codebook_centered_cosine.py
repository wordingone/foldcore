#!/usr/bin/env python3
"""
Step 347 -- ARC-AGI-3: centered cosine similarity.

Change to process(): subtract codebook mean before computing sims.
  mean_V       = self.V.mean(dim=0)  [if V.shape[0] > 1 else zeros]
  centered_x   = F.normalize(x - mean_V, dim=0)
  centered_V   = F.normalize(self.V - mean_V, dim=1)
  sims         = centered_V @ centered_x

The mean = timer-only component. Subtracting it cancels the timer,
leaving only per-action game signal. Centering IS state-derived —
it adapts as the codebook grows.

Everything else: class vote, attract, spawn, thresh — all on centered sims.

LS20 only. 5000 steps. Pure exploration first 500, then epsilon=0.10.
Script: scripts/run_step347_centered_cosine.py
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
# CompressedFold — centered cosine similarity
# ==============================================================================

class CompressedFold:
    def __init__(self, d, k=3, device=DEVICE):
        self.V      = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7
        self.k      = k
        self.d      = d
        self.device = device

    def _get_centered(self, x_norm=None):
        """
        Compute mean_V and centered versions of V (and optionally x).
        Returns (mean_V, centered_V_normed, centered_x_normed_or_None).
        centered_V_normed: shape (n, d), row-normalized after mean subtraction.
        """
        n = self.V.shape[0]
        if n > 1:
            mean_V = self.V.mean(dim=0)  # (d,)
        else:
            mean_V = torch.zeros(self.d, device=self.device)

        cV = self.V - mean_V.unsqueeze(0)         # (n, d)
        norms = cV.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cV_normed = cV / norms                     # (n, d)

        if x_norm is not None:
            cx = x_norm - mean_V                   # x is already L2-normed; center it
            cx_norm_val = cx.norm().clamp(min=1e-8)
            cx_normed = cx / cx_norm_val           # (d,)
            return mean_V, cV_normed, cx_normed

        return mean_V, cV_normed, None

    def _update_thresh(self):
        """Thresh computed in RAW space (uncentered) to avoid antipodal collapse at n=2."""
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

        mean_V, cV, cx = self._get_centered(x_norm=x)
        sims = cV @ cx   # (n,)

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
            # Spawn: store raw (un-centered) x
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
# Encoding (same as steps 344-346)
# ==============================================================================

def avgpool8(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(8, 8, 8, 8).mean(axis=(2, 3)).flatten()  # (64,)

def diff_encode(pooled_curr, pooled_prev):
    diff = pooled_curr - pooled_prev
    return torch.from_numpy(diff.astype(np.float32))

def is_zero(t, eps=1e-6):
    return float(t.norm()) < eps


# ==============================================================================
# Game runner
# ==============================================================================

def run_game(arc, game_id, max_steps=5000, max_resets=50,
             explore_steps=500, epsilon=0.10, k=3, verbose=True):
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
    sim_samples     = []   # (step, centered_sim_to_winner)
    lvl_step_start  = 0
    win             = False

    prev_pooled  = None
    prev_diff    = None
    prev_cls     = None
    prev_action  = None

    while total_steps < max_steps and total_resets < max_resets:
        if obs is None:
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            prev_pooled = prev_diff = prev_cls = prev_action = None
            lvl_step_start = total_steps
            continue

        if obs.state == GameState.GAME_OVER:
            game_over_count += 1
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            prev_pooled = prev_diff = prev_cls = prev_action = None
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

        curr_pooled = avgpool8(obs.frame)
        if prev_pooled is None:
            enc = torch.from_numpy(curr_pooled.astype(np.float32))
        else:
            enc = diff_encode(curr_pooled, prev_pooled)

        # Action selection
        if is_zero(enc):
            action   = random.choice(action_space)
            cls_used = action_space.index(action) if action in action_space else 0
        elif total_steps < explore_steps:
            # Pure exploration: stamp with action label
            action   = random.choice(action_space)
            cls_used = action_space.index(action) if action in action_space else 0
            fold.process(enc, label=cls_used)
        else:
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

        enc_this = enc
        cls_this = cls_used
        obs_levels_before = obs.levels_completed

        obs = env.step(action, data=data)
        total_steps += 1

        if obs is None: break

        # Codebook snapshot every 250 steps
        if total_steps % 250 == 0:
            n = fold.V.shape[0]
            mean_norm = 0.0
            if n > 1:
                mean_norm = float(fold.V.mean(dim=0).norm().item())
            cb_snapshots.append((total_steps, n, fold.thresh, mean_norm))
            if verbose:
                phase = "explore" if total_steps <= explore_steps else "exploit"
                print(f"    [step {total_steps:4d}] {phase:7s}"
                      f"  cb={n:3d}  thresh={fold.thresh:.4f}"
                      f"  mean_norm={mean_norm:.4f}"
                      f"  levels={total_levels}  go={game_over_count}", flush=True)

        # Positive stamp on level completion
        if obs is not None and obs.levels_completed > obs_levels_before:
            total_levels = obs.levels_completed
            steps_this   = total_steps - lvl_step_start
            steps_per_lvl.append(steps_this)
            lvl_step_start = total_steps
            if verbose:
                print(f"    [step {total_steps}] LEVEL {obs.levels_completed}"
                      f"  steps={steps_this}"
                      f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
            if not is_zero(enc_this):
                fold.process(enc_this, label=cls_this)

        if obs is not None and obs.state == GameState.WIN:
            win = True
            break

        prev_pooled = curr_pooled
        prev_diff   = enc_this
        prev_cls    = cls_this
        prev_action = action

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
        'fold':          fold,  # keep for post-analysis
    }


# ==============================================================================
# Post-run analysis: centered sim variance by action type
# ==============================================================================

def analyze_centering(arc, game_id, fold):
    """
    From same start state: collect diffs for each action, compute centered sims.
    Reports whether centering separates sprite-movement from timer-only actions.
    """
    from arcengine import GameState

    if fold.V.shape[0] < 2:
        print("  [centering analysis] codebook too small, skip", flush=True)
        return

    mean_V = fold.V.mean(dim=0)
    print(f"  codebook mean norm: {mean_V.norm():.4f}", flush=True)
    print(f"  (non-zero mean = timer component absorbed into codebook)", flush=True)
    print(flush=True)

    # Collect 20 diffs per action from fresh env
    env    = arc.make(game_id)
    obs    = env.reset()
    p_prev = avgpool8(obs.frame)

    print("  Centered sim stats per action (20 trials each):", flush=True)
    action_space = env.action_space
    for act in action_space:
        env2  = arc.make(game_id)
        obs2  = env2.reset()
        p0    = avgpool8(obs2.frame)

        centered_sims = []
        raw_sims      = []
        for _ in range(20):
            data = {}
            if act.is_complex():
                arr = np.array(obs2.frame[0])
                cy, cx = divmod(int(np.argmax(arr)), 64)
                data = {"x": cx, "y": cy}
            obs2 = env2.step(act, data=data)
            if obs2 is None: break
            p1   = avgpool8(obs2.frame)
            diff = torch.from_numpy((p1 - p0).astype(np.float32))
            if is_zero(diff):
                p0 = p1
                continue

            x_raw = F.normalize(diff.to(fold.device), dim=0)
            raw_s = float((fold.V @ x_raw).max().item()) if fold.V.shape[0] > 0 else 0.0

            # Centered
            diff_dev = diff.to(fold.device)
            cx_diff  = diff_dev - mean_V
            if cx_diff.norm() < 1e-8:
                p0 = p1
                continue
            cx_norm = F.normalize(cx_diff, dim=0)
            cV       = fold.V - mean_V.unsqueeze(0)
            cV_norms = cV.norm(dim=1, keepdim=True).clamp(min=1e-8)
            cV_normed = cV / cV_norms
            centered_s = float((cV_normed @ cx_norm).max().item()) if fold.V.shape[0] > 0 else 0.0

            raw_sims.append(raw_s)
            centered_sims.append(centered_s)
            p0 = p1

        if raw_sims:
            print(f"    {act.name}: raw_sim max={max(raw_sims):.4f} mean={np.mean(raw_sims):.4f}"
                  f"  centered_sim max={max(centered_sims):.4f} mean={np.mean(centered_sims):.4f}"
                  f"  (n={len(raw_sims)})", flush=True)
        else:
            print(f"    {act.name}: all zero diffs", flush=True)


# ==============================================================================
# Main
# ==============================================================================

def main():
    t0 = time.time()
    print("Step 347 -- ARC-AGI-3: centered cosine similarity", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("mean_V = codebook mean = timer component", flush=True)
    print("centered_x = F.normalize(x - mean_V), centered_V = F.normalize(V - mean_V)", flush=True)
    print("All sims (vote, attract, thresh) in centered space.", flush=True)
    print(flush=True)

    import arc_agi
    arc   = arc_agi.Arcade()
    games = arc.get_environments()
    ls20  = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if ls20 is None:
        print("ERROR: LS20 not found", flush=True)
        return

    print(f"Running: {ls20.title} ({ls20.game_id})", flush=True)
    print(f"max_steps=5000  explore_steps=500  epsilon=0.10", flush=True)
    print(flush=True)

    r = run_game(arc, ls20.game_id,
                 max_steps=5000, max_resets=50,
                 explore_steps=500, epsilon=0.10, k=3, verbose=True)

    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 347 SUMMARY", flush=True)
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
        phase = "explore" if s <= 500 else "exploit"
        print(f"  step {s:4d} [{phase:7s}]:  cb={cb:3d}  thresh={th:.4f}"
              f"  mean_norm={mn:.4f}", flush=True)

    print(flush=True)
    print("Post-run centering analysis:", flush=True)
    analyze_centering(arc, ls20.game_id, r['fold'])

    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
