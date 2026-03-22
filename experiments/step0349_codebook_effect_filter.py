#!/usr/bin/env python3
"""
Step 349 -- ARC-AGI-3: only learn from actions that change the state.

Filter: only call process() when state actually changed (atol=0.01 on raw avgpool).
  - ACTION3/4: timer changes ~0.009/step -> below atol -> filtered out
  - ACTION1/2: sprite moves 12 cells by ~0.05 each -> above atol -> stamped

Exploration bias: after bootstrap, epsilon exploration only picks from
action classes that have codebook entries.

Base: Step 348c (force-seed + unit-norm centering + absolute frame).

5000 steps on LS20.
Script: scripts/run_step349_effect_filter.py
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
# CompressedFold
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

    def classes_with_entries(self):
        """Return set of class labels that have codebook entries."""
        if self.V.shape[0] == 0:
            return set()
        return set(int(l) for l in self.labels.cpu().numpy())


# ==============================================================================
# Encoding (Step 348c: avgpool8 + unit-norm centering)
# ==============================================================================

def avgpool8(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(8, 8, 8, 8).mean(axis=(2, 3)).flatten()  # (64,), [0,1]

def centered_frame(pooled, fold):
    """Normalize pooled to unit sphere, then center by codebook mean."""
    t      = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if fold.V.shape[0] > 2:
        mean_V = fold.V.mean(dim=0).cpu()
        t_unit = t_unit - mean_V
    return t_unit  # process() normalizes again

def state_changed(pooled_prev, pooled_curr, atol=0.01):
    """True if any cell changed by more than atol in raw avgpool [0,1] space."""
    return not torch.allclose(
        torch.from_numpy(pooled_prev.astype(np.float32)),
        torch.from_numpy(pooled_curr.astype(np.float32)),
        atol=atol
    )


# ==============================================================================
# Game runner
# ==============================================================================

def run_game(arc, game_id, max_steps=5000, max_resets=50,
             bootstrap_steps=200, epsilon=0.10, k=3,
             effect_atol=0.01, verbose=True):
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
    filtered_counts  = {}   # actions filtered (no effect)
    stamped_counts   = {}   # actions stamped (had effect)
    lvl_step_start   = 0
    win              = False
    seeded           = False

    prev_pooled  = None

    while total_steps < max_steps and total_resets < max_resets:
        if obs is None:
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            prev_pooled  = None
            lvl_step_start = total_steps
            continue

        if obs.state == GameState.GAME_OVER:
            game_over_count += 1
            # No stamp on GAME_OVER
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            prev_pooled  = None
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
        enc         = centered_frame(curr_pooled, fold)

        # Force-seed: one entry per action class before bootstrap
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
            obs_prev_pooled   = curr_pooled
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
            if obs is not None and obs.state == GameState.WIN:
                win = True
            prev_pooled = obs_prev_pooled
            continue

        if not seeded:
            seeded = True

        # Action selection
        known_classes = fold.classes_with_entries()
        if total_steps < bootstrap_steps:
            # Random exploration — all actions allowed during bootstrap
            action   = random.choice(action_space)
            cls_used = action_space.index(action) if action in action_space else 0
        else:
            # Exploitation: epsilon-greedy, biased toward known action classes
            if random.random() < epsilon:
                # Bias: pick from actions that have codebook entries
                biased = [action_space[c] for c in known_classes if c < n_acts]
                if biased:
                    action = random.choice(biased)
                else:
                    action = random.choice(action_space)
                cls_used = action_space.index(action) if action in action_space else 0
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
        prev_pooled_step  = curr_pooled

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

        # Effect filter: only stamp if state changed
        curr_pooled_new = avgpool8(obs.frame) if obs is not None else curr_pooled
        if obs is not None:
            if state_changed(prev_pooled_step, curr_pooled_new, atol=effect_atol):
                enc_new = centered_frame(curr_pooled_new, fold)
                fold.process(enc_new, label=cls_used)
                stamped_counts[action.name] = stamped_counts.get(action.name, 0) + 1
            else:
                filtered_counts[action.name] = filtered_counts.get(action.name, 0) + 1

        # Level completion: no extra stamp (trust codebook)
        if obs is not None and obs.levels_completed > obs_levels_before:
            total_levels = obs.levels_completed
            steps_this   = total_steps - lvl_step_start
            steps_per_lvl.append(steps_this)
            lvl_step_start = total_steps
            if verbose:
                print(f"    [step {total_steps}] LEVEL {obs.levels_completed}"
                      f"  steps_this={steps_this}"
                      f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)

        if obs is not None and obs.state == GameState.WIN:
            win = True
            break

        prev_pooled = curr_pooled_new

    cls_dist = {}
    if fold.V.shape[0] > 0:
        for lbl in fold.labels.cpu().numpy():
            k_ = int(lbl)
            cls_dist[k_] = cls_dist.get(k_, 0) + 1

    return {
        'win':            win,
        'levels':         total_levels,
        'steps':          total_steps,
        'resets':         total_resets,
        'game_over':      game_over_count,
        'steps_per_level': steps_per_lvl,
        'cb_final':       fold.V.shape[0],
        'thresh_final':   fold.thresh,
        'cb_snapshots':   cb_snapshots,
        'cls_dist':       dict(sorted(cls_dist.items())),
        'action_counts':  action_counts,
        'filtered_counts': filtered_counts,
        'stamped_counts':  stamped_counts,
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    t0 = time.time()
    print("Step 349 -- ARC-AGI-3: effect filter (only stamp state-changing actions)", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("Filter: state_changed = not allclose(prev_pooled, curr_pooled, atol=0.01)", flush=True)
    print("Bias: epsilon exploration picks from known action classes only.", flush=True)
    print("Base: 348c (force-seed + unit-norm centering).", flush=True)
    print(flush=True)

    import arc_agi
    arc   = arc_agi.Arcade()
    games = arc.get_environments()
    ls20  = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if ls20 is None:
        print("ERROR: LS20 not found", flush=True)
        return

    print(f"Running: {ls20.title} ({ls20.game_id})", flush=True)
    print(f"max_steps=5000  bootstrap=200  epsilon=0.10  k=3  atol=0.01", flush=True)
    print(flush=True)

    r = run_game(arc, ls20.game_id,
                 max_steps=5000, max_resets=50,
                 bootstrap_steps=200, epsilon=0.10, k=3,
                 effect_atol=0.01, verbose=True)

    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 349 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"win={r['win']}  levels={r['levels']}  steps={r['steps']}", flush=True)
    print(f"resets={r['resets']}  game_over={r['game_over']}", flush=True)
    print(f"cb_final={r['cb_final']}  thresh={r['thresh_final']:.4f}", flush=True)
    print(flush=True)
    print(f"steps_per_level: {r['steps_per_level']}", flush=True)
    print(f"action_counts (all): {r['action_counts']}", flush=True)
    print(f"stamped_counts (had effect): {r['stamped_counts']}", flush=True)
    print(f"filtered_counts (no effect): {r['filtered_counts']}", flush=True)
    print(f"cls_dist: {r['cls_dist']}", flush=True)
    print(flush=True)

    print("Codebook over time:", flush=True)
    for s, cb, th, mn in r['cb_snapshots']:
        phase = "bootstrap" if s <= 200 else "exploit"
        print(f"  step {s:4d} [{phase:9s}]:  cb={cb:3d}  thresh={th:.4f}"
              f"  mean_norm={mn:.4f}", flush=True)

    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)

    # Verify filter worked
    print(flush=True)
    print("Filter verification:", flush=True)
    total_stamped  = sum(r['stamped_counts'].values())
    total_filtered = sum(r['filtered_counts'].values())
    print(f"  Total stamped:  {total_stamped}", flush=True)
    print(f"  Total filtered: {total_filtered}", flush=True)
    if total_stamped + total_filtered > 0:
        stamp_rate = total_stamped / (total_stamped + total_filtered) * 100
        print(f"  Stamp rate: {stamp_rate:.1f}%", flush=True)
    for act_name in sorted(set(list(r['stamped_counts'].keys()) +
                                list(r['filtered_counts'].keys()))):
        s_ = r['stamped_counts'].get(act_name, 0)
        f_ = r['filtered_counts'].get(act_name, 0)
        rate = f"{s_/(s_+f_)*100:.0f}%" if (s_+f_) > 0 else "n/a"
        print(f"    {act_name}: stamped={s_}  filtered={f_}  stamp_rate={rate}", flush=True)


if __name__ == '__main__':
    main()
