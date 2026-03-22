#!/usr/bin/env python3
"""
Step 417 — Constraint Map Validation against process_novelty() baseline.

Based on Step 353 OPTIMIZED (incremental Gram matrix).
7 runs: 1 baseline + 3 reverse tests + 3 forward tests.

Usage:
    python run_step417_constraint_validation.py baseline
    python run_step417_constraint_validation.py R1       # fixed thresh=0.95
    python run_step417_constraint_validation.py R2       # fixed lr=0.05
    python run_step417_constraint_validation.py R3       # both fixed
    python run_step417_constraint_validation.py F1       # argmin → index-based
    python run_step417_constraint_validation.py F2       # no centering
    python run_step417_constraint_validation.py F3       # k=1

Measurement: step count to Level 1. Not binary pass/fail.
  - Baseline ~26K = stochastic coverage.
  - Variant < 26K = substitution added directional signal.
  - Variant = 26K = substitution is neutral.
  - Variant > 26K or never = substitution degraded coverage.
"""

import sys
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC  = 256   # 16x16


# ==============================================================================
# Base: CompressedFold with incremental Gram (from Step 353 optimized)
# ==============================================================================

class CompressedFoldBase:
    def __init__(self, d, k=3, device=DEVICE):
        self.V      = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7
        self.k      = k
        self.d      = d
        self.device = device
        self.G      = torch.zeros(0, 0, device=device)

    def _recompute_row(self, idx):
        n = self.V.shape[0]
        if n == 0: return
        new_sims = self.V @ self.V[idx]
        self.G[idx, :] = new_sims
        self.G[:, idx] = new_sims
        self.G[idx, idx] = -float('inf')

    def _append_to_gram(self, new_entry):
        n = self.G.shape[0]
        if n == 0:
            self.G = torch.full((1, 1), -float('inf'), device=self.device)
            return
        new_sims = self.V[:n] @ new_entry
        self.G = torch.cat([self.G, new_sims.unsqueeze(0)], dim=0)
        new_col = torch.cat([new_sims, torch.tensor([-float('inf')], device=self.device)])
        self.G = torch.cat([self.G, new_col.unsqueeze(1)], dim=1)

    def _update_thresh(self):
        n = self.G.shape[0]
        if n < 2: return
        self.thresh = float(self.G.max(dim=1).values.median())

    def _get_lr(self, sim):
        return 1.0 - float(sim)

    def _get_action(self, sims, n_cls):
        cls_ids = torch.arange(n_cls, device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        masked = sims.unsqueeze(0) * one_hot + (1 - one_hot) * (-1e9)
        topk_k = min(self.k, masked.shape[1])
        scores = masked.topk(topk_k, dim=1).values.sum(dim=1)
        return scores.argmin().item()

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V      = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels,
                                  torch.tensor([label], device=self.device)])
        self._append_to_gram(x_n)
        self._update_thresh()

    def process_novelty(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            spawn_label = label if label is not None else 0
            self.V      = x.unsqueeze(0)
            self.labels = torch.tensor([spawn_label], device=self.device)
            self.G      = torch.full((1, 1), -float('inf'), device=self.device)
            return spawn_label

        sims  = self.V @ x
        n_cls = int(self.labels.max().item()) + 1

        prediction  = self._get_action(sims, n_cls)
        spawn_label = label if label is not None else prediction
        target_mask = (self.labels == prediction)

        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V      = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([spawn_label], device=self.device)])
            self._append_to_gram(x)
        else:
            target_sims = sims.clone()
            target_sims[~target_mask] = -float('inf')
            winner = target_sims.argmax().item()
            alpha  = self._get_lr(sims[winner].item())
            self.V[winner] = F.normalize(
                self.V[winner] + alpha * (x - self.V[winner]), dim=0)
            self._recompute_row(winner)

        self._update_thresh()
        return prediction


# ==============================================================================
# REVERSE VARIANTS: remove V-derivedness
# ==============================================================================

class VariantR1(CompressedFoldBase):
    """Fixed threshold = 0.95. Remove Gram-median adaptiveness."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thresh = 0.95

    def _update_thresh(self):
        pass  # threshold never changes


class VariantR2(CompressedFoldBase):
    """Fixed lr = 0.05. Remove 1-sim adaptiveness."""
    def _get_lr(self, sim):
        return 0.05


class VariantR3(CompressedFoldBase):
    """Fixed threshold AND fixed lr. Both constants."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thresh = 0.95

    def _update_thresh(self):
        pass

    def _get_lr(self, sim):
        return 0.05


# ==============================================================================
# FORWARD VARIANTS: remove process_novelty-specific features
# ==============================================================================

class VariantF1(CompressedFoldBase):
    """argmin → index-based action. No labels, no class scoring.
    Action = winner_index % n_actions. Tests if novelty-seeking is load-bearing."""

    def _get_action(self, sims, n_cls):
        # Ignore class structure entirely. Return winner index.
        return sims.argmax().item()

    def process_novelty(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            self.V      = x.unsqueeze(0)
            self.labels = torch.tensor([0], device=self.device)
            self.G      = torch.full((1, 1), -float('inf'), device=self.device)
            return 0

        sims  = self.V @ x
        winner = sims.argmax().item()
        action = winner  # caller will do % n_acts

        # Spawn or attract based on winner similarity
        if sims[winner] < self.thresh:
            self.V      = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([0], device=self.device)])
            self._append_to_gram(x)
        else:
            alpha = self._get_lr(sims[winner].item())
            self.V[winner] = F.normalize(
                self.V[winner] + alpha * (x - self.V[winner]), dim=0)
            self._recompute_row(winner)

        self._update_thresh()
        return action


class VariantF2(CompressedFoldBase):
    """No centering. Raw normalized input without subtracting codebook mean."""
    pass  # encoding change handled in run_game


class VariantF3(CompressedFoldBase):
    """k=1 instead of k=3. Single nearest neighbor instead of top-K scoring."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=1, **kwargs)


# ==============================================================================
# Encoding
# ==============================================================================

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


def raw_enc(pooled, fold):
    """F2 variant: normalize only, no centering."""
    t = torch.from_numpy(pooled.astype(np.float32))
    return F.normalize(t, dim=0)


# ==============================================================================
# Game runner
# ==============================================================================

def run_game(arc, game_id, variant_name, max_steps=50000, max_resets=500,
             k=3, verbose=True):
    from arcengine import GameState

    VARIANTS = {
        'baseline': CompressedFoldBase,
        'R1': VariantR1,
        'R2': VariantR2,
        'R3': VariantR3,
        'F1': VariantF1,
        'F2': VariantF2,
        'F3': VariantF3,
    }

    cls = VARIANTS[variant_name]
    fold = cls(d=D_ENC, k=k)
    enc_fn = raw_enc if variant_name == 'F2' else centered_enc

    env = arc.make(game_id)
    obs = env.reset()

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
        enc         = enc_fn(curr_pooled, fold)

        state_hash = hash(curr_pooled.tobytes())
        unique_states.add(state_hash)

        # Force-seed (skip for F1 which doesn't use labels)
        if not seeded and fold.V.shape[0] < n_acts and variant_name != 'F1':
            i = fold.V.shape[0]
            fold._force_add(enc, label=i)
            action   = action_space[i % n_acts]
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

        if total_steps % 5000 == 0:
            n = fold.V.shape[0]
            mean_norm = float(fold.V.mean(dim=0).norm().item()) if n > 0 else 0.0
            cb_snapshots.append((total_steps, n, fold.thresh, mean_norm, len(unique_states)))
            if verbose:
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
        'variant':         variant_name,
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


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else 'baseline'
    valid = ['baseline', 'R1', 'R2', 'R3', 'F1', 'F2', 'F3']
    if variant not in valid:
        print(f"Usage: {sys.argv[0]} [{' | '.join(valid)}]")
        sys.exit(1)

    DESCRIPTIONS = {
        'baseline': 'process_novelty() exact (incremental Gram optimization)',
        'R1': 'REVERSE: fixed thresh=0.95 (remove Gram-median adaptiveness)',
        'R2': 'REVERSE: fixed lr=0.05 (remove 1-sim adaptiveness)',
        'R3': 'REVERSE: both fixed (thresh=0.95, lr=0.05)',
        'F1': 'FORWARD: argmin → index-based action (no labels, no class scoring)',
        'F2': 'FORWARD: no centering (raw normalized input)',
        'F3': 'FORWARD: k=1 (single nearest neighbor, not top-3)',
    }

    t0 = time.time()
    print(f"Step 417 — Constraint Map Validation", flush=True)
    print(f"Variant: {variant} — {DESCRIPTIONS[variant]}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(flush=True)

    import arc_agi
    arc   = arc_agi.Arcade()
    games = arc.get_environments()
    ls20  = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if ls20 is None:
        print("ERROR: LS20 not found", flush=True)
        return

    print(f"Running: {ls20.title} ({ls20.game_id})", flush=True)
    print(f"max_steps=50000  k={'1 (F3)' if variant == 'F3' else '3'}", flush=True)
    print(flush=True)

    r = run_game(arc, ls20.game_id, variant_name=variant,
                 max_steps=50000, max_resets=500,
                 k=1 if variant == 'F3' else 3, verbose=True)

    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print(f"STEP 417 [{variant}] SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"win={r['win']}  levels={r['levels']}  steps={r['steps']}", flush=True)
    print(f"resets={r['resets']}  game_over={r['game_over']}", flush=True)
    print(f"cb_final={r['cb_final']}  thresh={r['thresh_final']:.4f}", flush=True)
    print(f"unique_states={r['unique_states']}", flush=True)
    print(flush=True)
    print(f"steps_per_level: {r['steps_per_level']}", flush=True)
    print(f"action_counts: {r['action_counts']}", flush=True)
    print(flush=True)

    print("Codebook over time:", flush=True)
    for s, cb, th, mn, uniq in r['cb_snapshots']:
        print(f"  step {s:6d}:  cb={cb:6d}  thresh={th:.4f}"
              f"  mean_norm={mn:.4f}  unique={uniq:5d}", flush=True)

    print(flush=True)
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

    if len(lns) >= 20:
        avg_last20 = sum(lns[-20:]) / 20
        print(f"Avg new states/life (last 20): {avg_last20:.1f}", flush=True)

    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
