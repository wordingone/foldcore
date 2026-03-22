#!/usr/bin/env python3
"""
Step 417 — Adaptive Substrate: Operations as Data (Phase 2 begins)

THE SEARCH SPACE (derived from R1-R6 + U1-U19 + I1-I9):
================
416 experiments mapped the feasible region for recursive self-improvement.
The binding constraint is R3: all modifiable aspects self-modified.

Current system (process(), 22 lines) fails R3 (operations hardcoded) and R4
(no self-testing). The correct search direction: move operations from hardcoded
Python into the codebook. The codebook must encode both DATA and RULES.

The irreducible interpreter (R6): compare, select, store, modify. These are
mathematical necessities — you can't store "how to add" as data. But WHICH
comparison, WHICH selection, WHICH update — those CAN be data.

THIS EXPERIMENT:
================
Tests the first step: per-entry operation flags that determine update behavior.

  ATTRACT (0): winner attracted toward observation (standard LVQ)
  FREEZE  (1): entry preserves successful knowledge (created on LEVEL_UP)
  REPEL   (2): entry marks dangerous state-action pair (created on GAME_OVER)

The flags are DATA in the codebook. Game events MODIFY them. Different flag
configurations → different update behavior → different effective algorithm.

At 16x16 with cosine saturation (sims ≈ 1.000), spatial locality is gone.
REPEL degrades from spatial blocking to statistical blocking: actions with
more death-entries get higher scores → argmin avoids them. This is still
useful — it discovers globally bad actions through accumulation.

Partial R3: the update rule is state-derived (per-entry flags).
R4 addressed: game-over is self-test (revert via REPEL), level-up is keep.

KILL CRITERION: Level 1 on LS20 within 50K steps (matching Step 353 baseline).
DISPROVES IF: Strictly worse than baseline (additions hurt, confirming U13).
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256  # 16x16


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline: CompressedFold from Step 353 (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class CompressedFold:
    """Pure novelty-seeking baseline. No operation flags. No game-event learning."""

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
            tgt = label if label is not None else 0
            self.V      = x.unsqueeze(0)
            self.labels = torch.tensor([tgt], device=self.device)
            return tgt

        sims  = self.V @ x
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
        else:
            target_sims = sims.clone()
            target_sims[~target_mask] = -float('inf')
            winner = target_sims.argmax().item()
            alpha  = 1.0 - float(sims[winner].item())
            self.V[winner] = F.normalize(
                self.V[winner] + alpha * (x - self.V[winner]), dim=0)

        self._update_thresh()
        return prediction

    def on_game_over(self):
        pass

    def on_level_up(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# New: AdaptiveFold — operations as data
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveFold:
    """
    Process() with per-entry operation flags.
    The update rule is DATA in the codebook, not hardcoded Python.

    Each entry has:
      - V[i]: pattern vector (for matching)
      - labels[i]: action class (for game action selection)
      - ops[i]: operation flag (determines what happens when this entry wins)

    The operation flags create three qualitatively different behaviors:
      ATTRACT: standard competitive learning (entry evolves toward observations)
      FREEZE:  entry is protected (preserves successful knowledge across levels)
      REPEL:   entry marks danger (boosts class score so argmin avoids it)

    Different configurations of flags = different effective algorithms.
    Game dynamics modify flags = system modifies its own learning algorithm.
    This is partial R3: the update rule is self-modified via gameplay.
    """

    OP_ATTRACT = 0
    OP_FREEZE  = 1
    OP_REPEL   = 2

    def __init__(self, d, k=3, repel_weight=0.5, device=DEVICE):
        self.V      = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.ops    = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7
        self.k      = k
        self.d      = d
        self.device = device
        self.repel_weight  = repel_weight
        self.last_winner   = -1
        self.recent_winners = []

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V      = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels,
                                 torch.tensor([label], device=self.device)])
        self.ops    = torch.cat([self.ops,
                                 torch.tensor([self.OP_ATTRACT], device=self.device)])
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
            tgt = label if label is not None else 0
            self.V      = x.unsqueeze(0)
            self.labels = torch.tensor([tgt], device=self.device)
            self.ops    = torch.tensor([self.OP_ATTRACT], device=self.device)
            return tgt

        sims  = self.V @ x
        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(n_cls, device=self.device)

        for c in range(n_cls):
            c_mask = (self.labels == c)
            if c_mask.sum() == 0: continue
            cs    = sims[c_mask]
            ops_c = self.ops[c_mask]

            # Non-REPEL entries: standard top-K positive scoring
            non_repel = (ops_c != self.OP_REPEL)
            if non_repel.sum() > 0:
                nrc = cs[non_repel]
                scores[c] = nrc.topk(min(self.k, len(nrc))).values.sum()

            # REPEL entries: boost score so argmin avoids this class
            # At saturation: degrades to count-based (sum of ~1.0 each)
            # Without saturation: proximity-weighted (nearby REPELs matter more)
            repel = (ops_c == self.OP_REPEL)
            if repel.sum() > 0:
                scores[c] += cs[repel].sum() * self.repel_weight

        prediction  = scores.argmin().item()
        spawn_label = label if label is not None else prediction

        # Update: find winner in target class, excluding REPEL entries
        target_mask = (self.labels == prediction) & (self.ops != self.OP_REPEL)

        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            # Spawn new entry with ATTRACT flag
            self.V      = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([spawn_label], device=self.device)])
            self.ops    = torch.cat([self.ops,
                                     torch.tensor([self.OP_ATTRACT], device=self.device)])
            new_idx = self.V.shape[0] - 1
            self.last_winner = new_idx
            self.recent_winners.append(new_idx)
        else:
            target_sims = sims.clone()
            target_sims[~target_mask] = -float('inf')
            winner = target_sims.argmax().item()

            self.last_winner = winner
            self.recent_winners.append(winner)

            op = self.ops[winner].item()
            if op == self.OP_ATTRACT:
                alpha = 1.0 - float(sims[winner].item())
                self.V[winner] = F.normalize(
                    self.V[winner] + alpha * (x - self.V[winner]), dim=0)
            # OP_FREEZE: no update — entry is protected
            # OP_REPEL: excluded by target_mask above

        # Cap recent_winners to prevent unbounded growth
        if len(self.recent_winners) > 200:
            self.recent_winners = self.recent_winners[-200:]

        self._update_thresh()
        return prediction

    def on_game_over(self):
        """Last ATTRACT winner → REPEL. Learn from failure.

        Only flips ATTRACT entries (not FREEZE — those are protected knowledge).
        The REPEL entry stays at the death-state position, permanently boosting
        its action class score. argmin avoids boosted classes.

        With cosine saturation: REPEL is statistical (death-count per action).
        Without saturation: REPEL is spatial (nearby deaths block actions).
        """
        if (self.last_winner >= 0
                and self.last_winner < len(self.ops)
                and self.ops[self.last_winner] == self.OP_ATTRACT):
            self.ops[self.last_winner] = self.OP_REPEL
        self.recent_winners.clear()

    def on_level_up(self):
        """Recent ATTRACT winners → FREEZE. Preserve success.

        FREEZE entries contribute normally to scoring but can't be attracted.
        They're permanent landmarks of successful behavior that persist across
        levels, enabling transfer (I5).
        """
        for w in self.recent_winners[-10:]:
            if (w >= 0 and w < len(self.ops)
                    and self.ops[w] == self.OP_ATTRACT):
                self.ops[w] = self.OP_FREEZE
        self.recent_winners.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Encoding (identical to Step 353)
# ═══════════════════════════════════════════════════════════════════════════════

def avgpool16(frame):
    """64x64 frame → 16x16 average pool → 256D flat vector."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(pooled, fold):
    """Normalize, then subtract codebook mean (U16: encode differences)."""
    t      = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if fold.V.shape[0] > 2:
        mean_V = fold.V.mean(dim=0).cpu()
        t_unit = t_unit - mean_V
    return t_unit


# ═══════════════════════════════════════════════════════════════════════════════
# Game runner (parameterized by fold instance)
# ═══════════════════════════════════════════════════════════════════════════════

def run_game(arc, game_id, fold, max_steps=50000, max_resets=500,
             k=3, verbose=True):
    from arcengine import GameState

    env = arc.make(game_id)
    obs = env.reset()

    total_steps     = 0
    total_resets    = 0
    total_levels    = 0
    game_over_count = 0
    steps_per_lvl   = []
    lvl_step_start  = 0
    seeded          = False
    win             = False
    unique_states   = set()
    action_counts   = {}
    snapshots       = []

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
            fold.on_game_over()
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

        pooled = avgpool16(obs.frame)
        enc    = centered_enc(pooled, fold)

        state_hash = hash(pooled.tobytes())
        unique_states.add(state_hash)

        # Force-seed: one entry per action class
        if not seeded and fold.V.shape[0] < n_acts:
            i = fold.V.shape[0]
            fold._force_add(enc, label=i)
            action = action_space[i % n_acts]
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
                          f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}",
                          flush=True)
            if obs is not None and obs.levels_completed > obs_levels_before:
                total_levels = obs.levels_completed
                fold.on_level_up()
                steps_this = total_steps - lvl_step_start
                steps_per_lvl.append(steps_this)
                lvl_step_start = total_steps
            if obs is not None and obs.state == GameState.WIN:
                win = True
            continue

        if not seeded:
            seeded = True

        # ── Main loop: process observation ──
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

        # Snapshot every 5000 steps
        if total_steps % 5000 == 0:
            op_dist = {}
            if hasattr(fold, 'ops') and fold.ops.shape[0] > 0:
                for name, val in [('A', 0), ('F', 1), ('R', 2)]:
                    op_dist[name] = int((fold.ops == val).sum().item())
            snap = (total_steps, fold.V.shape[0], fold.thresh,
                    len(unique_states), total_levels, game_over_count, op_dist)
            snapshots.append(snap)
            if verbose:
                ops_str = f"  ops={op_dist}" if op_dist else ""
                print(f"    [step {total_steps:6d}]"
                      f"  cb={fold.V.shape[0]:5d}  thresh={fold.thresh:.4f}"
                      f"  unique={len(unique_states):5d}"
                      f"  levels={total_levels}  go={game_over_count}"
                      f"{ops_str}", flush=True)

        if obs.levels_completed > obs_levels_before:
            total_levels = obs.levels_completed
            fold.on_level_up()
            steps_this = total_steps - lvl_step_start
            steps_per_lvl.append(steps_this)
            lvl_step_start = total_steps
            if verbose:
                print(f"    [step {total_steps}] LEVEL {obs.levels_completed}"
                      f"  steps_this={steps_this}"
                      f"  cb={fold.V.shape[0]}", flush=True)

        if obs.state == GameState.WIN:
            win = True
            break

    # ── Collect final stats ──
    cls_dist = {}
    if fold.V.shape[0] > 0:
        for lbl in fold.labels.cpu().numpy():
            cls_dist[int(lbl)] = cls_dist.get(int(lbl), 0) + 1

    op_dist = {}
    if hasattr(fold, 'ops') and fold.ops.shape[0] > 0:
        for name, val in [('ATTRACT', 0), ('FREEZE', 1), ('REPEL', 2)]:
            op_dist[name] = int((fold.ops == val).sum().item())

    # Per-class REPEL count
    repel_per_class = {}
    if hasattr(fold, 'ops') and fold.ops.shape[0] > 0:
        for c in range(int(fold.labels.max().item()) + 1):
            c_mask = (fold.labels == c) & (fold.ops == AdaptiveFold.OP_REPEL)
            repel_per_class[c] = int(c_mask.sum().item())

    return {
        'win':              win,
        'levels':           total_levels,
        'steps':            total_steps,
        'resets':           total_resets,
        'game_over':        game_over_count,
        'steps_per_level':  steps_per_lvl,
        'cb_final':         fold.V.shape[0],
        'thresh_final':     fold.thresh,
        'unique_states':    len(unique_states),
        'action_counts':    action_counts,
        'cls_dist':         dict(sorted(cls_dist.items())),
        'op_dist':          op_dist,
        'repel_per_class':  repel_per_class,
        'snapshots':        snapshots,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("Step 417 — Adaptive Substrate: Operations as Data", flush=True)
    print("=" * 70, flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Phase 2 begins. First experiment in the correct search space.", flush=True)
    print(flush=True)
    print("Search space: codebook encodes DATA + RULES.", flush=True)
    print("This experiment: per-entry operation flags (ATTRACT/FREEZE/REPEL).", flush=True)
    print("  REPEL created on GAME_OVER — learn from failure (R4).", flush=True)
    print("  FREEZE created on LEVEL_UP — preserve success (I5 transfer).", flush=True)
    print(flush=True)

    import arc_agi
    arc   = arc_agi.Arcade()
    games = arc.get_environments()
    ls20  = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if ls20 is None:
        print("ERROR: LS20 not found", flush=True)
        return

    MAX_STEPS = 50000
    K = 3

    # ── Run baseline ──
    print(f"{'-'*70}", flush=True)
    print(f"RUN 1: Baseline (CompressedFold, Step 353 equivalent)", flush=True)
    print(f"  game={ls20.title}  max_steps={MAX_STEPS}  k={K}", flush=True)
    print(f"{'-'*70}", flush=True)
    fold_base = CompressedFold(d=D_ENC, k=K)
    r_base = run_game(arc, ls20.game_id, fold_base,
                       max_steps=MAX_STEPS, k=K, verbose=True)
    print(flush=True)

    # ── Run adaptive ──
    print(f"{'-'*70}", flush=True)
    print(f"RUN 2: Adaptive (AdaptiveFold, operations as data)", flush=True)
    print(f"  game={ls20.title}  max_steps={MAX_STEPS}  k={K}"
          f"  repel_weight=0.5", flush=True)
    print(f"{'-'*70}", flush=True)
    fold_new = AdaptiveFold(d=D_ENC, k=K, repel_weight=0.5)
    r_new = run_game(arc, ls20.game_id, fold_new,
                      max_steps=MAX_STEPS, k=K, verbose=True)
    print(flush=True)

    # ── Summary ──
    elapsed = time.time() - t0
    print("=" * 70, flush=True)
    print("STEP 417 COMPARISON", flush=True)
    print("=" * 70, flush=True)

    for label, r in [("Baseline (CompressedFold)", r_base),
                     ("Adaptive (AdaptiveFold)",   r_new)]:
        print(flush=True)
        print(f"  {label}:", flush=True)
        print(f"    levels={r['levels']}  steps={r['steps']}"
              f"  game_over={r['game_over']}", flush=True)
        print(f"    cb={r['cb_final']}  thresh={r['thresh_final']:.4f}"
              f"  unique={r['unique_states']}", flush=True)
        print(f"    actions={r['action_counts']}", flush=True)
        print(f"    cls_dist={r['cls_dist']}", flush=True)
        if r['steps_per_level']:
            print(f"    steps_per_level={r['steps_per_level']}", flush=True)
        if r['op_dist']:
            print(f"    ops={r['op_dist']}", flush=True)
        if r['repel_per_class']:
            print(f"    repel_per_class={r['repel_per_class']}", flush=True)

    # ── Kill check ──
    print(flush=True)
    print("=" * 70, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 70, flush=True)

    base_l1 = r_base['levels'] >= 1
    new_l1  = r_new['levels'] >= 1

    print(f"  Baseline Level 1: {'FOUND' if base_l1 else 'NOT FOUND'}"
          f" (steps={r_base['steps']})", flush=True)
    print(f"  Adaptive Level 1: {'FOUND' if new_l1 else 'NOT FOUND'}"
          f" (steps={r_new['steps']})", flush=True)

    if new_l1 and not base_l1:
        print(f"  → ADAPTIVE WINS (found level, baseline didn't)", flush=True)
    elif base_l1 and not new_l1:
        print(f"  → BASELINE WINS (additions hurt — U13 confirmed)", flush=True)
    elif new_l1 and base_l1:
        base_steps = r_base['steps_per_level'][0] if r_base['steps_per_level'] else MAX_STEPS
        new_steps  = r_new['steps_per_level'][0] if r_new['steps_per_level'] else MAX_STEPS
        if new_steps < base_steps * 0.8:
            print(f"  → ADAPTIVE FASTER ({new_steps} vs {base_steps} steps to L1)",
                  flush=True)
        elif base_steps < new_steps * 0.8:
            print(f"  → BASELINE FASTER ({base_steps} vs {new_steps} steps to L1)",
                  flush=True)
        else:
            print(f"  → TIE (both found L1: baseline={base_steps},"
                  f" adaptive={new_steps})", flush=True)
    else:
        print(f"  → BOTH MISSED (stochastic — need more runs)", flush=True)

    # ── REPEL analysis ──
    if r_new['op_dist']:
        print(flush=True)
        print("REPEL ANALYSIS:", flush=True)
        total_entries = r_new['cb_final']
        n_repel = r_new['op_dist'].get('REPEL', 0)
        n_freeze = r_new['op_dist'].get('FREEZE', 0)
        print(f"  Total entries: {total_entries}", flush=True)
        print(f"  REPEL:  {n_repel} ({100*n_repel/max(1,total_entries):.1f}%)",
              flush=True)
        print(f"  FREEZE: {n_freeze} ({100*n_freeze/max(1,total_entries):.1f}%)",
              flush=True)
        if r_new['repel_per_class']:
            print(f"  REPEL per action: {r_new['repel_per_class']}", flush=True)
            vals = list(r_new['repel_per_class'].values())
            if max(vals) > 1.5 * min(max(1, v) for v in vals):
                print(f"  → ASYMMETRIC REPEL: system learned to avoid"
                      f" specific actions", flush=True)
            else:
                print(f"  → SYMMETRIC REPEL: deaths spread evenly"
                      f" (limited signal)", flush=True)

    print(flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)
    print(flush=True)

    # ── What this means for the search ──
    print("=" * 70, flush=True)
    print("SEARCH SPACE ASSESSMENT", flush=True)
    print("=" * 70, flush=True)
    print("This experiment tests: can per-entry operation flags (REPEL/FREEZE)", flush=True)
    print("improve exploration efficiency via game-event learning?", flush=True)
    print(flush=True)
    print("If REPEL helps: next step = non-saturated metric (Mahalanobis)", flush=True)
    print("  so REPEL blocking becomes spatial, not just statistical.", flush=True)
    print("If REPEL neutral: expected at saturation. Test on non-saturated", flush=True)
    print("  scale (a%b at d=2) where spatial REPEL can show full effect.", flush=True)
    print("If REPEL hurts: U13 confirmed for this mechanism. Revert,", flush=True)
    print("  try different direction (metric adaptation as first move).", flush=True)


if __name__ == '__main__':
    main()
