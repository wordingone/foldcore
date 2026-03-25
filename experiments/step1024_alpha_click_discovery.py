"""
Step 1024 — Alpha-Guided Click Target Discovery (Defense Track)

DEBATE EXPERIMENT (defense). Tests whether alpha encoding (R3, Step 895)
provides efficient WHERE discovery for click games.

Three phases:
  Phase 1 (500 steps): Random ACTION6 clicks on 8x8 grid. Record alpha-weighted
  frame diff per position. Alpha learns informative encoding dims from prediction error.
  Phase 2 (500 steps): Top-8 coarse positions → probe 4x4 subgrid around each.
  Phase 3 (remaining): Click all discovered targets by effect magnitude.

Test on FT09 and VC33. Same substrate, same config.
"""
import os, sys, time
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import numpy as np

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import arc_agi
from arcengine import GameAction, GameState

# ─── Constants ───
DIM = 256
PHASE1_STEPS = 500
PHASE2_STEPS = 500
MAX_STEPS = 50_000
TIME_CAP = 60
EMA_ALPHA = 0.1
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
TOP_K = 8
SUBGRID_RADIUS = 4

# ─── Encoding (from 674/800b) ───

def enc_frame(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()

# ─── Alpha-weighted probing substrate ───

class AlphaClickProbe:
    def __init__(self, seed=0):
        self.running_mean = np.zeros(DIM, np.float32)
        self.n_obs = 0
        self.prev_enc = None
        self.alpha_weights = np.ones(DIM, np.float32)  # attention per dim
        self.pred_w = np.zeros((DIM, DIM), np.float32)  # simple linear predictor
        # Per-position response tracking
        self.coarse_response = np.zeros(64, np.float32)
        self.coarse_counts = np.zeros(64, np.int32)
        self.fine_positions = []
        self.fine_response = {}
        self.discovered_targets = []
        self.phase = 'coarse'
        self.step = 0
        self.rng = np.random.RandomState(seed)

    def encode(self, frame):
        x = enc_frame(frame)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        centered = x - self.running_mean
        return centered

    def update_alpha(self, enc):
        """Update alpha weights from prediction error."""
        if self.prev_enc is not None:
            pred = self.pred_w @ self.prev_enc
            error = enc - pred
            # Update predictor (simple online gradient)
            lr = 0.01
            self.pred_w += lr * np.outer(error, self.prev_enc)
            # Alpha = normalized absolute prediction error per dim
            abs_err = np.abs(error)
            self.alpha_weights = 0.9 * self.alpha_weights + 0.1 * abs_err
        self.prev_enc = enc.copy()

    def weighted_change(self, enc1, enc2):
        """Alpha-weighted change magnitude."""
        diff = enc2 - enc1
        return float(np.sum(self.alpha_weights * np.abs(diff)))

    def act(self, frame):
        enc = self.encode(frame)
        self.update_alpha(enc)
        self.step += 1

        if self.phase == 'coarse' and self.step <= PHASE1_STEPS:
            # Random 8x8 grid probe
            idx = self.rng.randint(64)
            cx, cy = CLICK_GRID[idx]
            self._pending_idx = idx
            self._pending_enc = enc.copy()
            return cx, cy

        elif self.phase == 'coarse' and self.step == PHASE1_STEPS + 1:
            # Transition to fine
            avg_response = np.where(self.coarse_counts > 0,
                                    self.coarse_response / self.coarse_counts, 0)
            top_indices = np.argsort(avg_response)[-TOP_K:]
            self.fine_positions = []
            for idx in top_indices:
                bx, by = CLICK_GRID[idx]
                for dx in range(-SUBGRID_RADIUS, SUBGRID_RADIUS + 1, 2):
                    for dy in range(-SUBGRID_RADIUS, SUBGRID_RADIUS + 1, 2):
                        fx, fy = bx + dx, by + dy
                        if 0 <= fx < 64 and 0 <= fy < 64:
                            self.fine_positions.append((fx, fy))
            self.phase = 'fine'
            self._fine_idx = 0
            return self.fine_positions[0] if self.fine_positions else (32, 32)

        elif self.phase == 'fine' and self._fine_idx < len(self.fine_positions):
            pos = self.fine_positions[self._fine_idx]
            self._pending_fine = pos
            self._pending_enc = enc.copy()
            self._fine_idx += 1
            if self._fine_idx >= len(self.fine_positions):
                # Transition to execute
                sorted_fine = sorted(self.fine_response.items(),
                                     key=lambda x: x[1], reverse=True)
                self.discovered_targets = [pos for pos, _ in sorted_fine[:20]]
                self.phase = 'execute'
                self._exec_idx = 0
            return pos

        else:
            # Execute: cycle through discovered targets
            if self.discovered_targets:
                pos = self.discovered_targets[self._exec_idx % len(self.discovered_targets)]
                self._exec_idx += 1
                return pos
            return (32, 32)

    def observe_result(self, frame):
        """Called after action to record response."""
        enc = self.encode(frame)
        if hasattr(self, '_pending_enc'):
            change = self.weighted_change(self._pending_enc, enc)
            if self.phase == 'coarse' and hasattr(self, '_pending_idx'):
                self.coarse_response[self._pending_idx] += change
                self.coarse_counts[self._pending_idx] += 1
            elif self.phase == 'fine' and hasattr(self, '_pending_fine'):
                pos = self._pending_fine
                self.fine_response[pos] = self.fine_response.get(pos, 0) + change

# ─── Run one seed ───

def run_seed(arc, game_id, seed):
    np.random.seed(seed)
    env = arc.make(game_id)
    action6 = list(GameAction)[6]
    sub = AlphaClickProbe(seed=seed)

    obs = env.reset()
    steps = 0
    levels = 0
    l1_step = None
    t0 = time.time()

    while steps < MAX_STEPS:
        if time.time() - t0 > TIME_CAP:
            break
        if obs is None:
            obs = env.reset()
            continue
        if obs.state == GameState.GAME_OVER:
            obs = env.reset()
            continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset()
            continue

        cx, cy = sub.act(obs.frame)
        lvls_before = obs.levels_completed
        obs = env.step(action6, data={"x": cx, "y": cy})
        steps += 1

        if obs is not None and obs.frame:
            sub.observe_result(obs.frame)

        if obs is None:
            break
        if obs.levels_completed > lvls_before:
            levels = obs.levels_completed
            if l1_step is None:
                l1_step = steps

    elapsed = time.time() - t0
    return {
        'seed': seed, 'levels': levels, 'l1_step': l1_step,
        'steps': steps, 'elapsed': round(elapsed, 1),
        'phase': sub.phase, 'n_targets': len(sub.discovered_targets),
        'targets': sub.discovered_targets[:10],
    }

# ─── Main ───

def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()

    print("=== Step 1024: Alpha-Guided Click Target Discovery (Defense) ===")
    print(f"Phases: coarse={PHASE1_STEPS}, fine={PHASE2_STEPS}, then execute")
    print()

    for game_key in ['ft09', 'vc33']:
        game = next((e for e in envs if game_key in e.game_id.lower()), None)
        if game is None:
            print(f"SKIP — {game_key} not found")
            continue

        print(f"--- {game.game_id} ---")
        results = []
        for seed in range(5):
            r = run_seed(arc, game.game_id, seed)
            status = f"L{r['levels']}@{r['l1_step']}" if r['levels'] > 0 else "FAIL"
            print(f"  s{seed}: {status:12s}  targets={r['n_targets']}  phase={r['phase']}  "
                  f"steps={r['steps']}  {r['elapsed']}s")
            if r['targets']:
                print(f"    top targets: {r['targets'][:5]}")
            results.append(r)

        wins = sum(1 for r in results if r['levels'] > 0)
        print(f"  {game_key}: {wins}/5 wins")
        print()

    print("Step 1024 DONE")

if __name__ == "__main__":
    main()
