"""
Step 1025 — Mode Map WHERE + 800b HOW + W_pred WHEN (Defense Track)

Tests whether old framework components (800b delta, W_pred) add HOW/WHEN
capability on top of mode map WHERE discovery. Target: VC33 L2+.
"""
import os, sys, time
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import numpy as np
from scipy.ndimage import label as ndlabel

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import arc_agi
from arcengine import GameAction, GameState

# ─── Constants ───
MODE_WARMUP = 5000
HOW_PHASE = 2000
WHEN_PHASE = 3000
MIN_CLUSTER = 4
MAX_CLUSTER = 60
MAX_STEPS = 50_000
TIME_CAP = 120  # longer for L2+ attempts
BURST = 5
N_GRID = 64
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
DIM = 256
EMA_ALPHA = 0.1

# ─── Mode map (from 576/1023) ───
def update_freq(freq_arr, frame):
    arr = np.array(frame[0], dtype=np.int32)
    r, c = np.arange(64)[:, None], np.arange(64)[None, :]
    freq_arr[r, c, arr] += 1

def compute_mode(freq_arr):
    return np.argmax(freq_arr, axis=2).astype(np.int32)

def find_isolated_clusters(mode_arr):
    clusters = []
    for color in range(1, 16):
        mask = (mode_arr == color)
        if not mask.any(): continue
        labeled, n = ndlabel(mask)
        for cid in range(1, n + 1):
            region = (labeled == cid)
            sz = int(region.sum())
            if 4 <= sz <= 60:
                ys, xs = np.where(region)
                clusters.append({'cx_int': int(round(xs.mean())), 'cy_int': int(round(ys.mean())),
                                  'color': int(color), 'size': sz})
    return clusters

def enc_frame(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()

# ─── Combined substrate ───
class ModeMap800bWpred:
    def __init__(self, seed=0):
        self.freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.clusters = []
        self.phase = 'where'
        self.step = 0
        # 800b components
        self.running_mean = np.zeros(DIM, np.float32)
        self.n_obs = 0
        self.prev_enc = None
        self.zone_delta = {}       # zone_idx -> EMA delta
        self.zone_interactive = []  # sorted zones by interactivity
        # W_pred (simple linear forward model)
        self.W_pred = np.zeros((DIM, DIM), np.float32)
        self.zone_pred_error = {}  # zone_idx -> cumulative pred error
        # WHEN tracking
        self.ordering_scores = {}  # tuple(order) -> cumulative W_pred coherence
        self.current_order = []
        self.order_idx = 0
        self.best_order = None
        self.rng = np.random.RandomState(seed)

    def encode(self, frame):
        x = enc_frame(frame)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def act(self, frame):
        enc = self.encode(frame)
        self.step += 1

        # Phase 1: WHERE (mode map)
        if self.phase == 'where':
            update_freq(self.freq, frame)
            if self.step >= MODE_WARMUP:
                mode = compute_mode(self.freq)
                self.clusters = find_isolated_clusters(mode)
                if self.clusters:
                    self.phase = 'how'
                    self.zone_delta = {i: 1.0 for i in range(len(self.clusters))}
                    self.zone_pred_error = {i: 0.0 for i in range(len(self.clusters))}
                    self._how_zone = 0
                    self._how_burst = BURST
            a = self.rng.randint(N_GRID)
            cx, cy = CLICK_GRID[a]
            self.prev_enc = enc
            return cx, cy

        # Phase 2: HOW (800b delta per zone)
        elif self.phase == 'how':
            if self.prev_enc is not None:
                change = float(np.linalg.norm(enc - self.prev_enc))
                z = self._how_zone
                self.zone_delta[z] = (1 - EMA_ALPHA) * self.zone_delta[z] + EMA_ALPHA * change
                # W_pred update
                pred = self.W_pred @ self.prev_enc
                err = enc - pred
                self.W_pred += 0.001 * np.outer(err, self.prev_enc)
                self.zone_pred_error[z] += float(np.linalg.norm(err))

            self._how_burst -= 1
            if self._how_burst <= 0:
                self._how_zone = (self._how_zone + 1) % len(self.clusters)
                self._how_burst = BURST

            if self.step >= MODE_WARMUP + HOW_PHASE:
                # Classify zones: interactive = high delta
                sorted_zones = sorted(self.zone_delta.items(), key=lambda x: x[1], reverse=True)
                self.zone_interactive = [z for z, d in sorted_zones if d > 0.5]
                if not self.zone_interactive:
                    self.zone_interactive = [z for z, _ in sorted_zones[:5]]
                self.phase = 'when'
                self._when_step = 0
                self._gen_random_order()

            c = self.clusters[self._how_zone]
            self.prev_enc = enc
            return c['cx_int'], c['cy_int']

        # Phase 3: WHEN (ordering discovery)
        else:
            if self.prev_enc is not None:
                pred = self.W_pred @ self.prev_enc
                err = float(np.linalg.norm(enc - pred))
                self.W_pred += 0.001 * np.outer(enc - pred, self.prev_enc)
                order_key = tuple(self.current_order)
                self.ordering_scores[order_key] = self.ordering_scores.get(order_key, 0) + err

            self._when_step += 1
            # Try a new random ordering every N clicks
            if self.order_idx >= len(self.current_order) * BURST:
                self._gen_random_order()

            zone_idx = self.current_order[self.order_idx // BURST % len(self.current_order)]
            self.order_idx += 1
            c = self.clusters[zone_idx]
            self.prev_enc = enc
            return c['cx_int'], c['cy_int']

    def _gen_random_order(self):
        if self.zone_interactive:
            self.current_order = list(self.zone_interactive)
            self.rng.shuffle(self.current_order)
        else:
            self.current_order = list(range(len(self.clusters)))
            self.rng.shuffle(self.current_order)
        self.order_idx = 0

# ─── Run ───
def run_seed(arc, game_id, seed):
    np.random.seed(seed)
    env = arc.make(game_id)
    action6 = list(GameAction)[6]
    sub = ModeMap800bWpred(seed=seed)
    obs = env.reset()
    steps = levels = 0
    l1_step = l2_step = None
    t0 = time.time()

    while steps < MAX_STEPS:
        if time.time() - t0 > TIME_CAP: break
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); continue

        cx, cy = sub.act(obs.frame)
        lvls_before = obs.levels_completed
        obs = env.step(action6, data={"x": cx, "y": cy})
        steps += 1
        if obs is None: break
        if obs.levels_completed > lvls_before:
            levels = obs.levels_completed
            if l1_step is None: l1_step = steps
            if levels >= 2 and l2_step is None: l2_step = steps

    elapsed = time.time() - t0
    return {'seed': seed, 'levels': levels, 'l1_step': l1_step, 'l2_step': l2_step,
            'steps': steps, 'elapsed': round(elapsed, 1), 'phase': sub.phase,
            'n_clusters': len(sub.clusters), 'n_interactive': len(sub.zone_interactive)}

def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    if not vc33:
        print("SKIP"); return

    print("=== Step 1025: Mode Map + 800b + W_pred (Defense) ===")
    print(f"Phases: WHERE={MODE_WARMUP}, HOW={HOW_PHASE}, WHEN={WHEN_PHASE}")
    print(f"Game: {vc33.game_id}\n")

    results = []
    for seed in range(5):
        r = run_seed(arc, vc33.game_id, seed)
        status = f"L{r['levels']}" + (f"@{r['l1_step']}" if r['l1_step'] else "")
        l2 = f" L2@{r['l2_step']}" if r['l2_step'] else ""
        print(f"  s{seed}: {status:12s}{l2}  clusters={r['n_clusters']}  "
              f"interactive={r['n_interactive']}  phase={r['phase']}  {r['elapsed']}s")
        results.append(r)

    wins = sum(1 for r in results if r['levels'] > 0)
    l2 = sum(1 for r in results if r['l2_step'] is not None)
    print(f"\n  VC33: {wins}/5 L1, {l2}/5 L2+")
    print(f"  Comparison: 1023 mode-map-only = 5/5 L1, 0/5 L2+")
    if l2 > 0:
        print(f"  SIGNAL: 800b/W_pred adds L2+ capability beyond mode map alone")
    else:
        print(f"  NO L2 PROGRESS: HOW/WHEN additions don't help for VC33 L2+")
    print("\nStep 1025 DONE")

if __name__ == "__main__":
    main()
