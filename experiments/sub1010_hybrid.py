"""
sub1010_hybrid.py — Hybrid 800b + Attention + CC Zone Discovery.

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1010 --substrate experiments/sub1010_hybrid.py

FAMILY: hybrid-action (new — combines extraction-cc + attention-trajectory components)
R3 HYPOTHESIS: A substrate that uses 800b for coverage AND attention for temporal context
  can navigate LS20 (coverage) while providing sequential signal for FT09/VC33 (temporal credit).
  R3 on action selection: combined signal from two information sources adapts to game structure.

DESIGN (from Leo 2026-03-24):
  800b = coverage signal (delta_per_zone EMA) — preserves LS20 navigation
  Attention = temporal credit signal (buffer retrieval) — targets FT09/VC33
  CC zones = action space adaptation (discovered from game interaction)
  combined[a] = 800b_score[a] + beta * attn_score[a]
  beta = 0.0 for first WARMUP steps (coverage only), then 0.5 (hybrid)

KEY DIFFERENCES FROM 1007/1009:
  - 800b is BASE signal, attention is ADDITIVE (not replacement)
  - LS20 should be preserved: coverage still active when beta=0.5
  - CC zone discovery adapts action space without game-specific logic
  - One config — no game-conditional logic

KILL: LS20 L1 < 80% (coverage regression) AND FT09=0 AND VC33=0
ALIVE: FT09 or VC33 L1 > 0 on ANY seed = breakthrough
SUCCESS: chain score > 3/5
BUDGET: 10K steps/game, 10 seeds
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

# ─── Hyperparameters ───
ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM   # 320

# Encoding (994/916 style)
RUNNING_MEAN_ALPHA = 0.1
ETA_W = 0.01                 # W_pred learning rate
ALPHA_UPDATE_DELAY = 50      # steps before alpha starts updating
ALPHA_LO = 0.10
ALPHA_HI = 5.00

# Action selection
ALPHA_EMA = 0.10             # 800b delta EMA rate
INIT_DELTA = 1.0
EPS = 0.20                   # random exploration rate
TEMP = 0.10                  # softmax temperature
BETA = 0.5                   # attention weight in combined score
WARMUP = 200                 # steps before attention activates (beta = 0 before this)

# Attention buffer
MAX_BUFFER = 200
BUFFER_MIN = 10              # minimum entries before using attention

# CC zone discovery (same as 1008)
PHASE1_STEPS = 1000
REDISCOVERY_INTERVAL = 2000
MODE_EMA_ALPHA = 0.1
DEVIATION_PERCENTILE = 95
MIN_ZONE_PIXELS = 4
MAX_ZONES = 20
MIN_ZONES_FALLBACK = 4

CONFIG = {
    "ENC_DIM": ENC_DIM, "H_DIM": H_DIM,
    "RUNNING_MEAN_ALPHA": RUNNING_MEAN_ALPHA, "ETA_W": ETA_W,
    "ALPHA_EMA": ALPHA_EMA, "EPS": EPS, "TEMP": TEMP,
    "BETA": BETA, "WARMUP": WARMUP, "MAX_BUFFER": MAX_BUFFER,
    "PHASE1_STEPS": PHASE1_STEPS, "REDISCOVERY_INTERVAL": REDISCOVERY_INTERVAL,
}


# ─── Utilities ───

def _obs_to_gray(obs):
    """Convert obs to 2D grayscale. Handles (C,H,W) and (H,W,C)."""
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[1]:   # (C, H, W)
            arr = arr.mean(axis=0)
        else:                              # (H, W, C)
            arr = arr.mean(axis=2)
    return arr


def _find_cc(binary_mask):
    """4-connected BFS. Returns list of (size, cy, cx) sorted by size desc."""
    H, W = binary_mask.shape
    visited = np.zeros((H, W), dtype=bool)
    results = []
    for r0 in range(H):
        for c0 in range(W):
            if not binary_mask[r0, c0] or visited[r0, c0]:
                continue
            q = deque([(r0, c0)])
            visited[r0, c0] = True
            sum_r = sum_c = count = 0
            while q:
                r, c = q.popleft()
                sum_r += r; sum_c += c; count += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and binary_mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if count >= MIN_ZONE_PIXELS:
                results.append((count, sum_r // count, sum_c // count))
    results.sort(key=lambda x: -x[0])
    return results


def _centroid_to_action(cy, cx, H, W, n_actions):
    """Map zone centroid to game click action. None for directional games."""
    if n_actions <= 8:
        return None
    n_dir = 4
    n_click = n_actions - n_dir
    grid = int(round(n_click ** 0.5))
    row = min(int(cy * grid / H), grid - 1)
    col = min(int(cx * grid / W), grid - 1)
    return n_dir + row * grid + col


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def _softmax_sel(vals, temp, rng):
    x = np.array(vals, dtype=np.float32) / temp
    x -= x.max()
    e = np.exp(x)
    p = e / (e.sum() + 1e-12)
    return int(rng.choice(len(vals), p=p))


# ─── Substrate ───

class HybridActionSubstrate:
    """
    Hybrid 800b + Attention + CC Zone Discovery (Step 1010).

    Combines:
    - 994/916 encoding (running_mean + ESN h + alpha weighting + W_pred)
    - 800b delta EMA per zone (coverage signal)
    - Attention buffer (temporal credit signal)
    - CC zone discovery (action space adaptation)

    combined[zone] = 800b_score[zone] + beta * attn_score[zone]
    beta = 0 for first WARMUP steps (800b only), then 0.5 (hybrid)

    Interface contract (ChainRunner-compatible):
      process(obs: np.ndarray) -> int
      on_level_transition()
      set_game(n_actions: int)
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # 994/916 encoding
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM), dtype=np.float32)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_actions_game = 4

        # Action tracking (per zone)
        self._zone_actions = [0, 1, 2, 3]
        self._n_zone_actions = 4
        self._delta_per_zone = np.full(4, INIT_DELTA, dtype=np.float32)

        # Trajectory buffer (enc-keyed, zone-indexed)
        self._buf_K = deque(maxlen=MAX_BUFFER)   # past encs (ENC_DIM,)
        self._buf_Z = deque(maxlen=MAX_BUFFER)   # zone indices
        self._buf_D = deque(maxlen=MAX_BUFFER)   # deltas (alpha-weighted)

        # CC zone discovery
        self._pixel_ema = None
        self._frame_h = self._frame_w = None
        self._step_in_game = 0

        # Previous step state
        self._prev_ext = None
        self._prev_zone = None

    def set_game(self, n_actions: int):
        """Called on game switch. Reset per-game state; keep global running mean and W_h/W_x."""
        self._n_actions_game = n_actions
        self._step_in_game = 0

        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM), dtype=np.float32)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._pred_errors.clear()

        # Zone discovery reset
        self._pixel_ema = None
        self._frame_h = self._frame_w = None
        self._zone_actions = list(range(n_actions))
        self._n_zone_actions = n_actions
        self._delta_per_zone = np.full(n_actions, INIT_DELTA, dtype=np.float32)

        # Buffer reset (prevents cross-game contamination)
        self._buf_K.clear()
        self._buf_Z.clear()
        self._buf_D.clear()

        self._prev_ext = None
        self._prev_zone = None

    # ── CC Zone Discovery ──

    def _update_mode_map(self, obs):
        gray = _obs_to_gray(obs)
        if self._pixel_ema is None:
            self._frame_h, self._frame_w = gray.shape
            self._pixel_ema = gray.copy()
        else:
            self._pixel_ema += MODE_EMA_ALPHA * (gray - self._pixel_ema)
        return gray

    def _read_gray(self, obs):
        gray = _obs_to_gray(obs)
        if self._frame_h is None:
            self._frame_h, self._frame_w = gray.shape
        return gray

    def _run_zone_discovery(self, current_gray):
        if self._pixel_ema is None or self._frame_h is None:
            return
        if current_gray.shape != self._pixel_ema.shape:
            return
        dev = np.abs(current_gray - self._pixel_ema)
        thresh = np.percentile(dev, DEVIATION_PERCENTILE)
        mask = dev >= thresh
        components = _find_cc(mask)[:MAX_ZONES]

        valid_actions = []
        for _size, cy, cx in components:
            ga = _centroid_to_action(cy, cx, self._frame_h, self._frame_w, self._n_actions_game)
            if ga is not None:
                valid_actions.append(ga)

        if len(valid_actions) >= MIN_ZONES_FALLBACK:
            self._zone_actions = valid_actions
            self._n_zone_actions = len(valid_actions)
            self._delta_per_zone = np.full(self._n_zone_actions, INIT_DELTA, dtype=np.float32)
        else:
            self._zone_actions = list(range(self._n_actions_game))
            self._n_zone_actions = self._n_actions_game
            self._delta_per_zone = np.full(self._n_zone_actions, INIT_DELTA, dtype=np.float32)

    # ── Encoding ──

    def _encode(self, obs):
        """994/916 encoding: avgpool16 + running_mean + ESN h → ext (320D)."""
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        a = RUNNING_MEAN_ALPHA
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = (enc_raw - self._running_mean).astype(np.float32)
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        return enc, np.concatenate([enc, self.h]).astype(np.float32)

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY:
            return
        mean_errors = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(mean_errors)) or np.any(np.isinf(mean_errors)):
            return
        raw_alpha = np.sqrt(np.clip(mean_errors, 0, 1e6) + 1e-8)
        mean_raw = np.mean(raw_alpha)
        if mean_raw < 1e-8 or np.isnan(mean_raw):
            return
        self.alpha = np.clip(raw_alpha / mean_raw, ALPHA_LO, ALPHA_HI)

    # ── Main Process ──

    def process(self, obs: np.ndarray) -> int:
        self._step_in_game += 1

        # ── Phase 1: random + build mode map ──
        if self._step_in_game <= PHASE1_STEPS:
            current_gray = self._update_mode_map(obs)
            enc, ext = self._encode(obs)
            if self._step_in_game == PHASE1_STEPS:
                self._run_zone_discovery(current_gray)
            action = int(self._rng.randint(0, self._n_actions_game))
            self._prev_ext = ext
            self._prev_zone = action   # track raw action during Phase 1
            return action

        # ── Phase 2: hybrid action selection ──
        current_gray = self._read_gray(obs)
        steps_in_phase2 = self._step_in_game - PHASE1_STEPS
        if steps_in_phase2 % REDISCOVERY_INTERVAL == 0:
            self._run_zone_discovery(current_gray)

        enc, ext = self._encode(obs)

        # W_pred + alpha update
        if self._prev_ext is not None:
            pred = self.W_pred @ self._prev_ext
            error = (ext * self.alpha) - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W * np.outer(error, self._prev_ext)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

        # 800b delta update + buffer append
        if self._prev_ext is not None and self._prev_zone is not None:
            weighted_change = (ext - self._prev_ext) * self.alpha
            delta = float(np.linalg.norm(weighted_change))
            z = self._prev_zone
            if 0 <= z < self._n_zone_actions:
                self._delta_per_zone[z] = (
                    (1 - ALPHA_EMA) * self._delta_per_zone[z] + ALPHA_EMA * delta
                )
            # Append to buffer (use enc, not ext, as key for attention)
            if z < self._n_zone_actions:
                self._buf_K.append(enc.copy())
                self._buf_Z.append(z)
                self._buf_D.append(delta)

        # ── Hybrid action selection ──
        if self._rng.random() < EPS:
            z = int(self._rng.randint(0, self._n_zone_actions))
        else:
            # 800b score (always present)
            score_800b = self._delta_per_zone.copy()

            # Attention score (when buffer ready and past warmup)
            n_buf = len(self._buf_K)
            use_attention = (self._step_in_game > PHASE1_STEPS + WARMUP) and (n_buf >= BUFFER_MIN)
            beta = BETA if use_attention else 0.0

            if use_attention:
                K = np.stack(self._buf_K)              # (T, ENC_DIM)
                Z = np.array(self._buf_Z, dtype=np.int32)
                D = np.array(self._buf_D, dtype=np.float32)
                scores = K @ enc / (ENC_DIM ** 0.5) / TEMP
                attn = _softmax(scores)
                attn_score = np.zeros(self._n_zone_actions, dtype=np.float32)
                weighted_D = attn * D
                for t in range(len(Z)):
                    if Z[t] < self._n_zone_actions:
                        attn_score[Z[t]] += weighted_D[t]
                combined = score_800b + beta * attn_score
            else:
                combined = score_800b

            z = _softmax_sel(combined, TEMP, self._rng)

        action = self._zone_actions[z] if z < len(self._zone_actions) else \
            int(self._rng.randint(0, self._n_actions_game))

        self._prev_ext = ext.copy()
        self._prev_zone = z
        return action

    def on_level_transition(self):
        """Clear prev to avoid cross-episode delta corruption."""
        self._prev_ext = None
        self._prev_zone = None


SUBSTRATE_CLASS = HybridActionSubstrate
