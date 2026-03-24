"""
sub1008_cc_zone.py — CC Zone Discovery + 800b (Extraction 1).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1008 --substrate experiments/sub1008_cc_zone.py

FAMILY: extraction-cc (new)
R3 HYPOTHESIS: Mode map + connected component analysis discovers action spaces
  from game interaction — R3 on action representation.
  The substrate learns WHICH actions exist (click zones) from frame statistics.
  If CC discovery + 800b shows FT09 or VC33 L1 on any seed, it means zone
  discovery is an encoding component that survives the graph ban.

EXTRACTION PROTOCOL (bans/POLICY.md):
  Source: Step 576 (VC33 5/5, autonomous CC zone discovery)
  Killing finding: Graph ban negative transfer (cold > warm, Step 776)
  Test: CC discovery + 800b — no graph, no per-(state,action) storage
  One variable: action space discovery vs always-directional-actions

KILL: chain_score < 1006 baseline AND FT09=0 AND VC33=0
ALIVE: FT09 or VC33 L1 > 0 on ANY seed (unprecedented post-ban)
BUDGET: 10K steps/game, 10 seeds, all 5 phases

Jun directive 2026-03-24:
- Substrate defines the class. Harness (run_experiment.py) is the constant.
- Game order randomized per seed — cannot be bypassed
- chain_kill verdict: PASS / KILL / FAIL vs baseline_994.json
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

# ─── Hyperparameters ───
PHASE1_STEPS = 1000           # random exploration phase — builds mode map
REDISCOVERY_INTERVAL = 2000   # re-run zone discovery every N phase-2 steps
MODE_EMA_ALPHA = 0.1          # EMA rate for pixel mode estimation (Phase 1)
DEVIATION_PERCENTILE = 95     # top 5% of pixels by deviation = interactive
MIN_ZONE_PIXELS = 4           # minimum CC size (pixels) to count as a zone
MAX_ZONES = 20                # cap on discovered zones
MIN_ZONES_FALLBACK = 4        # if fewer valid zones found, use directional fallback
ALPHA_EMA = 0.10              # 800b delta EMA rate
INIT_DELTA = 1.0              # initial delta_per_action value
EPSILON = 0.20                # random exploration rate
SOFTMAX_TEMP = 0.10           # softmax temperature for zone selection

CONFIG = {
    "PHASE1_STEPS": PHASE1_STEPS,
    "REDISCOVERY_INTERVAL": REDISCOVERY_INTERVAL,
    "MODE_EMA_ALPHA": MODE_EMA_ALPHA,
    "DEVIATION_PERCENTILE": DEVIATION_PERCENTILE,
    "MAX_ZONES": MAX_ZONES,
    "ALPHA_EMA": ALPHA_EMA,
    "EPSILON": EPSILON,
    "SOFTMAX_TEMP": SOFTMAX_TEMP,
}


# ─── Frame utilities ───

def _obs_to_gray(obs):
    """Convert observation to 2D grayscale float32 array.

    Handles (C, H, W) and (H, W, C) formats. Detection: if shape[0] < shape[1],
    assume channels-first (C, H, W); otherwise channels-last (H, W, C).
    Works for C=1..N channels.
    """
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[1]:   # (C, H, W): C is smallest dim
            arr = arr.mean(axis=0)
        else:                              # (H, W, C)
            arr = arr.mean(axis=2)
    # arr is now (H, W)
    return arr


def _find_cc(binary_mask):
    """4-connected BFS connected components on binary mask.

    Returns list of (size, centroid_y, centroid_x) sorted by size descending.
    Only returns components with size >= MIN_ZONE_PIXELS.
    """
    H, W = binary_mask.shape
    visited = np.zeros((H, W), dtype=bool)
    results = []

    for r0 in range(H):
        for c0 in range(W):
            if not binary_mask[r0, c0] or visited[r0, c0]:
                continue
            # BFS
            q = deque([(r0, c0)])
            visited[r0, c0] = True
            sum_r = sum_c = count = 0
            while q:
                r, c = q.popleft()
                sum_r += r
                sum_c += c
                count += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and binary_mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if count >= MIN_ZONE_PIXELS:
                results.append((count, sum_r // count, sum_c // count))

    results.sort(key=lambda x: -x[0])  # largest first
    return results


def _centroid_to_action(cy, cx, H, W, n_actions):
    """Map zone centroid pixel (cy, cx) to game action index.

    For 68-action games (FT09, VC33): 4 dirs + 64 clicks on 8x8 grid over H×W frame.
    Returns the click action index, or None for directional-only games (n_actions <= 8).
    """
    if n_actions <= 8:
        return None  # directional game: no click mapping
    n_dir = 4
    n_click = n_actions - n_dir              # 64 for 68-action games
    grid = int(round(n_click ** 0.5))        # 8 for 64 clicks
    row = min(int(cy * grid / H), grid - 1)
    col = min(int(cx * grid / W), grid - 1)
    return n_dir + row * grid + col


def _softmax_sel(vals, temp, rng):
    x = np.array(vals, dtype=np.float32) / temp
    x -= x.max()
    e = np.exp(x)
    p = e / (e.sum() + 1e-12)
    return int(rng.choice(len(vals), p=p))


# ─── Substrate ───

class CCZoneSubstrate:
    """
    CC Zone Discovery + 800b substrate (Extraction 1, Step 1008).

    Phase 1 (1000 steps): random exploration. Builds pixel EMA (mode proxy).
    At step 1000: detect high-deviation pixels → connected components → zones.
    Phase 2: 800b delta EMA over discovered zone actions.
    Rediscovery every 2000 phase-2 steps (game may change structure across levels).
    Fallback: if < 4 valid click zones found, use original n_actions (directional).

    Interface contract (ChainRunner-compatible):
      process(obs: np.ndarray) -> int
      on_level_transition()
      set_game(n_actions: int)
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_game = 4    # game's n_actions (from set_game)
        self._step_in_game = 0

        # Pixel mode map (built during Phase 1, frozen after)
        self._pixel_ema = None      # (H, W) array — EMA of pixel values
        self._frame_h = None
        self._frame_w = None

        # Zone action table (set by _run_zone_discovery)
        self._zone_actions = []     # game action index per zone
        self._n_zone_actions = 0
        self._delta_per_zone = np.array([], dtype=np.float32)

        # Running mean encoding (global, persists across games — 800b pattern)
        self._running_mean = np.zeros(256, dtype=np.float32)
        self._n_obs_total = 0
        self._prev_enc = None
        self._prev_zone = None      # zone index of previous action

    def set_game(self, n_actions: int):
        """Called on game switch. Reset zone state; keep running mean."""
        self._n_actions_game = n_actions
        self._step_in_game = 0
        self._pixel_ema = None
        self._frame_h = None
        self._frame_w = None
        self._zone_actions = list(range(n_actions))   # fallback: all game actions
        self._n_zone_actions = n_actions
        self._delta_per_zone = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._prev_enc = None
        self._prev_zone = None

    def _update_mode_map(self, obs):
        """Update pixel EMA (Phase 1 only). Returns grayscale frame."""
        gray = _obs_to_gray(obs)
        if self._pixel_ema is None:
            self._frame_h, self._frame_w = gray.shape
            self._pixel_ema = gray.copy()
        else:
            self._pixel_ema += MODE_EMA_ALPHA * (gray - self._pixel_ema)
        return gray

    def _read_gray(self, obs):
        """Read grayscale frame without updating EMA (Phase 2 use)."""
        gray = _obs_to_gray(obs)
        if self._frame_h is None:
            self._frame_h, self._frame_w = gray.shape
        return gray

    def _run_zone_discovery(self, current_gray):
        """Detect high-deviation pixels, run CC, build zone action table."""
        if self._pixel_ema is None or self._frame_h is None:
            return
        if current_gray.shape != self._pixel_ema.shape:
            return  # obs shape mismatch (e.g., CIFAR phase vs game frame) — skip

        # Deviation from background mode
        dev = np.abs(current_gray - self._pixel_ema)
        thresh = np.percentile(dev, DEVIATION_PERCENTILE)
        mask = dev >= thresh

        components = _find_cc(mask)[:MAX_ZONES]

        # Map centroids to game click actions
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
            # Fallback: use original game action space (directional games, or no clear zones)
            self._zone_actions = list(range(self._n_actions_game))
            self._n_zone_actions = self._n_actions_game
            self._delta_per_zone = np.full(self._n_zone_actions, INIT_DELTA, dtype=np.float32)

    def _encode(self, obs):
        """Standard 256-dim encoding with global running mean centering."""
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs_total += 1
        a = 1.0 / self._n_obs_total
        self._running_mean += a * (enc_raw - self._running_mean)
        return (enc_raw - self._running_mean).astype(np.float32)

    def process(self, obs: np.ndarray) -> int:
        self._step_in_game += 1

        # ── Phase 1: random exploration + build mode map ──
        if self._step_in_game <= PHASE1_STEPS:
            current_gray = self._update_mode_map(obs)
            enc = self._encode(obs)
            self._prev_enc = enc

            if self._step_in_game == PHASE1_STEPS:
                # End of Phase 1: run zone discovery
                self._run_zone_discovery(current_gray)

            action = int(self._rng.randint(0, self._n_actions_game))
            self._prev_zone = action   # track raw action during Phase 1
            return action

        # ── Phase 2: zone-based 800b ──
        current_gray = self._read_gray(obs)

        # Periodic rediscovery
        steps_in_phase2 = self._step_in_game - PHASE1_STEPS
        if steps_in_phase2 % REDISCOVERY_INTERVAL == 0:
            self._run_zone_discovery(current_gray)

        enc = self._encode(obs)

        # 800b delta update
        if self._prev_enc is not None and self._prev_zone is not None:
            delta = float(np.linalg.norm(enc - self._prev_enc))
            z = self._prev_zone
            if 0 <= z < self._n_zone_actions:
                self._delta_per_zone[z] = (
                    (1 - ALPHA_EMA) * self._delta_per_zone[z] + ALPHA_EMA * delta
                )

        # Action selection
        if self._rng.random() < EPSILON:
            z = int(self._rng.randint(0, self._n_zone_actions))
        else:
            z = _softmax_sel(self._delta_per_zone, SOFTMAX_TEMP, self._rng)

        action = self._zone_actions[z] if z < len(self._zone_actions) else \
            int(self._rng.randint(0, self._n_actions_game))

        self._prev_enc = enc.copy()
        self._prev_zone = z
        return action

    def on_level_transition(self):
        """Clear prev to avoid cross-episode delta corruption."""
        self._prev_enc = None
        self._prev_zone = None


SUBSTRATE_CLASS = CCZoneSubstrate
