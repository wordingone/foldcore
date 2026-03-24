"""
sub1011_unconstrained.py — Unconstrained PRISM Diagnostic (Direction 2, Sub-mode B).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1011 --substrate experiments/sub1011_unconstrained.py

DIRECTION 2: ALL BANS AND CONSTRAINTS SUSPENDED (Jun directive 2026-03-24).
  - Codebook ban (Step 416): LIFTED
  - Graph ban (Step 777): LIFTED
  - Per-game tuning prohibition: LIFTED (auto-detection = adaptive, not hardcoded)
  - One-config rule: LIFTED (same substrate, per-game ADAPTIVE behavior)
  - R1 (reward/loss signals): ALLOWED
  - Budget cap extended: 30 min per game

PURPOSE: Capability ceiling measurement. Expected: chain_score 5/5 (all games L1).

DESIGN:
  Phase detection (1000 random steps per game):
  - n_actions <= 8: directional game (LS20-type) → 4-action argmin coverage
  - n_actions > 8: click game (FT09/VC33-type) → CC zone discovery + argmin
  - No set_game call (CIFAR): classification → argmax of running codebook

  Action selection:
  - Directional: argmin(zone_visits) with epsilon=0.1 (systematic 4-direction coverage)
  - Click: CC zone discovery → argmin(zone_visits) (systematic zone coverage)
  - Domain separation: running_mean resets on set_game() (per-domain centering)

STILL ACTIVE: Reproducibility (same code + seeds = same result). Honest reporting.
EXPECTED: LS20=100%, FT09>0%, VC33>0% (first post-ban FT09/VC33 L1)
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from collections import deque, defaultdict
from substrates.step0674 import _enc_frame

# ─── Hyperparameters ───
ENC_DIM = 256
RUNNING_MEAN_ALPHA = 0.1      # per-domain centering

# CC zone discovery
PHASE1_STEPS = 1000
REDISCOVERY_INTERVAL = 2000
MODE_EMA_ALPHA = 0.1
DEVIATION_PERCENTILE = 95
MIN_ZONE_PIXELS = 2           # lower threshold for finer zone detection
MAX_ZONES = 69                # up to 69 zones (matches FT09's full action space)
MIN_ZONES_FALLBACK = 1        # always use zones if any found

# Action selection
EPS = 0.10                    # epsilon for exploration
SOFTMAX_TEMP = 0.10

CONFIG = {
    "PHASE1_STEPS": PHASE1_STEPS,
    "MAX_ZONES": MAX_ZONES,
    "EPS": EPS,
    "MODE_EMA_ALPHA": MODE_EMA_ALPHA,
    "RUNNING_MEAN_ALPHA": RUNNING_MEAN_ALPHA,
}


# ─── Utilities ───

def _obs_to_gray(obs):
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[1]:
            arr = arr.mean(axis=0)
        else:
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
                    if (0 <= nr < H and 0 <= nc < W
                            and binary_mask[nr, nc] and not visited[nr, nc]):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if count >= MIN_ZONE_PIXELS:
                results.append((count, sum_r // count, sum_c // count))
    results.sort(key=lambda x: -x[0])
    return results


def _centroid_to_action(cy, cx, H, W, n_actions):
    """Map zone centroid to game click action index."""
    if n_actions <= 8:
        return None
    n_dir = 4
    n_click = n_actions - n_dir
    grid = int(round(n_click ** 0.5))
    row = min(int(cy * grid / H), grid - 1)
    col = min(int(cx * grid / W), grid - 1)
    return n_dir + row * grid + col


# ─── Substrate ───

class UnconstrainedSubstrate:
    """
    Unconstrained PRISM diagnostic. All bans lifted.

    Auto-detects game type from n_actions. Uses CC zone discovery for click
    games, directional argmin for navigation games. Global argmin over zone
    visit counts ensures systematic coverage of all discovered actions.

    This is the capability ceiling measurement — if this fails to get L1 on
    FT09/VC33, the games are fundamentally unsolvable under our substrate model.
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_game = 4
        self._step_in_game = 0

        # Per-domain encoding (running mean resets on set_game)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)

        # Zone discovery
        self._pixel_ema = None
        self._frame_h = self._frame_w = None
        self._zone_actions = [0, 1, 2, 3]
        self._n_zone_actions = 4

        # Visit counts per zone (for argmin — systematic coverage)
        self._zone_visits = np.zeros(4, dtype=np.int32)

        self._is_click_game = False

    def set_game(self, n_actions: int):
        """Game switch: reset per-domain state, detect game type."""
        self._n_actions_game = n_actions
        self._step_in_game = 0
        self._is_click_game = (n_actions > 8)

        # Per-domain centering reset
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)

        # Zone reset
        self._pixel_ema = None
        self._frame_h = self._frame_w = None
        self._zone_actions = list(range(n_actions))
        self._n_zone_actions = n_actions
        self._zone_visits = np.zeros(n_actions, dtype=np.int32)

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
        """CC zone discovery — lifted ban, no restrictions."""
        if self._pixel_ema is None or self._frame_h is None:
            return
        if current_gray.shape != self._pixel_ema.shape:
            return

        dev = np.abs(current_gray - self._pixel_ema)
        thresh = np.percentile(dev, DEVIATION_PERCENTILE)
        mask = dev >= thresh
        components = _find_cc(mask)[:MAX_ZONES]

        if not self._is_click_game:
            # Directional game: keep original 4 actions
            return

        valid_actions = []
        for _size, cy, cx in components:
            ga = _centroid_to_action(cy, cx, self._frame_h, self._frame_w,
                                     self._n_actions_game)
            if ga is not None:
                valid_actions.append(ga)

        if len(valid_actions) >= MIN_ZONES_FALLBACK:
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for a in valid_actions:
                if a not in seen:
                    seen.add(a)
                    deduped.append(a)
            self._zone_actions = deduped
            self._n_zone_actions = len(deduped)
            self._zone_visits = np.zeros(self._n_zone_actions, dtype=np.int32)
        # else: keep full n_actions fallback

    def _encode(self, obs):
        """Per-domain centered encoding."""
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        a = RUNNING_MEAN_ALPHA
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return (enc_raw - self._running_mean).astype(np.float32)

    def process(self, obs: np.ndarray) -> int:
        self._step_in_game += 1

        # Phase 1: random + build mode map
        if self._step_in_game <= PHASE1_STEPS:
            current_gray = self._update_mode_map(obs)
            self._encode(obs)   # warm up running mean

            if self._step_in_game == PHASE1_STEPS:
                self._run_zone_discovery(current_gray)

            return int(self._rng.randint(0, self._n_actions_game))

        # Phase 2: zone argmin (systematic coverage)
        current_gray = self._read_gray(obs)
        steps_in_phase2 = self._step_in_game - PHASE1_STEPS
        if steps_in_phase2 % REDISCOVERY_INTERVAL == 0:
            self._run_zone_discovery(current_gray)

        self._encode(obs)   # keep running mean current

        # Action selection: argmin over zone visits with epsilon exploration
        if self._rng.random() < EPS:
            z = int(self._rng.randint(0, self._n_zone_actions))
        else:
            # Argmin = least-visited zone (systematic coverage)
            z = int(np.argmin(self._zone_visits[:self._n_zone_actions]))

        action = self._zone_actions[z] if z < len(self._zone_actions) else \
            int(self._rng.randint(0, self._n_actions_game))

        if z < self._n_zone_actions:
            self._zone_visits[z] += 1

        return action

    def on_level_transition(self):
        """Keep zone visits persistent across episodes — coverage must accumulate."""
        pass  # visits persist: resetting on death caused action-0 bias


SUBSTRATE_CLASS = UnconstrainedSubstrate
