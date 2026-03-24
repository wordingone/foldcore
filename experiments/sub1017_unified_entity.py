"""
sub1017_unified_entity.py — Unconstrained PRISM Unified Entity (Direction 2, Sub-mode B).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1017 --substrate experiments/sub1017_unified_entity.py --steps 10000 --seeds 5

DIRECTION 2: ALL BANS AND CONSTRAINTS SUSPENDED (Jun directive 2026-03-24).
  - Codebook ban (Step 416): LIFTED
  - Graph ban (Step 777): LIFTED
  - Per-game tuning: LIFTED
  - One-config rule: LIFTED (auto-detection = adaptive, not hardcoded)
  - R1 (reward/loss signals): N/A
  - Budget cap extended: 30 min total

PURPOSE: Unified capability ceiling. One substrate, all games, randomized PRISM order.
  Can ONE entity solve ALL game types through PRISM?

DESIGN:
  Game type detection (first 500 steps):
    n_actions <= 8 OR n_actions == 5 (CIFAR) → non-click game
    n_actions > 8 → click game (FT09/VC33-type)

  Non-click games (LS20, CIFAR):
    Full 674 graph directly (avgpool16 + LSH + per-state argmin)
    No zone discovery needed — action space is already small

  Click games (FT09/VC33, n_actions > 8):
    Phase 2a: CC zone discovery (200 steps)
      - Systematically click each grid position
      - Measure frame diff per click → accumulated diff map
      - CC on diff mask → N_zones discovered
    Phase 3: 674 graph on discovered zones
      - TransitionTriggered674 with n_actions = N_zones
      - zone_idx from 674 → remapped to zone_actions[zone_idx]

KEY FIX FROM 1011: Per-state argmin via 674 graph. NOT global visit count argmin.
  At current encoded observation node → least-visited outgoing edge at THAT node.
  This is navigation, not round-robin.

STILL ACTIVE: Reproducibility (same code + seeds = same result).
EXPECTED: CIFAR=100%, LS20=100%, FT09=L1, VC33=L1 (chain 5/5).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from collections import deque
from substrates.step0674 import TransitionTriggered674, _enc_frame

# ─── Hyperparameters ───
PHASE1_STEPS = 500          # game type detection
CC_PROBE_STEPS = 200        # CC zone discovery budget (click games)
MIN_ZONE_PIXELS = 5         # minimum diff pixels to form a zone
MAX_ZONES = 20              # max zones to discover
MIN_ZONES_FALLBACK = 2      # min zones required; else use full action set


CONFIG = {
    "PHASE1_STEPS": PHASE1_STEPS,
    "CC_PROBE_STEPS": CC_PROBE_STEPS,
    "MIN_ZONE_PIXELS": MIN_ZONE_PIXELS,
    "MAX_ZONES": MAX_ZONES,
}


# ─── Utilities ───

def _obs_to_gray(obs):
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
                    if (0 <= nr < H and 0 <= nc < W
                            and binary_mask[nr, nc] and not visited[nr, nc]):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if count >= MIN_ZONE_PIXELS:
                results.append((count, sum_r // count, sum_c // count))
    results.sort(key=lambda x: -x[0])
    return results


def _centroid_to_action(cy, cx, H, W, n_actions):
    """Map zone centroid to click action index. None for non-click games."""
    if n_actions <= 8:
        return None
    n_dir = 4
    n_click = n_actions - n_dir
    grid = int(round(n_click ** 0.5))
    row = min(int(cy * grid / H), grid - 1)
    col = min(int(cx * grid / W), grid - 1)
    return n_dir + row * grid + col


# ─── Substrate ───

class UnifiedEntitySubstrate:
    """
    Unified Unconstrained Substrate (Step 1017).

    One entity, all games. Per-state 674 graph for navigation.
    CC zone discovery for click game action space reduction.
    All bans lifted (Direction 2).

    CIFAR/LS20 → 674 graph directly (small action spaces, no probing needed)
    FT09/VC33  → CC probing → 674 graph over discovered zones
    """

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._n_actions = 4
        self._step = 0

        # Game type
        self._is_click = False
        self._phase = 'detect'

        # Zone discovery (click games)
        self._zone_actions = list(range(4))
        self._probe_list = []
        self._probe_idx = 0
        self._probe_prev_gray = None
        self._probe_acc_diff = None
        self._frame_h = self._frame_w = None

        # 674 graph (initialized after phase 1/2)
        self._graph = None

    def set_game(self, n_actions: int):
        """Game switch: reset per-game state, detect type."""
        self._n_actions = n_actions
        self._step = 0
        self._is_click = (n_actions > 8)
        self._phase = 'detect'

        # Reset zone discovery
        self._zone_actions = list(range(n_actions))
        self._probe_list = []
        self._probe_idx = 0
        self._probe_prev_gray = None
        self._probe_acc_diff = None
        self._frame_h = self._frame_w = None

        # Reset graph (will be re-created with correct n_actions after zone discovery)
        self._graph = None

    def _init_graph(self, n_actions: int):
        """Initialize 674 graph with given action count."""
        self._graph = TransitionTriggered674(n_actions=n_actions, seed=self._seed)

    def _finalize_zones(self):
        """Run CC on accumulated probe diff → extract zone actions."""
        if self._probe_acc_diff is None or self._frame_h is None:
            return

        thresh = np.percentile(self._probe_acc_diff, 85)
        mask = self._probe_acc_diff >= thresh
        components = _find_cc(mask)[:MAX_ZONES]

        valid_actions = []
        for _size, cy, cx in components:
            ga = _centroid_to_action(cy, cx, self._frame_h, self._frame_w,
                                     self._n_actions)
            if ga is not None:
                valid_actions.append(ga)

        if len(valid_actions) >= MIN_ZONES_FALLBACK:
            seen = set()
            deduped = []
            for a in valid_actions:
                if a not in seen:
                    seen.add(a)
                    deduped.append(a)
            self._zone_actions = deduped
        # else: keep full n_actions fallback

    def _launch_run_phase(self):
        """After zone discovery: create graph and enter run phase."""
        self._init_graph(n_actions=len(self._zone_actions))
        self._phase = 'run'

    def process(self, obs: np.ndarray) -> int:
        self._step += 1

        # ── Phase 1: Game type detection (random actions) ──
        if self._step <= PHASE1_STEPS:
            # Track frame geometry for CC discovery
            gray = _obs_to_gray(obs)
            if self._frame_h is None:
                self._frame_h, self._frame_w = gray.shape

            if self._step == PHASE1_STEPS:
                if not self._is_click:
                    # Non-click: directional game or CIFAR
                    # Use 674 directly with full n_actions
                    self._init_graph(n_actions=self._n_actions)
                    self._phase = 'run'
                else:
                    # Click game: start CC probing
                    self._probe_list = list(range(4, self._n_actions))
                    self._probe_prev_gray = gray
                    self._probe_acc_diff = np.zeros_like(gray)
                    self._phase = 'probe'

            return int(self._rng.randint(0, self._n_actions))

        # ── Phase 2a: CC probing (click games) ──
        if self._phase == 'probe':
            gray = _obs_to_gray(obs)

            # Measure diff from last action
            if (self._probe_prev_gray is not None
                    and self._probe_acc_diff is not None
                    and gray.shape == self._probe_prev_gray.shape):
                self._probe_acc_diff += np.abs(gray - self._probe_prev_gray)
            self._probe_prev_gray = gray

            steps_in_probe = self._step - PHASE1_STEPS
            if self._probe_idx < len(self._probe_list):
                action = self._probe_list[self._probe_idx]
                self._probe_idx += 1
                # Finalize if budget exhausted or all positions probed
                if (steps_in_probe >= CC_PROBE_STEPS
                        or self._probe_idx >= len(self._probe_list)):
                    self._finalize_zones()
                    self._launch_run_phase()
                return action
            else:
                # Exhausted probe list
                self._finalize_zones()
                self._launch_run_phase()
                # Fall through to run phase this step

        # ── Phase 3: 674 per-state graph argmin ──
        if self._graph is not None:
            zone_idx = self._graph.process(obs)
            # Remap zone_idx → actual game action
            if zone_idx < len(self._zone_actions):
                return int(self._zone_actions[zone_idx])
            else:
                return int(self._rng.randint(0, self._n_actions))

        return int(self._rng.randint(0, self._n_actions))

    def on_level_transition(self):
        """Pass to graph on level transition (graph keeps edges, resets episode state)."""
        if self._graph is not None:
            self._graph.on_level_transition()


SUBSTRATE_CLASS = UnifiedEntitySubstrate
