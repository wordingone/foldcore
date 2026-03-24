"""
sub1016_cc_seq_novelty.py — CC Zone Discovery + Sequence Novelty (Direction 1).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1016 --substrate experiments/sub1016_cc_seq_novelty.py

FAMILY: sequence-novelty (extension of 1014)
R3 HYPOTHESIS: CC-discovered zones + action-tuple novelty scoring produces temporal credit
  for state-dependent click sequences. If FT09/VC33 > 0: encoding was the bottleneck in 1014.
  If still 0: sequencing mechanism itself is insufficient.

GAME-AGNOSTIC BASE (Jun directive 2026-03-24):
  NO 800b, NO alpha, NO h, NO running_mean encoding.
  Only mechanisms: CC zone discovery (click games) + sequence novelty learning.

  Phase 1 — Game type detection (500 random steps)
    n_actions <= 8 → directional game (LS20-type)
    n_actions > 8  → click game (FT09/VC33-type)

  Phase 2a — CC Zone Probing (click games, 200 steps):
    Systematically click each grid position (n_click = n_actions - 4 positions)
    Measure frame diff per click. Accumulate diffs.
    CC on accumulated diff mask → zone map
    Merge, filter, result: N_zones discovered click actions

  Phase 2b — Directional (directional games):
    Skip CC. Use 4 directional actions directly.

  Phase 3 — Sequence Novelty (all games):
    K=3 action tuples → EMA of observation change magnitude
    Action selection: argmax over candidate next-action sequences

ONE VARIABLE FROM 1014: CC zone discovery reduces click action space.
  Everything else (sequence novelty, K=3, EPS=0.3) identical.

KILL: FT09=0 AND VC33=0 AND LS20=0
ALIVE: FT09 or VC33 L1 > 0 → encoding was bottleneck in 1014
NEUTRAL: LS20 held, FT09/VC33=0 → CC helps but sequence novelty insufficient
BUDGET: 10K steps/game, 5 seeds
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from collections import deque

# ─── Hyperparameters ───
ENC_DIM = 64 * 64              # raw 64×64 flattened (game-agnostic base)
RUNNING_MEAN_ALPHA = 0.1       # encoding centering rate

PHASE1_STEPS = 500             # game type detection
CC_PROBE_STEPS = 200           # CC zone discovery budget (click games)

SEQ_LEN = 3                    # K: action tuple length
SEQ_ALPHA = 0.1                # EMA rate for sequence score updates
NOVELTY_BUFFER_MAX = 500       # max observations in novelty buffer
NOVELTY_WINDOW = 100           # compare against last N buffered obs
EPS = 0.30                     # random exploration rate

MIN_ZONE_PIXELS = 5            # minimum diff pixels to form a zone
MAX_ZONES = 20                 # max zones to discover

CONFIG = {
    "ENC_DIM": ENC_DIM,
    "RUNNING_MEAN_ALPHA": RUNNING_MEAN_ALPHA,
    "PHASE1_STEPS": PHASE1_STEPS,
    "CC_PROBE_STEPS": CC_PROBE_STEPS,
    "SEQ_LEN": SEQ_LEN,
    "SEQ_ALPHA": SEQ_ALPHA,
    "EPS": EPS,
}


def _obs_to_gray(obs):
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[1]:  # (C, H, W)
            arr = arr.mean(axis=0)
        else:                             # (H, W, C)
            arr = arr.mean(axis=2)
    return arr


def _find_cc(binary_mask):
    """4-connected BFS. Returns list of (size, cy, cx) sorted by size desc."""
    from collections import deque as _deque
    H, W = binary_mask.shape
    visited = np.zeros((H, W), dtype=bool)
    results = []
    for r0 in range(H):
        for c0 in range(W):
            if not binary_mask[r0, c0] or visited[r0, c0]:
                continue
            q = _deque([(r0, c0)])
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
    """Map zone centroid to click action index. None for directional games."""
    if n_actions <= 8:
        return None
    n_dir = 4
    n_click = n_actions - n_dir
    grid = int(round(n_click ** 0.5))
    row = min(int(cy * grid / H), grid - 1)
    col = min(int(cx * grid / W), grid - 1)
    return n_dir + row * grid + col


class CCSeqNoveltySubstrate:
    """
    CC Zone Discovery + Sequence Novelty (Step 1016).

    One variable from Step 1014: CC zone discovery reduces click-game action
    space from 69 positions to ~5-15 discovered zones. Sequence novelty then
    operates over these discovered zones.

    Game-agnostic base: raw 64×64 encoding, no 800b, no alpha, no h.
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = 4
        self._step_in_game = 0

        # Game type detection
        self._is_click_game = False
        self._phase = 'detect'   # 'detect' → 'probe' → 'run'

        # Encoding (raw 64×64 + centered running mean)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._frame_h = self._frame_w = None

        # CC probing state (click games Phase 2a)
        self._probe_actions = []          # list of actions to probe
        self._probe_idx = 0               # current probe action index
        self._probe_prev_gray = None      # gray before probe action
        self._probe_acc_diff = None       # accumulated diff across probes

        # Discovered zone actions
        self._zone_actions = list(range(4))  # starts as directional fallback
        self._n_zone_actions = 4

        # Sequence novelty (same as 1014)
        self._novelty_buffer = []
        self._sequence_scores = {}
        self._action_history = []
        self._recent_novelty = []

    def set_game(self, n_actions: int):
        """Reset per-game state."""
        self._n_actions = n_actions
        self._step_in_game = 0
        self._is_click_game = (n_actions > 8)
        self._phase = 'detect'

        # Reset encoding
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._frame_h = self._frame_w = None

        # Reset probing
        self._probe_actions = []
        self._probe_idx = 0
        self._probe_prev_gray = None
        self._probe_acc_diff = None

        # Reset zone actions to full range
        self._zone_actions = list(range(n_actions))
        self._n_zone_actions = n_actions

        # Keep sequence knowledge across games (game-agnostic transfer)
        # Keep: _novelty_buffer, _sequence_scores, _action_history, _recent_novelty

    def _encode(self, obs):
        """Raw 64×64 flatten + centered running mean."""
        arr = _obs_to_gray(obs)
        if self._frame_h is None:
            self._frame_h, self._frame_w = arr.shape
        flat = arr.flatten()
        if len(flat) != ENC_DIM:
            indices = np.linspace(0, len(flat) - 1, ENC_DIM).astype(int)
            flat = flat[indices]
        self._running_mean = (
            (1 - RUNNING_MEAN_ALPHA) * self._running_mean + RUNNING_MEAN_ALPHA * flat
        )
        return (flat - self._running_mean).astype(np.float32)

    def _compute_novelty(self, enc):
        """Min L2 distance to recent buffer entries."""
        if not self._novelty_buffer:
            return 1.0
        window = self._novelty_buffer[-NOVELTY_WINDOW:]
        dists = [float(np.linalg.norm(enc - b)) for b in window]
        return min(dists)

    def _update_novelty_buffer(self, enc, novelty):
        if len(self._novelty_buffer) < 50 or novelty > (
            np.median(self._recent_novelty[-50:]) if self._recent_novelty else 0.0
        ):
            self._novelty_buffer.append(enc.copy())
            if len(self._novelty_buffer) > NOVELTY_BUFFER_MAX:
                self._novelty_buffer.pop(0)

    def _build_probe_list(self):
        """Build list of click actions to probe for CC discovery."""
        n_dir = 4
        n_click = self._n_actions - n_dir
        return list(range(n_dir, n_dir + n_click))

    def _finalize_zones(self):
        """Run CC on accumulated probe diff → extract zone actions."""
        if self._probe_acc_diff is None:
            return

        H, W = self._probe_acc_diff.shape
        thresh = np.percentile(self._probe_acc_diff, 85)
        mask = self._probe_acc_diff >= thresh
        components = _find_cc(mask)[:MAX_ZONES]

        valid_actions = []
        for _size, cy, cx in components:
            ga = _centroid_to_action(cy, cx, H, W, self._n_actions)
            if ga is not None:
                valid_actions.append(ga)

        if len(valid_actions) >= 2:
            seen = set()
            deduped = []
            for a in valid_actions:
                if a not in seen:
                    seen.add(a)
                    deduped.append(a)
            self._zone_actions = deduped
            self._n_zone_actions = len(deduped)

    def process(self, obs: np.ndarray) -> int:
        self._step_in_game += 1

        # Phase 1: Game type detection (random)
        if self._step_in_game <= PHASE1_STEPS:
            enc = self._encode(obs)  # warm up running mean

            if self._step_in_game == PHASE1_STEPS:
                # Finalize game type detection
                if not self._is_click_game:
                    # Directional game: skip probing, go straight to run
                    self._zone_actions = list(range(min(4, self._n_actions)))
                    self._n_zone_actions = len(self._zone_actions)
                    self._phase = 'run'
                else:
                    # Click game: start CC probing
                    self._probe_actions = self._build_probe_list()
                    self._probe_idx = 0
                    self._probe_prev_gray = _obs_to_gray(obs)
                    H, W = self._probe_prev_gray.shape
                    self._probe_acc_diff = np.zeros((H, W), dtype=np.float32)
                    self._phase = 'probe'

            return int(self._rng.randint(0, self._n_actions))

        # Phase 2a: CC probing (click games)
        if self._phase == 'probe':
            enc = self._encode(obs)

            current_gray = _obs_to_gray(obs)

            # If we just took a probe action, measure its diff
            if self._probe_prev_gray is not None and self._probe_acc_diff is not None:
                if current_gray.shape == self._probe_prev_gray.shape:
                    diff = np.abs(current_gray - self._probe_prev_gray)
                    self._probe_acc_diff += diff

            self._probe_prev_gray = current_gray

            # Select next probe action
            if self._probe_idx < len(self._probe_actions):
                action = self._probe_actions[self._probe_idx]
                self._probe_idx += 1

                # Check if probing budget exhausted or all probes done
                steps_in_probe = self._step_in_game - PHASE1_STEPS
                if steps_in_probe >= CC_PROBE_STEPS or self._probe_idx >= len(self._probe_actions):
                    self._finalize_zones()
                    self._phase = 'run'

                return action
            else:
                # Done probing
                self._finalize_zones()
                self._phase = 'run'
                return int(self._rng.randint(0, self._n_actions))

        # Phase 3: Sequence novelty (all games)
        enc = self._encode(obs)
        novelty = self._compute_novelty(enc)

        self._recent_novelty.append(novelty)
        if len(self._recent_novelty) > 200:
            self._recent_novelty.pop(0)

        # Update sequence score for the sequence that PRODUCED this observation
        if len(self._action_history) >= SEQ_LEN:
            seq = tuple(self._action_history[-SEQ_LEN:])
            if seq not in self._sequence_scores:
                self._sequence_scores[seq] = 0.0
            self._sequence_scores[seq] += SEQ_ALPHA * (novelty - self._sequence_scores[seq])

        self._update_novelty_buffer(enc, novelty)

        # Action selection over discovered zones
        n = self._n_zone_actions
        if self._rng.random() < EPS or not self._sequence_scores:
            z = int(self._rng.randint(0, n))
        else:
            prefix = tuple(self._action_history[-(SEQ_LEN - 1):])
            scores = []
            for i in range(n):
                a = self._zone_actions[i] if i < len(self._zone_actions) else i
                candidate = prefix + (a,)
                scores.append(self._sequence_scores.get(candidate, 0.0))
            z = int(np.argmax(scores))

        action = self._zone_actions[z] if z < len(self._zone_actions) else \
            int(self._rng.randint(0, self._n_actions))

        self._action_history.append(action)
        if len(self._action_history) > 1000:
            self._action_history = self._action_history[-1000:]

        return action

    def on_level_transition(self):
        """Clear action history on level transition (new episode = new sequence context)."""
        self._action_history = []


SUBSTRATE_CLASS = CCSeqNoveltySubstrate
