"""
sub1131_defense_v46.py — Saliency-driven click targeting (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1131 --substrate experiments/sub1131_defense_v46.py

FAMILY: Observation-driven click targeting (NEW defense approach)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: v43-v45 probed clicks in SPATIAL GRID patterns (blind to
observation content). But games with clickable elements (buttons, tiles,
controls) render them as VISUALLY DISTINCT regions. The observation ITSELF
tells us where to click.

v46 computes saliency from the initial observation: blocks with values
far from the global mean are likely interactive elements. Click those first.
Zero learned params — saliency is computed from raw observation, not
learned from prediction error (that's prosecution's alpha approach).

Architecture:
- Phase 1 (steps 1-50): keyboard explore (same as v30)
- If keyboard works → reactive on keyboard
- Phase 2 (steps 51-100): compute saliency from current observation.
  Click top-16 most salient blocks (highest absolute deviation from mean).
  2 reps each = 32 steps.
- Phase 2b (steps 101-132): click top-16 LEAST salient blocks (darkest/
  most uniform regions might be clickable backgrounds). 2 reps each.
- Phase 3 (step 133+): reactive switching on responsive click targets.

Key difference from v43: v43 probes ALL blocks uniformly. v46 probes
observation-salient blocks FIRST. If interactive elements are visually
distinct, v46 finds them faster.

KILL: ARC ≤ 0 on click games.
SUCCESS: ARC > 0 — observation saliency predicts interactive elements.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
KB_EXPLORE = 50
TOP_K = 16
PROBE_REPS = 2
MAX_PATIENCE = 20
CHANGE_THRESH = 0.5


def _obs_to_enc(obs):
    """avgpool4: 64x64 → 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


def _block_to_click_action(block_idx):
    """Block center → click action index."""
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class SaliencyClickSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._has_clicks = False
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None

        # Phase management
        self._phase = "keyboard"
        self._kb_total_change = 0.0
        self._kb_responsive = []

        # Saliency click probing
        self._salient_blocks = []     # sorted by saliency (highest first)
        self._anti_salient_blocks = []  # sorted by anti-saliency (lowest first)
        self._probe_list = []  # blocks to probe
        self._probe_idx = 0
        self._probe_step = 0
        self._probe_pre_enc = None
        self._responsive_clicks = []

        # Reactive switching
        self._active_actions = []
        self._current_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def _compute_saliency(self, enc):
        """Compute per-block saliency: absolute deviation from global mean."""
        mean_val = enc.mean()
        saliency = np.abs(enc - mean_val)
        # Most salient blocks (highest deviation = likely interactive elements)
        sorted_desc = np.argsort(saliency)[::-1]
        self._salient_blocks = list(sorted_desc[:TOP_K].astype(int))
        # Least salient blocks (lowest deviation = maybe clickable backgrounds)
        self._anti_salient_blocks = list(sorted_desc[-TOP_K:].astype(int))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_dist = 0.0
            self._current_idx = self._rng.randint(min(self._n_actions_env, N_KB))
            return self._current_idx

        dist = self._dist_to_initial(enc)

        # === Phase 1: Keyboard explore ===
        if self._phase == "keyboard":
            if self._prev_enc is not None:
                change = float(np.sum(np.abs(enc - self._prev_enc)))
                self._kb_total_change += change
                prev_a = (self.step_count - 1) % min(self._n_actions_env, N_KB)
                if change > CHANGE_THRESH and prev_a not in self._kb_responsive:
                    self._kb_responsive.append(prev_a)

            if self.step_count <= KB_EXPLORE:
                action = self.step_count % min(self._n_actions_env, N_KB)
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return action

            # Keyboard done
            if len(self._kb_responsive) >= 2:
                self._phase = "reactive"
                self._active_actions = self._kb_responsive[:]
                self._current_idx = 0
            elif self._has_clicks:
                # Compute saliency from current observation
                self._compute_saliency(enc)
                # Probe salient blocks first, then anti-salient
                self._probe_list = self._salient_blocks + self._anti_salient_blocks
                self._phase = "saliency_probe"
                self._probe_idx = 0
                self._probe_step = 0
                self._probe_pre_enc = enc.copy()
            else:
                self._phase = "reactive"
                self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                self._current_idx = 0

        # === Phase 2: Saliency-guided click probing ===
        if self._phase == "saliency_probe":
            if self._probe_step > 0 and self._probe_pre_enc is not None:
                if self._probe_step == PROBE_REPS:
                    probe_change = float(np.sum(np.abs(enc - self._probe_pre_enc)))
                    if probe_change > CHANGE_THRESH:
                        block = self._probe_list[self._probe_idx]
                        self._responsive_clicks.append(block)
                    self._probe_idx += 1
                    self._probe_step = 0
                    self._probe_pre_enc = enc.copy()

            if self._probe_idx >= len(self._probe_list):
                # Probing done — build action set
                self._active_actions = self._kb_responsive[:]
                for block in self._responsive_clicks:
                    self._active_actions.append(_block_to_click_action(block))
                if not self._active_actions:
                    self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                self._phase = "reactive"
                self._current_idx = 0
            else:
                self._probe_step += 1
                block = self._probe_list[self._probe_idx]
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return _block_to_click_action(block)

        # === Phase 3: Reactive switching ===
        if self._phase == "reactive":
            n_active = len(self._active_actions)
            if n_active == 0:
                self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                n_active = len(self._active_actions)

            progress = (self._prev_dist - dist) > 1e-4 if self._prev_dist is not None else False
            no_change = abs(self._prev_dist - dist) < 1e-6 if self._prev_dist is not None else True

            self._steps_on_action += 1
            if progress:
                self._consecutive_progress += 1
                self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
                self._actions_tried_this_round = 0
            else:
                self._consecutive_progress = 0
                if self._steps_on_action >= self._patience or no_change:
                    self._actions_tried_this_round += 1
                    self._steps_on_action = 0
                    self._patience = 3
                    if self._actions_tried_this_round >= n_active:
                        self._current_idx = self._rng.randint(n_active)
                        self._actions_tried_this_round = 0
                    else:
                        self._current_idx = (self._current_idx + 1) % n_active

            action = self._active_actions[self._current_idx % n_active]
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # Fallback
        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._current_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Keep phase and active_actions


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "top_k": TOP_K,
    "probe_reps": PROBE_REPS,
    "max_patience": MAX_PATIENCE,
    "change_thresh": CHANGE_THRESH,
    "family": "observation-driven click targeting",
    "tag": "defense v46 (ℓ₁ saliency-driven: click visually distinct blocks first)",
}

SUBSTRATE_CLASS = SaliencyClickSubstrate
