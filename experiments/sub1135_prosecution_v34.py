"""
sub1135_prosecution_v34.py — Random feature nonlinear forward model (prosecution: ℓ_π)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1135 --substrate experiments/sub1135_prosecution_v34.py

FAMILY: Nonlinear forward model (NEW prosecution family from literature)
Tagged: prosecution (ℓ_π)
R3 HYPOTHESIS: Random feature expansion (cos(W_random @ enc + bias)) captures
nonlinear action-effect relationships that linear W_fwd models miss. Per-action
forward models in nonlinear feature space enable game UNDERSTANDING, not just
change detection. Random features approximate kernel methods (Rahimi & Recht 2007).

Architecture:
- enc = avgpool4 (256D)
- phi(enc) = cos(W_random @ enc + bias) — FIXED random projection (256D → 256D)
  W_random is FROZEN (R1 compliant). Nonlinear feature map.
- Per-action forward model: W_fwd[a] (256 × 256) for each keyboard action.
  Updated: W_fwd[a] += lr * outer(phi_next - W_fwd[a] @ phi_curr, phi_curr) / (phi_curr^2 + eps)
- Action selection: pick action where ||predicted_phi_next - phi_current|| is largest
  (most predicted change = most informative action)

Phase 1 (0-100): random keyboard warmup — collect per-action data
Phase 2 (100+): forward-model-guided action selection

KILL: ARC ≤ v24 (0.0045 on lucky draw).
SUCCESS: ARC > 0 on previously 0% game type.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
WARMUP_STEPS = 100
FWD_LR = 0.01
MAX_PATIENCE = 20


def _obs_to_enc(obs):
    """avgpool4 + center: 64x64 → 16x16 = 256D, zero-centered."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    enc -= enc.mean()
    return enc


class RandomFeatureForwardSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._game_number = 0

        # FROZEN random projection (R1 compliant — never updated)
        self._W_random = self._rng.randn(N_DIMS, N_DIMS).astype(np.float32) * 0.5
        self._bias = self._rng.uniform(0, 2 * np.pi, N_DIMS).astype(np.float32)

        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_phi = None
        self._prev_action = None
        self._prev_dist = None

        # Per-action forward models (7 × 256 × 256)
        n_kb = min(self._n_actions_env, N_KB)
        self._W_fwd = [np.eye(N_DIMS, dtype=np.float32) * 0.99 for _ in range(n_kb)]

        # Reactive switching fallback
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def _phi(self, enc):
        """Random feature map: phi(enc) = cos(W_random @ enc + bias)."""
        return np.cos(self._W_random @ enc + self._bias)

    def _update_forward_model(self, action, phi_curr, phi_next):
        """Update W_fwd[action] with outer product learning rule."""
        if action >= len(self._W_fwd):
            return
        pred = self._W_fwd[action] @ phi_curr
        error = phi_next - pred
        norm_sq = np.dot(phi_curr, phi_curr) + 1e-8
        self._W_fwd[action] += FWD_LR * np.outer(error, phi_curr) / norm_sq
        self.r3_updates += 1
        self.att_updates_total += 1

    def _predict_change(self, phi_curr):
        """For each action, predict how much phi would change."""
        n_kb = min(self._n_actions_env, N_KB)
        changes = np.zeros(n_kb, dtype=np.float32)
        for a in range(n_kb):
            pred_next = self._W_fwd[a] @ phi_curr
            changes[a] = float(np.sum(np.abs(pred_next - phi_curr)))
        return changes

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)
        phi = self._phi(enc)
        n_kb = min(self._n_actions_env, N_KB)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_phi = phi.copy()
            self._prev_dist = 0.0
            self._prev_action = self._rng.randint(n_kb)
            return self._prev_action

        dist = self._dist_to_initial(enc)

        # Update forward model with observed transition
        if self._prev_phi is not None and self._prev_action is not None:
            self._update_forward_model(self._prev_action, self._prev_phi, phi)

        # === Phase 1: Warmup — random keyboard actions ===
        if self.step_count <= WARMUP_STEPS:
            action = self.step_count % n_kb
            self._prev_enc = enc.copy()
            self._prev_phi = phi.copy()
            self._prev_action = action
            self._prev_dist = dist
            return action

        # === Phase 2: Forward-model-guided action selection ===
        # Predict which action would cause the most change
        predicted_changes = self._predict_change(phi)

        # Use predicted change to BIAS action selection, but keep reactive switching
        # for exploitation when progress is detected
        progress = (self._prev_dist - dist) > 1e-4 if self._prev_dist is not None else False
        no_change = abs(self._prev_dist - dist) < 1e-6 if self._prev_dist is not None else True

        self._steps_on_action += 1

        if progress:
            # Current action is working — keep it (reactive exploitation)
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
        else:
            self._consecutive_progress = 0
            if self._steps_on_action >= self._patience or no_change:
                self._steps_on_action = 0
                self._patience = 3
                self._actions_tried_this_round += 1

                if self._actions_tried_this_round >= n_kb:
                    # All actions tried — pick predicted best
                    self._current_action = int(np.argmax(predicted_changes))
                    self._actions_tried_this_round = 0
                else:
                    # Try next action by predicted change (descending)
                    sorted_actions = np.argsort(predicted_changes)[::-1]
                    idx = self._actions_tried_this_round % n_kb
                    self._current_action = int(sorted_actions[idx])

        self._prev_enc = enc.copy()
        self._prev_phi = phi.copy()
        self._prev_action = self._current_action
        self._prev_dist = dist
        return self._current_action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_phi = None
        self._prev_dist = None
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Keep W_fwd and W_random across levels (ℓ_π)


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "warmup_steps": WARMUP_STEPS,
    "fwd_lr": FWD_LR,
    "max_patience": MAX_PATIENCE,
    "family": "nonlinear forward model",
    "tag": "prosecution v34 (ℓ_π random feature forward model: cos(W@enc+b) → per-action W_fwd)",
}

SUBSTRATE_CLASS = RandomFeatureForwardSubstrate
