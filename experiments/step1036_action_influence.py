"""
Step 1036 — Action-Influence Substrate (Debate v3: Prosecution)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1036 --substrate experiments/step1036_action_influence.py

FAMILY: action-influence (NEW — first post-D2 atomic substrate)
R3 HYPOTHESIS: Per-pixel running statistics can discover WHERE (interactive zones),
  WHAT STATE is normal (target), and WHICH actions affect which regions — all from
  interaction alone, with one config, for all games.
  If action-conditional change tracking works, then L1 on 2+ games from a SINGLE
  process() function because the substrate discovers both click targets (FT09/VC33)
  and navigation strategy (LS20) from the same per-pixel statistics.
  Falsified if: L1 = 0 on all games after 10K steps (influence maps too noisy to
  produce useful action selection).

KILL: L1 < 1 on all games after 10K steps
SUCCESS: L1 >= 1 on 2+ games (matches scripted pipeline, proves atomic substrate viable)
BUDGET: 10K steps/game, 5 seeds, all PRISM phases

DEBATE CONTEXT: Prosecution (Leo) claims D2 discovery mechanism IS encodable as
atomic substrate. This file IS the substrate. If it works → prosecution wins
(D2-grounded framework produces a working substrate with RHAE data).

HARNESS REQUIREMENT: This substrate uses ACTION6 (click) with coordinates.
The harness must support: when process() returns 5 (ACTION6), read self._click_xy
and send env.step(ACTION6, data={"x": x, "y": y}).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

# ─── Hyperparameters ───
# ONE config for all games. No per-game branches.
ALPHA_CHANGE = 0.99       # change_map decay (per-pixel change frequency)
ALPHA_TARGET = 0.999      # target_map decay (per-pixel "normal" state)
ALPHA_INFLUENCE = 0.1     # influence map learning rate
WARMUP_STEPS = 200        # random exploration before using scores
EPSILON_START = 1.0       # initial exploration rate
EPSILON_END = 0.05        # final exploration rate
EPSILON_DECAY = 2000      # steps to decay epsilon
CLICK_THRESHOLD = 0.01    # min mismatch to trigger click action
SMOOTH_KERNEL = 5         # spatial smoothing kernel size


def _spatial_smooth(arr, k=SMOOTH_KERNEL):
    """Fast spatial averaging via cumsum (box filter)."""
    if k <= 1:
        return arr
    pad = k // 2
    padded = np.pad(arr, pad, mode='edge')
    cs = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    h, w = arr.shape
    out = cs[k:h+k, k:w+k] - cs[:h, k:w+k] - cs[k:h+k, :w] + cs[:h, :w]
    return out / (k * k)


class ActionInfluenceSubstrate:
    """
    Atomic substrate: discovers WHERE, WHAT, and WHICH from running statistics.

    State:
      change_map (64x64): per-pixel running average of frame differences
      target_map (64x64x16): per-pixel running mean of color values (normal state)
      influence (7 x 64x64): per-action influence map (which pixels each action affects)
      prev_obs (64x64): previous frame
      prev_action (int): last action taken

    Action selection:
      For each action a: score[a] = sum(influence[a] * change_map * mismatch)
      If best action is ACTION6 (click): _click_xy = argmax(change_map * mismatch)
      Else: return action index

    No phases. No game detection. No hardcoded mappings.
    Game type emerges from which actions build influence.

    Interface contract (ChainRunner-compatible):
      process(obs: np.ndarray) -> int
      on_level_transition()
      set_game(n_actions: int)
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = 7
        self._step = 0
        self._init_maps()

    def _init_maps(self):
        """Initialize all per-pixel maps."""
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.target_map = None  # initialized on first obs
        self.influence = np.zeros((self._n_actions, 64, 64), dtype=np.float32)
        self.prev_obs = None
        self.prev_action = None
        self._click_xy = (32, 32)  # default click position (center)
        self._step = 0

    def set_game(self, n_actions: int):
        """Called on game switch. Reset per-game state."""
        self._n_actions = min(n_actions, 7)  # cap at 7 (ACTION1-7)
        self._init_maps()

    def process(self, obs: np.ndarray) -> int:
        """Process one observation, return action index."""
        obs = np.asarray(obs, dtype=np.float32)

        # Handle multi-channel obs: arc_agi returns frame[0] as 64x64 grid, values 0-15
        if obs.ndim == 3:
            obs = obs[0] if obs.shape[0] < obs.shape[-1] else obs[:, :, 0]
        if obs.ndim != 2:
            obs = obs.reshape(64, 64) if obs.size == 4096 else obs.ravel()[:4096].reshape(64, 64)

        # Initialize target on first observation
        if self.target_map is None:
            self.target_map = obs.copy()

        # ─── Update statistics ───
        if self.prev_obs is not None:
            diff = np.abs(obs - self.prev_obs)

            # change_map: per-pixel running change frequency
            self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff

            # influence: which pixels did the last action affect?
            if self.prev_action is not None and self.prev_action < self._n_actions:
                self.influence[self.prev_action] = (
                    (1 - ALPHA_INFLUENCE) * self.influence[self.prev_action]
                    + ALPHA_INFLUENCE * diff
                )

        # target_map: per-pixel running mean (what's "normal")
        self.target_map = ALPHA_TARGET * self.target_map + (1 - ALPHA_TARGET) * obs

        # ─── Compute mismatch ───
        mismatch = np.abs(obs - self.target_map)

        # Combined signal: interactive AND currently different from normal
        signal = self.change_map * mismatch

        # Smooth spatially
        smooth_signal = _spatial_smooth(signal)

        self.prev_obs = obs.copy()
        self._step += 1

        # ─── Epsilon exploration ───
        epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * self._step / EPSILON_DECAY)

        if self._step < WARMUP_STEPS or self._rng.random() < epsilon:
            # Random action — for click actions, use random position
            action = self._rng.randint(0, self._n_actions)
            if action == 5:  # ACTION6 = click
                self._click_xy = (self._rng.randint(0, 64), self._rng.randint(0, 64))
            self.prev_action = action
            return action

        # ─── Score each action ───
        scores = np.zeros(self._n_actions, dtype=np.float32)
        for a in range(self._n_actions):
            # score = how much does this action affect wrong interactive regions?
            scores[a] = np.sum(self.influence[a] * smooth_signal)

        # Check if click action (ACTION6, index 5) is best
        best_action = int(np.argmax(scores))

        # For click action: set click target to argmax of smoothed signal
        if best_action == 5 or (scores.max() < CLICK_THRESHOLD and self._n_actions > 5):
            # Use spatial signal to find click target
            flat_idx = np.argmax(smooth_signal)
            cy, cx = np.unravel_index(flat_idx, (64, 64))
            self._click_xy = (int(cx), int(cy))
            best_action = 5  # force click

        self.prev_action = best_action
        return best_action

    def on_level_transition(self):
        """Called on level completion. Reset target to allow fast adaptation."""
        # Fast reset: target_map partially reset to allow quick adaptation to new level
        # Don't fully reset — some cross-level knowledge may help
        if self.target_map is not None and self.prev_obs is not None:
            self.target_map = 0.5 * self.target_map + 0.5 * self.prev_obs
        # Don't reset change_map or influence — these are learned knowledge


# Exposed as CONFIG dict — picked up by run_experiment.py for save_results()
CONFIG = {
    "alpha_change": ALPHA_CHANGE,
    "alpha_target": ALPHA_TARGET,
    "alpha_influence": ALPHA_INFLUENCE,
    "warmup_steps": WARMUP_STEPS,
    "epsilon_start": EPSILON_START,
    "epsilon_end": EPSILON_END,
    "epsilon_decay": EPSILON_DECAY,
    "click_threshold": CLICK_THRESHOLD,
    "smooth_kernel": SMOOTH_KERNEL,
    "family": "action-influence",
    "debate": "prosecution",
}

# Explicit substrate declaration
SUBSTRATE_CLASS = ActionInfluenceSubstrate
