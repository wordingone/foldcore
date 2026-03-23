"""
step0900_attractor.py -- Attractor Landscape Navigation.

R3 hypothesis: energy landscape over observation space provides navigation without
counting or matching. sigma decay IS R3 (resolution increases with experience).

Architecture:
- Energy surface E(x) = sum_i exp(-||x - obs_i||^2 / (2 * sigma^2))
  High E = familiar. Low E = novel.
- Max stored observations: MAX_OBS=200 (reservoir sampling if exceeded).
- sigma starts at 1.0, decays as 1/sqrt(n_stored) — kernel width narrows.
- Action selection: predict next obs (delta momentum: next ≈ current + delta_recent)
  for each action (simple: no forward model, just momentum prediction).
  Pick action that leads to LOWEST E (most novel region).
- 20% epsilon exploration.

Dynamical systems approach. Sigma decay = R3 (resolution increases with experience).
Kill criterion: L1 < random (36.4/seed) at end of 10K steps.
Protocol: cold only, 10K steps, 10 seeds.

Note: With ENC_DIM=256 and gaussian kernel, energy = exp(-||delta||^2/2σ²) is sensitive
to normalization. Using L2 norm of centered enc (std ≈ 1 after centering).
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000
N_ACTIONS = 4
MAX_OBS = 200
SIGMA_INIT = 1.0
ENC_DIM = 256
EPSILON = 0.20


class AttractorLandscape900(BaseSubstrate):
    """Kernel-based energy landscape. sigma decays as state count grows."""

    def __init__(self, n_actions=N_ACTIONS, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        self._obs_store = []  # stored observations (up to MAX_OBS)
        self._n_stored = 0
        self._sigma = SIGMA_INIT
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_delta = None
        self._prev_action = None; self._last_enc = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self._running_mean = (1 - alpha) * self._running_mean + alpha * enc_raw
        return enc_raw - self._running_mean

    def _add_obs(self, enc):
        """Add to obs store (reservoir sampling at capacity)."""
        self._n_stored += 1
        if len(self._obs_store) < MAX_OBS:
            self._obs_store.append(enc.copy())
        else:
            # Reservoir sampling
            idx = self._rng.randint(0, self._n_stored)
            if idx < MAX_OBS:
                self._obs_store[idx] = enc.copy()
        # Update sigma
        self._sigma = max(SIGMA_INIT / np.sqrt(max(1, self._n_stored)), 0.01)

    def _energy(self, x):
        """Compute energy E(x) = sum_i exp(-||x - obs_i||^2 / (2σ²))."""
        if not self._obs_store:
            return 0.0
        store = np.array(self._obs_store, dtype=np.float32)  # (M, D)
        diffs = store - x  # (M, D)
        dists_sq = np.sum(diffs**2, axis=1) / ENC_DIM  # normalize by dim
        energies = np.exp(-dists_sq / (2 * self._sigma**2 + 1e-8))
        return float(energies.sum())

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc
        self._add_obs(enc)

        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            # Predict next enc using momentum: next ≈ current + (current - prev)
            if self._prev_enc is not None:
                delta = enc - self._prev_enc
            else:
                delta = np.zeros(ENC_DIM, np.float32)
            # Weight delta by action (simplified: action shifts in random direction)
            # For n_actions=4: use cardinal delta perturbations
            # This is a proxy for "what would this action do" without a model
            # Use: prediction = current + delta * action_weight
            # Action weights: 0=stop(0.0), 1=forward(1.0), 2=back(-1.0), 3=side(0.5)
            action_weights = [0.0, 1.0, -1.0, 0.5]

            best_a = 0; best_score = float('inf')
            for a in range(self._n_actions):
                # Predicted next obs: current + action_weight * recent_delta
                pred_next = enc + action_weights[a] * delta
                e = self._energy(pred_next)
                if e < best_score:
                    best_score = e; best_a = a
            action = best_a

        if self._prev_enc is not None:
            self._prev_delta = enc - self._prev_enc
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        self._obs_store = []
        self._n_stored = 0
        self._sigma = SIGMA_INIT
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_delta = None
        self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        self._prev_enc = None; self._prev_delta = None; self._prev_action = None

    def get_state(self): return {}
    def set_state(self, s): pass
    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


print("=" * 70)
print("STEP 900 — ATTRACTOR LANDSCAPE NAVIGATION")
print("=" * 70)
print(f"Energy E(x) = sum_i exp(-||x-obs_i||^2/2σ²). sigma={SIGMA_INIT} → decays as 1/sqrt(n).")
print(f"Action: pick action with minimum predicted energy (most novel). eps={EPSILON}.")
print(f"Kill criterion: L1 < random (36.4/seed) at 10K steps.")

t0 = time.time()
comps = []

for ts in TEST_SEEDS:
    substrate_seed = ts % 4
    sub = AttractorLandscape900(n_actions=N_ACTIONS, seed=substrate_seed)
    sub.reset(substrate_seed)
    env = make_game(); obs = env.reset(seed=ts * 1000)
    step = 0; completions = 0; current_level = 0

    while step < TEST_STEPS:
        if obs is None:
            obs = env.reset(seed=ts * 1000); current_level = 0
            sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            sub.on_level_transition()
        if done:
            obs = env.reset(seed=ts * 1000); current_level = 0
            sub.on_level_transition()

    comps.append(completions)
    print(f"  seed={ts:3d}: L1={completions:4d}  sigma={sub._sigma:.4f}  n_stored={sub._n_stored}")

mean_L1 = np.mean(comps)
print(f"\nMean L1: {mean_L1:.1f}/seed  (random=36.4)")
print(f"Kill check: {'ALIVE' if mean_L1 >= 36.4 else 'KILLED (L1<random)'}")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 900 DONE")
