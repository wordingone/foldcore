"""
step0898_lz_complexity.py -- Lempel-Ziv Complexity Novelty.

R3 hypothesis: navigation without ANY model — pure information theory.
Compressibility of observation sequence as action signal. The LZ compression
dictionary IS the self-modifying state (grows with each novel observation).

Architecture:
- Maintain obs history as byte sequence (hash each obs to 8 bits).
- For each candidate action: simulate appending action_byte to history.
- Compute compression ratio of augmented sequence (zlib level=1).
- Pick action that INCREASES compression ratio most (= least compressible = most novel).
- Window: last WINDOW_SIZE observations.

No model. No counting. No matching. Pure compression.
R3: the compression dictionary grows with observations (self-modification).
R2 compliant: self-supervised on observation stream.
Graph ban compliant: no per-(state,action) data structures.

Protocol: cold test only (compression dictionary grows from scratch each run).
10K steps, 10 seeds (substrate_seed = seed%4 for varied RNG).
Metric: L1, action variance (does compression differentiate actions?).
Kill criterion: if compression ratio variance across actions < 0.001.
"""
import sys, time, zlib
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000
N_ACTIONS = 4
WINDOW_SIZE = 500
ENC_DIM = 256
EPSILON = 0.10  # small; LZ should guide well


def obs_to_byte(enc):
    """Hash enc to single byte (0-255)."""
    # Use first 8 elements, quantize to 0-255
    vals = np.clip((enc[:8] * 16 + 128), 0, 255).astype(np.uint8)
    return int(vals[0])  # single byte per obs (simple but distinctive)


def compression_ratio(byte_seq):
    """zlib compression ratio of byte sequence."""
    if len(byte_seq) < 4:
        return 1.0
    data = bytes(byte_seq)
    compressed = zlib.compress(data, level=1)
    return len(compressed) / len(data)


class LZNovelty898(BaseSubstrate):
    """LZ compression-based novelty. No model, no counting."""

    def __init__(self, n_actions=N_ACTIONS, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        self._history = deque(maxlen=WINDOW_SIZE)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_action = None; self._last_enc = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self._running_mean = (1 - alpha) * self._running_mean + alpha * enc_raw
        return enc_raw - self._running_mean

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc
        b = obs_to_byte(enc)
        self._history.append(b)

        if self._rng.random() < self._epsilon:
            return int(self._rng.randint(0, self._n_actions))

        # For each action: simulate appending action_byte to history
        # Action byte = action index (0-3)
        hist_list = list(self._history)
        base_ratio = compression_ratio(hist_list)
        best_a = 0; best_ratio_gain = -1.0
        action_ratios = []
        for a in range(self._n_actions):
            # Augment with action byte
            aug = hist_list + [a]
            ratio = compression_ratio(aug)
            action_ratios.append(ratio)
            if ratio > best_ratio_gain:
                best_ratio_gain = ratio; best_a = a

        self._prev_action = best_a
        self._last_action_ratios = action_ratios
        return best_a

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        self._history = deque(maxlen=WINDOW_SIZE)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        self._prev_action = None

    def get_state(self): return {}
    def set_state(self, s): pass
    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


print("=" * 70)
print("STEP 898 — LZ COMPLEXITY NOVELTY")
print("=" * 70)
print(f"No model. Compression ratio of obs history drives action selection.")
print(f"Window={WINDOW_SIZE} obs. Action: argmax compression ratio increase.")
print(f"Kill criterion: action ratio variance < 0.001.")

t0 = time.time()
comps = []; ratio_vars = []

for ts in TEST_SEEDS:
    substrate_seed = ts % 4
    sub = LZNovelty898(n_actions=N_ACTIONS, seed=substrate_seed)
    sub.reset(substrate_seed)
    env = make_game(); obs = env.reset(seed=ts * 1000)
    step = 0; completions = 0; current_level = 0
    action_ratio_history = []

    while step < TEST_STEPS:
        if obs is None:
            obs = env.reset(seed=ts * 1000); current_level = 0
            sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        if hasattr(sub, '_last_action_ratios') and len(sub._last_action_ratios) == N_ACTIONS:
            action_ratio_history.append(sub._last_action_ratios)
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            sub.on_level_transition()
        if done:
            obs = env.reset(seed=ts * 1000); current_level = 0
            sub.on_level_transition()

    # Compute ratio variance (differentiation between actions)
    if action_ratio_history:
        ratios_arr = np.array(action_ratio_history)  # (T, n_actions)
        mean_ratio_var = float(np.mean(np.var(ratios_arr, axis=1)))
    else:
        mean_ratio_var = 0.0

    comps.append(completions); ratio_vars.append(mean_ratio_var)
    print(f"  seed={ts:3d}: L1={completions:4d}  ratio_variance={mean_ratio_var:.5f}")

mean_L1 = np.mean(comps); mean_rv = np.mean(ratio_vars)
print(f"\nMean L1: {mean_L1:.1f}/seed  (random=36.4)")
print(f"Mean ratio variance: {mean_rv:.5f} (kill threshold: 0.001)")
print(f"Kill check: {'ALIVE' if mean_rv >= 0.001 else 'KILLED (variance<0.001)'}")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 898 DONE")
