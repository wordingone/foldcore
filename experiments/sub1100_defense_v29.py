"""
sub1100_defense_v29.py — Argmin-over-frequency with coarse state hashing

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1100 --substrate experiments/sub1100_defense_v29.py

FAMILY: Frequency-counting exploration (NEW — not reactive switching)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: Reactive switching (v21-v28, 7 failures) is structurally
limited — it requires distance-to-initial as progress signal, which misses
games where progress isn't measurable by L1 distance. Argmin-over-frequency
needs NO progress signal: it explores by visiting LEAST-SEEN states. This
works for ANY game regardless of progress structure.

ARCHITECTURE:
- avgpool8 (64D) encoding — same pooling as v21
- COARSE hash: 4 quadrants × 4 brightness levels = 256 buckets
  (designed for ~5.5 visits per (bucket,action) pair in 10K steps)
- Visit count table: dict of (bucket, action) → count
- Action selection: argmin over visit counts for current bucket
  (ties broken randomly)
- Zero learned params, zero gradient updates
- State grows (visit counts accumulate) but processing rules are FIXED

WHY THIS IS ℓ₁ (defense):
- Hash function: FIXED (not learned)
- Selection rule: FIXED (argmin over counts)
- Counts: accumulated STATE, not learned parameters
- No W matrices, no alpha, no outer products

WHY THIS IS DIFFERENT FROM PROSECUTION:
- Prosecution: learned W_fwd predicts action effects → selects max-change
- Defense: counts visits → selects least-visited (exploration, not prediction)

WHY THIS IS DIFFERENT FROM v21 (reactive switching):
- v21: distance-to-initial progress signal, round-robin action cycling
- v29: no progress signal at all, argmin over frequency

BANS: Graph and codebook bans LIFTED (Jun, 2026-03-25). Per-(state,action)
counting is explicitly allowed.

KILL: L1 < 2/5 (worse than v21 baseline on typical draw).
SUCCESS: L1 > 2/5 on same draw type OR ANY L2+.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 8
N_BLOCKS = 8
N_DIMS = N_BLOCKS * N_BLOCKS  # 64
N_KB = 7
N_QUANT = 4  # brightness quantization levels per quadrant


def _obs_to_enc(obs):
    """avgpool8: 64x64 → 8x8 = 64D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class ArgminFrequencySubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        # Visit count table: (bucket_hash, action) → count
        self._visit_counts = {}

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._init_state()

    def _obs_to_bucket(self, enc):
        """Coarse hash: 4 quadrants × N_QUANT brightness levels.

        Quadrants: top-left, top-right, bottom-left, bottom-right
        Each quadrant = 16 encoding dims (4×4 blocks in the 8×8 grid).
        Mean brightness quantized to N_QUANT levels.
        Total buckets: N_QUANT^4 = 256.
        """
        quadrants = []
        # 8×8 grid split into 4×4 quadrants
        for qy in range(2):
            for qx in range(2):
                vals = []
                for by in range(4):
                    for bx in range(4):
                        idx = (qy * 4 + by) * N_BLOCKS + (qx * 4 + bx)
                        vals.append(enc[idx])
                q_mean = np.mean(vals)
                # Quantize to 0..N_QUANT-1
                level = int(q_mean * N_QUANT)
                level = max(0, min(N_QUANT - 1, level))
                quadrants.append(level)
        return tuple(quadrants)

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(obs)
        bucket = self._obs_to_bucket(enc)

        # Get visit counts for all actions in this bucket
        counts = np.zeros(self._n_actions, dtype=np.float32)
        for a in range(self._n_actions):
            counts[a] = self._visit_counts.get((bucket, a), 0)

        # Argmin: pick least-visited action (ties broken randomly)
        min_count = counts.min()
        candidates = np.where(counts == min_count)[0]
        action = int(self._rng.choice(candidates))

        # Update visit count
        self._visit_counts[(bucket, action)] = counts[action] + 1

        return action

    def on_level_transition(self):
        # Keep visit counts across levels (cross-level transfer)
        # But reset step count
        self.step_count = 0


CONFIG = {
    "n_dims": N_DIMS,
    "n_quant": N_QUANT,
    "n_buckets": N_QUANT ** 4,
    "family": "frequency-counting exploration",
    "tag": "defense v29 (ℓ₁ argmin-over-frequency, coarse hash, 256 buckets)",
}

SUBSTRATE_CLASS = ArgminFrequencySubstrate
