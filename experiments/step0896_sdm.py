"""
step0896_sdm.py -- Sparse Distributed Memory (SDM) Forward Model.

R3 hypothesis: distributed address-based storage provides non-linear prediction
without neural networks. Memory writes across nearby addresses = non-linear dynamics.

Architecture:
- N_ADDRESSES=500 hard addresses (random binary vectors, 256-bit) for runtime.
- Write: at step t, write (weighted_enc_t → next_enc_{t+1}) to all addresses within
  Hamming distance RADIUS of binarized(enc_t ⊕ onehot(action)).
- Read: retrieve predicted next_enc by summing stored values at nearby addresses.
- Delta rule on stored values: converges, unlike raw overwrite.
- Action: argmax ||predicted_next - current_enc|| (prediction-contrast novelty).
- visited_set for hash-based novelty (secondary signal).

NOT codebook (no cosine, no attractor, no unit sphere — binary Hamming on random addresses).
NOT graph (no per-(state,action) counting).

Kill criterion: if retrieval accuracy < 15% at 5K steps (address space too sparse).
Protocol: cold only, 10K steps, 10 seeds (substrate_seed=seed%4).
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
N_ADDRESSES = 500  # runtime: Hamming scan over 500 addresses per step
RADIUS = 80        # Hamming radius for write (out of 256 bits)
ETA_SDM = 0.1
ENC_DIM = 256
EPSILON = 0.20


class SDM896(BaseSubstrate):
    """Sparse Distributed Memory forward model."""

    def __init__(self, n_actions=N_ACTIONS, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        rng = np.random.RandomState(seed)
        self._rng = np.random.RandomState(seed)
        # Hard addresses: (N_ADDRESSES, 256) binary
        self._addresses = (rng.random((N_ADDRESSES, ENC_DIM)) > 0.5).astype(np.float32)
        # Address-action memories: (N_ADDRESSES, n_actions, ENC_DIM)
        self._memory = np.zeros((N_ADDRESSES, n_actions, ENC_DIM), dtype=np.float32)
        self._mem_counts = np.zeros((N_ADDRESSES, n_actions), dtype=np.int32)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_enc_bin = None
        self._prev_action = None; self._last_enc = None
        self._step = 0

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self._running_mean = (1 - alpha) * self._running_mean + alpha * enc_raw
        return enc_raw - self._running_mean

    def _binarize(self, enc):
        return (enc > 0).astype(np.float32)

    def _nearby_addresses(self, addr_bin):
        """Return indices of hard addresses within Hamming radius."""
        # Hamming distance = number of bit differences
        diffs = np.abs(self._addresses - addr_bin).sum(axis=1)  # (N_ADDRESSES,)
        return np.where(diffs <= RADIUS)[0]

    def _read(self, enc_bin, action):
        """Read predicted next enc from nearby addresses."""
        nearby = self._nearby_addresses(enc_bin)
        if len(nearby) == 0:
            return None, 0
        # Weighted sum by count
        total = self._mem_counts[nearby, action].sum()
        if total == 0:
            return None, 0
        weighted = (self._memory[nearby, action] * self._mem_counts[nearby, action:action+1]).sum(axis=0)
        return weighted / (total + 1e-8), len(nearby)

    def _write(self, enc_bin, action, next_enc):
        """Write (enc_bin, action) → next_enc to nearby addresses."""
        nearby = self._nearby_addresses(enc_bin)
        for idx in nearby:
            # Delta update
            pred = self._memory[idx, action]
            err = next_enc - pred
            self._memory[idx, action] += ETA_SDM * err
            self._mem_counts[idx, action] = min(self._mem_counts[idx, action] + 1, 1000)

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc
        enc_bin = self._binarize(enc)

        # Write previous transition
        if self._prev_enc_bin is not None and self._prev_action is not None:
            self._write(self._prev_enc_bin, self._prev_action, enc)

        self._step += 1

        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            # Pick action with max predicted change (prediction-contrast)
            best_a = 0; best_score = -1.0
            for a in range(self._n_actions):
                pred, n_nearby = self._read(enc_bin, a)
                if pred is None:
                    score = 1.0  # unknown = novel, high score
                else:
                    score = float(np.sum((pred - enc)**2))
                if score > best_score:
                    best_score = score; best_a = a
            action = best_a

        self._prev_enc = enc.copy(); self._prev_enc_bin = enc_bin.copy()
        self._prev_action = action
        return action

    def retrieval_accuracy(self, n_test=50):
        """Estimate retrieval accuracy on stored (enc_bin, action) pairs."""
        if self._prev_enc_bin is None:
            return None
        # Check how well memory predicts the last observation
        total_err = 0.0; total_norm = 0.0; count = 0
        for a in range(self._n_actions):
            pred, _ = self._read(self._prev_enc_bin, a)
            if pred is not None and self._last_enc is not None:
                err = float(np.sum((pred - self._last_enc)**2))
                norm = float(np.sum(self._last_enc**2)) + 1e-8
                total_err += err; total_norm += norm; count += 1
        if count == 0: return None
        return float(1.0 - total_err / total_norm) * 100.0

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        rng = np.random.RandomState(seed)
        self._rng = np.random.RandomState(seed)
        self._addresses = (rng.random((N_ADDRESSES, ENC_DIM)) > 0.5).astype(np.float32)
        self._memory = np.zeros((N_ADDRESSES, self._n_actions, ENC_DIM), dtype=np.float32)
        self._mem_counts = np.zeros((N_ADDRESSES, self._n_actions), dtype=np.int32)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_enc_bin = None
        self._prev_action = None; self._last_enc = None; self._step = 0

    def on_level_transition(self):
        self._prev_enc = None; self._prev_enc_bin = None; self._prev_action = None

    def get_state(self): return {}
    def set_state(self, s): pass
    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


print("=" * 70)
print("STEP 896 — SPARSE DISTRIBUTED MEMORY (SDM)")
print("=" * 70)
print(f"N_addresses={N_ADDRESSES}, Hamming radius={RADIUS}. Prediction-contrast action.")
print(f"Kill criterion: retrieval accuracy < 15% at 5K steps.")

t0 = time.time()
comps = []; acc_at_5k = []

for ts in TEST_SEEDS:
    substrate_seed = ts % 4
    sub = SDM896(n_actions=N_ACTIONS, seed=substrate_seed)
    sub.reset(substrate_seed)
    env = make_game(); obs = env.reset(seed=ts * 1000)
    step = 0; completions = 0; current_level = 0
    retr_5k = None

    while step < TEST_STEPS:
        if obs is None:
            obs = env.reset(seed=ts * 1000); current_level = 0
            sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        obs, _, done, info = env.step(action); step += 1
        if step == 5000:
            retr_5k = sub.retrieval_accuracy()
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            sub.on_level_transition()
        if done:
            obs = env.reset(seed=ts * 1000); current_level = 0
            sub.on_level_transition()

    final_acc = sub.retrieval_accuracy()
    comps.append(completions)
    acc_at_5k.append(retr_5k)
    print(f"  seed={ts:3d}: L1={completions:4d}  retr_acc@5K={retr_5k:.1f}%" if retr_5k else f"  seed={ts:3d}: L1={completions:4d}  retr_acc@5K=N/A")

mean_L1 = np.mean(comps)
valid_acc = [a for a in acc_at_5k if a is not None]
mean_acc = np.mean(valid_acc) if valid_acc else None
print(f"\nMean L1: {mean_L1:.1f}/seed  (random=36.4)")
if mean_acc:
    print(f"Mean retrieval accuracy @5K: {mean_acc:.1f}%")
    print(f"Kill check: {'ALIVE' if mean_acc >= 15.0 else 'KILLED (acc<15%)'}")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 896 DONE")
