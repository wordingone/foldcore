"""
step0899_population_predictors.py -- Population of Micro-Predictors.

R3 hypothesis: R3 via population selection, not gradient. The ensemble's prediction
changes through evolutionary pressure. Selection IS self-modification.

Architecture:
- N=10 predictors, each W_i: (256, 256+n_actions). Initialized random (small).
- Every step: all 10 predict next obs. Track individual pred errors (window=50).
- Every SELECTION_INTERVAL=200 steps: selection event.
  - Rank by recent pred accuracy.
  - Kill bottom 3. Duplicate top 3 with Gaussian noise (sigma=0.01).
  - Middle 4 kept as-is.
- Action: top predictor picks action maximizing predicted novelty (visited_set).
- 20% epsilon exploration.

R3 via selection pressure, not gradient descent.
Kill criterion: if population diversity collapses (std of predictor W norms < 0.001) at 5K.

Protocol: cold test (no warm/pretrain — R3 via evolution, not transfer).
10K steps, 10 seeds (substrate_seed = seed%4).
Metric: L1, population diversity, which predictor dominates.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000
N_ACTIONS = 4
N_PREDICTORS = 10
SELECTION_INTERVAL = 200
ERROR_WINDOW = 50
ETA = 0.01
ENC_DIM = 256
EPSILON = 0.20
MUTATION_SIGMA = 0.01


def enc_hash(enc):
    coarse = (enc[:32] * 8).astype(np.int16)  # slightly finer than step893
    return hash(coarse.tobytes())


class PopulationPredictors899(BaseSubstrate):
    """Population of micro-predictors with evolutionary selection."""

    def __init__(self, n_actions=N_ACTIONS, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        inp_dim = ENC_DIM + n_actions
        # Initialize population with small random weights
        self.W_pop = [self._rng.randn(ENC_DIM, inp_dim).astype(np.float32) * 0.01
                      for _ in range(N_PREDICTORS)]
        # Per-predictor recent error windows
        self.error_windows = [deque(maxlen=ERROR_WINDOW) for _ in range(N_PREDICTORS)]
        self._step = 0
        self.visited = set()
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self._running_mean = (1 - alpha) * self._running_mean + alpha * enc_raw
        return enc_raw - self._running_mean

    def _inp(self, enc, action):
        a_oh = np.zeros(self._n_actions, np.float32); a_oh[action] = 1.0
        return np.concatenate([enc, a_oh])

    def _selection_event(self):
        # Score each predictor by recent mean error
        scores = [np.mean(w) if len(w) > 0 else 1.0 for w in self.error_windows]
        ranked = np.argsort(scores)  # ascending = lower error = better
        best_3 = ranked[:3]; worst_3 = ranked[-3:]
        # Replace worst 3 with noisy copies of best 3
        for i, (bad, good) in enumerate(zip(worst_3, best_3)):
            self.W_pop[bad] = self.W_pop[good].copy() + \
                self._rng.randn(*self.W_pop[good].shape).astype(np.float32) * MUTATION_SIGMA
            self.error_windows[bad] = deque(maxlen=ERROR_WINDOW)

    def population_diversity(self):
        norms = [float(np.linalg.norm(W)) for W in self.W_pop]
        return float(np.std(norms))

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc
        h = enc_hash(enc)
        self.visited.add(h)
        self._step += 1

        # Update all predictors with delta rule
        if self._prev_enc is not None and self._prev_action is not None:
            inp = self._inp(self._prev_enc, self._prev_action)
            for i, W in enumerate(self.W_pop):
                pred = W @ inp
                err = pred - enc
                W -= ETA * np.outer(err, inp)
                self.error_windows[i].append(float(np.mean(np.abs(err))))

        # Selection event
        if self._step % SELECTION_INTERVAL == 0:
            self._selection_event()

        # Action selection: best predictor picks novel action
        scores = [np.mean(w) if len(w) > 0 else 1.0 for w in self.error_windows]
        best_predictor_idx = int(np.argmin(scores))
        W_best = self.W_pop[best_predictor_idx]

        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            best_a = 0; best_score = -1.0
            for a in range(self._n_actions):
                pred = W_best @ self._inp(enc, a)
                h_pred = enc_hash(pred)
                score = 0.0 if h_pred in self.visited else 1.0
                pred_err_mag = float(np.linalg.norm(pred - enc))
                score += 0.1 * pred_err_mag
                if score > best_score:
                    best_score = score; best_a = a
            action = best_a

        self._prev_enc = enc.copy(); self._prev_action = action
        return action

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        inp_dim = ENC_DIM + self._n_actions
        self.W_pop = [self._rng.randn(ENC_DIM, inp_dim).astype(np.float32) * 0.01
                      for _ in range(N_PREDICTORS)]
        self.error_windows = [deque(maxlen=ERROR_WINDOW) for _ in range(N_PREDICTORS)]
        self._step = 0
        self.visited = set()
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        self.visited = set()
        self._prev_enc = None; self._prev_action = None

    def get_state(self): return {}
    def set_state(self, s): pass
    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


print("=" * 70)
print("STEP 899 — POPULATION OF MICRO-PREDICTORS")
print("=" * 70)
print(f"N={N_PREDICTORS} predictors. Selection every {SELECTION_INTERVAL} steps: kill worst 3, duplicate best 3.")
print(f"Action: best predictor argmax(novelty). Kill: diversity < 0.001 at 5K steps.")

t0 = time.time()
comps = []; diversities = []

for ts in TEST_SEEDS:
    substrate_seed = ts % 4
    sub = PopulationPredictors899(n_actions=N_ACTIONS, seed=substrate_seed)
    sub.reset(substrate_seed)
    env = make_game(); obs = env.reset(seed=ts * 1000)
    step = 0; completions = 0; current_level = 0
    div_at_5k = None

    while step < TEST_STEPS:
        if obs is None:
            obs = env.reset(seed=ts * 1000); current_level = 0
            sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        obs, _, done, info = env.step(action); step += 1
        if step == 5000:
            div_at_5k = sub.population_diversity()
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            sub.on_level_transition()
        if done:
            obs = env.reset(seed=ts * 1000); current_level = 0
            sub.on_level_transition()

    final_div = sub.population_diversity()
    comps.append(completions); diversities.append(final_div)
    print(f"  seed={ts:3d}: L1={completions:4d}  diversity: 5K={div_at_5k:.4f} end={final_div:.4f}")

mean_L1 = np.mean(comps); mean_div = np.mean(diversities)
print(f"\nMean L1: {mean_L1:.1f}/seed  (random=36.4)")
print(f"Mean final diversity: {mean_div:.4f} (kill threshold: 0.001)")
print(f"Kill check: {'ALIVE' if mean_div >= 0.001 else 'KILLED (diversity<0.001)'}")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 899 DONE")
