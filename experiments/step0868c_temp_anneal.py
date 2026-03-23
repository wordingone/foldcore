"""
step0868c_temp_anneal.py -- Temperature Annealing for 800b softmax.

Step 868b problem: T=0.1 too sharp from step 1. If one action gets slightly higher
delta early (by luck), softmax locks onto it before all actions differentiate.
Result: 5/10 seeds get L1=0 (cold-start lottery).

Fix (Leo mail 2628): anneal temperature 1.0 → 0.1 over first 5K steps.
  T = max(0.1, 1.0 - 0.9 * (step / 5000))
  - step 0: T=1.0 → near-uniform exploration (all actions ~equal)
  - step 2500: T=0.55 → moderate preference
  - step 5000+: T=0.1 → sharp exploitation (like 868b)

Architecture: identical to step868b (800b change EMA + softmax, varied substrate_seeds).
Only change: annealed temperature instead of fixed T=0.1.

Protocol: 25K, 10 seeds, substrate_seed=seed (varied). Same as 868b.
Compare directly: 868c vs 868b. Key metric: zero-seed fraction.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000
N_ACTIONS = 4
INIT_DELTA = 1.0
ALPHA_EMA = 0.10
EPSILON = 0.20
T_START = 1.0
T_END = 0.10
ANNEAL_STEPS = 5_000


class TempAnneal868c(BaseSubstrate):
    """800b change EMA + temperature-annealed softmax."""

    def __init__(self, n_actions=4, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self._epsilon = epsilon
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._running_mean = np.zeros(256, np.float32)
        self._n_obs = 0
        self._step = 0
        self._prev_enc = None; self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def _temperature(self):
        return max(T_END, T_START - (T_START - T_END) * (self._step / ANNEAL_STEPS))

    def _select_action(self, delta):
        T = self._temperature()
        d = delta / (np.sum(delta) + 1e-8)
        exp_d = np.exp(d / T)
        probs = exp_d / exp_d.sum()
        return int(self._rng.choice(self._n_actions, p=probs))

    def process(self, observation):
        enc = self._encode(observation)
        self._step += 1
        if self._prev_enc is not None and self._prev_action is not None:
            change = float(np.sum((enc - self._prev_enc) ** 2))
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * change)
        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = self._select_action(self.delta_per_action)
        self._prev_enc = enc.copy(); self._prev_action = action
        return action

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        self._prev_enc = None; self._prev_action = None
        self.delta_per_action = np.full(self._n_actions, INIT_DELTA, np.float32)
        self._running_mean = np.zeros(256, np.float32)
        self._n_obs = 0; self._step = 0

    def on_level_transition(self):
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
print("STEP 868c — 800b SOFTMAX WITH TEMPERATURE ANNEALING (1.0 → 0.1)")
print("=" * 70)
print(f"Anneals T from 1.0 (uniform) to 0.1 (sharp) over first {ANNEAL_STEPS} steps.")
print(f"Fixes 868b cold-start lottery (5/10 seeds at L1=0).")
print(f"25K steps, 10 seeds, substrate_seed=seed (varied).")

t0 = time.time()
comps = []

for ts in TEST_SEEDS:
    sub = TempAnneal868c(n_actions=N_ACTIONS, seed=ts)
    sub.reset(ts)
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
    print(f"  seed={ts:3d}: L1={completions:4d}")

mean_L1 = np.mean(comps); std_L1 = np.std(comps)
zero_seeds = sum(1 for x in comps if x == 0)
print(f"\nMean L1: {mean_L1:.1f}/seed  std={std_L1:.1f}  zero={zero_seeds}/{len(comps)}")
print(f"  {comps}")
print(f"\nComparison:")
print(f"  868b (fixed T=0.1): mean=72.1 std=112 zero=5/10")
print(f"  868c (annealed T): mean={mean_L1:.1f} std={std_L1:.1f} zero={zero_seeds}/{len(comps)}")
print(f"  Annealing {'FIXES zero-seed problem' if zero_seeds < 5 else 'does NOT fix zero-seed problem'}")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 868c DONE")
