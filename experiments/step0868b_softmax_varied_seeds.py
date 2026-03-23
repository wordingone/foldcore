"""
step0868b_softmax_varied_seeds.py -- Step 868 validation with varied substrate seeds.

Step 868 softmax_01 reported 379/seed (25K, seeds 6-10) but substrate_seed=0 always.
n_eff=1: all seeds share the same random substrate. This MAY inflate results.

Step 868b: same softmax_01 mechanism, but substrate_seed=seed (varied).
Validates whether 379/seed is real or an n_eff=1 artifact.

Architecture: identical to step868 OpponentProcess (800b + softmax T=0.1).
- delta_per_action: EMA of (enc_t - enc_{t-1})^2 sum
- Action: normalized-delta softmax T=0.1 + epsilon=0.20

Protocol: 10 seeds (1-10), 25K steps, substrate_seed=seed (varied).
Compare to step868 baseline: softmax_01 @ seed=0 → 379/seed.
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
SOFTMAX_T = 0.1


class SoftmaxNav868b(BaseSubstrate):
    """800b change EMA + softmax T=0.1. Identical to step868 softmax_01."""

    def __init__(self, n_actions=4, seed=0):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._running_mean = np.zeros(256, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def _select_action(self, delta):
        """Softmax T=0.1 on normalized delta — same as step868 softmax_01."""
        d = delta / (np.sum(delta) + 1e-8)
        exp_d = np.exp(d / SOFTMAX_T)
        probs = exp_d / exp_d.sum()
        return int(self._rng.choice(self._n_actions, p=probs))

    def process(self, observation):
        enc = self._encode(observation)
        if self._prev_enc is not None and self._prev_action is not None:
            change = float(np.sum((enc - self._prev_enc) ** 2))
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * change)
        if self._rng.random() < EPSILON:
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
        self._n_obs = 0

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
print("STEP 868b — SOFTMAX_01 VARIED SEEDS VALIDATION")
print("=" * 70)
print(f"Validating step868 softmax_01=379/seed (seed=0, n_eff=1).")
print(f"Now using substrate_seed=seed (varied). 10 seeds, 25K steps.")

t0 = time.time()
comps = []

for ts in TEST_SEEDS:
    sub_seed = ts  # varied (not 0)
    sub = SoftmaxNav868b(n_actions=N_ACTIONS, seed=sub_seed)
    sub.reset(sub_seed)
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
    print(f"  seed={ts:3d} (sub_seed={sub_seed}): L1={completions:4d}")

mean_L1 = np.mean(comps)
print(f"\nMean L1 (varied seeds): {mean_L1:.1f}/seed  ({comps})")
print(f"Step 868 baseline: softmax_01 seed=0 = 379/seed")
print(f"Validation: {'CONFIRMED (within 10%)' if abs(mean_L1 - 379) / 379 < 0.10 else 'DISCREPANCY — n_eff issue' if mean_L1 < 340 else 'CLOSE'}")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 868b DONE")
