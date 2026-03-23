"""
step0868d_l2norm_baseline.py -- Plain 800b with L2 norm delta (control for 895g).

CONFOUND FROM 895g: step895g (dual-stream) got mean=213.9/seed vs 868b's 72.1/seed.
But 895g changed TWO things vs 868b:
  1. Architecture: separated alpha stream from navigation
  2. Metric: raw_change = norm(enc - prev_enc)  [L2 norm]
     vs 868b: change = sum((enc - prev_enc)^2)  [squared sum, no sqrt]

This experiment isolates the metric change. Architecture = plain 800b (no alpha, no W).
Only change vs 868b: use L2 norm instead of squared sum.

If 868d ≈ 213/seed → the 895g improvement is the metric, not the architecture.
If 868d ≈ 72/seed  → the improvement IS from dual-stream (alpha distortion avoided).

Protocol: LS20, 25K, 10 seeds, substrate_seed=seed (varied, same as 868b).
No pretrain, no transfer. Pure navigation baseline with L2 norm delta.
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
SOFTMAX_TEMP = 0.10


def softmax_action(delta, temp):
    x = delta / temp
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


class Plain800b_L2(BaseSubstrate):
    """Plain 800b change-tracking with L2 norm delta (no alpha, no W)."""

    def __init__(self, n_actions=4, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self._epsilon = epsilon
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

    def process(self, observation):
        enc = self._encode(observation)

        if self._prev_enc is not None and self._prev_action is not None:
            # L2 NORM delta (vs 868b squared sum)
            change = float(np.linalg.norm(enc - self._prev_enc))
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * change)

        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            probs = softmax_action(self.delta_per_action, SOFTMAX_TEMP)
            action = int(self._rng.choice(self._n_actions, p=probs))

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
print("STEP 868d — PLAIN 800b WITH L2 NORM DELTA (CONTROL FOR 895g)")
print("=" * 70)
print(f"Isolates metric change from architecture change.")
print(f"868b used squared-sum. 895g used L2 norm. Which explains the 3x gap?")
print(f"25K steps, 10 seeds, substrate_seed=seed (varied). No pretrain.")

t0 = time.time()
comps = []

for ts in TEST_SEEDS:
    sub = Plain800b_L2(n_actions=N_ACTIONS, seed=ts)
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
print(f"\nComparison (controls):")
print(f"  868b (squared-sum delta, 25K):      mean=72.1  std=112  zero=5/10")
print(f"  868d (L2-norm delta, 25K):          mean={mean_L1:.1f}  std={std_L1:.1f}  zero={zero_seeds}/{len(comps)}")
print(f"  895g (dual-stream L2-norm, 25K):    mean=213.9 std=67.4  zero=0/10")
print(f"\nInterpretation:")
if abs(mean_L1 - 213.9) < 30:
    print(f"  868d ≈ 895g: improvement is FROM L2-NORM METRIC, not dual-stream")
elif abs(mean_L1 - 72.1) < 30:
    print(f"  868d ≈ 868b: improvement IS from dual-stream (alpha separation matters)")
else:
    print(f"  Ambiguous: 868d={mean_L1:.1f}, between 72.1 and 213.9. Partial effect.")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 868d DONE")
