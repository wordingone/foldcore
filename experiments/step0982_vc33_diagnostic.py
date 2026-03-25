"""
step0982_vc33_diagnostic.py -- VC33 standalone diagnostic with unmodified 916.

FAMILY: Diagnostic (mirrors Step 969 FT09 standalone)
R3 HYPOTHESIS: Not a hypothesis test — pure diagnostic. Run UNMODIFIED 916 substrate
on VC33 standalone at 10K and 25K steps. Determines whether VC33 failure in chain is
budget-limited (solvable) or mechanism-limited (structural gap like FT09).

Pre-ban VC33: SOLVED with 3-zone mapping + argmin (Step 505, 3/3).
Post-ban: never tested standalone with 916.

Key question: if n_actions is small → 800b should work. If large (68) → same coverage
problem as FT09 (68^7 ordered sequence).

If VC33 > 0 at ANY budget → budget-limited (solvable).
If VC33 = 0 at 25K → mechanism-limited (same as FT09).

No kill criterion. Diagnostic only.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

# Exact 916 constants
ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM   # 320
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
ALPHA_LO = 0.10
ALPHA_HI = 5.00
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
TEST_SEEDS = list(range(1, 11))


def softmax_action(delta, temp, rng):
    x = delta / temp; x = x - np.max(x)
    e = np.exp(x); probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


class RecurrentTrajectory916:
    """Unmodified 916: fixed-random recurrent h + clamped alpha + 800b change-tracking."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM + n_actions), dtype=np.float32)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        self._prev_ext = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        return np.concatenate([enc, self.h]).astype(np.float32)

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY: return
        mean_errors = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(mean_errors)) or np.any(np.isinf(mean_errors)): return
        raw_alpha = np.sqrt(np.clip(mean_errors, 0, 1e6) + 1e-8)
        mean_raw = np.mean(raw_alpha)
        if mean_raw < 1e-8 or np.isnan(mean_raw): return
        self.alpha = np.clip(raw_alpha / mean_raw, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        ext_enc = self._encode(obs)
        if self._prev_ext is not None and self._prev_action is not None:
            inp = np.concatenate([self._prev_ext * self.alpha,
                                   one_hot(self._prev_action, self._n_actions)])
            pred = self.W_pred @ inp
            error = (ext_enc * self.alpha) - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0: error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()
            weighted_delta = (ext_enc - self._prev_ext) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (1 - ALPHA_EMA) * self.delta_per_action[a] + ALPHA_EMA * change
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_action(self.delta_per_action, SOFTMAX_TEMP, self._rng)
        self._prev_ext = ext_enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None
        self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))


def make_env(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except Exception:
        import util_arcagi3; return util_arcagi3.make(name)


def run_vc33(n_steps, seeds, n_actions=68):
    results = []
    for seed in seeds:
        env = make_env("VC33")
        obs = env.reset(seed=seed * 1000)
        # Try to detect n_actions from env
        try:
            n_act = env.action_space.n
        except Exception:
            n_act = n_actions
        sub = RecurrentTrajectory916(n_actions=n_act, seed=seed)
        step = 0; completions = 0; level = 0
        while step < n_steps:
            if obs is None:
                obs = env.reset(seed=seed * 1000); sub.on_level_transition(); continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % n_act
            obs, _, done, info = env.step(action); step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                completions += cl - level; level = cl; sub.on_level_transition()
            if done:
                obs = env.reset(seed=seed * 1000); level = 0; sub.on_level_transition()
        results.append(completions)
        print(f"  seed={seed}: completions={completions:4d}  alpha_conc={sub.alpha_conc():.2f}  n_actions={n_act}")
    return results


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("STEP 982 — VC33 STANDALONE DIAGNOSTIC (unmodified 916)")
    print("=" * 70)
    vc33_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/vc33') if len(d) >= 8), '?')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), '?')
    print(f"Game versions: VC33={vc33_hash}  FT09={ft09_hash} (reference)")
    print(f"Mirror of Step 969 FT09 standalone. 10 seeds. 10K + 25K.")
    print(f"Hypothesis: VC33=0 at 25K → mechanism-limited. VC33>0 → budget-limited.")
    print()

    t0 = time.time()

    print("--- VC33 @ 10K (10 seeds) ---")
    r10k = run_vc33(10_000, TEST_SEEDS)
    print(f"  10K summary: mean={np.mean(r10k):.1f}  nonzero={sum(1 for x in r10k if x > 0)}/10  {r10k}")

    print()
    print("--- VC33 @ 25K (10 seeds) ---")
    r25k = run_vc33(25_000, TEST_SEEDS)
    print(f"  25K summary: mean={np.mean(r25k):.1f}  nonzero={sum(1 for x in r25k if x > 0)}/10  {r25k}")

    print()
    print("=" * 70)
    print("STEP 982 RESULTS:")
    print(f"  VC33 @ 10K: mean={np.mean(r10k):.1f}/seed  nonzero={sum(1 for x in r10k if x > 0)}/10  {r10k}")
    print(f"  VC33 @ 25K: mean={np.mean(r25k):.1f}/seed  nonzero={sum(1 for x in r25k if x > 0)}/10  {r25k}")
    print()
    if any(x > 0 for x in r25k):
        print("  VERDICT: BUDGET-LIMITED — VC33 is solvable with 916, needs more budget in chain.")
    elif any(x > 0 for x in r10k):
        print("  VERDICT: PARTIALLY BUDGET-LIMITED — signal at 10K, more steps needed.")
    else:
        print("  VERDICT: MECHANISM-LIMITED — VC33=0 at 25K, same structural gap as FT09.")
    print(f"  FT09 reference (Step 969): 0/10 at 25K — mechanism-limited confirmed.")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("STEP 982 DONE")
