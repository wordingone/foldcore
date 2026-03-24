"""Step 953: Hebbian RNN softmax action selection (reassessment experiment).

R3 hypothesis: Softmax resolves the argmax winner-take-all lock by providing soft
exploitation that scales with W_a signal strength. When W_a ≈ 0 → uniform distribution
(natural exploration without schedule). As W_a grows → soft preference. When W_a large
→ near-greedy. No epsilon, no schedule — adaptation is automatic.

Changes from 948 base (h_dim=64, fixed W_h — one change):
  - Softmax action selection replaces argmax + epsilon
  - Temperature T=1.0
  - W_a: same positive-only Hebbian as 948
  - W_h: fixed random (same as 948)
  - No epsilon parameter

6th Hebbian RNN experiment (justified: different action selection mechanism, never tested).
Kill: LS20 ≤1/10 seeds → family DEAD.
Success: 3+ seeds L1>0 → softmax fixes the lock.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
ETA_PRED = 0.01
ETA_ACTION = 0.001
SOFTMAX_TEMP = 1.0
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def softmax_sample(scores, temp, rng):
    x = scores / temp
    x = x - np.max(x)  # numerical stability
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(scores), p=probs))


class HebbianSoftmax953:
    """Hebbian RNN with softmax action selection. No epsilon, no argmax."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Fixed random recurrent weights (same as 948)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        self.W_pred = np.zeros((ENC_DIM, H_DIM), dtype=np.float32)
        self.W_a = np.zeros((n_actions, H_DIM), dtype=np.float32)

        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._prev_h = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = sigmoid(self.W_h @ self.h + self.W_x @ enc)
        return enc

    def process(self, obs):
        enc = self._encode(obs)

        if self._prev_h is not None and self._prev_action is not None:
            pred = self.W_pred @ self._prev_h
            error = enc - pred
            delta = float(np.linalg.norm(error))

            if delta > 10.0:
                error = error * (10.0 / delta)
                delta = 10.0

            if not np.any(np.isnan(error)):
                # W_pred: same as 948
                self.W_pred += ETA_PRED * np.outer(error, self._prev_h)

                # W_a: same positive-only Hebbian as 948
                self.W_a[self._prev_action] += ETA_ACTION * delta * self.h

        # Softmax action selection — always sample, no epsilon threshold
        # W_a ≈ 0 → uniform; W_a large → near-greedy; no winner-take-all
        scores = self.W_a @ self.h
        action = softmax_sample(scores, SOFTMAX_TEMP, self._rng)

        self._prev_h = self.h.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_h = None
        self._prev_action = None

    def w_a_norm(self):
        return float(np.linalg.norm(self.W_a))


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except Exception:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps):
    results = []
    w_norms = []
    for seed in seeds:
        sub = HebbianSoftmax953(n_actions=n_actions, seed=seed)
        env = make_game(game_name)
        obs = env.reset(seed=seed * 1000)
        step = 0; completions = 0; current_level = 0
        while step < n_steps:
            if obs is None:
                obs = env.reset(seed=seed * 1000)
                sub.on_level_transition()
                continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
            obs, _, done, info = env.step(action)
            step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > current_level:
                completions += (cl - current_level)
                current_level = cl
                sub.on_level_transition()
            if done:
                obs = env.reset(seed=seed * 1000)
                current_level = 0
                sub.on_level_transition()
        results.append(completions)
        w_norms.append(sub.w_a_norm())
        print(f"    seed={seed}: L1={completions:4d}  W_a_norm={sub.w_a_norm():.3f}")
    return results, w_norms


if __name__ == "__main__":
    import os, time

    print("=" * 70)
    print("STEP 953 — HEBBIAN RNN SOFTMAX (reassessment experiment)")
    print("=" * 70)
    t0 = time.time()

    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), 'unknown')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), 'unknown')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}")
    print(f"H_DIM={H_DIM}  ETA_PRED={ETA_PRED}  ETA_ACTION={ETA_ACTION}  SOFTMAX_TEMP={SOFTMAX_TEMP}")
    print()

    print("--- LS20 (4 actions, 10K steps) ---")
    ls20_r, ls20_norms = run_game("LS20", 4, TEST_SEEDS, TEST_STEPS)
    ls20_mean = np.mean(ls20_r)
    ls20_nonzero = sum(1 for x in ls20_r if x > 0)
    print(f"  LS20: L1={ls20_mean:.1f}/seed  nonzero={ls20_nonzero}/10  W_a_norm={np.mean(ls20_norms):.3f}")
    print(f"  {ls20_r}")

    print()
    print("--- FT09 (68 actions, 10K steps) ---")
    ft09_r, ft09_norms = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS)
    ft09_mean = np.mean(ft09_r)
    ft09_nonzero = sum(1 for x in ft09_r if x > 0)
    print(f"  FT09: L1={ft09_mean:.1f}/seed  nonzero={ft09_nonzero}/10  W_a_norm={np.mean(ft09_norms):.3f}")
    print(f"  {ft09_r}")

    print()
    print("=" * 70)
    print("STEP 953 RESULTS (vs 948: LS20 nonzero=1/10, mean=9.6):")

    if ls20_nonzero >= 3:
        verdict = f"SUCCESS — {ls20_nonzero}/10 seeds, softmax fixes the lock → iterate"
    elif ls20_nonzero <= 1:
        verdict = f"KILL — {ls20_nonzero}/10 seeds → Hebbian RNN FAMILY DEAD → ESN next"
    else:
        verdict = f"MARGINAL — {ls20_nonzero}/10 seeds (need 3+ for success)"

    print(f"  LS20: L1={ls20_mean:.1f}/seed  {ls20_r}")
    print(f"  FT09: L1={ft09_mean:.1f}/seed  {ft09_r}")
    print(f"  VERDICT: {verdict}")

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("STEP 953 DONE")
