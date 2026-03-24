"""Step 955: ESN + sigmoid activation (restore positive accumulation).

R3 hypothesis: Sigmoid activation restores positive accumulation (h ∈ [0,1]) while
ESN's fixed W_h (spectral radius 0.9) provides strong recurrence from step 1.
Combines 948's proven activation with ESN's stronger trajectory encoding.

954 showed: tanh h ∈ [-1,1] → mixed-sign W_a updates → cancellation → W_a_norm=2.2.
Expected: sigmoid h ∈ [0,1] → all-positive W_a updates → accumulation → W_a_norm≈20.

One change from 954: sigmoid instead of tanh. Everything else identical.

Kill: LS20 ≤1/10 seeds → activation isn't the issue.
Success: 3+ seeds → sigmoid + ESN = robust → iterate on reservoir params.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
SPECTRAL_RADIUS = 0.9
SPARSITY = 0.1
ETA_PRED = 0.01
ETA_ACTION = 0.001
EPSILON = 0.20
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def make_reservoir(h_dim, spectral_radius, sparsity, rs):
    W = rs.randn(h_dim, h_dim).astype(np.float32)
    eigvals = np.abs(np.linalg.eigvals(W))
    W *= spectral_radius / eigvals.max()
    mask = (rs.random(W.shape) < sparsity).astype(np.float32)
    W *= mask
    return W


class ESNSigmoid955:
    """ESN + sigmoid h. Fixed W_h/W_x, Hebbian W_a, tanh→sigmoid."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        self.W_h = make_reservoir(H_DIM, SPECTRAL_RADIUS, SPARSITY, rs)
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
        self.h = sigmoid(self.W_h @ self.h + self.W_x @ enc)  # sigmoid, not tanh
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
                self.W_pred += ETA_PRED * np.outer(error, self._prev_h)
                self.W_a[self._prev_action] += ETA_ACTION * delta * self.h

        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            scores = self.W_a @ self.h
            action = int(np.argmax(scores))

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
        sub = ESNSigmoid955(n_actions=n_actions, seed=seed)
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
    print("STEP 955 — ESN + SIGMOID (restore positive accumulation)")
    print("=" * 70)
    t0 = time.time()

    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), 'unknown')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), 'unknown')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}")
    print(f"H_DIM={H_DIM}  SR={SPECTRAL_RADIUS}  SPARSITY={SPARSITY}  ACTIVATION=sigmoid")
    print(f"ETA_PRED={ETA_PRED}  ETA_ACTION={ETA_ACTION}  EPSILON={EPSILON}")
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
    print(f"STEP 955 RESULTS (vs 954: LS20 nonzero=1/10, W_a_norm=1.9):")

    if ls20_nonzero >= 3:
        verdict = f"SUCCESS — {ls20_nonzero}/10 seeds, sigmoid + ESN = robust → iterate reservoir"
    elif ls20_nonzero <= 1:
        verdict = f"KILL — {ls20_nonzero}/10 seeds, activation not the issue"
    else:
        verdict = f"MARGINAL — {ls20_nonzero}/10 seeds (need 3+ for success)"

    print(f"  LS20: L1={ls20_mean:.1f}/seed  {ls20_r}")
    print(f"  FT09: L1={ft09_mean:.1f}/seed  {ft09_r}")
    print(f"  VERDICT: {verdict}")

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("STEP 955 DONE")
