"""Step 950: Hebbian RNN warm-start W_a (break W_pred/W_a deadlock).

R3 hypothesis: Temporal sequencing of learning (W_pred first, then W_a) produces
more robust action selection because W_a updates are informed by trained prediction
errors, not random noise. Pure random during warm-up trains W_pred on diverse
trajectories; Hebbian W_a updates start from a substrate that actually predicts.

Changes from 949 (h_dim=128):
  - WARM_UP=1000 steps: pure random action, W_pred+W_h learn, W_a frozen
  - After warm-up: epsilon-greedy via argmax(W_a @ h), W_a Hebbian updates begin
  - W_h also learns: W_h += lr * 0.1 * outer(error[:H_DIM], h)

Kill: LS20 all seeds L1=0 → warm-start doesn't break deadlock → deadlock is structural.
Success: 3+ seeds L1>0 → deadlock was temporal.
Neutral: still 1/10 → W_a symmetry breaking needs more.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 128
ETA_PRED = 0.01
ETA_ACTION = 0.001
EPSILON = 0.20
WARM_UP = 1000
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class HebbianWarmStart950:
    """Hebbian RNN with warm-start: W_pred trains first, W_a after WARM_UP steps."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Recurrent weights — learned (not fixed this time)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # Prediction and action mapping
        self.W_pred = np.zeros((ENC_DIM, H_DIM), dtype=np.float32)
        self.W_a = np.zeros((n_actions, H_DIM), dtype=np.float32)

        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._step = 0
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
        self._step += 1

        if self._prev_h is not None and self._prev_action is not None:
            pred = self.W_pred @ self._prev_h
            error = enc - pred
            delta = float(np.linalg.norm(error))

            if delta > 10.0:
                error = error * (10.0 / delta)
                delta = 10.0

            if not np.any(np.isnan(error)):
                # W_pred always learns
                self.W_pred += ETA_PRED * np.outer(error, self._prev_h)

                # W_h also learns (recurrence needs training too)
                self.W_h += ETA_PRED * 0.1 * np.outer(error[:H_DIM], self._prev_h)

                # W_a: Hebbian only after warm-up
                if self._step >= WARM_UP:
                    self.W_a[self._prev_action] += ETA_ACTION * delta * self.h

        # Action selection
        if self._step < WARM_UP:
            # Pure random during warm-up — trains W_pred on diverse trajectories
            action = int(self._rng.randint(0, self._n_actions))
        else:
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
        sub = HebbianWarmStart950(n_actions=n_actions, seed=seed)
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
    print("STEP 950 — HEBBIAN RNN WARM-START (W_pred first, W_a after 1K steps)")
    print("=" * 70)
    t0 = time.time()

    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), 'unknown')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), 'unknown')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}")
    print(f"H_DIM={H_DIM}  WARM_UP={WARM_UP}  ETA_PRED={ETA_PRED}  ETA_ACTION={ETA_ACTION}  EPSILON={EPSILON}")
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
    print("STEP 950 RESULTS (vs 949: LS20 nonzero=1/10, mean=9.6):")

    any_ls20 = any(x > 0 for x in ls20_r)
    if not any_ls20:
        verdict = "KILL — warm-start fails, deadlock is structural"
    elif ls20_nonzero >= 3:
        verdict = f"SUCCESS — {ls20_nonzero}/10 seeds nonzero, deadlock was temporal"
    else:
        verdict = f"NEUTRAL — {ls20_nonzero}/10 seeds, W_a symmetry breaking needs more"

    print(f"  LS20: L1={ls20_mean:.1f}/seed  {ls20_r}")
    print(f"  FT09: L1={ft09_mean:.1f}/seed  {ft09_r}")
    print(f"  VERDICT: {verdict}")

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("STEP 950 DONE")
