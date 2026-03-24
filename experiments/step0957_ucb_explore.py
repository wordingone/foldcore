"""Step 957: UCB action selection (principled exploration).

R3 hypothesis: UCB's provable exploration bounds ensure all actions are sampled
proportionally from every state, breaking specific-sequence dependence that makes
1/10 seeds lucky under epsilon-greedy.

Global N_a (action counts) — not per-state (ban-safe). UCB bonus decays naturally
as N_a grows. No epsilon, no RNG in action selection.

948 base + UCB replaces argmax+epsilon. W_h also trained per Leo spec.
Kill: LS20 ≤1/10. Success: 3+/10.
"""
import sys, math
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
ETA_PRED = 0.01
ETA_ACTION = 0.001
UCB_C = 2.0
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class UCBExplore957:
    """948 base + UCB action selection. Global N_a, no per-state data."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        rs = np.random.RandomState(seed + 10000)

        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.W_pred = np.zeros((ENC_DIM, H_DIM), dtype=np.float32)
        self.W_a = np.zeros((n_actions, H_DIM), dtype=np.float32)
        self.N_a = np.zeros(n_actions, dtype=np.float32)  # global action counts

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

    def _get_action(self):
        if self._step < self._n_actions:
            return self._step  # try each action once first
        action_scores = self.W_a @ self.h
        ucb_bonus = UCB_C * np.sqrt(math.log(self._step + 1) / (self.N_a + 1))
        return int(np.argmax(action_scores + ucb_bonus))

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
                self.W_pred += ETA_PRED * np.outer(error, self._prev_h)
                self.W_h += ETA_PRED * 0.1 * np.outer(error[:H_DIM], self._prev_h)
                self.W_a[self._prev_action] += ETA_ACTION * delta * self.h

        action = self._get_action()
        self.N_a[action] += 1
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
    for seed in seeds:
        sub = UCBExplore957(n_actions=n_actions, seed=seed)
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
        print(f"    seed={seed}: L1={completions:4d}  W_a_norm={sub.w_a_norm():.3f}  N_a={sub.N_a.astype(int).tolist()}")
    return results


if __name__ == "__main__":
    import os, time
    print("=" * 70)
    print("STEP 957 — UCB EXPLORATION")
    print("=" * 70)
    t0 = time.time()
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), 'unknown')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), 'unknown')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}")
    print(f"UCB_C={UCB_C}  ETA_PRED={ETA_PRED}  ETA_ACTION={ETA_ACTION}")
    print()

    print("--- LS20 (4 actions, 10K steps) ---")
    ls20_r = run_game("LS20", 4, TEST_SEEDS, TEST_STEPS)
    ls20_nz = sum(1 for x in ls20_r if x > 0)
    print(f"  LS20: L1={np.mean(ls20_r):.1f}/seed  nonzero={ls20_nz}/10  {ls20_r}")

    print()
    print("--- FT09 (68 actions, 10K steps) ---")
    ft09_r = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS)
    ft09_nz = sum(1 for x in ft09_r if x > 0)
    print(f"  FT09: L1={np.mean(ft09_r):.1f}/seed  nonzero={ft09_nz}/10  {ft09_r}")

    print()
    print("=" * 70)
    verdict = f"SUCCESS — {ls20_nz}/10" if ls20_nz >= 3 else f"KILL — {ls20_nz}/10"
    print(f"  VERDICT: {verdict}")
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("STEP 957 DONE")
