"""Step 958: Ensemble selection — accept 1/10, select winner from K=5 instances.

R3 hypothesis: If 1/10 bootstrap rate is structural, running K=5 parallel instances
for 2K steps and selecting the one with highest W_a row variance (most differentiated)
amplifies the bootstrap rate from 10% to 1-(0.9^5)=41%.

K=5 instances, BOOTSTRAP_STEPS=2000. Winner = highest W_a.var(axis=1).mean().
Continue winner for remaining 8K steps.
Total budget: 5*2K + 8K = 18K steps per seed (~1.8x normal budget).

Kill: LS20 ≤1/10. Success: 3+/10.
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
EPSILON = 0.20
K = 5
BOOTSTRAP_STEPS = 2000
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class HebbianBase:
    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)
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

    def w_a_row_var(self):
        return float(np.var(self.W_a, axis=1).mean())

    def w_a_norm(self):
        return float(np.linalg.norm(self.W_a))


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except Exception:
        import util_arcagi3; return util_arcagi3.make(name)


def run_ensemble(game_name, n_actions, seed, n_steps, k, bootstrap_steps):
    """Run K instances for bootstrap_steps, select winner, continue to n_steps."""
    instances = []
    envs = []
    obs_list = []

    # Init K instances with different sub-seeds
    for ki in range(k):
        sub = HebbianBase(n_actions=n_actions, seed=seed * 100 + ki)
        env = make_game(game_name)
        obs = env.reset(seed=seed * 1000)
        instances.append(sub)
        envs.append(env)
        obs_list.append(obs)

    # Run all K for bootstrap_steps
    step_counts = [0] * k
    completions = [0] * k
    current_levels = [0] * k

    for t in range(bootstrap_steps):
        for ki in range(k):
            if obs_list[ki] is None:
                obs_list[ki] = envs[ki].reset(seed=seed * 1000)
                instances[ki].on_level_transition()
                continue
            action = instances[ki].process(np.asarray(obs_list[ki], dtype=np.float32)) % n_actions
            obs_list[ki], _, done, info = envs[ki].step(action)
            step_counts[ki] += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > current_levels[ki]:
                completions[ki] += cl - current_levels[ki]
                current_levels[ki] = cl
                instances[ki].on_level_transition()
            if done:
                obs_list[ki] = envs[ki].reset(seed=seed * 1000)
                current_levels[ki] = 0
                instances[ki].on_level_transition()

    # Select winner: highest W_a row variance
    scores = [inst.w_a_row_var() for inst in instances]
    winner_idx = int(np.argmax(scores))
    winner = instances[winner_idx]
    winner_env = envs[winner_idx]
    winner_obs = obs_list[winner_idx]
    winner_completions = completions[winner_idx]
    winner_level = current_levels[winner_idx]

    # Continue winner for remaining steps
    remaining = n_steps - bootstrap_steps
    for t in range(remaining):
        if winner_obs is None:
            winner_obs = winner_env.reset(seed=seed * 1000)
            winner.on_level_transition()
            continue
        action = winner.process(np.asarray(winner_obs, dtype=np.float32)) % n_actions
        winner_obs, _, done, info = winner_env.step(action)
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > winner_level:
            winner_completions += cl - winner_level
            winner_level = cl
            winner.on_level_transition()
        if done:
            winner_obs = winner_env.reset(seed=seed * 1000)
            winner_level = 0
            winner.on_level_transition()

    return winner_completions, winner_idx, scores[winner_idx], winner.w_a_norm()


if __name__ == "__main__":
    import os, time

    print("=" * 70)
    print("STEP 958 — ENSEMBLE SELECTION (K=5, bootstrap 2K steps)")
    print("=" * 70)
    t0 = time.time()

    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), 'unknown')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), 'unknown')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}")
    print(f"K={K}  BOOTSTRAP_STEPS={BOOTSTRAP_STEPS}  total_budget={K*BOOTSTRAP_STEPS + (TEST_STEPS-BOOTSTRAP_STEPS)}")
    print()

    print("--- LS20 (4 actions, 10K budget) ---")
    ls20_r = []
    for seed in TEST_SEEDS:
        completions, winner_idx, var_score, w_norm = run_ensemble("LS20", 4, seed, TEST_STEPS, K, BOOTSTRAP_STEPS)
        ls20_r.append(completions)
        print(f"    seed={seed}: L1={completions:4d}  winner=k{winner_idx}  W_a_var={var_score:.4f}  W_a_norm={w_norm:.3f}")
    ls20_nz = sum(1 for x in ls20_r if x > 0)
    print(f"  LS20: L1={np.mean(ls20_r):.1f}/seed  nonzero={ls20_nz}/10  {ls20_r}")

    print()
    print("--- FT09 (68 actions, 10K budget) ---")
    ft09_r = []
    for seed in TEST_SEEDS:
        completions, winner_idx, var_score, w_norm = run_ensemble("FT09", 68, seed, TEST_STEPS, K, BOOTSTRAP_STEPS)
        ft09_r.append(completions)
        print(f"    seed={seed}: L1={completions:4d}  winner=k{winner_idx}  W_a_var={var_score:.4f}  W_a_norm={w_norm:.3f}")
    ft09_nz = sum(1 for x in ft09_r if x > 0)
    print(f"  FT09: L1={np.mean(ft09_r):.1f}/seed  nonzero={ft09_nz}/10  {ft09_r}")

    print()
    print("=" * 70)
    verdict = f"SUCCESS — {ls20_nz}/10" if ls20_nz >= 3 else f"KILL — {ls20_nz}/10"
    print(f"  VERDICT: {verdict}")
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("STEP 958 DONE")
