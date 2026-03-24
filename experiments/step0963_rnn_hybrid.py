"""Step 963: 916 + Trainable RNN trajectory state.

FAMILY: 916-hybrid (800b action, trainable RNN encoding)

R3 HYPOTHESIS: Trainable recurrent W_h provides adaptive trajectory context that
fixed-random W_h (916) cannot. If W_h learns to emphasize dimensions that track
sequential progress, alpha concentration improves → 800b navigates better.

916 baseline: Fixed W_h/W_x (tanh), ext_enc=[enc, h] (320D), W_pred(320,320+n),
softmax over delta_per_action (temp=0.1), alpha from pred errors. 25K steps → 290.7.

963 changes from 916:
- W_h TRAINED: W_h += ETA_W * 0.01 * outer(error[:H_DIM], h) [vs fixed in 916]
- sigmoid activation [vs tanh in 916]
- enc_extended = [alpha * enc, h] (alpha applied in concat)
- W_pred (256, 320): predicts enc from enc_extended [vs 916's 320→320+n]
- delta = norm(error) for 800b [vs alpha-weighted obs change in 916]
- argmin action selection [vs softmax in 916]

KILL: LS20 < 916@10K fresh baseline.
SUCCESS: LS20 > 916@10K → trainable W_h adds trajectory context.

NOTE: 916 was measured at 25K steps. Kill criterion uses fresh 916@10K baseline.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM  # 320
ETA_W = 0.01
ETA_ACTION = 0.001
ALPHA_LO = 0.10
ALPHA_HI = 5.00
ALPHA_UPDATE_DELAY = 50
INIT_DELTA = 1.0
EPSILON = 0.20
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class RNNHybrid963:
    """800b action + trained recurrent h in enc_extended. One change from 916: W_h trained."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Recurrent weights — W_h TRAINED (smaller init), W_x fixed
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.01
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.01

        # Forward model: predicts enc (256D) from enc_extended (320D)
        self.W_pred = np.zeros((ENC_DIM, EXT_DIM), dtype=np.float32)

        # Alpha on enc dims (256D, applied when constructing enc_extended)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)

        # 800b change tracking
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)

        # State
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._prev_h = None
        self._prev_enc_ext = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = sigmoid(self.W_h @ self.h + self.W_x @ enc)
        enc_ext = np.concatenate([self.alpha * enc, self.h]).astype(np.float32)
        return enc, enc_ext

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY:
            return
        mean_errors = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(mean_errors)) or np.any(np.isinf(mean_errors)):
            return
        raw_alpha = np.sqrt(np.clip(mean_errors, 0, 1e6) + 1e-8)
        mean_raw = np.mean(raw_alpha)
        if mean_raw < 1e-8 or np.isnan(mean_raw):
            return
        self.alpha = np.clip(raw_alpha / mean_raw, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        enc, enc_ext = self._encode(obs)

        if self._prev_enc_ext is not None and self._prev_action is not None:
            pred = self.W_pred @ self._prev_enc_ext
            error = enc - pred  # 256D
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
                err_norm = 10.0
            if not np.any(np.isnan(error)):
                self.W_pred += ETA_W * np.outer(error, self._prev_enc_ext)
                # Train W_h from first H_DIM dims of error
                self.W_h += ETA_W * 0.01 * np.outer(error[:H_DIM], self._prev_h)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

                # 800b: EMA of prediction error norm per action
                a = self._prev_action
                self.delta_per_action[a] = (0.99 * self.delta_per_action[a]
                                             + 0.01 * err_norm)

        # Action: argmin of delta_per_action + epsilon
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = int(np.argmin(self.delta_per_action))

        self._prev_h = self.h.copy()
        self._prev_enc_ext = enc_ext.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_h = None
        self._prev_enc_ext = None
        self._prev_action = None
        # h persists across transitions (trajectory context)

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def w_a_norm(self):
        return float(np.linalg.norm(self.delta_per_action))


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except Exception:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps):
    results = []
    for seed in seeds:
        sub = RNNHybrid963(n_actions=n_actions, seed=seed)
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
        print(f"    seed={seed}: L1={completions:4d}  alpha_conc={sub.alpha_conc():.2f}")
    return results


if __name__ == "__main__":
    import os, time
    print("=" * 70)
    print("STEP 963 — RNN HYBRID (800b + trainable W_h trajectory state)")
    print("=" * 70)
    t0 = time.time()
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), 'unknown')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), 'unknown')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}")
    print(f"ETA_W={ETA_W}  EPSILON={EPSILON}  TEST_STEPS={TEST_STEPS}")
    print(f"Architecture: enc_ext=[alpha*enc, h] (320D), W_pred(256,320), argmin 800b")
    print()

    print("--- LS20 (4 actions, 10K steps) ---")
    ls20_r = run_game("LS20", 4, TEST_SEEDS, TEST_STEPS)
    ls20_mean = np.mean(ls20_r)
    ls20_nz = sum(1 for x in ls20_r if x > 0)
    print(f"  LS20: L1={ls20_mean:.1f}/seed  nonzero={ls20_nz}/10  {ls20_r}")

    print()
    print("--- FT09 (68 actions, 10K steps) ---")
    ft09_r = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS)
    ft09_mean = np.mean(ft09_r)
    ft09_nz = sum(1 for x in ft09_r if x > 0)
    print(f"  FT09: L1={ft09_mean:.1f}/seed  nonzero={ft09_nz}/10  {ft09_r}")

    print()
    print("=" * 70)
    print(f"  LS20: {ls20_mean:.1f}/seed  FT09: {ft09_mean:.1f}/seed")
    print(f"  (Compare to 916@25K: LS20=290.7. Fresh 916@10K baseline needed.)")
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("STEP 963 DONE")
