"""Step 946: Additive action bias — augment 916 prediction without disruption.

R3 hypothesis: Additive action-specific prediction bias increases discriminative
capacity without disrupting the alpha feedback loop. W_action captures
game-agnostic action-consequence patterns. Remaining error after W_action
correction concentrates alpha on NOVEL action outcomes — not all outcomes.

Mechanism: Keep 916 exactly as-is. Add W_action @ a_onehot as correction
to 916's prediction. When W_action=0 (initial state), this IS 916.

Kill: LS20 < 72.7 (916 baseline at 10K) → KILL (must not regress).
Chain kill on LS20 + FT09.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM   # 320
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


def softmax_action(delta, temp, rng):
    x = delta / temp
    x = x - np.max(x)
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class AdditiveActionBias946:
    """916 + additive W_action bias. 916 internals untouched.

    W_action @ a_onehot adds a per-action prediction correction.
    Both W_pred and W_action updated from combined error.
    At init: W_action=0 → identical to 916.
    """

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Fixed random recurrent weights (never trained)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # 916's W_pred — unchanged
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM + n_actions), dtype=np.float32)

        # Additive action bias: EXT_DIM × n_actions, init=0 (→ IS 916 at start)
        self.W_action = np.zeros((EXT_DIM, n_actions), dtype=np.float32)

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
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY:
            return
        mean_errors = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(mean_errors)) or np.any(np.isinf(mean_errors)):
            return
        raw_alpha = np.sqrt(np.clip(mean_errors, 0, 1e6) + 1e-8)
        mean_raw = np.mean(raw_alpha)
        if mean_raw < 1e-8 or np.isnan(mean_raw):
            return
        self.alpha = raw_alpha / mean_raw
        self.alpha = np.clip(self.alpha, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        ext_enc = self._encode(obs)

        if self._prev_ext is not None and self._prev_action is not None:
            a_vec = one_hot(self._prev_action, self._n_actions)

            # 916's W_pred (UNCHANGED): input = [alpha*prev_ext, action_onehot]
            inp = np.concatenate([self._prev_ext * self.alpha, a_vec])
            pred_916 = self.W_pred @ inp

            # Additive action bias (NEW): per-action correction, no state key
            pred_bonus = self.W_action @ a_vec

            # Combined prediction
            pred = pred_916 + pred_bonus

            # Error from combined prediction
            error = (ext_enc * self.alpha) - pred

            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                # Update W_pred (916's rule — unchanged)
                self.W_pred -= ETA_W * np.outer(error, inp)
                # Update W_action from same error
                self.W_action -= ETA_W * np.outer(error, a_vec)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            # 800b: unchanged from 916
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

    def w_action_norm(self):
        """Frobenius norm of W_action — how much has it learned?"""
        return float(np.linalg.norm(self.W_action))


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps):
    results = []
    concs = []
    w_norms = []
    for seed in seeds:
        sub = AdditiveActionBias946(n_actions=n_actions, seed=seed)
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
        concs.append(sub.alpha_conc())
        w_norms.append(sub.w_action_norm())
        print(f"    seed={seed}: L1={completions:4d}  alpha_conc={sub.alpha_conc():.2f}  W_action_norm={sub.w_action_norm():.3f}")
    return results, concs, w_norms


if __name__ == "__main__":
    import os
    import time

    print("=" * 70)
    print("STEP 946 — ADDITIVE ACTION BIAS")
    print("=" * 70)
    print("916 kept intact. W_action @ a_onehot added as prediction correction.")
    print("Base: 916. 10K steps. LS20 + FT09 chain kill check.")
    t0 = time.time()

    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), 'unknown')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), 'unknown')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}")
    print()

    print("--- LS20 (4 actions, 10K steps) ---")
    ls20_results, ls20_concs, ls20_norms = run_game("LS20", 4, TEST_SEEDS, TEST_STEPS)
    ls20_mean = np.mean(ls20_results)
    ls20_std = np.std(ls20_results)
    ls20_zeros = sum(1 for x in ls20_results if x == 0)
    print(f"  LS20: L1={ls20_mean:.1f}/seed  std={ls20_std:.1f}  zero={ls20_zeros}/10  alpha_conc={np.mean(ls20_concs):.2f}")
    print(f"  W_action_norm={np.mean(ls20_norms):.3f}  {ls20_results}")

    print("\n--- FT09 (68 actions, 10K steps) ---")
    ft09_results, ft09_concs, ft09_norms = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS)
    ft09_mean = np.mean(ft09_results)
    ft09_std = np.std(ft09_results)
    ft09_zeros = sum(1 for x in ft09_results if x == 0)
    print(f"  FT09: L1={ft09_mean:.1f}/seed  std={ft09_std:.1f}  zero={ft09_zeros}/10  alpha_conc={np.mean(ft09_concs):.2f}")
    print(f"  W_action_norm={np.mean(ft09_norms):.3f}  {ft09_results}")

    print(f"\n{'='*70}")
    print(f"STEP 946 RESULTS:")
    print(f"  LS20: L1={ls20_mean:.1f}/seed  (916@10K baseline: 72.7)")
    print(f"  FT09: L1={ft09_mean:.1f}/seed  (916@10K: 0.0)")
    print(f"  alpha_conc: LS20={np.mean(ls20_concs):.2f}  FT09={np.mean(ft09_concs):.2f}")
    print(f"  W_action learned: LS20={np.mean(ls20_norms):.3f}  FT09={np.mean(ft09_norms):.3f}")

    if ls20_mean < 72.7:
        print(f"  KILL: LS20 regressed below 916@10K baseline")
    elif ft09_mean > 0:
        if ls20_mean >= 72.7:
            print(f"  PASS: LS20 stable, FT09 improved")
        else:
            print(f"  CHAIN KILL: FT09 improved but LS20 degraded")
    else:
        print(f"  PASS: LS20 maintains baseline. FT09 still 0.")

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("STEP 946 DONE")
