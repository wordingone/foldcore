"""
step0972_action_conditioned_wpred.py -- Action-conditioned W_pred forward model.

FAMILY: Action-conditioned prediction (forward model on 916 base)
R3 HYPOTHESIS: Conditioning W_pred on prev_action creates a per-action forward model.
delta[a] = ||enc - W_pred @ [h, enc_ext, one_hot(a)]|| = prediction error for action a.
800b running_mean tracks per-action prediction quality. Actions with inaccurate
predictions (novel territory) have high delta → softmax selection drives exploration.
W_pred generalizes across states (parametric matrix), not per-(state,action) table.

One change from 916: W_pred input gains one_hot(prev_action).
- 916 W_pred: (320, 320) predicts ext_enc from prev_ext
- 972 W_pred: (256, 64+320+n_actions) predicts enc from [h, enc_ext, one_hot(a)]

Ban-safe: W_pred is a parametric model, not per-(state,action) data structure.
Running_mean is per-action global (same as 916). No per-state data.

Kill: LS20 < 290.7 (action conditioning hurts) OR FT09 = 0/10.
Success: FT09 > 0 (first post-ban FT09 signal).

Run: FT09 (68 actions) + LS20 (4 actions). 25K, 10 seeds, cold start.
"""
import sys, time
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
ALPHA_LO = 0.10
ALPHA_HI = 5.00
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


def softmax_action(delta, temp, rng):
    x = delta / temp
    x = x - np.max(x)
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class ActionConditionedPred972:
    """916 + action-conditioned W_pred: predicts enc from [h, enc_ext, one_hot(action)]."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Fixed random reservoir (same as 916)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # Action-conditioned forward model
        # Input: [h(64), enc_ext(320), one_hot(n_actions)] → output: enc(256)
        pred_input_dim = H_DIM + EXT_DIM + n_actions
        self.W_pred = np.zeros((ENC_DIM, pred_input_dim), dtype=np.float32)

        # Alpha on enc (same as 916, applied to prediction target)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)

        # 800b running mean per action (delta = prediction error)
        self.running_mean = np.full(n_actions, INIT_DELTA, dtype=np.float32)

        # Recurrent state
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean_enc = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

        self._prev_h = None
        self._prev_ext = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean_enc = (1 - a) * self._running_mean_enc + a * enc_raw
        enc = enc_raw - self._running_mean_enc
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        ext_enc = np.concatenate([enc, self.h]).astype(np.float32)
        return enc, ext_enc

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
        enc, ext_enc = self._encode(obs)

        if self._prev_h is not None and self._prev_action is not None:
            # Action-conditioned prediction: predict enc from [prev_h, prev_ext, one_hot(prev_a)]
            one_hot_a = np.zeros(self._n_actions, dtype=np.float32)
            one_hot_a[self._prev_action] = 1.0
            pred_input = np.concatenate([self._prev_h, self._prev_ext, one_hot_a])
            pred = self.W_pred @ pred_input  # (256,)

            # Error vs alpha-weighted enc
            error = (enc * self.alpha) - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
                err_norm = 10.0
            if not np.any(np.isnan(error)):
                # Gradient descent: minimize ||target - pred||^2
                self.W_pred += ETA_W * np.outer(error, pred_input)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            # 800b: delta = prediction error (action-conditioned novelty signal)
            delta = err_norm
            a = self._prev_action
            self.running_mean[a] = (1 - ALPHA_EMA) * self.running_mean[a] + ALPHA_EMA * delta

        # Softmax action selection (high running_mean = high novelty = preferred)
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_action(self.running_mean, SOFTMAX_TEMP, self._rng)

        self._prev_h = self.h.copy()
        self._prev_ext = ext_enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_h = None
        self._prev_ext = None
        self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def running_mean_max(self):
        return float(np.max(self.running_mean))

    def running_mean_spread(self):
        return float(np.max(self.running_mean) - np.min(self.running_mean))


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except Exception:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps):
    results = []
    concs = []
    for seed in seeds:
        sub = ActionConditionedPred972(n_actions=n_actions, seed=seed)
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
        print(f"    seed={seed}: L1={completions:4d}  alpha_conc={sub.alpha_conc():.2f}"
              f"  rm_spread={sub.running_mean_spread():.4f}")
    return results, concs


print("=" * 70)
print("STEP 972 — ACTION-CONDITIONED W_pred (forward model per action)")
print("=" * 70)
print("W_pred: (256, 64+320+n_actions). Predicts enc from [h, enc_ext, one_hot(a)].")
print("delta = prediction error. 800b softmax over novelty. 916 base + action conditioning.")
print("FT09 + LS20, 25K, 10 seeds cold.")
t0 = time.time()

# FT09 (68 actions)
print("\n--- FT09 (68 actions, 25K, 10 seeds) ---")
ft09_results, ft09_concs = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS)
ft09_mean = np.mean(ft09_results)
ft09_std = np.std(ft09_results)
ft09_zeros = sum(1 for x in ft09_results if x == 0)
print(f"  FT09: L1={ft09_mean:.1f}/seed  std={ft09_std:.1f}  zero={ft09_zeros}/10  alpha_conc={np.mean(ft09_concs):.2f}")
print(f"  {ft09_results}")

# LS20 (4 actions)
print("\n--- LS20 (4 actions, 25K, 10 seeds) ---")
ls20_results, ls20_concs = run_game("LS20", 4, TEST_SEEDS, TEST_STEPS)
ls20_mean = np.mean(ls20_results)
ls20_std = np.std(ls20_results)
ls20_zeros = sum(1 for x in ls20_results if x == 0)
print(f"  LS20: L1={ls20_mean:.1f}/seed  std={ls20_std:.1f}  zero={ls20_zeros}/10  alpha_conc={np.mean(ls20_concs):.2f}")
print(f"  {ls20_results}")

print(f"\n{'='*70}")
print(f"STEP 972 RESULTS (916@25K: LS20=290.7/0/10, FT09=0.0/10):")
print(f"  FT09: L1={ft09_mean:.1f}/seed  zero={ft09_zeros}/10  (baseline: 0.0/10)")
print(f"  LS20: L1={ls20_mean:.1f}/seed  zero={ls20_zeros}/10  (baseline: 290.7/0/10)")

if ft09_zeros < 10 and ls20_mean >= 290.7 * 0.9:
    verdict = f"SUCCESS — FT09 signal ({10-ft09_zeros}/10) AND LS20={ls20_mean:.1f} (≥90% baseline)"
elif ft09_zeros < 10:
    verdict = f"FT09-SIGNAL — {10-ft09_zeros}/10 nonzero but LS20={ls20_mean:.1f} degraded"
elif ls20_mean >= 290.7 * 0.9:
    verdict = f"LS20-ONLY — LS20={ls20_mean:.1f} holds but FT09=0"
else:
    verdict = f"KILL — LS20={ls20_mean:.1f} (<261.6) and FT09=0"
print(f"  VERDICT: {verdict}")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 972 DONE")
