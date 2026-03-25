"""
step0971_two_stream.py -- Two-stream action selection: separate exploration + exploitation.

FAMILY: Sequential credit (two-stream variant)
R3 HYPOTHESIS: 800b exploration stream (running_mean, untouched) and eligibility
trace exploitation stream (trace_score, separate buffer) can coexist. When LAMBDA
is small, exploration dominates early; trace_score biases toward recently-successful
sequences after interesting events. Two signals don't interfere.

Combined: action_score[a] = -running_mean[a] + LAMBDA * trace_score[a]
argmax(action_score) = exploration when trace_score flat, exploitation when trace active.

Kill: LS20 < 290.7 (traces still corrupt exploration even when separate).
Success: FT09 > 0 AND LS20 ≈ 290.7 (two streams coexist).

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
TRACE_DECAY = 0.9
TRACE_SCORE_DECAY = 0.999
TRACE_CREDIT_LR = 0.01
DELTA_PERCENTILE = 90
LAMBDA = 0.1   # exploitation weight (small — exploration dominates early)
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


class TwoStream971:
    """916 + two-stream action selection: 800b exploration + trace exploitation."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Fixed random reservoir (same as 916)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # Trainable forward model (same as 916)
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM), dtype=np.float32)

        # Alpha on ext_enc (same as 916)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)

        # Stream 1: 800b exploration (NEVER modified by traces)
        self.running_mean = np.full(n_actions, INIT_DELTA, dtype=np.float32)

        # Stream 2: eligibility exploitation (SEPARATE)
        self.trace_score = np.zeros(n_actions, dtype=np.float32)
        self.e = np.zeros(n_actions, dtype=np.float32)
        self._delta_history = []

        # Recurrent state
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean_enc = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

        self._prev_ext = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean_enc = (1 - a) * self._running_mean_enc + a * enc_raw
        enc = enc_raw - self._running_mean_enc
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
        self.alpha = np.clip(raw_alpha / mean_raw, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        ext_enc = self._encode(obs)

        if self._prev_ext is not None and self._prev_action is not None:
            # Forward model update (same as 916)
            pred = self.W_pred @ self._prev_ext
            error = (ext_enc * self.alpha) - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W * np.outer(error, self._prev_ext)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            # 800b delta on ext_enc
            weighted_delta = (ext_enc - self._prev_ext) * self.alpha
            delta = float(np.linalg.norm(weighted_delta))
            a = self._prev_action

            # Stream 1: 800b exploration (UNTOUCHED by traces)
            self.running_mean[a] = (1 - ALPHA_EMA) * self.running_mean[a] + ALPHA_EMA * delta

            # Stream 2: eligibility trace exploitation
            self.e *= TRACE_DECAY
            self.e[a] += 1.0
            self._delta_history.append(delta)

            # Credit recent actions on large delta
            if len(self._delta_history) > 100:
                threshold = np.percentile(self._delta_history, DELTA_PERCENTILE)
                if delta > threshold:
                    self.trace_score += TRACE_CREDIT_LR * self.e

            # Slow decay of trace_score to prevent saturation
            self.trace_score *= TRACE_SCORE_DECAY

        # Combined action score: exploration + exploitation
        action_score = -self.running_mean + LAMBDA * self.trace_score

        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = int(np.argmax(action_score))

        self._prev_ext = ext_enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None
        self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def trace_max(self):
        return float(np.max(self.trace_score))


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except Exception:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps):
    results = []
    concs = []
    for seed in seeds:
        sub = TwoStream971(n_actions=n_actions, seed=seed)
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
        print(f"    seed={seed}: L1={completions:4d}  alpha_conc={sub.alpha_conc():.2f}  trace_max={sub.trace_max():.4f}")
    return results, concs


print("=" * 70)
print("STEP 971 — TWO-STREAM (separate 800b exploration + trace exploitation)")
print("=" * 70)
print("Stream1: running_mean (800b, UNTOUCHED). Stream2: trace_score (separate).")
print("action_score = -running_mean + LAMBDA * trace_score. argmax selection.")
print(f"LAMBDA={LAMBDA}  TRACE_DECAY={TRACE_DECAY}  TRACE_SCORE_DECAY={TRACE_SCORE_DECAY}")
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
print(f"STEP 971 RESULTS (916@25K: LS20=290.7/0/10, FT09=0.0/10):")
print(f"  FT09: L1={ft09_mean:.1f}/seed  zero={ft09_zeros}/10  (baseline: 0.0/10)")
print(f"  LS20: L1={ls20_mean:.1f}/seed  zero={ls20_zeros}/10  (baseline: 290.7/0/10)")

if ft09_zeros < 10 and ls20_mean >= 290.7 * 0.9:
    verdict = f"SUCCESS — FT09={10-ft09_zeros}/10 nonzero AND LS20={ls20_mean:.1f} (≥90% baseline)"
elif ft09_zeros < 10:
    verdict = f"FT09-ONLY — signal {10-ft09_zeros}/10 but LS20={ls20_mean:.1f} degraded"
elif ls20_mean >= 290.7 * 0.9:
    verdict = f"LS20-ONLY — LS20={ls20_mean:.1f} holds but FT09=0"
else:
    verdict = f"KILL — LS20={ls20_mean:.1f} (<261.6) and FT09=0"
print(f"  VERDICT: {verdict}")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 971 DONE")
