"""
step0970_eligibility_traces.py -- Global eligibility traces for sequential credit.

FAMILY: Sequential credit (new mechanism on 916 base)
R3 HYPOTHESIS: Eligibility traces assign credit to action SEQUENCES, not individual
actions. If correct actions produce intermediate visual changes, traces discover
sequences progressively. When a large delta occurs (top 10%), recent actions
(high e) get credit → running_mean decreases → selected more often.

One addition to 916: global eligibility e[a] (per-action, not per-(state,action)).
On large delta events, credit recent actions proportional to e[a].

Note: Leo's spec shows argmin action selection; 916 uses softmax. Implemented
with argmin as specified. Softmax variant can be tested if argmin kills.

Kill: FT09=0/10 at 25K AND LS20 < 290.7 (traces hurt navigation).
Success: FT09 > 0 on any seed (first FT09 signal post-ban).
Diagnostic: LS20 improves but FT09 stays 0 → traces help but FT09 needs more.

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
DELTA_PERCENTILE = 90
TRACE_CREDIT_LR = 0.1
TRACE_MIN = 0.01
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


class EligibilityTraces970:
    """916 + global eligibility traces for sequential credit assignment."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Fixed random reservoir (same as 916)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # Trainable forward model on extended encoding (same as 916)
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM), dtype=np.float32)

        # Alpha on ext_enc (persistent)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)

        # 800b running mean per action (delta tracking)
        self.running_mean = np.full(n_actions, INIT_DELTA, dtype=np.float32)

        # Global eligibility trace (per-action, not per-(state,action))
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

            # Standard 800b EMA update
            self.running_mean[a] = (1 - ALPHA_EMA) * self.running_mean[a] + ALPHA_EMA * delta

            # Eligibility trace: decay all, boost prev_action
            self.e *= TRACE_DECAY
            self.e[a] += 1.0

            # Delta percentile threshold — credit sequences on large deltas
            self._delta_history.append(delta)
            if len(self._delta_history) > 100:
                threshold = np.percentile(self._delta_history, DELTA_PERCENTILE)
                if delta > threshold:
                    # Credit recent actions proportionally to eligibility
                    for ai in range(self._n_actions):
                        if self.e[ai] > TRACE_MIN:
                            self.running_mean[ai] -= TRACE_CREDIT_LR * self.e[ai]
                    # Prevent negative running_mean
                    np.clip(self.running_mean, 0.0, None, out=self.running_mean)

        # Action selection: argmin (as per spec)
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = int(np.argmin(self.running_mean))

        self._prev_ext = ext_enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None
        self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def e_max(self):
        return float(np.max(self.e))


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except Exception:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps):
    results = []
    concs = []
    for seed in seeds:
        sub = EligibilityTraces970(n_actions=n_actions, seed=seed)
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
        print(f"    seed={seed}: L1={completions:4d}  alpha_conc={sub.alpha_conc():.2f}  e_max={sub.e_max():.3f}")
    return results, concs


print("=" * 70)
print("STEP 970 — ELIGIBILITY TRACES (sequential credit assignment)")
print("=" * 70)
print("R3: global e[a] decays per step, +1 for prev_action. On top-10% delta,")
print("credit recent actions: running_mean[a] -= TRACE_CREDIT_LR * e[a].")
print("916 base + traces. FT09 + LS20, 25K, 10 seeds cold.")
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
print(f"STEP 970 RESULTS (916@25K: LS20=290.7/0/10, FT09=0.0/10):")
print(f"  FT09: L1={ft09_mean:.1f}/seed  zero={ft09_zeros}/10  (baseline: 0.0/10)")
print(f"  LS20: L1={ls20_mean:.1f}/seed  zero={ls20_zeros}/10  (baseline: 290.7/0/10)")

if ft09_zeros < 10:
    verdict = f"SUCCESS — FT09 signal: {10-ft09_zeros}/10 nonzero seeds"
elif ls20_mean >= 290.7:
    verdict = f"LS20-ONLY — traces help LS20 ({ls20_mean:.1f}) but FT09 still 0"
elif ft09_zeros == 10 and ls20_mean < 290.7:
    verdict = f"KILL — LS20={ls20_mean:.1f} (<290.7) and FT09=0"
else:
    verdict = f"PARTIAL"
print(f"  VERDICT: {verdict}")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 970 DONE")
