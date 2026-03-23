"""
step0895_pred_error_attention.py -- Prediction-Error Attention (PEA).

R3 hypothesis: per-dimension prediction error drives encoding self-modification.
The substrate discovers which observation dimensions are informative WITHOUT human
prescription. On FT09: should concentrate alpha on the 1.3% dynamic pixels.
On LS20: should remain roughly uniform (all dims change in grid-world).

Architecture:
- W: (256, 256+n_actions). Forward model on alpha-weighted encoding. Delta rule.
- alpha: (256,). Attention weights, start uniform. Updated from per-dim error history.
- running_mean: centering (standard pipeline).
- visited_set: per-obs hash(weighted_enc.tobytes()). Pass graph ban.

Process:
1. enc = avgpool16(obs) - running_mean  (256D centered)
2. weighted_enc = enc * alpha
3. Forward model: pred = W @ concat(weighted_enc, onehot(a))
4. Update W via delta rule on current step error
5. Update alpha from pred_errors deque (sqrt concentration, clamped to [0.01, 10])
6. Action = argmax(novelty + 0.1 * ||pred_next - weighted_enc||), eps=0.20 random

Alpha update: every step when pred_errors has >=50 samples.
  mean_errors = mean(pred_errors, axis=0)  # per-dimension
  raw_alpha = sqrt(mean_errors + 1e-8)
  alpha = raw_alpha / mean(raw_alpha)  # normalize
  alpha = clip(alpha, 0.01, 10.0)

Kill criterion: if max(alpha)/min(alpha) < 2.0 after 5K steps, signal too weak.

R3_cf protocol:
- Pretrain: seeds 1-5, 5K steps. W + alpha trained.
- Cold: fresh W + alpha (uniform), seeds 6-10.
- Warm: pretrained W + alpha (W_alpha transfer), fresh running_mean.
- Also: W_only transfer variant (reset alpha to uniform, keep W).
- Metric: L1 completions, pred accuracy (MSE), alpha concentration (max/min ratio).

FT09 CRITICAL: track alpha distribution at steps 1K, 5K per seed (runtime cap).
LS20: full R3_cf protocol, 10K steps per seed.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

PRETRAIN_SEEDS = list(range(1, 6))
TEST_SEEDS = list(range(6, 11))
PRETRAIN_STEPS_LS20 = 5_000
TEST_STEPS_LS20 = 10_000
PRETRAIN_STEPS_FT09 = 2_000   # runtime cap: 68 actions × steps
TEST_STEPS_FT09 = 5_000       # runtime cap
N_ACTIONS_LS20 = 4
N_ACTIONS_FT09 = 68
ETA_W = 0.01
ETA_ALPHA_DELAY = 50  # steps before alpha starts updating
ENC_DIM = 256
EPSILON = 0.20


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32)
    v[a] = 1.0
    return v


class PredErrorAttention895(BaseSubstrate):
    """Prediction-Error Attention substrate. alpha self-modifies based on prediction errors."""

    def __init__(self, n_actions=4, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._running_count = 0
        self._pred_errors = deque(maxlen=200)
        self.visited = set()
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None
        self._last_weighted_enc = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._running_count += 1
        lr = min(0.01, 1.0 / self._running_count)
        self._running_mean += lr * (enc_raw - self._running_mean)
        return enc_raw - self._running_mean

    def _encode_for_pred(self, obs):
        return _enc_frame(np.asarray(obs, dtype=np.float32)) - self._running_mean

    def _update_alpha(self):
        if len(self._pred_errors) < ETA_ALPHA_DELAY:
            return
        mean_errors = np.mean(self._pred_errors, axis=0)
        raw_alpha = np.sqrt(mean_errors + 1e-8)
        mean_raw = np.mean(raw_alpha)
        if mean_raw < 1e-8:
            return
        self.alpha = raw_alpha / mean_raw
        self.alpha = np.clip(self.alpha, 0.01, 10.0)

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc
        weighted_enc = enc * self.alpha
        self._last_weighted_enc = weighted_enc

        # Update W from previous step
        if self._prev_enc is not None and self._prev_action is not None:
            prev_weighted = self._prev_enc * self.alpha
            inp = np.concatenate([prev_weighted, one_hot(self._prev_action, self._n_actions)])
            pred = self.W @ inp
            error = weighted_enc - pred
            self.W -= ETA_W * np.outer(error, inp)
            self._pred_errors.append(np.abs(error))
            self._update_alpha()

        # Add current obs hash to visited
        obs_hash = hash(weighted_enc.tobytes())
        self.visited.add(obs_hash)

        # Action selection
        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            best_action = 0
            best_score = -1.0
            for a in range(self._n_actions):
                inp_a = np.concatenate([weighted_enc, one_hot(a, self._n_actions)])
                pred_next = self.W @ inp_a
                pred_hash = hash(pred_next.tobytes())
                novelty = 0.0 if pred_hash in self.visited else 1.0
                err_mag = float(np.linalg.norm(pred_next - weighted_enc))
                score = novelty + 0.1 * err_mag
                if score > best_score:
                    best_score = score
                    best_action = a
            action = best_action

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def alpha_concentration(self):
        """max(alpha)/min(alpha) ratio. >2.0 = signal detected."""
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def alpha_top_dims(self, k=10):
        """Indices of top-k alpha dimensions."""
        return np.argsort(self.alpha)[-k:][::-1].tolist()

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + self._n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._running_count = 0
        self._pred_errors = deque(maxlen=200)
        self.visited = set()
        self._prev_enc = None; self._prev_action = None
        self._last_enc = None; self._last_weighted_enc = None

    def on_level_transition(self):
        self.visited = set()
        self._prev_enc = None; self._prev_action = None

    def get_state(self):
        return {"W": self.W.copy(), "alpha": self.alpha.copy(),
                "running_mean": self._running_mean.copy(), "count": self._running_count}

    def set_state(self, s):
        self.W = s["W"].copy(); self.alpha = s["alpha"].copy()
        self._running_mean = s["running_mean"].copy()
        self._running_count = s["count"]

    def frozen_elements(self): return []


def make_ls20():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def make_ft09():
    try:
        import arcagi3; return arcagi3.make("FT09")
    except:
        import util_arcagi3; return util_arcagi3.make("FT09")


def run_phase(substrate, env_fn, n_actions, env_seed, n_steps, alpha_checkpoints=None):
    """Run substrate for n_steps. alpha_checkpoints: list of step counts to record alpha."""
    env = env_fn()
    obs = env.reset(seed=env_seed)
    step = 0; completions = 0; current_level = 0
    pred_errors_mse = []
    alpha_records = {}  # step -> alpha concentration

    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed)
            substrate.on_level_transition()
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)
        prev_enc = substrate._prev_enc
        prev_action = substrate._prev_action

        action = substrate.process(obs_arr) % n_actions
        obs_next, _, done, info = env.step(action)
        step += 1

        # Alpha checkpoint tracking
        if alpha_checkpoints and step in alpha_checkpoints:
            alpha_records[step] = substrate.alpha_concentration()

        # Pred accuracy (MSE-based)
        if prev_enc is not None and prev_action is not None and obs_next is not None:
            actual_enc = substrate._encode_for_pred(np.asarray(obs_next, dtype=np.float32))
            prev_weighted = prev_enc * substrate.alpha
            inp = np.concatenate([prev_weighted, one_hot(prev_action, substrate.n_actions)])
            pred = substrate.W @ inp
            actual_weighted = actual_enc * substrate.alpha
            err = float(np.sum((pred - actual_weighted)**2))
            norm = float(np.sum(actual_weighted**2)) + 1e-8
            pred_errors_mse.append((err, norm))

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            substrate.on_level_transition()
        if done:
            obs_next = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition()
        obs = obs_next

    pred_acc = None
    if pred_errors_mse:
        te = sum(e for e, n in pred_errors_mse)
        tn = sum(n for e, n in pred_errors_mse)
        pred_acc = float(1.0 - te / tn) * 100.0
    return completions, pred_acc, alpha_records


print("=" * 70)
print("STEP 895 — PREDICTION-ERROR ATTENTION (alpha self-modification)")
print("=" * 70)
print(f"alpha(256): concentration = sqrt(mean_pred_error_per_dim). Passes graph ban.")
print(f"W: linear forward model on alpha-weighted encoding. Delta rule, eta={ETA_W}.")
print(f"Action: argmax(novelty + 0.1 * pred_err_mag). eps={EPSILON}.")
print(f"Kill criterion: max(alpha)/min(alpha) < 2.0 at 5K steps.")

t0 = time.time()

# ======================== LS20 ========================
print(f"\n--- LS20 (n_actions={N_ACTIONS_LS20}, pretrain {PRETRAIN_STEPS_LS20}/seed, test {TEST_STEPS_LS20}/seed) ---")

sub_p_ls20 = PredErrorAttention895(n_actions=N_ACTIONS_LS20, seed=0)
sub_p_ls20.reset(0)
for ps in PRETRAIN_SEEDS:
    sub_p_ls20.on_level_transition()
    env = make_ls20(); obs = env.reset(seed=ps * 1000); s = 0
    while s < PRETRAIN_STEPS_LS20:
        if obs is None:
            obs = env.reset(seed=ps * 1000); sub_p_ls20.on_level_transition(); continue
        action = sub_p_ls20.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS_LS20
        obs, _, done, _ = env.step(action); s += 1
        if done:
            obs = env.reset(seed=ps * 1000); sub_p_ls20.on_level_transition()
saved_ls20 = sub_p_ls20.get_state()
print(f"  LS20 pretrain done ({time.time()-t0:.1f}s). alpha_conc={sub_p_ls20.alpha_concentration():.2f}")

cold_comps_ls20 = []; cold_accs_ls20 = []; cold_conc_ls20 = []
warm_comps_ls20 = []; warm_accs_ls20 = []; warm_conc_ls20 = []

for ts in TEST_SEEDS:
    # Cold
    sub_c = PredErrorAttention895(n_actions=N_ACTIONS_LS20, seed=0)
    sub_c.reset(0)
    c_comp, c_acc, _ = run_phase(sub_c, make_ls20, N_ACTIONS_LS20, ts * 1000, TEST_STEPS_LS20)
    cold_comps_ls20.append(c_comp); cold_accs_ls20.append(c_acc)
    cold_conc_ls20.append(sub_c.alpha_concentration())

    # Warm W+alpha transfer (fresh running_mean)
    sub_w = PredErrorAttention895(n_actions=N_ACTIONS_LS20, seed=0)
    sub_w.reset(0)
    sub_w.W = saved_ls20["W"].copy()
    sub_w.alpha = saved_ls20["alpha"].copy()
    w_comp, w_acc, _ = run_phase(sub_w, make_ls20, N_ACTIONS_LS20, ts * 1000, TEST_STEPS_LS20)
    warm_comps_ls20.append(w_comp); warm_accs_ls20.append(w_acc)
    warm_conc_ls20.append(sub_w.alpha_concentration())

mc_ls = np.mean(cold_comps_ls20); mw_ls = np.mean(warm_comps_ls20)
mc_acc_ls = np.mean([a for a in cold_accs_ls20 if a is not None]) if any(a is not None for a in cold_accs_ls20) else None
mw_acc_ls = np.mean([a for a in warm_accs_ls20 if a is not None]) if any(a is not None for a in warm_accs_ls20) else None
mc_conc_ls = np.mean(cold_conc_ls20); mw_conc_ls = np.mean(warm_conc_ls20)

print(f"  cold: L1={mc_ls:.1f}/seed  pred_acc={mc_acc_ls:.2f}%  alpha_conc={mc_conc_ls:.2f}")
print(f"  warm: L1={mw_ls:.1f}/seed  pred_acc={mw_acc_ls:.2f}%  alpha_conc={mw_conc_ls:.2f}")
if mc_acc_ls is not None and mw_acc_ls is not None:
    r3_pass = mw_acc_ls > mc_acc_ls
    print(f"  R3_cf (pred_acc): {'PASS' if r3_pass else 'FAIL'} ({mc_acc_ls:.2f}% → {mw_acc_ls:.2f}%, delta={mw_acc_ls-mc_acc_ls:+.2f}%)")
print(f"  Kill check: max_conc={max(cold_conc_ls20):.2f}. {'ALIVE' if max(cold_conc_ls20) >= 2.0 else 'KILLED (conc<2.0)'}")
print(f"  Pretrain alpha top dims: {sub_p_ls20.alpha_top_dims(5)}")

# ======================== FT09 ========================
print(f"\n--- FT09 (n_actions={N_ACTIONS_FT09}, test {TEST_STEPS_FT09}/seed, cold only) ---")
print(f"  [Alpha tracking at steps 1K, {TEST_STEPS_FT09//5}K, {TEST_STEPS_FT09//2}K, {TEST_STEPS_FT09}K]")

# FT09 is substrate_seed-varied (seed % 4) and cold-only (runtime cap with 68 actions)
ft09_concs = []; ft09_comps = []; ft09_accs = []
alpha_checkpoints_ft09 = {1000, TEST_STEPS_FT09//5, TEST_STEPS_FT09//2, TEST_STEPS_FT09}

for ts in range(1, 11):
    substrate_seed = ts % 4
    sub_ft = PredErrorAttention895(n_actions=N_ACTIONS_FT09, seed=substrate_seed)
    sub_ft.reset(substrate_seed)
    c_comp, c_acc, alpha_rec = run_phase(sub_ft, make_ft09, N_ACTIONS_FT09, ts * 1000,
                                          TEST_STEPS_FT09, alpha_checkpoints=alpha_checkpoints_ft09)
    ft09_comps.append(c_comp); ft09_accs.append(c_acc)
    ft09_concs.append(sub_ft.alpha_concentration())
    conc_at_1k = alpha_rec.get(1000, 0); conc_at_end = alpha_rec.get(TEST_STEPS_FT09, 0)
    print(f"  seed={ts:3d}: L1={c_comp:4d}  alpha_conc: 1K={conc_at_1k:.2f} → end={conc_at_end:.2f}  top_dims={sub_ft.alpha_top_dims(3)}")

valid_ft09_acc = [a for a in ft09_accs if a is not None]
mean_ft09_conc = np.mean(ft09_concs)
print(f"\n  FT09 Mean L1: {np.mean(ft09_comps):.1f}/seed")
print(f"  FT09 Mean alpha_conc: {mean_ft09_conc:.2f} (kill threshold: 2.0)")
if valid_ft09_acc:
    print(f"  FT09 Mean pred_acc: {np.mean(valid_ft09_acc):.2f}%")
print(f"  Kill check: {'ALIVE' if mean_ft09_conc >= 2.0 else 'KILLED (conc<2.0)'}")

print(f"\nBaseline: 800b cold=~300/seed (25K). Random=36.4/seed (25K). ~15/seed at 10K.")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 895 DONE")
