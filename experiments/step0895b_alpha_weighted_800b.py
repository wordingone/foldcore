"""
step0895b_alpha_weighted_800b.py -- Alpha-Weighted 800b.

R3 hypothesis: alpha self-modification (from step895) makes 800b's EMA mechanism
sensitive to informative dimensions. Alpha discovers WHICH dims change productively.
800b discovers WHICH action produces the most change IN THOSE DIMS.

Architecture:
- self.alpha: (256,) attention weights. Updated from per-dim W prediction errors.
  Same alpha update as step895 (sqrt concentration, clamped [0.01, 10]).
- W: (256, 256+n_actions) forward model. Delta rule. For alpha update only.
- delta_per_action: (n_actions,) EMA of alpha-weighted change per action.
  `weighted_change_a = ||( enc_t - enc_{t-1}) * alpha||^2`
  `delta[a] = (1-ALPHA) * delta[a] + ALPHA * weighted_change_a`
- Action: argmax(delta_per_action) with epsilon=0.20.

This is the simplest combination of step895's R3 (alpha) and step800b's mechanism (EMA convergence).
Step895 found alpha concentrates on FT09's dynamic dims. Alpha-weighted 800b should converge
to the action that produces change IN THOSE DIMS (= the productive click).

R3_cf protocol: W + alpha transfer (warm), fresh running_mean. Cold: fresh all.
Seeds 1-5 pretrain (W + alpha) 5K steps. Seeds 6-10 test 10K steps cold/warm.
Metric: L1, alpha concentration (max/min ratio), R3_cf.
Also run FT09 (cold only, 5K steps, 10 seeds, varied substrate_seed).
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
PRETRAIN_STEPS = 5_000
TEST_STEPS = 10_000
N_ACTIONS_LS20 = 4
N_ACTIONS_FT09 = 68
ETA_W = 0.01
ALPHA_EMA = 0.10          # EMA for delta_per_action (same as 800b)
INIT_DELTA = 1.0          # INIT_DELTA for delta_per_action (same as 800b)
ALPHA_UPDATE_DELAY = 50   # steps before alpha starts updating
ENC_DIM = 256
EPSILON = 0.20


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


class AlphaWeighted800b(BaseSubstrate):
    """Alpha attention weights + 800b EMA per-action delta mechanism."""

    def __init__(self, n_actions=4, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        # Forward model for alpha update
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        # Attention weights
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        # 800b EMA per-action delta (on ALPHA-WEIGHTED change)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self._running_mean = (1 - alpha) * self._running_mean + alpha * enc_raw
        return enc_raw - self._running_mean

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
        self.alpha = np.clip(self.alpha, 0.01, 10.0)

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc

        if self._prev_enc is not None and self._prev_action is not None:
            # Update W (for alpha update)
            inp = np.concatenate([self._prev_enc * self.alpha, one_hot(self._prev_action, self._n_actions)])
            pred = self.W @ inp
            error = (enc * self.alpha) - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            # Update delta_per_action with alpha-weighted change
            weighted_change = float(np.sum(((enc - self._prev_enc) * self.alpha) ** 2))
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a] +
                                         ALPHA_EMA * weighted_change)

        # Action selection: argmax(delta_per_action) with epsilon
        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = int(np.argmax(self.delta_per_action))

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def alpha_concentration(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def alpha_top_dims(self, k=5):
        return np.argsort(self.alpha)[-k:][::-1].tolist()

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + self._n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.delta_per_action = np.full(self._n_actions, INIT_DELTA, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def get_state(self):
        return {"W": self.W.copy(), "alpha": self.alpha.copy(),
                "delta": self.delta_per_action.copy(),
                "running_mean": self._running_mean.copy(), "n_obs": self._n_obs}

    def set_state(self, s):
        self.W = s["W"].copy(); self.alpha = s["alpha"].copy()
        self.delta_per_action = s["delta"].copy()
        self._running_mean = s["running_mean"].copy(); self._n_obs = s["n_obs"]

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


def run_phase(substrate, env_fn, n_actions, env_seed, n_steps):
    env = env_fn(); obs = env.reset(seed=env_seed)
    step = 0; completions = 0; current_level = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); substrate.on_level_transition(); continue
        action = substrate.process(np.asarray(obs, dtype=np.float32)) % n_actions
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            substrate.on_level_transition()
        if done:
            obs = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition()
    return completions


print("=" * 70)
print("STEP 895b — ALPHA-WEIGHTED 800b (alpha self-mod + EMA convergence)")
print("=" * 70)
print(f"Alpha: sqrt(mean_per_dim_pred_error). EMA delta: ||( enc_t - enc_{{t-1}}) * alpha||^2.")
print(f"Action: argmax(alpha-weighted EMA delta). eps={EPSILON}.")
print(f"R3_cf: W+alpha transfer (warm), fresh running_mean. Pretrain {PRETRAIN_STEPS}/seed.")

t0 = time.time()

# ======================== LS20 ========================
print(f"\n--- LS20 (n_actions={N_ACTIONS_LS20}, test {TEST_STEPS}/seed) ---")

sub_p = AlphaWeighted800b(n_actions=N_ACTIONS_LS20, seed=0)
sub_p.reset(0)
for ps in PRETRAIN_SEEDS:
    sub_p.on_level_transition()
    env = make_ls20(); obs = env.reset(seed=ps * 1000); s = 0
    while s < PRETRAIN_STEPS:
        if obs is None:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition(); continue
        action = sub_p.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS_LS20
        obs, _, done, _ = env.step(action); s += 1
        if done:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition()
saved = sub_p.get_state()
print(f"  Pretrain done ({time.time()-t0:.1f}s). alpha_conc={sub_p.alpha_concentration():.2f} top_dims={sub_p.alpha_top_dims(3)}")
print(f"  delta_per_action={sub_p.delta_per_action.tolist()}")

cold_comps = []; warm_comps = []
cold_concs = []; warm_concs = []

for ts in TEST_SEEDS:
    sub_c = AlphaWeighted800b(n_actions=N_ACTIONS_LS20, seed=0)
    sub_c.reset(0)
    c_comp = run_phase(sub_c, make_ls20, N_ACTIONS_LS20, ts * 1000, TEST_STEPS)
    cold_comps.append(c_comp); cold_concs.append(sub_c.alpha_concentration())

    sub_w = AlphaWeighted800b(n_actions=N_ACTIONS_LS20, seed=0)
    sub_w.reset(0)
    sub_w.W = saved["W"].copy(); sub_w.alpha = saved["alpha"].copy()
    # Keep delta_per_action from pretrain too (it converged to meaningful values)
    sub_w.delta_per_action = saved["delta"].copy()
    w_comp = run_phase(sub_w, make_ls20, N_ACTIONS_LS20, ts * 1000, TEST_STEPS)
    warm_comps.append(w_comp); warm_concs.append(sub_w.alpha_concentration())

mc = np.mean(cold_comps); mw = np.mean(warm_comps)
mc_conc = np.mean(cold_concs); mw_conc = np.mean(warm_concs)
print(f"  cold: L1={mc:.1f}/seed  alpha_conc={mc_conc:.2f}  ({cold_comps})")
print(f"  warm: L1={mw:.1f}/seed  alpha_conc={mw_conc:.2f}  ({warm_comps})")
print(f"  L1 delta (warm-cold): {mw-mc:+.1f}/seed")
print(f"  vs 800b baseline: cold~300/seed (25K), warm~300/seed (10K~120/seed)")

# ======================== FT09 (cold only) ========================
print(f"\n--- FT09 (n_actions={N_ACTIONS_FT09}, test {min(TEST_STEPS, 5000)}/seed, cold, varied substrate_seed) ---")
ft09_comps = []; ft09_concs = []

for ts in range(1, 11):
    substrate_seed = ts % 4
    sub_ft = AlphaWeighted800b(n_actions=N_ACTIONS_FT09, seed=substrate_seed)
    sub_ft.reset(substrate_seed)
    c_comp = run_phase(sub_ft, make_ft09, N_ACTIONS_FT09, ts * 1000, min(TEST_STEPS, 5000))
    ft09_comps.append(c_comp); ft09_concs.append(sub_ft.alpha_concentration())
    print(f"  seed={ts:3d}: L1={c_comp:4d}  alpha_conc={sub_ft.alpha_concentration():.2f}  top_dims={sub_ft.alpha_top_dims(3)}")

print(f"\n  FT09 Mean L1: {np.mean(ft09_comps):.1f}/seed  alpha_conc: {np.mean(ft09_concs):.2f}")

print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 895b DONE")
