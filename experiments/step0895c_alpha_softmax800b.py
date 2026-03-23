"""
step0895c_alpha_softmax800b.py -- Alpha-800b with Softmax T=0.1 (Correct Spec).

R3 hypothesis: alpha self-modification (from step895) + PROVEN 800b softmax navigation.
Decoupled: prediction error drives alpha (R3). Softmax(T=0.1) drives action selection.

vs step895b (argmax): argmax collapses to single action once delta diverges slightly.
  softmax T=0.1 maintains exploration proportional to delta differences.

Architecture:
- alpha: (256,) attention weights from per-dim prediction error. Same as step895.
- W: (256, 256+n_actions) forward model. Delta rule with gradient clip. Alpha update only.
- delta_per_action: EMA of ||(enc_t - enc_{t-1}) * alpha|| (L2 norm, not squared).
  Mail 2617 spec: change = norm(weighted_enc - prev_enc * alpha).
- Action: softmax(delta_per_action / T=0.1) with epsilon=0.20. THE proven 800b mechanism.

R3_cf protocol:
- Pretrain seeds 1-5: 5K steps, accumulate W + alpha.
- Warm: transfer W+alpha. Reset: running_mean, delta_per_action=1.0, n_obs=0.
- Cold: fresh all. Test seeds 6-10, 10K steps.
- FT09: cold only, 5K steps, 10 seeds, substrate_seed=seed%4.

Critical: warm alpha starts concentrated on informative dims immediately.
Cold alpha starts uniform and must re-discover dims from scratch.
If warm reconcentrates FASTER -> R3 transfer confirmed.
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
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
ENC_DIM = 256
EPSILON = 0.20
SOFTMAX_TEMP = 0.10


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


def softmax_action(delta, temp):
    x = delta / temp
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


class AlphaSoftmax800b(BaseSubstrate):
    """Alpha attention (R3) + 800b softmax navigation (decoupled)."""

    def __init__(self, n_actions=4, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
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
            # Update W (for alpha update only)
            inp = np.concatenate([self._prev_enc * self.alpha,
                                   one_hot(self._prev_action, self._n_actions)])
            pred = self.W @ inp
            error = (enc * self.alpha) - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            # Delta: L2 norm of alpha-weighted change (mail 2617 spec)
            weighted_delta = (enc - self._prev_enc) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * change)

        # Action: softmax(T=0.1) — proven 800b navigation mechanism
        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            probs = softmax_action(self.delta_per_action, SOFTMAX_TEMP)
            action = int(self._rng.choice(self._n_actions, p=probs))

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


def run_phase(substrate, env_fn, n_actions, env_seed, n_steps,
              alpha_checkpoints=None):
    """Run n_steps, optionally recording alpha_conc at checkpoints."""
    env = env_fn(); obs = env.reset(seed=env_seed)
    step = 0; completions = 0; current_level = 0
    alpha_log = {}
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); substrate.on_level_transition(); continue
        action = substrate.process(np.asarray(obs, dtype=np.float32)) % n_actions
        obs, _, done, info = env.step(action); step += 1
        if alpha_checkpoints and step in alpha_checkpoints:
            alpha_log[step] = substrate.alpha_concentration()
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            substrate.on_level_transition()
        if done:
            obs = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition()
    return completions, alpha_log


print("=" * 70)
print("STEP 895c — ALPHA-800b SOFTMAX (CORRECT SPEC: T=0.1, L2 delta norm)")
print("=" * 70)
print(f"Alpha: sqrt(per-dim pred_error). Delta: L2 norm of (enc_delta * alpha).")
print(f"Action: softmax(delta/T=0.1) + eps={EPSILON}. Decoupled R3 + navigation.")
print(f"R3_cf: W+alpha warm transfer, fresh running_mean+delta.")

t0 = time.time()
ALPHA_CKS = {1000, 5000, 10000}

# ======================== LS20 ========================
print(f"\n--- LS20 (n_actions={N_ACTIONS_LS20}) ---")

# Pretrain
sub_p = AlphaSoftmax800b(n_actions=N_ACTIONS_LS20, seed=0)
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
print(f"  Pretrain done ({time.time()-t0:.1f}s). alpha_conc={sub_p.alpha_concentration():.2f} top={sub_p.alpha_top_dims(3)}")
print(f"  delta_per_action={sub_p.delta_per_action.tolist()}")

cold_comps = []; warm_comps = []
cold_concs = []; warm_concs = []
cold_conc_series = []; warm_conc_series = []

for ts in TEST_SEEDS:
    # Cold
    sub_c = AlphaSoftmax800b(n_actions=N_ACTIONS_LS20, seed=ts % 4)
    sub_c.reset(ts % 4)
    c_comp, c_alpha = run_phase(sub_c, make_ls20, N_ACTIONS_LS20, ts * 1000,
                                 TEST_STEPS, ALPHA_CKS)
    cold_comps.append(c_comp); cold_concs.append(sub_c.alpha_concentration())
    cold_conc_series.append(c_alpha)

    # Warm: W + alpha transfer, fresh running_mean + delta
    sub_w = AlphaSoftmax800b(n_actions=N_ACTIONS_LS20, seed=ts % 4)
    sub_w.reset(ts % 4)
    sub_w.W = saved["W"].copy()
    sub_w.alpha = saved["alpha"].copy()
    # running_mean, delta_per_action, n_obs all stay FRESH (reset)
    w_comp, w_alpha = run_phase(sub_w, make_ls20, N_ACTIONS_LS20, ts * 1000,
                                 TEST_STEPS, ALPHA_CKS)
    warm_comps.append(w_comp); warm_concs.append(sub_w.alpha_concentration())
    warm_conc_series.append(w_alpha)

mc = np.mean(cold_comps); mw = np.mean(warm_comps)
print(f"  cold: L1={mc:.1f}/seed  alpha_conc={np.mean(cold_concs):.2f}  {cold_comps}")
print(f"  warm: L1={mw:.1f}/seed  alpha_conc={np.mean(warm_concs):.2f}  {warm_comps}")
print(f"  L1 delta (warm-cold): {mw-mc:+.1f}/seed")

# Alpha reconcentration speed (warm vs cold)
for ts_i, ts in enumerate(TEST_SEEDS):
    c_s = cold_conc_series[ts_i]; w_s = warm_conc_series[ts_i]
    c_str = " ".join(f"@{k}:{v:.2f}" for k, v in sorted(c_s.items()))
    w_str = " ".join(f"@{k}:{v:.2f}" for k, v in sorted(w_s.items()))
    print(f"  seed {ts}: cold alpha_conc [{c_str}] | warm [{w_str}]")

# ======================== FT09 (cold only) ========================
print(f"\n--- FT09 (n_actions={N_ACTIONS_FT09}, cold only, 5K steps, varied substrate_seed) ---")
ft09_comps = []; ft09_concs = []
FT09_STEPS = min(TEST_STEPS, 5000)

for ts in range(1, 11):
    sub_seed = ts % 4
    sub_ft = AlphaSoftmax800b(n_actions=N_ACTIONS_FT09, seed=sub_seed)
    sub_ft.reset(sub_seed)
    c_comp, _ = run_phase(sub_ft, make_ft09, N_ACTIONS_FT09, ts * 1000, FT09_STEPS)
    ft09_comps.append(c_comp); ft09_concs.append(sub_ft.alpha_concentration())
    print(f"  seed={ts:3d}: L1={c_comp:4d}  alpha_conc={sub_ft.alpha_concentration():.2f}  top={sub_ft.alpha_top_dims(3)}")

print(f"\n  FT09 mean L1: {np.mean(ft09_comps):.1f}/seed  alpha_conc: {np.mean(ft09_concs):.2f}")

print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 895c DONE")
