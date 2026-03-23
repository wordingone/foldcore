"""
step0895h_clamped_neff10.py -- 895e rerun with substrate_seed=seed (n_eff=10).

895e (n_eff=4, sub_seed=seed%4) showed: warm=309.7/seed, cold=278.9/seed, zero=0/10.
This run fixes the n_eff=4 confound: substrate_seed=seed gives n_eff=10 unique configs.

R3 hypothesis: clamped alpha (0.1-5.0) with alpha-weighted delta outperforms raw L2 norm.
  868d (raw L2 norm, n_eff=10): 203.9/seed — TRUE baseline.
  895e (clamped alpha, n_eff=4): warm=309.7/seed — 52% over baseline but n_eff=4.
  895h (clamped alpha, n_eff=10): expected warm>203.9 if alpha helps navigation.

Architecture: identical to step895e (alpha = clip(sqrt(mean_err), 0.1, 5.0)).
Only change: sub_seed = ts (not ts % 4).

Protocol: LS20, 25K, 10 seeds (sub_seed=seed). Pretrain seeds 1-5 (5K).
Key metric: warm vs cold vs 868d baseline (203.9/seed).
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

PRETRAIN_SEEDS = list(range(1, 6))
TEST_SEEDS = list(range(1, 11))
PRETRAIN_STEPS = 5_000
TEST_STEPS = 25_000
N_ACTIONS = 4
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
ENC_DIM = 256
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


def softmax_action(delta, temp):
    x = delta / temp
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


class AlphaClamped895h(BaseSubstrate):
    """Identical to 895e. Only sub_seed=seed (n_eff=10) in test loop."""

    def __init__(self, n_actions=4, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._seed = seed
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self._alpha_pred_acc = deque(maxlen=200)
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
        self.alpha = np.clip(self.alpha, ALPHA_LO, ALPHA_HI)

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc

        if self._prev_enc is not None and self._prev_action is not None:
            inp = np.concatenate([self._prev_enc * self.alpha,
                                   one_hot(self._prev_action, self._n_actions)])
            pred = self.W @ inp
            error = (enc * self.alpha) - pred

            alpha_enc_norm = float(np.linalg.norm(enc * self.alpha)) + 1e-8
            self._alpha_pred_acc.append(1.0 - float(np.linalg.norm(error)) / alpha_enc_norm)

            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            weighted_delta = (enc - self._prev_enc) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * change)

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

    def alpha_pred_acc_mean(self):
        if not self._alpha_pred_acc:
            return None
        return float(np.mean(list(self._alpha_pred_acc)[-100:]))

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + self._n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.delta_per_action = np.full(self._n_actions, INIT_DELTA, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self._alpha_pred_acc = deque(maxlen=200)
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


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


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
print("STEP 895h — CLAMPED ALPHA (0.1-5.0) WITH n_eff=10 (substrate_seed=seed)")
print("=" * 70)
print(f"Fixes 895e n_eff=4 confound. Confirms: clamped alpha > raw L2 norm (203.9)?")
print(f"25K steps, 10 seeds, sub_seed=seed (not seed%4). Pretrain 5 seeds 5K.")

t0 = time.time()

sub_p = AlphaClamped895h(n_actions=N_ACTIONS, seed=0)
sub_p.reset(0)
for ps in PRETRAIN_SEEDS:
    sub_p.on_level_transition()
    env = make_game(); obs = env.reset(seed=ps * 1000); s = 0
    while s < PRETRAIN_STEPS:
        if obs is None:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition(); continue
        action = sub_p.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        obs, _, done, _ = env.step(action); s += 1
        if done:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition()
saved = sub_p.get_state()
print(f"  Pretrain done ({time.time()-t0:.1f}s). alpha_conc={sub_p.alpha_concentration():.2f} top={sub_p.alpha_top_dims(3) if hasattr(sub_p, 'alpha_top_dims') else 'N/A'}")

cold_comps = []; warm_comps = []
cold_concs = []; warm_concs = []
cold_pa = []; warm_pa = []

for ts in TEST_SEEDS:
    sub_seed = ts  # ← FIXED: substrate_seed=seed, n_eff=10

    sub_c = AlphaClamped895h(n_actions=N_ACTIONS, seed=sub_seed)
    sub_c.reset(sub_seed)
    c_comp = run_phase(sub_c, make_game, N_ACTIONS, ts * 1000, TEST_STEPS)
    cold_comps.append(c_comp); cold_concs.append(sub_c.alpha_concentration())
    cold_pa.append(sub_c.alpha_pred_acc_mean())

    sub_w = AlphaClamped895h(n_actions=N_ACTIONS, seed=sub_seed)
    sub_w.reset(sub_seed)
    sub_w.W = saved["W"].copy()
    sub_w.alpha = saved["alpha"].copy()
    w_comp = run_phase(sub_w, make_game, N_ACTIONS, ts * 1000, TEST_STEPS)
    warm_comps.append(w_comp); warm_concs.append(sub_w.alpha_concentration())
    warm_pa.append(sub_w.alpha_pred_acc_mean())
    print(f"  seed={ts}: cold L1={c_comp:4d} warm L1={w_comp:4d}  cold_conc={sub_c.alpha_concentration():.2f}")

mc = np.mean(cold_comps); mw = np.mean(warm_comps)
sc = np.std(cold_comps); sw = np.std(warm_comps)
zc = sum(1 for x in cold_comps if x == 0)
zw = sum(1 for x in warm_comps if x == 0)
mcp = np.mean([p for p in cold_pa if p is not None]) if any(p is not None for p in cold_pa) else 0.0
mwp = np.mean([p for p in warm_pa if p is not None]) if any(p is not None for p in warm_pa) else 0.0
print(f"\ncold: L1={mc:.1f}/seed  std={sc:.1f}  zero={zc}/{len(cold_comps)}  alpha_conc={np.mean(cold_concs):.2f}  pred_acc={mcp:.3f}")
print(f"      {cold_comps}")
print(f"warm: L1={mw:.1f}/seed  std={sw:.1f}  zero={zw}/{len(warm_comps)}  alpha_conc={np.mean(warm_concs):.2f}  pred_acc={mwp:.3f}")
print(f"      {warm_comps}")
print(f"L1 delta: {mw-mc:+.1f}/seed")
print(f"\nComparison (true baselines):")
print(f"  868b (squared-sum, n_eff=10):        72.1/seed  std=112  zero=5/10  [wrong metric]")
print(f"  868d (raw L2 norm, n_eff=10):       203.9/seed  std=106  zero=1/10  [TRUE baseline]")
print(f"  895e cold (clamped, n_eff=4):        278.9/seed std=62   zero=0/10")
print(f"  895e warm (clamped, n_eff=4):        309.7/seed std=32.5 zero=0/10")
print(f"  895h cold (clamped, n_eff=10):      {mc:.1f}/seed std={sc:.1f} zero={zc}/{len(cold_comps)}")
print(f"  895h warm (clamped, n_eff=10):      {mw:.1f}/seed std={sw:.1f} zero={zw}/{len(warm_comps)}")
if mw > 203.9:
    print(f"  CONFIRMED: clamped alpha outperforms raw L2 norm (895h warm {mw:.1f} > 868d 203.9)")
else:
    print(f"  NOT CONFIRMED: 895h warm {mw:.1f} <= 868d 203.9 (n_eff=4 result was artifact)")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 895h DONE")
