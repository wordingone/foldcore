"""
step0923_vc33_diagnostic.py -- VC33 diagnostic with 916 architecture.

R3 hypothesis: Recurrent trajectory encoding (916) + clamped alpha + 800b may help VC33
where 895h failed (L1=0, Step 914). VC33 is zone-based (not tile-puzzle like FT09).
Zone transitions should produce visible observation changes → 800b change-tracking MAY work.

VC33 metadata: 7 baseline_actions = [6,13,31,59,92,24,82]. Full gym: N_ACTIONS=68 (from chain test).
Leo: "7 actions (vs 68)" — VC33 is potentially more tractable than FT09.

Architecture: Step 916 (recurrent h, echo-state) + clamped alpha + 800b.
Run: VC33 only, 25K, 10 seeds cold. Also baseline with 895h (cold) for comparison.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM
N_ACTIONS = 68  # VC33 full gym space (from step 914)
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


def softmax_action(delta, temp, rng):
    x = delta / temp; x -= np.max(x); e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class Recurrent916:
    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM + n_actions), dtype=np.float32)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        self._prev_ext = None; self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1-a)*self._running_mean + a*enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        return np.concatenate([enc, self.h]).astype(np.float32)

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY: return
        me = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(me)) or np.any(np.isinf(me)): return
        ra = np.sqrt(np.clip(me, 0, 1e6) + 1e-8)
        mr = np.mean(ra)
        if mr < 1e-8 or np.isnan(mr): return
        self.alpha = np.clip(ra / mr, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        ext = self._encode(obs)
        if self._prev_ext is not None and self._prev_action is not None:
            inp = np.concatenate([self._prev_ext * self.alpha,
                                   one_hot(self._prev_action, self._n_actions)])
            pred = self.W_pred @ inp
            error = (ext * self.alpha) - pred
            en = float(np.linalg.norm(error))
            if en > 10.0: error *= 10.0/en
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()
            weighted_delta = (ext - self._prev_ext) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (1-ALPHA_EMA)*self.delta_per_action[a] + ALPHA_EMA*change
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_action(self.delta_per_action, SOFTMAX_TEMP, self._rng)
        self._prev_ext = ext.copy(); self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None; self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def top_delta_actions(self, k=7):
        return list(np.argsort(self.delta_per_action)[-k:])


class Clamped895h:
    """895h cold (comparison baseline) for VC33."""
    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        self._prev_enc = None; self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1; a = 1.0/self._n_obs
        self._running_mean = (1-a)*self._running_mean + a*enc_raw
        return enc_raw - self._running_mean

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY: return
        me = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(me)) or np.any(np.isinf(me)): return
        ra = np.sqrt(np.clip(me, 0, 1e6) + 1e-8); mr = np.mean(ra)
        if mr < 1e-8 or np.isnan(mr): return
        self.alpha = np.clip(ra / mr, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        enc = self._encode(obs)
        if self._prev_enc is not None and self._prev_action is not None:
            inp = np.concatenate([self._prev_enc * self.alpha,
                                   one_hot(self._prev_action, self._n_actions)])
            pred = self.W @ inp; error = (enc * self.alpha) - pred
            en = float(np.linalg.norm(error))
            if en > 10.0: error *= 10.0/en
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error)); self._update_alpha()
            weighted_delta = (enc - self._prev_enc) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (1-ALPHA_EMA)*self.delta_per_action[a] + ALPHA_EMA*change
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_action(self.delta_per_action, SOFTMAX_TEMP, self._rng)
        self._prev_enc = enc.copy(); self._prev_action = action
        return action

    def on_level_transition(self): self._prev_enc = None; self._prev_action = None
    def alpha_conc(self): return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))


VC33_BASELINE = [6, 13, 31, 59, 92, 24, 82]


def make_game():
    try:
        import arcagi3; return arcagi3.make("VC33")
    except:
        import util_arcagi3; return util_arcagi3.make("VC33")


def run_game(SubClass, seeds, n_steps):
    results = []; concs = []
    for seed in seeds:
        sub = SubClass(n_actions=N_ACTIONS, seed=seed)
        env = make_game()
        obs = env.reset(seed=seed * 1000)
        step = 0; completions = 0; current_level = 0
        while step < n_steps:
            if obs is None:
                obs = env.reset(seed=seed * 1000); sub.on_level_transition(); continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
            obs, _, done, info = env.step(action)
            step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > current_level:
                completions += (cl - current_level); current_level = cl
                sub.on_level_transition()
            if done:
                obs = env.reset(seed=seed * 1000); current_level = 0
                sub.on_level_transition()
        results.append(completions); concs.append(sub.alpha_conc())
        top7 = sub.top_delta_actions(7) if hasattr(sub, 'top_delta_actions') else []
        overlap = [a for a in top7 if a in VC33_BASELINE]
        print(f"    seed={seed}: L1={completions:4d}  alpha_conc={sub.alpha_conc():.2f}  "
              f"top7_delta={top7}  baseline_overlap={overlap}")
    return results, concs


print("=" * 70)
print("STEP 923 — VC33 DIAGNOSTIC (916 recurrent + 895h comparison)")
print("=" * 70)
print(f"VC33 baseline_actions={VC33_BASELINE}  N_ACTIONS={N_ACTIONS}")
print(f"VC33 is zone-based (not tile-puzzle). 800b may work if zones produce visible change.")
t0 = time.time()

print("\n--- 916 Recurrent h (ext_enc=320D) ---")
r916, c916 = run_game(Recurrent916, TEST_SEEDS, TEST_STEPS)
m916 = np.mean(r916); z916 = sum(1 for x in r916 if x == 0)
print(f"  916: L1={m916:.1f}/seed  std={np.std(r916):.1f}  zero={z916}/10  alpha_conc={np.mean(c916):.2f}")
print(f"  {r916}")

print("\n--- 895h cold (256D baseline) ---")
r895, c895 = run_game(Clamped895h, TEST_SEEDS, TEST_STEPS)
m895 = np.mean(r895); z895 = sum(1 for x in r895 if x == 0)
print(f"  895h: L1={m895:.1f}/seed  std={np.std(r895):.1f}  zero={z895}/10  alpha_conc={np.mean(c895):.2f}")
print(f"  {r895}")

print(f"\n{'='*70}")
print(f"STEP 923 RESULTS (VC33 diagnostic):")
print(f"  916 recurrent: L1={m916:.1f}/seed  zero={z916}/10")
print(f"  895h cold:     L1={m895:.1f}/seed  zero={z895}/10")
print(f"  Prior 895h (Step 914 chain, 5 seeds): 0.0/seed 5/5 zeros")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 923 DONE")
