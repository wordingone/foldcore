"""
step0996_fast_adapt_sweep.py -- Sweep FAST_ADAPT_STEPS: 200 vs 500 vs 1000.

FAMILY: Adaptive learning rate (994 base, duration sweep)
R3 HYPOTHESIS: Is 500-step fast adapt optimal? Test 200/500/1000 steps.
All use same 994 mechanism: h-novelty > 2x EMA → fast phase at 1.5x ETA_W.
Standalone LS20@10K, 10 seeds each variant.

994 result: FAST_ADAPT_STEPS=500 → LS20=83.8 (new chain best).
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256; H_DIM = 64; EXT_DIM = ENC_DIM + H_DIM
ETA_W = 0.01; ALPHA_EMA = 0.10; INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50; ALPHA_LO = 0.10; ALPHA_HI = 5.00
EPSILON = 0.20; SOFTMAX_TEMP = 0.10
ETA_H_EMA = 0.50; H_NOV_EMA = 0.99; SPIKE_THRESHOLD = 2.0; FAST_ETA_FACTOR = 1.5
TEST_SEEDS = list(range(1, 11))


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp; x -= np.max(x)
    e = np.exp(x); probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class FastAdaptVariant:
    def __init__(self, seed, fast_adapt_steps):
        self._fast_adapt_steps = fast_adapt_steps
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM), dtype=np.float32)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._n_actions = 4
        self.delta_per_action = np.full(4, INIT_DELTA, dtype=np.float32)
        self._prev_ext = None; self._prev_action = None
        self._h_ema = np.zeros(H_DIM, dtype=np.float32)
        self._h_novelty_ema = 1.0
        self._fast_adapt_countdown = 0

    def set_game(self, n_actions):
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._n_actions = n_actions
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._prev_ext = None; self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1; a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        self._h_ema = (1 - ETA_H_EMA) * self._h_ema + ETA_H_EMA * self.h
        return np.concatenate([enc, self.h]).astype(np.float32)

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY: return
        mean_errors = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(mean_errors)) or np.any(np.isinf(mean_errors)): return
        raw_alpha = np.sqrt(np.clip(mean_errors, 0, 1e6) + 1e-8)
        mean_raw = np.mean(raw_alpha)
        if mean_raw < 1e-8 or np.isnan(mean_raw): return
        self.alpha = np.clip(raw_alpha / mean_raw, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        ext_enc = self._encode(obs)
        h_novelty = float(np.linalg.norm(self.h - self._h_ema))
        self._h_novelty_ema = H_NOV_EMA * self._h_novelty_ema + (1 - H_NOV_EMA) * h_novelty
        if h_novelty > SPIKE_THRESHOLD * self._h_novelty_ema:
            self._fast_adapt_countdown = self._fast_adapt_steps
        eta_adaptive = ETA_W * FAST_ETA_FACTOR if self._fast_adapt_countdown > 0 else ETA_W
        if self._fast_adapt_countdown > 0: self._fast_adapt_countdown -= 1

        if self._prev_ext is not None and self._prev_action is not None:
            pred = self.W_pred @ self._prev_ext
            error = (ext_enc * self.alpha) - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0: error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W_pred -= eta_adaptive * np.outer(error, self._prev_ext)
                self._pred_errors.append(np.abs(error)); self._update_alpha()
            weighted_delta = (ext_enc - self._prev_ext) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (1 - ALPHA_EMA) * self.delta_per_action[a] + ALPHA_EMA * change
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_sel(self.delta_per_action, SOFTMAX_TEMP, self._rng)
        self._prev_ext = ext_enc.copy(); self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None; self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))


def make_env(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except Exception:
        import util_arcagi3; return util_arcagi3.make(name)


def run_ls20(fast_adapt_steps, seeds, n_steps=10_000):
    results = []
    for seed in seeds:
        sub = FastAdaptVariant(seed=seed, fast_adapt_steps=fast_adapt_steps)
        sub.set_game(4)
        env = make_env("LS20"); obs = env.reset(seed=seed * 1000)
        step = 0; completions = 0; level = 0
        while step < n_steps:
            if obs is None: obs = env.reset(seed=seed * 1000); sub.on_level_transition(); continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % 4
            obs, _, done, info = env.step(action); step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level: completions += cl - level; level = cl; sub.on_level_transition()
            if done: obs = env.reset(seed=seed * 1000); level = 0; sub.on_level_transition()
        results.append(completions)
    return results


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("STEP 996 — FAST ADAPT DURATION SWEEP (200 / 500 / 1000 steps)")
    print("=" * 70)
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    print(f"Game version: LS20={ls20_hash}. Standalone LS20@10K. 10 seeds each.")
    print(f"994 reference: FAST_ADAPT_STEPS=500 → LS20=83.8")
    print()

    t0 = time.time()
    for steps in [200, 500, 1000]:
        print(f"--- FAST_ADAPT_STEPS={steps} ---")
        res = run_ls20(steps, TEST_SEEDS)
        print(f"  LS20: mean={np.mean(res):.1f}  nonzero={sum(1 for x in res if x>0)}/10  {res}")
        print()

    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("STEP 996 DONE")
