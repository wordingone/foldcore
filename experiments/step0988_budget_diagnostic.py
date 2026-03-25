"""
step0988_budget_diagnostic.py -- Budget diagnostic: does 800b converge for 68 actions at 50K?

FAMILY: Diagnostic (mirrors Step 969 FT09 + Step 982 VC33, extended budget)
R3 HYPOTHESIS: None — pure diagnostic. 800b needs N steps to cover N actions. At 25K
(Step 969): FT09=0/10. Does 50K change anything?

Unmodified 965 (Chain916, h-reset, W_pred -=). Standalone per game.
LS20=10K (control). FT09=50K. VC33=50K. 10 seeds each.

Jun approved extended diagnostic budget.
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
TEST_SEEDS = list(range(1, 11))


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp; x -= np.max(x)
    e = np.exp(x); probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class Chain916_988:
    """Exact 965: W_pred -=, change-tracking delta, h-reset."""
    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
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
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._prev_ext = None; self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1; a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
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
        if self._prev_ext is not None and self._prev_action is not None:
            pred = self.W_pred @ self._prev_ext
            error = (ext_enc * self.alpha) - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0: error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W * np.outer(error, self._prev_ext)
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


def run_game(game, n_actions, seeds, n_steps):
    results = []
    for seed in seeds:
        sub = Chain916_988(n_actions=n_actions, seed=seed)
        env = make_env(game); obs = env.reset(seed=seed * 1000)
        step = 0; completions = 0; level = 0
        while step < n_steps:
            if obs is None: obs = env.reset(seed=seed * 1000); sub.on_level_transition(); continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
            obs, _, done, info = env.step(action); step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                completions += cl - level; level = cl; sub.on_level_transition()
            if done: obs = env.reset(seed=seed * 1000); level = 0; sub.on_level_transition()
        results.append(completions)
        print(f"  seed={seed}: completions={completions:4d}  alpha_conc={sub.alpha_conc():.2f}")
    return results


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("STEP 988 — BUDGET DIAGNOSTIC (965 standalone, 50K FT09/VC33)")
    print("=" * 70)
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), '?')
    vc33_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/vc33') if len(d) >= 8), '?')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}  VC33={vc33_hash}")
    print(f"Unmodified 965. Standalone. LS20=10K, FT09=50K, VC33=50K. 10 seeds.")
    print(f"Ref: Step 969 FT09@25K=0/10. Step 982 VC33@25K=0/10.")
    print()

    t0 = time.time()

    print("--- LS20 (4 actions, 10K) [control] ---")
    ls = run_game("LS20", 4, TEST_SEEDS, 10_000)
    print(f"  LS20@10K: mean={np.mean(ls):.1f}  nonzero={sum(1 for x in ls if x>0)}/10  {ls}")

    print()
    print("--- FT09 (68 actions, 50K) ---")
    ft = run_game("FT09", 68, TEST_SEEDS, 50_000)
    print(f"  FT09@50K: mean={np.mean(ft):.1f}  nonzero={sum(1 for x in ft if x>0)}/10  {ft}")

    print()
    print("--- VC33 (68 actions, 50K) ---")
    vc = run_game("VC33", 68, TEST_SEEDS, 50_000)
    print(f"  VC33@50K: mean={np.mean(vc):.1f}  nonzero={sum(1 for x in vc if x>0)}/10  {vc}")

    print()
    print("=" * 70)
    print("STEP 988 RESULTS:")
    print(f"  LS20@10K: mean={np.mean(ls):.1f}  nonzero={sum(1 for x in ls if x>0)}/10  {ls}")
    print(f"  FT09@50K: mean={np.mean(ft):.1f}  nonzero={sum(1 for x in ft if x>0)}/10  {ft}")
    print(f"  VC33@50K: mean={np.mean(vc):.1f}  nonzero={sum(1 for x in vc if x>0)}/10  {vc}")
    if any(x > 0 for x in ft):
        print(f"  FT09: BUDGET-LIMITED at 50K — signal found!")
    else:
        print(f"  FT09: MECHANISM-LIMITED confirmed at 50K (0/10 at 25K AND 50K)")
    if any(x > 0 for x in vc):
        print(f"  VC33: BUDGET-LIMITED at 50K — signal found!")
    else:
        print(f"  VC33: MECHANISM-LIMITED confirmed at 50K (0/10 at 25K AND 50K)")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("STEP 988 DONE")
