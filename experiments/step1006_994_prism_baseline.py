"""
step1006_994_prism_baseline.py — 994 baseline through randomized PRISM.

FAMILY: Baseline establishment (infrastructure)
R3 HYPOTHESIS: N/A — this establishes the reference point for all future chain_kill verdicts.
  994 (fast-adapt h-novelty) is the best known substrate. Run it through randomized PRISM
  (C mode: Split-CIFAR-100 × 2, LS20, FT09, VC33) with game order shuffled per seed.
  Save as chain_results/baseline_994.json.

KILL: N/A (this IS the baseline)
SUCCESS: completes all 5 phases × 10 seeds, saves baseline_994.json
BUDGET: 10K steps/game, 10 seeds, 5 phases (~50K steps total)

Jun directive 2026-03-24: game order randomized, names hidden during run.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame
from substrates.chain import ChainRunner, make_prism

# 994 hyperparameters (frozen)
ENC_DIM = 256; H_DIM = 64; EXT_DIM = ENC_DIM + H_DIM
ETA_W = 0.01; ALPHA_EMA = 0.10; INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50; ALPHA_LO = 0.10; ALPHA_HI = 5.00
EPSILON = 0.20; SOFTMAX_TEMP = 0.10
ETA_H_EMA = 0.50; H_NOV_EMA = 0.99; SPIKE_THRESHOLD = 2.0
FAST_ADAPT_STEPS = 500; FAST_ETA_FACTOR = 1.5

N_STEPS = 10_000
N_SEEDS = 10
STEP_NUM = 1006


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp; x -= np.max(x)
    e = np.exp(x); probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class FastAdapt994:
    """994 substrate — frozen reference implementation. DO NOT MODIFY."""

    def __init__(self, seed: int = 0):
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

    def set_game(self, n_actions: int):
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
            self._fast_adapt_countdown = FAST_ADAPT_STEPS
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


if __name__ == "__main__":
    import json

    print("=" * 70)
    print("STEP 1006 — 994 BASELINE THROUGH RANDOMIZED PRISM")
    print("Establishes canonical baseline for all future chain_kill verdicts.")
    print("=" * 70)
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), '?')
    vc33_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/vc33') if len(d) >= 8), '?')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}  VC33={vc33_hash}")
    print(f"Budget: {N_STEPS} steps/game, {N_SEEDS} seeds, randomized order (PRISM-C)")
    print()

    t0 = time.time()

    chain = make_prism(n_steps=N_STEPS)
    runner = ChainRunner(chain=chain, n_seeds=N_SEEDS, verbose=True)
    aggregated = runner.run(FastAdapt994, substrate_kwargs={})

    # Save as canonical baseline (chain_kill=None — this IS the baseline)
    out_path = runner.save_results(
        aggregated=aggregated,
        substrate_name="FastAdapt994",
        step=STEP_NUM,
        config={
            "ENC_DIM": ENC_DIM, "H_DIM": H_DIM,
            "ETA_W": ETA_W, "ALPHA_EMA": ALPHA_EMA,
            "EPSILON": EPSILON, "SOFTMAX_TEMP": SOFTMAX_TEMP,
            "FAST_ADAPT_STEPS": FAST_ADAPT_STEPS, "FAST_ETA_FACTOR": FAST_ETA_FACTOR,
        },
        chain_kill={"verdict": "BASELINE"},
    )

    # Also write to canonical baseline path
    baseline_path = 'B:/M/the-search/chain_results/baseline_994.json'
    import shutil
    shutil.copy(out_path, baseline_path)
    print(f"Baseline saved to: {baseline_path}")

    print()
    print("=" * 70)
    print("STEP 1006 RESULTS (994 PRISM BASELINE):")
    for name, data in aggregated.items():
        if isinstance(data, dict) and 'l1_rate' in data:
            print(f"  {name}: L1={data['l1_rate']:.0%}  avg_t={data['mean_elapsed']:.1f}s")
    phases_passed = sum(1 for v in aggregated.values()
                       if isinstance(v, dict) and v.get('l1_rate', 0) > 0)
    phases_total = len([v for v in aggregated.values() if isinstance(v, dict) and 'l1_rate' in v])
    print(f"  Chain score: {phases_passed}/{phases_total}")
    print(f"  Results: {out_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")
    print("STEP 1006 DONE")
