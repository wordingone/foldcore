"""
step0983_hires_encoding.py -- Higher-resolution encoding (avgpool2 = 1024 dims).

FAMILY: Encoding augmentation (916 base, h-reset)
R3 HYPOTHESIS: avgpool4 (kernel=4, 16x16=256 dims) collapses 4x4 pixel blocks, destroying
spatial correspondence between click positions and their pixel effects. With avgpool2
(kernel=2, 32x32=1024 dims), alpha can attend to the 8x8 block regions where specific
clicks cause pixel changes. Alpha concentrates on click-responsive spatial regions →
substrate discovers which actions affect which locations.

For FT09/VC33: clicking a specific tile → pixels change in that region → high prediction
error in those 8x8 blocks → high alpha on those dims → 800b selects actions that produce
changes in attended regions.

Changes from 965: ENC_DIM=1024 (avgpool2). EXT_DIM=1088. W_pred(1088,1088).
W_x(64,1024). h-reset on game switch preserved.

Kill: LS20 < 67.0 (higher res hurts navigation).
Success: FT09 > 0 OR VC33 > 0 (spatial precision enables click discovery).

Standalone on FT09, VC33, LS20 at 10K. 10 seeds each.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque

ENC_DIM = 1024   # avgpool2: 32x32
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM   # 1088
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
ALPHA_LO = 0.10
ALPHA_HI = 5.00
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
TEST_SEEDS = list(range(1, 11))


def _enc_frame_hires(frame):
    """avgpool2: 32x32 = 1024 dims. 4x spatial resolution vs baseline _enc_frame."""
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3:
        # (C,H,W) → (H,W,C)
        if frame.shape[0] <= 4 and frame.shape[1] > 4 and frame.shape[2] > 4:
            frame = frame.transpose(1, 2, 0)
        a = frame[:, :, 0] / 15.0 if frame.max() > 1 else frame[:, :, 0]
        h, w = a.shape
        ph, pw = max(h // 2, 1), max(w // 2, 1)  # 2x2 blocks
        pad_h, pad_w = ph * 2, pw * 2
        if h < pad_h or w < pad_w:
            buf = np.zeros((pad_h, pad_w), dtype=np.float32)
            buf[:min(h, pad_h), :min(w, pad_w)] = a[:min(h, pad_h), :min(w, pad_w)]
            a = buf
        pooled = a[:ph*2, :pw*2].reshape(ph, 2, pw, 2).mean(axis=(1, 3))
        x = pooled.flatten()[:ENC_DIM]
        if len(x) < ENC_DIM:
            x = np.pad(x, (0, ENC_DIM - len(x)))
    else:
        x = frame.flatten()[:ENC_DIM].astype(np.float32)
        if len(x) < ENC_DIM:
            x = np.pad(x, (0, ENC_DIM - len(x)))
    return (x - x.mean()).astype(np.float32)


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp; x -= np.max(x)
    e = np.exp(x); probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class HiResEnc983:
    """916 base + h-reset + avgpool2 (1024-dim encoding). EXT_DIM=1088."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM), dtype=np.float32)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self._running_mean_enc = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._prev_ext = None
        self._prev_action = None

    def reset(self):
        """h-reset (965 fix). W_pred/alpha/enc_mean persist."""
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = np.full(self._n_actions, INIT_DELTA, dtype=np.float32)
        self._prev_ext = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame_hires(obs)
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean_enc = (1 - a) * self._running_mean_enc + a * enc_raw
        enc = enc_raw - self._running_mean_enc
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
            if err_norm > 10.0: error = error * (10.0 / err_norm); err_norm = 10.0
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W * np.outer(error, self._prev_ext)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()
            weighted_delta = (ext_enc - self._prev_ext) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (1 - ALPHA_EMA) * self.delta_per_action[a] + ALPHA_EMA * change
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_sel(self.delta_per_action, SOFTMAX_TEMP, self._rng)
        self._prev_ext = ext_enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None
        self._prev_action = None

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
        sub = HiResEnc983(n_actions=n_actions, seed=seed)
        sub.reset()
        env = make_env(game)
        obs = env.reset(seed=seed * 1000)
        step = 0; completions = 0; level = 0
        while step < n_steps:
            if obs is None:
                obs = env.reset(seed=seed * 1000); sub.on_level_transition(); continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
            obs, _, done, info = env.step(action); step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                completions += cl - level; level = cl; sub.on_level_transition()
            if done:
                obs = env.reset(seed=seed * 1000); level = 0; sub.on_level_transition()
        results.append(completions)
        print(f"  seed={seed}: completions={completions:4d}  alpha_conc={sub.alpha_conc():.2f}")
    return results


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("STEP 983 — HIRES ENCODING (avgpool2, 1024 dims)")
    print("=" * 70)
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), '?')
    vc33_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/vc33') if len(d) >= 8), '?')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}  VC33={vc33_hash}")
    print(f"ENC_DIM={ENC_DIM}  EXT_DIM={EXT_DIM}  W_pred({EXT_DIM},{EXT_DIM})")
    print(f"Standalone (not chain). 10K steps, 10 seeds each.")
    print(f"Kill: LS20<67.0  Success: FT09>0 OR VC33>0")
    print()

    t0 = time.time()

    print("--- LS20 (4 actions, 10K, 10 seeds) [control] ---")
    ls = run_game("LS20", 4, TEST_SEEDS, 10_000)
    print(f"  LS20: mean={np.mean(ls):.1f}  nonzero={sum(1 for x in ls if x > 0)}/10  {ls}")
    t1 = time.time()
    print(f"  (elapsed so far: {t1-t0:.1f}s)")

    print()
    print("--- FT09 (68 actions, 10K, 10 seeds) ---")
    ft = run_game("FT09", 68, TEST_SEEDS, 10_000)
    print(f"  FT09: mean={np.mean(ft):.1f}  nonzero={sum(1 for x in ft if x > 0)}/10  {ft}")
    t2 = time.time()
    print(f"  (elapsed so far: {t2-t0:.1f}s)")

    print()
    print("--- VC33 (68 actions, 10K, 10 seeds) ---")
    vc = run_game("VC33", 68, TEST_SEEDS, 10_000)
    print(f"  VC33: mean={np.mean(vc):.1f}  nonzero={sum(1 for x in vc if x > 0)}/10  {vc}")

    print()
    print("=" * 70)
    print("STEP 983 RESULTS (baseline: LS20=74.7@10K standalone, FT09=0/VC33=0):")
    print(f"  LS20: mean={np.mean(ls):.1f}/seed  nonzero={sum(1 for x in ls if x > 0)}/10  {ls}")
    print(f"  FT09: mean={np.mean(ft):.1f}/seed  nonzero={sum(1 for x in ft if x > 0)}/10  {ft}")
    print(f"  VC33: mean={np.mean(vc):.1f}/seed  nonzero={sum(1 for x in vc if x > 0)}/10  {vc}")

    ls_verdict = (f"LS20 PASS ({np.mean(ls):.1f} >= 67.0)"
                  if np.mean(ls) >= 67.0 else f"LS20 DEGRADED ({np.mean(ls):.1f} < 67.0)")
    ft_verdict = f"FT09 SIGNAL ({sum(1 for x in ft if x>0)}/10)" if any(x>0 for x in ft) else "FT09 ZERO"
    vc_verdict = f"VC33 SIGNAL ({sum(1 for x in vc if x>0)}/10)" if any(x>0 for x in vc) else "VC33 ZERO"
    print(f"  {ls_verdict}  |  {ft_verdict}  |  {vc_verdict}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("STEP 983 DONE")
