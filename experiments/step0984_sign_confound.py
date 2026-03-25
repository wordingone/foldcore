"""
step0984_sign_confound.py -- Baseline reproduction + W_pred sign confound test.

PART A: Reproduce step0965 EXACTLY (Chain916, W_pred -=, h-reset).
Expected: LS20=67.0. Confirms codebase integrity.

PART B: Step 977 (action momentum) with W_pred -= (corrected sign).
Original 977 used += (gradient descent). 965 uses -= (gradient ascent → errors stay volatile).
Tests whether sign confound explains 977's LS20 degradation (47.9 → ?).

If 977-corrected ≈ 67.0 → sign was the real cause, mechanism is fine.
If 977-corrected still degrades → mechanism change is the real cause.

Chain: CIFAR(1K) → LS20(10K) → FT09(10K) → VC33(10K) → CIFAR(1K). 10 seeds each.
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
MOMENTUM_THRESHOLD = 2.0; MOMENTUM_STEPS = 3
TEST_SEEDS = list(range(1, 11)); PHASE_STEPS = 10_000; CIFAR_STEPS = 1_000


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp; x -= np.max(x)
    e = np.exp(x); probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


# ── PART A: Exact 965 reproduction ──────────────────────────────────────────

class Chain916_984A:
    """Exact step0965 reproduction. W_pred -=. Change tracking for delta_per_action."""

    def __init__(self, seed):
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
                self.W_pred -= ETA_W * np.outer(error, self._prev_ext)  # 965: -=
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


# ── PART B: 977 momentum with corrected -= sign ──────────────────────────────

class Momentum977Corrected:
    """977 action momentum with W_pred -= (corrected sign). Tests sign confound."""

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM), dtype=np.float32)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self._running_mean_enc = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._n_actions = 4; self.running_mean = None
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._prev_ext = None; self._prev_action = None
        self._momentum = 0; self._momentum_action = 0

    def set_game(self, n_actions):
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._n_actions = n_actions
        self.running_mean = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._prev_ext = None; self._prev_action = None
        self._momentum = 0; self._momentum_action = 0

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1; a = 1.0 / self._n_obs
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
        delta = 0.0
        if self._prev_ext is not None and self._prev_action is not None:
            target = ext_enc * self.alpha
            pred = self.W_pred @ self._prev_ext
            error = target - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0: error = error * (10.0 / err_norm); err_norm = 10.0
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W * np.outer(error, self._prev_ext)  # CORRECTED: -=
                self._pred_errors.append(np.abs(error)); self._update_alpha()
            delta = err_norm
            a = self._prev_action
            self.running_mean[a] = (1 - ALPHA_EMA) * self.running_mean[a] + ALPHA_EMA * delta
            if delta > MOMENTUM_THRESHOLD:
                self._momentum = MOMENTUM_STEPS; self._momentum_action = a
        if self._momentum > 0:
            action = self._momentum_action; self._momentum -= 1
        elif self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_sel(self.running_mean, SOFTMAX_TEMP, self._rng)
        self._prev_ext = ext_enc.copy(); self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None; self._prev_action = None
        self._momentum = 0

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))


# ── Shared infrastructure ────────────────────────────────────────────────────

def make_env(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except Exception:
        import util_arcagi3; return util_arcagi3.make(name)


def load_cifar():
    try:
        import torchvision, torchvision.transforms as T
        ds = torchvision.datasets.CIFAR100('B:/M/the-search/data', train=False,
                                            download=True, transform=T.ToTensor())
        imgs = np.array([np.array(ds[i][0]).transpose(1, 2, 0) for i in range(len(ds))], dtype=np.float32)
        lbls = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
        return imgs, lbls
    except Exception as e:
        print(f"  CIFAR load failed: {e}"); return None, None


def run_cifar(sub, imgs, lbls, seed, n_steps):
    if imgs is None: return 0.0
    sub.set_game(100)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(imgs))[:n_steps]
    correct = sum(1 for i in idx if sub.process(imgs[i]) % 100 == lbls[i])
    return correct / len(idx)


def run_arc(sub, game, n_actions, seed, n_steps):
    sub.set_game(n_actions)
    env = make_env(game); obs = env.reset(seed=seed)
    step = 0; completions = 0; level = 0
    while step < n_steps:
        if obs is None: obs = env.reset(seed=seed); sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            completions += cl - level; level = cl; sub.on_level_transition()
        if done: obs = env.reset(seed=seed); level = 0; sub.on_level_transition()
    return completions


def run_chain(SubClass, label, seeds, n_steps, cifar_steps, cifar_imgs, cifar_lbls):
    c1l, lsl, ftl, vcl, c2l = [], [], [], [], []
    for seed in seeds:
        sub = SubClass(seed=seed)
        c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, cifar_steps)
        l = run_arc(sub, "LS20", 4, seed * 1000, n_steps)
        f = run_arc(sub, "FT09", 68, seed * 1000, n_steps)
        v = run_arc(sub, "VC33", 68, seed * 1000, n_steps)
        c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, cifar_steps)
        c1l.append(c1); lsl.append(l); ftl.append(f); vcl.append(v); c2l.append(c2)
        print(f"  [{label}] seed={seed}: C1={c1:.3f} LS20={l:4d} FT09={f:4d} VC33={v:4d} C2={c2:.3f}"
              f"  alpha_conc={sub.alpha_conc():.1f}")
    return c1l, lsl, ftl, vcl, c2l


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("STEP 984 — BASELINE REPRO (965-exact) + 977 SIGN CONFOUND TEST")
    print("=" * 70)
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), '?')
    vc33_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/vc33') if len(d) >= 8), '?')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}  VC33={vc33_hash}")
    print(f"Part A: 965 exact (W_pred -=, change tracking). Expected LS20=67.0.")
    print(f"Part B: 977 momentum (W_pred -= corrected). Tests sign confound.")
    print()

    t0 = time.time()
    cifar_imgs, cifar_lbls = load_cifar()

    print("--- PART A: 965 exact reproduction ---")
    c1a, lsa, fta, vca, c2a = run_chain(Chain916_984A, "965-exact", TEST_SEEDS,
                                         PHASE_STEPS, CIFAR_STEPS, cifar_imgs, cifar_lbls)

    print()
    print("--- PART B: 977 momentum with corrected -= sign ---")
    c1b, lsb, ftb, vcb, c2b = run_chain(Momentum977Corrected, "977-corrected", TEST_SEEDS,
                                          PHASE_STEPS, CIFAR_STEPS, cifar_imgs, cifar_lbls)

    print()
    print("=" * 70)
    print("STEP 984 RESULTS:")
    print(f"  Part A (965 exact):       LS20={np.mean(lsa):.1f}/seed  nonzero={sum(1 for x in lsa if x>0)}/10  {lsa}")
    print(f"  Part B (977 -= corrected): LS20={np.mean(lsb):.1f}/seed  nonzero={sum(1 for x in lsb if x>0)}/10  {lsb}")
    print(f"  FT09-A: {fta}  FT09-B: {ftb}")
    print(f"  VC33-A: {vca}  VC33-B: {vcb}")
    print()
    a_pass = np.mean(lsa) >= 60.3  # 90% of 67.0
    b_vs_a = np.mean(lsb) / max(np.mean(lsa), 0.1)
    print(f"  Baseline integrity: {'PASS' if a_pass else 'DRIFT — investigate codebase'} (A LS20={np.mean(lsa):.1f}, expected ~67.0)")
    print(f"  Sign confound: 977-corrected is {b_vs_a*100:.0f}% of 965-exact "
          f"({'MECHANISM is real cause' if b_vs_a < 0.85 else 'SIGN WAS the cause'})")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("STEP 984 DONE")
