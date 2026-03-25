"""
step1003_osc_800b.py -- Oscillatory encoding + 800b action selection (hybrid).

FAMILY: Oscillatory dynamics
R3 HYPOTHESIS: Oscillatory phase features (96-dim, trajectory-dependent) provide
better state representation for 800b action discrimination than recurrent h (64-dim).
Both use the same 800b mechanism (delta_per_action, alpha, softmax). Only the state
representation differs: 916 uses concat(enc, h), 1003 uses concat(enc, osc_features).

1001/1002 failure: phase-coherence credit can't differentiate actions — no positive
feedback. 800b's argmax amplification IS the critical mechanism. This tests whether
oscillatory encoding adds value TO 800b, not as a replacement.

Kill: LS20 < 72.7 (must beat 916 baseline at 10K).
Budget: 10K, 10 seeds, LS20 standalone.
If > 72.7: oscillatory encoding helps → family alive.
If ≤ 72.7: encoding adds nothing → family dead.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

# Oscillator parameters (same as 1001/1002)
N_OSC = 32
MU = 1.0
DT = 0.1
N_SUBSTEPS = 5
TAU_F = 50
LR_WIJ = 0.01
OMEGA_LO = 0.5
OMEGA_HI = 2.0
DELTA_OBS_EMA_RATE = 0.99

# Dimensions
ENC_DIM = 256
FEAT_DIM = N_OSC * 3        # 96: [cos(phi), sin(phi), r]
EXT_DIM = ENC_DIM + FEAT_DIM  # 352

# 800b parameters (same as 916)
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
ALPHA_LO = 0.10
ALPHA_HI = 5.00
EPSILON = 0.20
SOFTMAX_TEMP = 0.10

TEST_SEEDS = list(range(1, 11))
PHASE_STEPS = 10_000
CIFAR_STEPS = 1_000


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp; x -= np.max(x)
    e = np.exp(x); probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class OscillatoryEncoding1003:
    """916 architecture with oscillatory features replacing recurrent h.

    State: concat(enc, osc_features) = 352-dim (vs 916's 320-dim enc+h).
    Action selection: 800b delta_per_action + softmax (identical to 916).
    Oscillator coupling: W_ij learned via delta_obs * e_ij (from 1002).
    W_pred: gradient ASCENT — same as 916, drives persistent alpha calibration.
    """

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Oscillators
        self._omega = rs.uniform(OMEGA_LO, OMEGA_HI, N_OSC).astype(np.float32)
        phases = rs.uniform(0, 2 * np.pi, N_OSC).astype(np.float32)
        self._z = (np.cos(phases) + 1j * np.sin(phases)).astype(np.complex64)
        self._W_ij = np.zeros((N_OSC, N_OSC), dtype=np.float32)
        self._W_in = rs.randn(N_OSC, ENC_DIM).astype(np.float32) * 0.1
        self._e_ij = np.zeros((N_OSC, N_OSC), dtype=np.float32)
        self._delta_obs_ema = 1.0

        # 800b (same as 916)
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM), dtype=np.float32)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)
        self._n_actions = 4
        self.delta_per_action = np.full(4, INIT_DELTA, dtype=np.float32)

        # Encoding
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._prev_ext = None
        self._prev_action = None
        self._prev_enc = None

    def set_game(self, n_actions):
        self._n_actions = n_actions
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        phases = self._rng.uniform(0, 2 * np.pi, N_OSC).astype(np.float32)
        self._z = (np.cos(phases) + 1j * np.sin(phases)).astype(np.complex64)
        self._prev_ext = None
        self._prev_action = None
        self._prev_enc = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return (enc_raw - self._running_mean).astype(np.float32)

    def _step_oscillators(self, enc):
        z = self._z.astype(np.complex128)
        omega = self._omega.astype(np.float64)
        W_ij = self._W_ij.astype(np.float64)
        input_drive = (self._W_in.astype(np.float64) @ enc.astype(np.float64)).astype(np.complex128)
        for _ in range(N_SUBSTEPS):
            r2 = np.real(z * np.conj(z))
            dz = (MU + 1j * omega) * z - r2 * z + W_ij @ z + input_drive
            z = z + DT * dz
            r = np.abs(z)
            too_large = r > 5.0
            if np.any(too_large):
                z[too_large] = z[too_large] / r[too_large] * 5.0
        bad = ~np.isfinite(np.abs(z))
        if np.any(bad):
            phases = self._rng.uniform(0, 2 * np.pi, np.sum(bad))
            z[bad] = np.cos(phases) + 1j * np.sin(phases)
        self._z = z.astype(np.complex64)

    def _update_eligibility(self):
        phi = np.angle(self._z).astype(np.float32)
        dphi = phi[:, None] - phi[None, :]
        coherence = np.cos(dphi)
        alpha = 1.0 / TAU_F
        self._e_ij = (1 - alpha) * self._e_ij + alpha * coherence

    def _get_features(self):
        phi = np.angle(self._z).astype(np.float32)
        r = np.abs(self._z).astype(np.float32)
        return np.concatenate([np.cos(phi), np.sin(phi), r])

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY: return
        mean_errors = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(mean_errors)) or np.any(np.isinf(mean_errors)): return
        raw_alpha = np.sqrt(np.clip(mean_errors, 0, 1e6) + 1e-8)
        mean_raw = np.mean(raw_alpha)
        if mean_raw < 1e-8 or np.isnan(mean_raw): return
        self.alpha = np.clip(raw_alpha / mean_raw, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        enc = self._encode(obs)
        self._step_oscillators(enc)
        self._update_eligibility()
        features = self._get_features()
        ext_enc = np.concatenate([enc, features]).astype(np.float32)  # 352-dim

        # W_ij update: delta_obs modulator (from 1002, kept for oscillator coupling)
        if self._prev_enc is not None:
            delta_obs = float(np.linalg.norm(enc - self._prev_enc))
            self._delta_obs_ema = DELTA_OBS_EMA_RATE * self._delta_obs_ema + (1 - DELTA_OBS_EMA_RATE) * delta_obs
            M_osc = max(0.0, delta_obs - self._delta_obs_ema)
            if M_osc > 1e-8:
                self._W_ij += LR_WIJ * M_osc * self._e_ij
                np.clip(self._W_ij, -5.0, 5.0, out=self._W_ij)

        # 800b: identical to 916 (W_pred gradient ASCENT, alpha, delta_per_action)
        if self._prev_ext is not None and self._prev_action is not None:
            pred = self.W_pred @ self._prev_ext
            error = (ext_enc * self.alpha) - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0: error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W * np.outer(error, self._prev_ext)  # ASCENT (same as 916)
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
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None
        self._prev_action = None
        self._prev_enc = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))


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


def run_chain(seeds, n_steps, cifar_steps, cifar_imgs, cifar_lbls):
    c1l, lsl, c2l = [], [], []
    for seed in seeds:
        sub = OscillatoryEncoding1003(seed=seed)
        c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, cifar_steps)
        l = run_arc(sub, "LS20", 4, seed * 1000, n_steps)
        c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, cifar_steps)
        c1l.append(c1); lsl.append(l); c2l.append(c2)
        print(f"  seed={seed}: C1={c1:.3f} LS20={l:4d} C2={c2:.3f}"
              f"  alpha_conc={sub.alpha_conc():.1f} delta_ema={sub._delta_obs_ema:.2f}")
    return c1l, lsl, c2l


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("STEP 1003 — OSC ENCODING + 800b (hybrid: osc features + 916 action)")
    print("=" * 70)
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    print(f"Game version: LS20={ls20_hash}")
    print(f"ext_enc = concat(enc[256], osc_features[96]) = 352-dim")
    print(f"W_pred: {EXT_DIM}x{EXT_DIM} ASCENT. 800b: delta_per_action + softmax T={SOFTMAX_TEMP}")
    print(f"Kill: LS20 < 72.7 (916 baseline). Success: LS20 > 72.7")
    print()

    t0 = time.time()
    cifar_imgs, cifar_lbls = load_cifar()
    c1, ls, c2 = run_chain(TEST_SEEDS, PHASE_STEPS, CIFAR_STEPS, cifar_imgs, cifar_lbls)

    print()
    print("=" * 70)
    print("STEP 1003 RESULTS (916 baseline @10K = 72.7):")
    print(f"  CIFAR-1: {np.mean(c1):.3f}")
    print(f"  LS20:    {np.mean(ls):.1f}/seed  nonzero={sum(1 for x in ls if x > 0)}/10  {ls}")
    print(f"  CIFAR-2: {np.mean(c2):.3f}")
    ls_v = "PASS (beats 916)" if np.mean(ls) > 72.7 else "KILL (no improvement over 916)"
    ls_sig = f"nonzero={sum(1 for x in ls if x > 0)}/10" if any(x > 0 for x in ls) else "ZERO"
    print(f"  {ls_v}  |  {ls_sig}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("STEP 1003 DONE")
