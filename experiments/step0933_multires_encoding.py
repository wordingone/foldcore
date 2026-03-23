"""
step0933_multires_encoding.py -- Multi-resolution encoding + alpha self-selection.

R3 hypothesis: Alpha self-selects the appropriate spatial resolution per game.
FT09 needs fine-grained tile-level discrimination (pool8/pool4).
LS20 needs coarse spatial navigation (pool16).
Multi-resolution gives alpha the CHOICE — alpha IS the resolution selector.

Architecture: 895h (clamped alpha + 800b, NO recurrent h). ONLY change: encoding.
Current: avgpool 4x4 patches → 16x16 = 256D.
New:     pool16 (256D) + pool8 (64D) + pool4 (16D) = 336D total.

Alpha covers all 336 dims. Concentration per band = diagnostic of resolution preference.

Kill criteria:
1. LS20 L1/seed < 200 (>20% below 914 baseline 248.6) → KILL
2. Chain kill: improves one game at cost of another → KILL
3. Alpha uniform across bands → inconclusive (not a kill)

Run: PRISM-light chain. 10K steps/game. 5 seeds. 5-min cap per seed.
Compare: 914 chain 895h (256D, ALPHA_LO=0.1, 25K): LS20=248.6.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque

TEST_SEEDS = list(range(1, 6))
PHASE_STEPS = 10_000
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
ENC_DIM = 336   # pool16(256) + pool8(64) + pool4(16)
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00

# Resolution band indices (for diagnostic)
BAND_16 = slice(0, 256)    # 4x4 patches → 16x16
BAND_8  = slice(256, 320)  # 8x8 patches → 8x8
BAND_4  = slice(320, 336)  # 16x16 patches → 4x4


def _enc_frame_multires(frame: np.ndarray) -> np.ndarray:
    """Multi-resolution encoding: pool16 + pool8 + pool4.

    Handles any input shape (flat 1D, 2D grid, 3D HxWxC) by normalizing to 2D
    and padding to minimum 64x64 if needed.
    """
    f = np.asarray(frame, dtype=np.float32)

    # Extract 2D spatial array
    if f.ndim == 1:
        side = max(int(np.ceil(np.sqrt(len(f)))), 1)
        padded = np.zeros(side * side, dtype=np.float32)
        padded[:len(f)] = f
        a = padded.reshape(side, side)
    elif f.ndim == 2:
        a = f
    else:
        # 3D (H, W, C) — use channel 0
        a = f[:, :, 0]

    # Normalize to 0-1
    mx = a.max()
    if mx > 1.0:
        a = a / 15.0 if mx <= 15.0 else a / 255.0

    # Pad to minimum 64x64 so all three pools work
    h, w = a.shape
    H = max(h, 64); W = max(w, 64)
    if h < H or w < W:
        padded = np.zeros((H, W), dtype=np.float32)
        padded[:h, :w] = a
        a = padded
    h, w = a.shape

    # pool16: (h/16) x (w/16) tiles → 16x16 = 256D
    pool16 = a.reshape(16, h // 16, 16, w // 16).mean(axis=(1, 3)).flatten()

    # pool8: (h/8) x (w/8) tiles → 8x8 = 64D
    pool8 = a.reshape(8, h // 8, 8, w // 8).mean(axis=(1, 3)).flatten()

    # pool4: (h/4) x (w/4) tiles → 4x4 = 16D
    pool4 = a.reshape(4, h // 4, 4, w // 4).mean(axis=(1, 3)).flatten()

    x = np.concatenate([pool16, pool8, pool4])  # 336D
    return x - x.mean()


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp; x -= np.max(x); e = np.exp(x)
    return int(rng.choice(len(delta), p=e / (e.sum() + 1e-12)))


class Chain933:
    """895h with 336D multi-resolution encoding. Alpha + 800b. No recurrent h.
    Alpha (336D) and running_mean persist across games. W + delta reset per game.
    """

    def __init__(self, seed=0):
        self._rng = np.random.RandomState(seed)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        self._n = None; self.W = None; self.delta_per_action = None
        self._prev_enc = None; self._prev_action = None

    def set_game(self, n_actions):
        self._n = n_actions
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._pred_errors.clear()
        self._prev_enc = None; self._prev_action = None

    def reset_seed(self, seed):
        self._rng = np.random.RandomState(seed)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0; self._pred_errors.clear()
        self._n = None; self.W = None; self.delta_per_action = None
        self._prev_enc = None; self._prev_action = None

    def _encode(self, obs):
        enc = _enc_frame_multires(obs)
        self._n_obs += 1; a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc
        return enc - self._running_mean

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
                                   one_hot(self._prev_action, self._n)])
            pred = self.W @ inp; error = (enc * self.alpha) - pred
            en = float(np.linalg.norm(error))
            if en > 10.0: error *= 10.0 / en
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error)); self._update_alpha()
            change = float(np.linalg.norm((enc - self._prev_enc) * self.alpha))
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * change)
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n))
        else:
            action = softmax_sel(self.delta_per_action, SOFTMAX_TEMP, self._rng)
        self._prev_enc = enc.copy(); self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def alpha_band_conc(self):
        """Mean alpha per resolution band (diagnostic)."""
        return {
            "pool16": float(np.mean(self.alpha[BAND_16])),
            "pool8":  float(np.mean(self.alpha[BAND_8])),
            "pool4":  float(np.mean(self.alpha[BAND_4])),
        }


def make_env(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


def load_cifar():
    try:
        import torchvision, torchvision.transforms as T
        ds = torchvision.datasets.CIFAR100('B:/M/the-search/data', train=False,
                                            download=True, transform=T.ToTensor())
        imgs = np.array([np.array(ds[i][0]).transpose(1, 2, 0)
                         for i in range(len(ds))], dtype=np.float32)
        lbls = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
        return imgs, lbls
    except Exception as e:
        print(f"CIFAR load failed: {e}"); return None, None


def run_cifar(sub, imgs, lbls, seed, n_steps):
    if imgs is None: return 0.0
    sub.set_game(100); rng = np.random.RandomState(seed)
    idx = rng.permutation(len(imgs))[:n_steps]
    correct = sum(1 for i in idx if sub.process(imgs[i]) % 100 == lbls[i])
    return correct / len(idx)


def run_arc(sub, game, n_actions, seed, n_steps):
    sub.set_game(n_actions); env = make_env(game)
    obs = env.reset(seed=seed); step = 0; completions = 0; level = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=seed); sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            completions += cl - level; level = cl; sub.on_level_transition()
        if done:
            obs = env.reset(seed=seed); level = 0; sub.on_level_transition()
    return completions


print("=" * 70)
print("STEP 933 — MULTI-RESOLUTION ENCODING (pool16+pool8+pool4=336D)")
print("=" * 70)
print(f"R3: alpha self-selects resolution per game. ENC_DIM=336 ({BAND_16}=pool16, "
      f"{BAND_8}=pool8, {BAND_4}=pool4).")
print(f"Architecture: 895h (alpha+800b, no h). W+delta reset per game. Alpha persists.")
t0 = time.time()

cifar_imgs, cifar_lbls = load_cifar()
cifar1 = []; ls20 = []; ft09 = []; vc33 = []; cifar2 = []

for seed in TEST_SEEDS:
    sub = Chain933(seed=seed)
    sub.reset_seed(seed)
    print(f"\n  Seed {seed}:")

    c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, PHASE_STEPS)
    cifar1.append(c1)
    b = sub.alpha_band_conc()
    print(f"    CIFAR-1: acc={c1:.4f}  alpha_conc={sub.alpha_conc():.2f}"
          f"  bands=[p16:{b['pool16']:.2f} p8:{b['pool8']:.2f} p4:{b['pool4']:.2f}]")

    l = run_arc(sub, "LS20", 4, seed * 1000, PHASE_STEPS)
    ls20.append(l)
    b = sub.alpha_band_conc()
    print(f"    LS20:    L1={l:4d}  alpha_conc={sub.alpha_conc():.2f}"
          f"  bands=[p16:{b['pool16']:.2f} p8:{b['pool8']:.2f} p4:{b['pool4']:.2f}]")

    f = run_arc(sub, "FT09", 68, seed * 1000, PHASE_STEPS)
    ft09.append(f)
    b = sub.alpha_band_conc()
    print(f"    FT09:    L1={f:4d}  alpha_conc={sub.alpha_conc():.2f}"
          f"  bands=[p16:{b['pool16']:.2f} p8:{b['pool8']:.2f} p4:{b['pool4']:.2f}]")

    v = run_arc(sub, "VC33", 68, seed * 1000, PHASE_STEPS)
    vc33.append(v)
    b = sub.alpha_band_conc()
    print(f"    VC33:    L1={v:4d}  alpha_conc={sub.alpha_conc():.2f}"
          f"  bands=[p16:{b['pool16']:.2f} p8:{b['pool8']:.2f} p4:{b['pool4']:.2f}]")

    c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, PHASE_STEPS)
    cifar2.append(c2)
    b = sub.alpha_band_conc()
    print(f"    CIFAR-2: acc={c2:.4f}  alpha_conc={sub.alpha_conc():.2f}"
          f"  bands=[p16:{b['pool16']:.2f} p8:{b['pool8']:.2f} p4:{b['pool4']:.2f}]")

print(f"\n{'=' * 70}")
print(f"STEP 933 RESULTS (multires 336D, PRISM-light, 10K/phase, 5 seeds):")
print(f"  CIFAR-1 acc: {np.mean(cifar1):.4f}  {[f'{x:.3f}' for x in cifar1]}")
print(f"  LS20  L1:    {np.mean(ls20):.1f}/seed  std={np.std(ls20):.1f}  zero={sum(1 for x in ls20 if x == 0)}/5  {ls20}")
print(f"  FT09  L1:    {np.mean(ft09):.1f}/seed  zero={sum(1 for x in ft09 if x == 0)}/5  {ft09}")
print(f"  VC33  L1:    {np.mean(vc33):.1f}/seed  zero={sum(1 for x in vc33 if x == 0)}/5  {vc33}")
print(f"  CIFAR-2 acc: {np.mean(cifar2):.4f}  {[f'{x:.3f}' for x in cifar2]}")
print(f"\nComparison (chain, 5 seeds):")
print(f"  914 895h (256D, 25K): LS20=248.6  FT09=0  VC33=0")
print(f"  933 multires (336D, 10K): LS20={np.mean(ls20):.1f}  FT09={np.mean(ft09):.1f}  VC33={np.mean(vc33):.1f}")
print(f"\nKill criterion: LS20 < 200 or chain kill (improves one, degrades another) → KILL.")
print(f"Total elapsed: {time.time() - t0:.1f}s")
print("STEP 933 DONE")
