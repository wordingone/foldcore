"""
step0939_gfs.py -- Growing Feature Space via online PCA.

R3 hypothesis: PCA on rolling observation buffer discovers dominant variance
directions in the current environment. Adding these as explicit features lets
alpha concentrate on them (R3: substrate modifies HOW it processes observations,
not just WHAT it stores). Unlike random R@h projection, eigenvectors are
semantically grounded — they capture real structure.

Key distinction from h-projection (938c-938e):
  938c-e: R is random → arbitrary action ranking → noise
  939: eigenvectors come from DATA → capture actual environment structure

Architecture: 916 base + dynamic feature growth.
  obs_buffer = deque(maxlen=1000) of raw enc_base (256D each), resets per game
  extra_features = list of eigenvectors (256D), persists across games (like alpha)
  extra_projs = [enc_base @ ef for ef in extra_features]  (K scalars)
  ext_enc = concat([enc_base(256), h(64), extra_projs(K)]) = (320+K)D

  PCA check every 1000 steps:
    cov = (buf - buf_mean).T @ (buf - buf_mean) / 1000
    eigenvalues, eigenvectors = eigh(cov)  -- ascending order
    if eigenvalues[-1] > PCA_RATIO * eigenvalues[-2]: add top eigenvector

  On feature addition:
    alpha appended with 1.0 (new dim starts unweighted)
    W_pred grows: zero row + zero column at new position
    _prev_ext = None (avoid size mismatch for delta)

  set_game() resets: W_pred, h, delta_per_action, obs_buffer, _prev_ext
  Persists across games: alpha, extra_features, running_mean, W_h, W_x

Kill: LS20 < 250 at 10K → KILL (worse than 914 baseline).
Signal: features_added > 0 AND VC33 > 0 (PCA captured relevant structure).
Run: PRISM-light Mode C, 10K/phase, 2 seeds.
"""
import sys, time, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
BASE_EXT_DIM = ENC_DIM + H_DIM   # 320 (before extra features)
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
OBS_BUFFER_SIZE = 1000
PCA_CHECK_INTERVAL = 1000
PCA_RATIO = 2.0           # dominant eigenvalue must be PCA_RATIO× the next
MAX_EXTRA_FEATURES = 16   # cap at 16 extra dims (avoids runaway growth)
TEST_SEEDS = list(range(1, 3))
PHASE_STEPS = 10_000
DIAG_STEPS = {1_000, 5_000, 10_000}


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


def softmax_action(vals, temp, rng):
    x = vals / temp; x -= np.max(x); e = np.exp(x)
    return int(rng.choice(len(vals), p=e / (e.sum() + 1e-12)))


class Chain939:
    """916 architecture + growing feature space via online PCA.

    extra_features: list of eigenvectors (256D each), persists across games.
    obs_buffer: rolling 1000-obs window of raw enc_base, resets per game.
    W_pred + h + delta + obs_buffer: reset per game.
    alpha + extra_features + running_mean: persist across games.
    """

    def __init__(self, seed=0):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Fixed recurrent weights (never trained)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # Persistent state (across games)
        self.alpha = np.ones(BASE_EXT_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        self.extra_features = []   # list of 256D eigenvectors

        # Per-game state (reset in set_game)
        self._n = None
        self.W_pred = None
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = None
        self._prev_ext = None
        self._prev_action = None
        self.obs_buffer = deque(maxlen=OBS_BUFFER_SIZE)

        self._phase_step = 0
        self._diag_log = {}
        self._features_added = 0   # count per phase

    def _ext_dim(self):
        return BASE_EXT_DIM + len(self.extra_features)

    def set_game(self, n_actions):
        self._n = n_actions
        d = self._ext_dim()
        self.W_pred = np.zeros((d, d + n_actions), dtype=np.float32)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._prev_ext = None
        self._prev_action = None
        self.obs_buffer = deque(maxlen=OBS_BUFFER_SIZE)
        self._phase_step = 0
        self._diag_log = {}
        self._features_added = 0

    def reset_seed(self, seed):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.alpha = np.ones(BASE_EXT_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors.clear()
        self.extra_features = []
        self._n = None; self.W_pred = None
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = None
        self._prev_ext = None; self._prev_action = None
        self.obs_buffer = deque(maxlen=OBS_BUFFER_SIZE)
        self._phase_step = 0; self._diag_log = {}; self._features_added = 0

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        # Extra projections (K scalars, one per eigenvector)
        if self.extra_features:
            extra = np.array([float(enc @ ef) for ef in self.extra_features], dtype=np.float32)
            ext_enc = np.concatenate([enc, self.h, extra])
        else:
            ext_enc = np.concatenate([enc, self.h])
        return enc, ext_enc.astype(np.float32)

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY:
            return
        me = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(me)) or np.any(np.isinf(me)):
            return
        ra = np.sqrt(np.clip(me, 0, 1e6) + 1e-8)
        mr = np.mean(ra)
        if mr < 1e-8 or np.isnan(mr):
            return
        self.alpha = np.clip(ra / mr, ALPHA_LO, ALPHA_HI)

    def _try_add_feature(self, enc_base):
        """PCA check: if top eigenvalue > PCA_RATIO * second, add top eigenvector."""
        if len(self.extra_features) >= MAX_EXTRA_FEATURES:
            return False
        if len(self.obs_buffer) < OBS_BUFFER_SIZE:
            return False
        buf = np.array(self.obs_buffer, dtype=np.float32)   # (1000, 256)
        buf_mean = buf.mean(axis=0)
        buf_c = buf - buf_mean
        cov = (buf_c.T @ buf_c) / len(buf)   # (256, 256)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending
        except np.linalg.LinAlgError:
            return False
        if eigenvalues[-1] <= 0 or eigenvalues[-2] <= 0:
            return False
        if eigenvalues[-1] > PCA_RATIO * eigenvalues[-2]:
            new_ef = eigenvectors[:, -1].astype(np.float32)  # top eigenvector
            self.extra_features.append(new_ef)
            self._grow_model()
            return True
        return False

    def _grow_model(self):
        """Expand W_pred and alpha by 1 dimension (appended at end of ext_enc)."""
        d_old = self._ext_dim() - 1   # before this new feature was added
        d_new = d_old + 1              # after

        # W_pred: (d_old, d_old+n) → (d_new, d_new+n)
        # New row (output dim) at end, new column (input dim) at position d_old (before one-hot block)
        W_old = self.W_pred   # (d_old, d_old + n_actions)
        # Insert column at position d_old (end of ext_enc part, before n_actions one-hot)
        W_new_col = np.concatenate([
            W_old[:, :d_old],
            np.zeros((d_old, 1), dtype=np.float32),
            W_old[:, d_old:]
        ], axis=1)  # (d_old, d_new + n_actions)
        # Append zero row at end (new output dim)
        W_new = np.concatenate([
            W_new_col,
            np.zeros((1, d_new + self._n), dtype=np.float32)
        ], axis=0)  # (d_new, d_new + n_actions)
        self.W_pred = W_new

        # Alpha: append 1.0 for new dim
        self.alpha = np.append(self.alpha, np.float32(1.0))

        # Invalidate prev_ext and pred_errors (size changed — old vectors wrong dim)
        self._prev_ext = None
        self._pred_errors.clear()
        self._features_added += 1

    def process(self, obs):
        enc, ext_enc = self._encode(obs)
        self._phase_step += 1

        # Store raw enc in obs_buffer (for PCA)
        self.obs_buffer.append(enc.copy())

        if self._prev_ext is not None and self._prev_action is not None:
            # W_pred training (916 style)
            if len(self._prev_ext) == len(ext_enc):  # safety: same size
                inp = np.concatenate([self._prev_ext * self.alpha,
                                       one_hot(self._prev_action, self._n)])
                pred = self.W_pred @ inp
                error = (ext_enc * self.alpha) - pred
                en = float(np.linalg.norm(error))
                if en > 10.0:
                    error = error * (10.0 / en)
                if not np.any(np.isnan(error)):
                    self.W_pred -= ETA_W * np.outer(error, inp)
                    self._pred_errors.append(np.abs(error))
                    self._update_alpha()

                # Delta: 800b on alpha-weighted ext_enc
                ext_change = float(np.linalg.norm(self.alpha * (ext_enc - self._prev_ext)))
                a = self._prev_action
                self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                             + ALPHA_EMA * ext_change)

        # PCA check every PCA_CHECK_INTERVAL steps
        if self._phase_step % PCA_CHECK_INTERVAL == 0:
            added = self._try_add_feature(enc)
            if added:
                # Re-encode with new feature
                _, ext_enc = self._encode(obs)

        # Diagnostics
        if self._phase_step in DIAG_STEPS:
            self._diag_log[self._phase_step] = {
                "alpha_conc": float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8)),
                "delta_spread": float(self.delta_per_action.max() - self.delta_per_action.min()),
                "ext_dim": self._ext_dim(),
                "features_added": self._features_added,
                "extra_features_total": len(self.extra_features),
            }

        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n))
        else:
            action = softmax_action(self.delta_per_action, SOFTMAX_TEMP, self._rng)

        self._prev_ext = ext_enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None
        self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def diag_str(self):
        parts = []
        for step in sorted(self._diag_log):
            d = self._diag_log[step]
            parts.append(f"@{step//1000}K: alpha_conc={d['alpha_conc']:.2f}"
                         f" delta_spr={d['delta_spread']:.3f}"
                         f" ext_dim={d['ext_dim']}"
                         f" feats={d['extra_features_total']}")
        return "  ".join(parts) if parts else "n/a"


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
    sub.set_game(100)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(imgs))[:n_steps]
    correct = sum(1 for i in idx if sub.process(imgs[i]) % 100 == lbls[i])
    return correct / len(idx)


def run_arc(sub, game, n_actions, seed, n_steps):
    sub.set_game(n_actions)
    env = make_env(game)
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


# --- Run ---
print("=" * 70)
print("STEP 939 — GROWING FEATURE SPACE (online PCA)")
print("=" * 70)
print(f"PCA every {PCA_CHECK_INTERVAL} steps. Ratio={PCA_RATIO}. Max extra={MAX_EXTRA_FEATURES}.")
print(f"2 seeds. PRISM-light Mode C, 10K/phase.")
t0 = time.time()

cifar_imgs, cifar_lbls = load_cifar()
cifar1 = []; ls20 = []; ft09 = []; vc33 = []; cifar2 = []

for seed in TEST_SEEDS:
    sub = Chain939(seed=seed)
    sub.reset_seed(seed)
    print(f"\n  Seed {seed}:")

    c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, PHASE_STEPS)
    cifar1.append(c1)
    print(f"    CIFAR-1: acc={c1:.4f}  alpha_conc={sub.alpha_conc():.2f}"
          f"  ext_dim={sub._ext_dim()}  feats={len(sub.extra_features)}")
    print(f"             diag: {sub.diag_str()}")

    l = run_arc(sub, "LS20", 4, seed * 1000, PHASE_STEPS)
    ls20.append(l)
    print(f"    LS20:    L1={l:4d}  alpha_conc={sub.alpha_conc():.2f}"
          f"  ext_dim={sub._ext_dim()}  feats={len(sub.extra_features)}")
    print(f"             diag: {sub.diag_str()}")

    f = run_arc(sub, "FT09", 68, seed * 1000, PHASE_STEPS)
    ft09.append(f)
    print(f"    FT09:    L1={f:4d}  alpha_conc={sub.alpha_conc():.2f}"
          f"  ext_dim={sub._ext_dim()}  feats={len(sub.extra_features)}")
    print(f"             diag: {sub.diag_str()}")

    v = run_arc(sub, "VC33", 68, seed * 1000, PHASE_STEPS)
    vc33.append(v)
    print(f"    VC33:    L1={v:4d}  alpha_conc={sub.alpha_conc():.2f}"
          f"  ext_dim={sub._ext_dim()}  feats={len(sub.extra_features)}")
    print(f"             diag: {sub.diag_str()}")

    c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, PHASE_STEPS)
    cifar2.append(c2)
    print(f"    CIFAR-2: acc={c2:.4f}  alpha_conc={sub.alpha_conc():.2f}"
          f"  ext_dim={sub._ext_dim()}  feats={len(sub.extra_features)}")
    print(f"             diag: {sub.diag_str()}")

print(f"\n{'=' * 70}")
print(f"STEP 939 RESULTS (GFS PCA, PRISM-light, 10K/phase, {len(TEST_SEEDS)} seeds):")
print(f"  LS20  L1: {np.mean(ls20):.1f}/seed  {ls20}")
print(f"  FT09  L1: {np.mean(ft09):.1f}/seed  {ft09}")
print(f"  VC33  L1: {np.mean(vc33):.1f}/seed  {vc33}")
print(f"  CIFAR-1:  {np.mean(cifar1):.4f}")
print(f"\nBaselines (chain, 5 seeds):")
print(f"  914 (10K): LS20=248.6    916 (25K): LS20=290.7")
print(f"\nKill: LS20 < 250 at 10K. Signal: features_added > 0 AND VC33 > 0.")
print(f"Total elapsed: {time.time() - t0:.1f}s")
print("STEP 939 DONE")
