"""
step0939b_gfs_warmup.py -- Growing Feature Space + warm-up exclusion.

R3 hypothesis: Zero-init W_pred rows cause bootstrap failure (939 root cause).
Warm-up exclusion: new PCA dims are excluded from alpha/delta for WARM_UP=1000
steps. W_pred trains on ALL dims always. After warm-up, dim enters alpha/delta
with well-initialized W_pred weights → no alpha spike.

ONE VARIABLE CHANGE from 939: dim_birth_step tracking + masked alpha/delta.
Everything else identical.

Architecture:
  - 916 base: enc(256) + h(64) + W_pred + alpha[0.1-5.0] + 800b delta
  - PCA every 1000 steps on rolling 1000-obs buffer
  - PCA_RATIO=2.0: add top eigenvector if dominant
  - ext_enc = [enc_base(256) | h(64) | extra_projs(K)] = (320+K)D
  - WARM_UP=1000: new dims excluded from alpha/delta for 1000 steps
    - W_pred trains on ALL dims always (warming + graduated)
    - alpha=1.0 for warming dims (not updated from pred errors)
    - delta = ||alpha_eff * (ext - prev_ext)|| with alpha_eff=0 for warming dims

dim_birth_step[i] = phase_step when extra_features[i] was added.
  In set_game(): existing features get birth=-WARM_UP (graduated at step 0).
  New features added mid-game: birth=current_phase_step → warm-up starts.

Kill: LS20 < 250 → KILL (bootstrap was the only issue → GFS dead).
Signal: LS20 >= 250 → warm-up fixes it → GFS family alive.
Diagnostic: alpha_conc at 100, 1000, 5000, 10000 — compare to 939 (was 50 from step 1).
Run: PRISM-light Mode C, 10K/phase, 2 seeds.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
BASE_EXT_DIM = ENC_DIM + H_DIM   # 320
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
PCA_RATIO = 2.0
MAX_EXTRA_FEATURES = 16
WARM_UP = 1000           # steps to exclude new dim from alpha/delta
TEST_SEEDS = list(range(1, 3))
PHASE_STEPS = 10_000
DIAG_STEPS = {100, 1_000, 5_000, 10_000}


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


def softmax_action(vals, temp, rng):
    x = vals / temp; x -= np.max(x); e = np.exp(x)
    return int(rng.choice(len(vals), p=e / (e.sum() + 1e-12)))


class Chain939b:
    """916 + GFS PCA growth + warm-up exclusion for new dims.

    dim_birth_step[i]: phase_step when extra_features[i] was added.
    Warm-up dims: alpha=1.0 (not updated), contribute 0 to delta.
    W_pred trains on ALL dims regardless.
    """

    def __init__(self, seed=0):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # Persistent across games
        self.alpha = np.ones(BASE_EXT_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        self.extra_features = []       # list of 256D eigenvectors
        self.dim_birth_step = []       # birth phase_step per extra feature

        # Per-game state
        self._n = None
        self.W_pred = None
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = None
        self._prev_ext = None
        self._prev_action = None
        self.obs_buffer = deque(maxlen=OBS_BUFFER_SIZE)

        self._phase_step = 0
        self._diag_log = {}
        self._features_added = 0

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
        # Existing features: mark as graduated (born -WARM_UP steps ago → diff=WARM_UP >= WARM_UP)
        self.dim_birth_step = [-WARM_UP] * len(self.extra_features)

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
        self.dim_birth_step = []
        self._n = None; self.W_pred = None
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = None
        self._prev_ext = None; self._prev_action = None
        self.obs_buffer = deque(maxlen=OBS_BUFFER_SIZE)
        self._phase_step = 0; self._diag_log = {}; self._features_added = 0

    def _get_graduated_mask(self):
        """Bool mask: True for dims that have completed WARM_UP steps."""
        mask = np.ones(self._ext_dim(), dtype=bool)
        for i, birth in enumerate(self.dim_birth_step):
            if (self._phase_step - birth) < WARM_UP:
                mask[BASE_EXT_DIM + i] = False
        return mask

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        if self.extra_features:
            extra = np.array([float(enc @ ef) for ef in self.extra_features],
                             dtype=np.float32)
            ext_enc = np.concatenate([enc, self.h, extra])
        else:
            ext_enc = np.concatenate([enc, self.h])
        return enc, ext_enc.astype(np.float32)

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY:
            return
        me = np.mean(list(self._pred_errors), axis=0)
        if np.any(np.isnan(me)) or np.any(np.isinf(me)):
            return
        mask = self._get_graduated_mask()
        ra = np.sqrt(np.clip(me, 0, 1e6) + 1e-8)
        mr = np.mean(ra[mask]) if np.any(mask) else np.mean(ra)
        if mr < 1e-8 or np.isnan(mr):
            return
        new_alpha_all = np.clip(ra / mr, ALPHA_LO, ALPHA_HI)
        # Only apply to graduated dims; warming dims stay at 1.0
        self.alpha = np.where(mask, new_alpha_all, np.float32(1.0))

    def _try_add_feature(self):
        if len(self.extra_features) >= MAX_EXTRA_FEATURES:
            return False
        if len(self.obs_buffer) < OBS_BUFFER_SIZE:
            return False
        buf = np.array(self.obs_buffer, dtype=np.float32)
        buf_c = buf - buf.mean(axis=0)
        cov = (buf_c.T @ buf_c) / len(buf)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return False
        if eigenvalues[-1] <= 0 or eigenvalues[-2] <= 0:
            return False
        if eigenvalues[-1] > PCA_RATIO * eigenvalues[-2]:
            new_ef = eigenvectors[:, -1].astype(np.float32)
            self.extra_features.append(new_ef)
            self.dim_birth_step.append(self._phase_step)
            self._grow_model()
            return True
        return False

    def _grow_model(self):
        """Expand W_pred and alpha by 1 dim (appended at end of ext_enc)."""
        d_old = self._ext_dim() - 1   # ext_dim already incremented by extra_features.append
        d_new = d_old + 1
        # Insert zero column at position d_old (before n_actions one-hot block)
        W_new = np.concatenate([
            np.concatenate([self.W_pred[:, :d_old],
                            np.zeros((d_old, 1), dtype=np.float32),
                            self.W_pred[:, d_old:]], axis=1),
            np.zeros((1, d_new + self._n), dtype=np.float32)
        ], axis=0)
        self.W_pred = W_new
        # New dim starts at alpha=1.0 (warm-up phase)
        self.alpha = np.append(self.alpha, np.float32(1.0))
        # Clear pred_errors (old vectors wrong dim)
        self._pred_errors.clear()
        # Invalidate prev_ext (size changed)
        self._prev_ext = None
        self._features_added += 1

    def process(self, obs):
        enc, ext_enc = self._encode(obs)
        self._phase_step += 1
        self.obs_buffer.append(enc.copy())

        if self._prev_ext is not None and self._prev_action is not None:
            if len(self._prev_ext) == len(ext_enc):
                # W_pred training: ALL dims (alpha=1.0 for warming dims → normal training weight)
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

                # Delta: ONLY graduated dims (warming dims contribute 0)
                mask = self._get_graduated_mask()
                alpha_eff = self.alpha * mask.astype(np.float32)
                ext_change = float(np.linalg.norm(alpha_eff * (ext_enc - self._prev_ext)))
                a = self._prev_action
                self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                             + ALPHA_EMA * ext_change)

        # PCA check every interval
        if self._phase_step % PCA_CHECK_INTERVAL == 0:
            added = self._try_add_feature()
            if added:
                _, ext_enc = self._encode(obs)   # re-encode with new feature

        # Diagnostics
        if self._phase_step in DIAG_STEPS:
            mask = self._get_graduated_mask()
            warming = int((~mask).sum())
            self._diag_log[self._phase_step] = {
                "alpha_conc": float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8)),
                "alpha_conc_grad": float(
                    np.max(self.alpha[mask]) / (np.min(self.alpha[mask]) + 1e-8)
                ) if np.any(mask) else 1.0,
                "delta_spread": float(self.delta_per_action.max() - self.delta_per_action.min()),
                "ext_dim": self._ext_dim(),
                "features_added": self._features_added,
                "warming": warming,
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
            parts.append(f"@{step}: alpha_conc={d['alpha_conc']:.2f}"
                         f"(grad={d['alpha_conc_grad']:.2f})"
                         f" delta_spr={d['delta_spread']:.3f}"
                         f" dim={d['ext_dim']} feats={d['features_added']}"
                         f" warm={d['warming']}")
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


print("=" * 70)
print("STEP 939b — GFS + WARM-UP EXCLUSION")
print("=" * 70)
print(f"WARM_UP={WARM_UP}. New PCA dims excluded from alpha/delta for {WARM_UP} steps.")
print(f"W_pred trains on ALL dims always. PCA_RATIO={PCA_RATIO}. MAX={MAX_EXTRA_FEATURES}.")
print(f"2 seeds. PRISM-light Mode C, 10K/phase.")
t0 = time.time()

cifar_imgs, cifar_lbls = load_cifar()
cifar1 = []; ls20 = []; ft09 = []; vc33 = []; cifar2 = []

for seed in TEST_SEEDS:
    sub = Chain939b(seed=seed)
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
print(f"STEP 939b RESULTS (GFS+warmup, PRISM-light, 10K/phase, {len(TEST_SEEDS)} seeds):")
print(f"  LS20  L1: {np.mean(ls20):.1f}/seed  {ls20}")
print(f"  FT09  L1: {np.mean(ft09):.1f}/seed  {ft09}")
print(f"  VC33  L1: {np.mean(vc33):.1f}/seed  {vc33}")
print(f"  CIFAR-1:  {np.mean(cifar1):.4f}")
print(f"\nBaselines: 914 (10K) LS20=248.6  |  916 (25K) LS20=290.7")
print(f"Kill: LS20 < 250. Signal: LS20 >= 250 (warmup fixed bootstrap).")
print(f"Total elapsed: {time.time() - t0:.1f}s")
print("STEP 939b DONE")
