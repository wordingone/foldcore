"""
step0942_frame_diff.py -- Frame differencing as observation preprocessing.

R3 hypothesis: |obs_t - obs_{t-1}| focuses the substrate on WHAT CHANGED rather
than what exists. Alpha then concentrates on changed-pixel dimensions (R3: modifying
HOW the substrate sees). In LS20, agent movement creates large diffs at agent position.
In FT09, tile clicks create localized diffs. In VC33, near-static obs → diffs focused
on the 1.3% that changes.

ONE VARIABLE CHANGE from 916: feed |obs_t - obs_{t-1}| into _enc_frame instead of obs_t.
Everything else identical (alpha, W_pred, h, 800b delta selector, PRISM-light chain).

last_obs is global state (NOT per-observation indexed — gate 5 clean).
Resets to None on set_game: first step diff = zeros.

Kill: LS20 < 250. Signal: LS20 >= 250 (diff-encoding useful for navigation).
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
EXT_DIM = ENC_DIM + H_DIM   # 320
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
TEST_SEEDS = list(range(1, 3))
PHASE_STEPS = 10_000
DIAG_STEPS = {1_000, 5_000, 10_000}


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


def softmax_action(vals, temp, rng):
    x = vals / temp; x -= np.max(x); e = np.exp(x)
    return int(rng.choice(len(vals), p=e / (e.sum() + 1e-12)))


class Chain942:
    """916 architecture. Preprocessing: feed |obs_t - obs_{t-1}| instead of obs_t.

    last_obs resets on set_game (first diff = zeros).
    Alpha + running_mean + W_h + W_x persist across games.
    W_pred + h + delta + last_obs reset per game.
    """

    def __init__(self, seed=0):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)

        self._n = None
        self.W_pred = None
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = None
        self.last_obs = None    # reset per game; diff = zeros on first step
        self._prev_ext = None
        self._prev_action = None

        self._phase_step = 0
        self._diag_log = {}

    def set_game(self, n_actions):
        self._n = n_actions
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM + n_actions), dtype=np.float32)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self.last_obs = None
        self._prev_ext = None
        self._prev_action = None
        self._phase_step = 0
        self._diag_log = {}

    def reset_seed(self, seed):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors.clear()
        self._n = None; self.W_pred = None
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = None
        self.last_obs = None
        self._prev_ext = None; self._prev_action = None
        self._phase_step = 0; self._diag_log = {}

    def _encode(self, obs):
        obs_f = np.asarray(obs, dtype=np.float32)
        # Frame differencing: |obs_t - obs_{t-1}|
        if self.last_obs is not None:
            diff = np.abs(obs_f - self.last_obs)
        else:
            diff = np.zeros_like(obs_f)
        self.last_obs = obs_f.copy()

        enc_raw = _enc_frame(diff)
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        ext_enc = np.concatenate([enc, self.h]).astype(np.float32)
        return ext_enc

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

    def process(self, obs):
        ext_enc = self._encode(obs)
        self._phase_step += 1

        if self._prev_ext is not None and self._prev_action is not None:
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

            ext_change = float(np.linalg.norm(self.alpha * (ext_enc - self._prev_ext)))
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * ext_change)

        if self._phase_step in DIAG_STEPS:
            self._diag_log[self._phase_step] = {
                "alpha_conc": float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8)),
                "delta_spread": float(self.delta_per_action.max() - self.delta_per_action.min()),
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
        self.last_obs = None   # reset diff at level boundary

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def diag_str(self):
        parts = []
        for step in sorted(self._diag_log):
            d = self._diag_log[step]
            parts.append(f"@{step//1000}K: alpha_conc={d['alpha_conc']:.2f}"
                         f" delta_spr={d['delta_spread']:.3f}")
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
print("STEP 942 — FRAME DIFFERENCING PREPROCESSING")
print("=" * 70)
print("One variable change from 916: feed |obs_t - obs_{t-1}| to _enc_frame.")
print(f"2 seeds. PRISM-light Mode C, 10K/phase.")
t0 = time.time()

cifar_imgs, cifar_lbls = load_cifar()
cifar1 = []; ls20 = []; ft09 = []; vc33 = []; cifar2 = []

for seed in TEST_SEEDS:
    sub = Chain942(seed=seed)
    sub.reset_seed(seed)
    print(f"\n  Seed {seed}:")

    c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, PHASE_STEPS)
    cifar1.append(c1)
    print(f"    CIFAR-1: acc={c1:.4f}  alpha_conc={sub.alpha_conc():.2f}")
    print(f"             diag: {sub.diag_str()}")

    l = run_arc(sub, "LS20", 4, seed * 1000, PHASE_STEPS)
    ls20.append(l)
    print(f"    LS20:    L1={l:4d}  alpha_conc={sub.alpha_conc():.2f}")
    print(f"             diag: {sub.diag_str()}")

    f = run_arc(sub, "FT09", 68, seed * 1000, PHASE_STEPS)
    ft09.append(f)
    print(f"    FT09:    L1={f:4d}  alpha_conc={sub.alpha_conc():.2f}")
    print(f"             diag: {sub.diag_str()}")

    v = run_arc(sub, "VC33", 68, seed * 1000, PHASE_STEPS)
    vc33.append(v)
    print(f"    VC33:    L1={v:4d}  alpha_conc={sub.alpha_conc():.2f}")
    print(f"             diag: {sub.diag_str()}")

    c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, PHASE_STEPS)
    cifar2.append(c2)
    print(f"    CIFAR-2: acc={c2:.4f}  alpha_conc={sub.alpha_conc():.2f}")
    print(f"             diag: {sub.diag_str()}")

print(f"\n{'=' * 70}")
print(f"STEP 942 RESULTS (frame-diff, PRISM-light, 10K/phase, {len(TEST_SEEDS)} seeds):")
print(f"  LS20  L1: {np.mean(ls20):.1f}/seed  {ls20}")
print(f"  FT09  L1: {np.mean(ft09):.1f}/seed  {ft09}")
print(f"  VC33  L1: {np.mean(vc33):.1f}/seed  {vc33}")
print(f"  CIFAR-1:  {np.mean(cifar1):.4f}")
print(f"\nBaselines: 914 (10K) LS20=248.6  |  916 (25K) LS20=290.7")
print(f"Kill: LS20 < 250.")
print(f"Total elapsed: {time.time() - t0:.1f}s")
print("STEP 942 DONE")
