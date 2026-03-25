"""
step0938d_alpha_enc_delta_h.py -- Alpha-weighted enc-only delta + h perturbation.

R3 hypothesis: Alpha-weighted enc-only delta preserves discrimination (what 916 has)
without h contamination (what 938c fixed), plus h position-dependence through R@h.
As alpha changes (R3), delta discrimination changes AND h projection changes.

938c failure: raw enc delta loses discrimination (alpha removed).
938d fix: delta uses alpha[:ENC_DIM] * (enc - prev_enc) — alpha provides discrimination,
enc-only means h never contaminates delta.

Architecture: 916 encoding.
  alpha_enc = alpha[:ENC_DIM]  (256D, no h dims)
  delta_enc_change = ||alpha_enc * (enc_t - enc_{t-1})||
  delta_ema[a] = 0.9*delta[a] + 0.1*delta_enc_change
  h_bias = |R @ h|  (R fixed, seed=42, shape n_actions×H_DIM)
  score = delta_ema + beta * h_bias
  action = softmax(score/T) with epsilon=0.20

Kill: LS20 < 250/seed at best beta → KILL.
SOTA: LS20 > 290 (916 standalone) → better than current best.
Run: PRISM-light Mode C, 10K/phase, 2 seeds. Beta sweep [0.1, 0.5, 1.0].
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
R_SEED = 42
TEST_SEEDS = list(range(1, 3))
PHASE_STEPS = 10_000
DIAG_STEPS = {1_000, 5_000, 10_000}
BETAS = [0.1, 0.5, 1.0]


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


def softmax_action(vals, temp, rng):
    x = vals / temp; x -= np.max(x); e = np.exp(x)
    return int(rng.choice(len(vals), p=e / (e.sum() + 1e-12)))


class Chain938d:
    """916 encoding. delta_ema on alpha-weighted enc (no h) + beta * |R@h|.

    Key fix over 938c: delta uses alpha[:ENC_DIM] * (enc - prev_enc),
    NOT raw enc. Alpha provides discrimination without h contamination.
    """

    def __init__(self, seed=0, beta=0.5):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)
        self.beta = beta

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
        self.R = None
        self._prev_ext = None
        self._prev_enc = None
        self._prev_action = None

        self._phase_step = 0
        self._diag_log = {}

    def set_game(self, n_actions):
        self._n = n_actions
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM + n_actions), dtype=np.float32)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        rr = np.random.RandomState(R_SEED)
        self.R = rr.randn(n_actions, H_DIM).astype(np.float32) * 0.1
        self._prev_ext = None
        self._prev_enc = None
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
        self.delta_per_action = None; self.R = None
        self._prev_ext = None; self._prev_enc = None; self._prev_action = None
        self._phase_step = 0; self._diag_log = {}

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        ext_enc = np.concatenate([enc, self.h]).astype(np.float32)
        return enc, ext_enc

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
        enc, ext_enc = self._encode(obs)
        self._phase_step += 1

        if self._prev_ext is not None and self._prev_action is not None:
            # W_pred training (same as 916, alpha-weighted ext_enc)
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

            # Delta: alpha-weighted enc-ONLY change (THE FIX: alpha discrimination, no h)
            alpha_enc = self.alpha[:ENC_DIM]   # 256D only
            enc_change = float(np.linalg.norm(alpha_enc * (enc - self._prev_enc)))
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * enc_change)

        # h-projected perturbation: |R @ h|
        h_bias = np.abs(self.R @ self.h)

        # Combined score
        score = self.delta_per_action + self.beta * h_bias

        if self._phase_step in DIAG_STEPS:
            self._diag_log[self._phase_step] = {
                "alpha_conc": float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8)),
                "delta_spread": float(self.delta_per_action.max() - self.delta_per_action.min()),
                "h_bias_spread": float(h_bias.max() - h_bias.min()),
                "score_spread": float(score.max() - score.min()),
            }

        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n))
        else:
            action = softmax_action(score, SOFTMAX_TEMP, self._rng)

        self._prev_ext = ext_enc.copy()
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None
        self._prev_enc = None
        self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def diag_str(self):
        parts = []
        for step in sorted(self._diag_log):
            d = self._diag_log[step]
            parts.append(f"@{step//1000}K: alpha_conc={d['alpha_conc']:.2f}"
                         f" delta_spr={d['delta_spread']:.3f}"
                         f" h_spr={d['h_bias_spread']:.3f}"
                         f" score_spr={d['score_spread']:.3f}")
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
print("STEP 938d — ALPHA-ENC-ONLY DELTA + h PERTURBATION")
print("=" * 70)
print("Fix: delta uses alpha[:ENC_DIM]*(enc-prev_enc). Alpha discrimination + no h contamination.")
print(f"Beta sweep: {BETAS}. 2 seeds each.")
t0 = time.time()

cifar_imgs, cifar_lbls = load_cifar()
all_results = {}

for beta in BETAS:
    print(f"\n{'─'*60}")
    print(f"  BETA = {beta}")
    print(f"{'─'*60}")
    cifar1 = []; ls20 = []; ft09 = []; vc33 = []; cifar2 = []

    for seed in TEST_SEEDS:
        sub = Chain938d(seed=seed, beta=beta)
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

    all_results[beta] = {"ls20": ls20, "ft09": ft09, "vc33": vc33,
                         "cifar1": cifar1, "cifar2": cifar2}

print(f"\n{'=' * 70}")
print(f"STEP 938d RESULTS (alpha-enc-only delta + h, PRISM-light, 10K/phase, {len(TEST_SEEDS)} seeds):")
print(f"{'Beta':>6}  {'LS20':>10}  {'FT09':>8}  {'VC33':>8}  CIFAR-1")
for beta in BETAS:
    r = all_results[beta]
    print(f"  {beta:>4}:  LS20={np.mean(r['ls20']):6.1f}/seed"
          f"  FT09={np.mean(r['ft09']):5.1f}"
          f"  VC33={np.mean(r['vc33']):5.1f}"
          f"  CIFAR={np.mean(r['cifar1']):.4f}")
print(f"\nBaselines (chain, 5 seeds):")
print(f"  914 (800b alpha-ext_enc, 10K): LS20=248.6")
print(f"  916 (800b alpha-ext_enc, 25K): LS20=290.7")
print(f"  938c (raw-enc delta + h, 10K): LS20=65.5 best")
print(f"\nKill: LS20 < 250 at best beta. SOTA: LS20 > 290.")
print(f"Total elapsed: {time.time() - t0:.1f}s")
print("STEP 938d DONE")
