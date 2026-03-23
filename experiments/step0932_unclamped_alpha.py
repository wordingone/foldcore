"""
step0932_unclamped_alpha.py -- Unclamped alpha (allow dims to die).

R3 hypothesis: removing alpha lower bound allows encoding to self-prune to
informative dimensions per game. Dims unused across many transitions → alpha→0
→ effectively dead. Encoding dimensionality adapts per game domain.

Architecture: 895h + ALPHA_LO=0.0 (was 0.1). All else identical.
- CIFAR: maybe 5 class-discriminative dims survive → less noise
- LS20: most dims survive (all informative for navigation)
- FT09: ~3 dims survive → 800b sees only puzzle-relevant change
- VC33: unknown

Key measurement: how many dims have alpha > 0.01 (effectively alive)?

Run: PRISM-light CIFAR→LS20→FT09→VC33→CIFAR. 25K/phase. 5 seeds.
Compare: 914 (895h chain, ALPHA_LO=0.1, LS20=248.6/seed).
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

TEST_SEEDS = list(range(1, 6))
PHASE_STEPS = 25_000
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
ENC_DIM = 256
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.00   # KEY CHANGE: was 0.10, now 0.0 — dims can die
ALPHA_HI = 5.00


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


def softmax_action(delta, temp):
    x = delta / temp; x = x - np.max(x); e = np.exp(x)
    return e / (e.sum() + 1e-12)


class Chain932:
    """895h with ALPHA_LO=0.0 — encoding self-prunes dead dims.
    Alpha and running_mean persist across games. W + delta reset per game.
    """

    def __init__(self, seed=0):
        self._rng = np.random.RandomState(seed)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        self._n_actions = None
        self.W = None
        self.delta_per_action = None
        self._prev_enc = None
        self._prev_action = None

    def set_game(self, n_actions):
        self._n_actions = n_actions
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._pred_errors.clear()
        self._prev_enc = None
        self._prev_action = None

    def reset_seed(self, seed):
        self._rng = np.random.RandomState(seed)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._pred_errors.clear()
        self._n_actions = None; self.W = None; self.delta_per_action = None
        self._prev_enc = None; self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1; a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

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
                                   one_hot(self._prev_action, self._n_actions)])
            pred = self.W @ inp; error = (enc * self.alpha) - pred
            en = float(np.linalg.norm(error))
            if en > 10.0: error *= 10.0 / en
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()
            change = float(np.linalg.norm((enc - self._prev_enc) * self.alpha))
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * change)
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            probs = softmax_action(self.delta_per_action, SOFTMAX_TEMP)
            action = int(self._rng.choice(self._n_actions, p=probs))
        self._prev_enc = enc.copy(); self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def alive_dims(self, threshold=0.01):
        return int(np.sum(self.alpha > threshold))


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
print("STEP 932 — UNCLAMPED ALPHA (allow dims to die, ALPHA_LO=0.0)")
print("=" * 70)
print(f"Key change: ALPHA_LO=0.0 (was 0.1). Dims can die → encoding self-prunes.")
print(f"Architecture: 895h chain. Alpha persists. W+delta reset per game.")
print(f"Measurement: alive_dims = count(alpha > 0.01).")
t0 = time.time()

cifar_imgs, cifar_lbls = load_cifar()
cifar1 = []; ls20 = []; ft09 = []; vc33 = []; cifar2 = []

for seed in TEST_SEEDS:
    sub = Chain932(seed=seed)
    sub.reset_seed(seed)
    print(f"\n  Seed {seed}:")

    c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, PHASE_STEPS)
    cifar1.append(c1)
    print(f"    CIFAR-1: acc={c1:.4f}  alive={sub.alive_dims():3d}/256  alpha_conc={sub.alpha_conc():.2f}")

    l = run_arc(sub, "LS20", 4, seed * 1000, PHASE_STEPS)
    ls20.append(l)
    print(f"    LS20:    L1={l:4d}  alive={sub.alive_dims():3d}/256  alpha_conc={sub.alpha_conc():.2f}")

    f = run_arc(sub, "FT09", 68, seed * 1000, PHASE_STEPS)
    ft09.append(f)
    print(f"    FT09:    L1={f:4d}  alive={sub.alive_dims():3d}/256  alpha_conc={sub.alpha_conc():.2f}")

    v = run_arc(sub, "VC33", 68, seed * 1000, PHASE_STEPS)
    vc33.append(v)
    print(f"    VC33:    L1={v:4d}  alive={sub.alive_dims():3d}/256  alpha_conc={sub.alpha_conc():.2f}")

    c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, PHASE_STEPS)
    cifar2.append(c2)
    print(f"    CIFAR-2: acc={c2:.4f}  alive={sub.alive_dims():3d}/256  alpha_conc={sub.alpha_conc():.2f}")

print(f"\n{'=' * 70}")
print(f"STEP 932 RESULTS (unclamped alpha, PRISM-light, 25K/phase, 5 seeds):")
print(f"  CIFAR-1 acc: {np.mean(cifar1):.4f}  {[f'{x:.3f}' for x in cifar1]}")
print(f"  LS20  L1:    {np.mean(ls20):.1f}/seed  std={np.std(ls20):.1f}  zero={sum(1 for x in ls20 if x == 0)}/5  {ls20}")
print(f"  FT09  L1:    {np.mean(ft09):.1f}/seed  zero={sum(1 for x in ft09 if x == 0)}/5  {ft09}")
print(f"  VC33  L1:    {np.mean(vc33):.1f}/seed  zero={sum(1 for x in vc33 if x == 0)}/5  {vc33}")
print(f"  CIFAR-2 acc: {np.mean(cifar2):.4f}  {[f'{x:.3f}' for x in cifar2]}")
print(f"\nComparison (chain, 25K, 5 seeds):")
print(f"  914 895h chain (ALPHA_LO=0.1):  LS20=248.6  FT09=0  VC33=0")
print(f"  932 unclamped  (ALPHA_LO=0.0):  LS20={np.mean(ls20):.1f}  FT09={np.mean(ft09):.1f}  VC33={np.mean(vc33):.1f}")
print(f"\nKill criterion: LS20 < 200 (>20% below 914 baseline) → KILL.")
print(f"Total elapsed: {time.time() - t0:.1f}s")
print("STEP 932 DONE")
