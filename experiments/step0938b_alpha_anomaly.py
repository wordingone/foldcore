"""
step0938b_alpha_anomaly.py -- Alpha-Anomaly Reactive Selector.

R3 hypothesis: Alpha-weighted anomaly score provides position-dependent action
selection WITHOUT per-observation memory. As alpha changes (R3), the anomaly
pattern changes → different actions at different positions. The action selector
adapts because the observation encoding changes as alpha reweights dimensions.

Architecture: Full 916 encoding pipeline (alpha, running_mean, W_h, W_x, h).
800b per-action delta EMA replaced entirely with anomaly-reactive selector.

Formula:
  anomaly[d] = alpha[d] * enc[d]^2   (enc already centered: enc_raw - running_mean)
  top_k_dims = argsort(anomaly)[-K:]  where K = min(8, n_actions)
  action_score[d % n_actions] += anomaly[d]  for d in top_k_dims
  action = argmax(action_score)

No per-observation storage. No per-action EMA. Purely reactive:
f(current_obs, global_alpha, global_running_mean) → action.

Kill criteria: LS20 L1=0 AND no better than random (random=248/seed at 10K) → KILL.
Interesting: LS20 L1 >= 50/seed (anomaly has weak navigation signal).
Run: PRISM-light Mode C, 10K steps/phase, 2 seeds minimum.
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
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.20
ALPHA_LO = 0.10
ALPHA_HI = 5.00
TOP_K = 8
TEST_SEEDS = list(range(1, 3))   # 2 seeds minimum per spec
PHASE_STEPS = 10_000
DIAG_STEPS = {1_000, 5_000, 10_000}


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


class Chain938b:
    """916 encoding pipeline. Action selector = alpha-anomaly reactive (no per-obs storage).

    Alpha + running_mean persist across games. W_pred + h reset per game.
    No delta_per_action, no per-observation tables.
    """

    def __init__(self, seed=0):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        self.alpha = np.ones(ENC_DIM, dtype=np.float32)   # enc dims only
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)

        self._n = None
        self.W_pred = None
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._prev_ext = None
        self._prev_action = None

        self._phase_step = 0
        self._diag_log = {}

    def set_game(self, n_actions):
        self._n = n_actions
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM + n_actions), dtype=np.float32)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._prev_ext = None
        self._prev_action = None
        self._phase_step = 0
        self._diag_log = {}

    def reset_seed(self, seed):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors.clear()
        self._n = None; self.W_pred = None
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._prev_ext = None; self._prev_action = None
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
        ra = np.sqrt(np.clip(me[:ENC_DIM], 0, 1e6) + 1e-8)
        mr = np.mean(ra)
        if mr < 1e-8 or np.isnan(mr):
            return
        self.alpha = np.clip(ra / mr, ALPHA_LO, ALPHA_HI)

    def _select_anomaly_action(self, enc):
        """Reactive: anomaly[d] = alpha[d] * enc[d]^2. Top-K dims → action scores."""
        anomaly = self.alpha * enc ** 2   # 256D, all non-negative
        k = min(TOP_K, self._n)
        top_k = np.argsort(anomaly)[-k:]   # indices of top-K anomalous dims
        action_score = np.zeros(self._n, dtype=np.float32)
        for d in top_k:
            action_score[int(d) % self._n] += anomaly[d]
        return int(np.argmax(action_score)), anomaly, action_score

    def process(self, obs):
        enc, ext_enc = self._encode(obs)
        self._phase_step += 1

        if self._prev_ext is not None and self._prev_action is not None:
            # W_pred training (same as 916 — alpha weighting on EXT_DIM)
            alpha_ext = np.concatenate([self.alpha, np.ones(H_DIM, dtype=np.float32)])
            inp = np.concatenate([self._prev_ext * alpha_ext,
                                   one_hot(self._prev_action, self._n)])
            pred = self.W_pred @ inp
            error = (ext_enc * alpha_ext) - pred
            en = float(np.linalg.norm(error))
            if en > 10.0:
                error = error * (10.0 / en)
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

        # Anomaly-reactive action selection
        anomaly_action, anomaly, action_score = self._select_anomaly_action(enc)

        # Diagnostic snapshots
        if self._phase_step in DIAG_STEPS:
            self._diag_log[self._phase_step] = {
                "alpha_conc": float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8)),
                "anomaly_max": float(anomaly.max()),
                "anomaly_spread": float(anomaly.max() - anomaly.min()),
                "score_spread": float(action_score.max() - action_score.min()),
                "top_action": anomaly_action,
            }

        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n))
        else:
            action = anomaly_action

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
                         f" anom_spread={d['anomaly_spread']:.3f}"
                         f" score_spread={d['score_spread']:.3f}"
                         f" top_a={d['top_action']}")
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
print("STEP 938b — ALPHA-ANOMALY REACTIVE SELECTOR")
print("=" * 70)
print("R3: anomaly[d]=alpha[d]*enc[d]^2 → top-K dims → action_score[d%n].")
print("Purely reactive: no per-obs storage, no per-action EMA.")
t0 = time.time()

cifar_imgs, cifar_lbls = load_cifar()
cifar1 = []; ls20 = []; ft09 = []; vc33 = []; cifar2 = []

for seed in TEST_SEEDS:
    sub = Chain938b(seed=seed)
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
print(f"STEP 938b RESULTS (alpha-anomaly reactive, PRISM-light, 10K/phase, {len(TEST_SEEDS)} seeds):")
print(f"  CIFAR-1 acc: {np.mean(cifar1):.4f}  {[f'{x:.3f}' for x in cifar1]}")
print(f"  LS20  L1:    {np.mean(ls20):.1f}/seed  std={np.std(ls20):.1f}  zero={sum(1 for x in ls20 if x == 0)}/{len(TEST_SEEDS)}  {ls20}")
print(f"  FT09  L1:    {np.mean(ft09):.1f}/seed  zero={sum(1 for x in ft09 if x == 0)}/{len(TEST_SEEDS)}  {ft09}")
print(f"  VC33  L1:    {np.mean(vc33):.1f}/seed  zero={sum(1 for x in vc33 if x == 0)}/{len(TEST_SEEDS)}  {vc33}")
print(f"  CIFAR-2 acc: {np.mean(cifar2):.4f}  {[f'{x:.3f}' for x in cifar2]}")
print(f"\nComparison (chain, 5 seeds):")
print(f"  914 (800b EMA, 10K):        LS20=248.6  (random baseline ~248)")
print(f"  916 (800b EMA, 25K):        LS20=290.7")
print(f"  938b (anomaly reactive, 10K): LS20={np.mean(ls20):.1f}  FT09={np.mean(ft09):.1f}  VC33={np.mean(vc33):.1f}")
print(f"\nKill: LS20=0 AND no better than random (248/seed). Interesting: LS20>=50/seed.")
print(f"Total elapsed: {time.time() - t0:.1f}s")
print("STEP 938b DONE")
