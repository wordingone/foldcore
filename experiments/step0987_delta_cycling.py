"""
step0987_delta_cycling.py -- Delta-drop sequential action cycling.

FAMILY: Delta-gated selection (965 exact base)
R3 HYPOTHESIS: When change-delta drops below 50% of recent peak, advance to next action
sequentially. "Current action exhausted → try next." Discovers ordered sequences IF correct
actions produce delta spikes. No change to W_pred(320), 800b, EXT_DIM, or 10K budget.

peak_delta decays per step (PEAK_DECAY=0.95). Cycling fires only when peak is meaningful
(peak_delta > 1.0) and delta dropped significantly. ~5-10% of steps affected.

Kill: LS20 < 67.0. Success: FT09 > 0 OR VC33 > 0.
Chain: CIFAR(1K) → LS20(10K) → FT09(10K) → VC33(10K) → CIFAR(1K). 10 seeds.
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
PEAK_DECAY = 0.95; DROP_RATIO = 0.50; PEAK_MIN = 1.0
TEST_SEEDS = list(range(1, 11)); PHASE_STEPS = 10_000; CIFAR_STEPS = 1_000


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp; x -= np.max(x)
    e = np.exp(x); probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class DeltaCycling987:
    """965 base + delta-drop sequential cycling. ONE change from 965."""

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
        self._peak_delta = 0.0  # tracks recent peak of change-delta

    def set_game(self, n_actions):
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._n_actions = n_actions
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._prev_ext = None; self._prev_action = None
        self._peak_delta = 0.0

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
        change = 0.0
        if self._prev_ext is not None and self._prev_action is not None:
            pred = self.W_pred @ self._prev_ext
            error = (ext_enc * self.alpha) - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0: error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W * np.outer(error, self._prev_ext)
                self._pred_errors.append(np.abs(error)); self._update_alpha()
            weighted_delta = (ext_enc - self._prev_ext) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (1 - ALPHA_EMA) * self.delta_per_action[a] + ALPHA_EMA * change

        # Update peak (decay each step, then update with current change)
        self._peak_delta = max(self._peak_delta * PEAK_DECAY, change)

        # Action selection: delta-drop cycling OR standard 800b
        if (self._prev_action is not None
                and self._peak_delta > PEAK_MIN
                and change < DROP_RATIO * self._peak_delta):
            # Delta dropped from peak: current action exhausted → advance to next
            action = (self._prev_action + 1) % self._n_actions
        elif self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_sel(self.delta_per_action, SOFTMAX_TEMP, self._rng)

        self._prev_ext = ext_enc.copy(); self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None; self._prev_action = None
        self._peak_delta = 0.0

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
    c1l, lsl, ftl, vcl, c2l = [], [], [], [], []
    for seed in seeds:
        sub = DeltaCycling987(seed=seed)
        c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, cifar_steps)
        l = run_arc(sub, "LS20", 4, seed * 1000, n_steps)
        f = run_arc(sub, "FT09", 68, seed * 1000, n_steps)
        v = run_arc(sub, "VC33", 68, seed * 1000, n_steps)
        c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, cifar_steps)
        c1l.append(c1); lsl.append(l); ftl.append(f); vcl.append(v); c2l.append(c2)
        print(f"  seed={seed}: C1={c1:.3f} LS20={l:4d} FT09={f:4d} VC33={v:4d} C2={c2:.3f}"
              f"  alpha_conc={sub.alpha_conc():.1f}")
    return c1l, lsl, ftl, vcl, c2l


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("STEP 987 — DELTA-DROP SEQUENTIAL CYCLING")
    print("=" * 70)
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), '?')
    vc33_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/vc33') if len(d) >= 8), '?')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}  VC33={vc33_hash}")
    print(f"965 base + peak tracking. Cycling fires when change < {DROP_RATIO}*peak AND peak>{PEAK_MIN}.")
    print(f"peak_decay={PEAK_DECAY}. No change to W_pred(320), EXT_DIM, 800b, or 10K budget.")
    print(f"Kill: LS20<67.0  Success: FT09>0 OR VC33>0")
    print()

    t0 = time.time()
    cifar_imgs, cifar_lbls = load_cifar()
    c1, ls, ft, vc, c2 = run_chain(TEST_SEEDS, PHASE_STEPS, CIFAR_STEPS, cifar_imgs, cifar_lbls)

    print()
    print("=" * 70)
    print("STEP 987 RESULTS (965 chain: LS20=67.0):")
    print(f"  CIFAR-1: {np.mean(c1):.3f}")
    print(f"  LS20:    {np.mean(ls):.1f}/seed  nonzero={sum(1 for x in ls if x > 0)}/10  {ls}")
    print(f"  FT09:    {np.mean(ft):.1f}/seed  nonzero={sum(1 for x in ft if x > 0)}/10  {ft}")
    print(f"  VC33:    {np.mean(vc):.1f}/seed  nonzero={sum(1 for x in vc if x > 0)}/10  {vc}")
    print(f"  CIFAR-2: {np.mean(c2):.3f}")
    ls_v = f"LS20 PASS ({np.mean(ls):.1f})" if np.mean(ls) >= 60.3 else f"LS20 DEGRADED ({np.mean(ls):.1f})"
    ft_v = f"FT09 SIGNAL ({sum(1 for x in ft if x>0)}/10)" if any(x>0 for x in ft) else "FT09 ZERO"
    vc_v = f"VC33 SIGNAL ({sum(1 for x in vc if x>0)}/10)" if any(x>0 for x in vc) else "VC33 ZERO"
    print(f"  {ls_v}  |  {ft_v}  |  {vc_v}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("STEP 987 DONE")
