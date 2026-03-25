"""
step0986_multipass_chain.py -- Multi-pass chain iteration.

FAMILY: Chain structure modification (965 exact mechanism)
R3 HYPOTHESIS: 5 passes × 2K steps/game instead of 1 pass × 10K. Total budget identical.
W_pred/alpha persist across games AND across passes. h resets each game switch (965 fix).
Cross-game transfer: LS20's navigation learning (W_pred, alpha) may prime FT09/VC33.
Multiple short exposures to each game vs one long exposure.

Kill: LS20 < 67.0 (2K windows too short). Success: FT09>0 OR VC33>0 (cross-game transfer).
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
N_PASSES = 5; STEPS_PER_GAME_PER_PASS = 2000; CIFAR_PER_PASS = 200
TEST_SEEDS = list(range(1, 11))


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp; x -= np.max(x)
    e = np.exp(x); probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class Chain916_986:
    """Exact 965 mechanism. W_pred -=. Change-tracking delta. h-reset on set_game."""

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
                self.W_pred -= ETA_W * np.outer(error, self._prev_ext)
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


def run_cifar_steps(sub, imgs, lbls, rng, n_steps):
    if imgs is None: return 0
    sub.set_game(100)
    idx = rng.permutation(len(imgs))[:n_steps]
    return sum(1 for i in idx if sub.process(imgs[i]) % 100 == lbls[i])


def run_arc_steps(sub, env, game_env, n_actions, seed, n_steps):
    """Run n_steps on existing env state, or reset if needed."""
    sub.set_game(n_actions)
    obs = game_env.reset(seed=seed)
    step = 0; completions = 0; level = 0
    while step < n_steps:
        if obs is None: obs = game_env.reset(seed=seed); sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
        obs, _, done, info = game_env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            completions += cl - level; level = cl; sub.on_level_transition()
        if done: obs = game_env.reset(seed=seed); level = 0; sub.on_level_transition()
    return completions


def run_multipass_chain(seeds, n_passes, steps_per_game, cifar_per_pass, cifar_imgs, cifar_lbls):
    lsl, ftl, vcl = [], [], []
    for seed in seeds:
        sub = Chain916_986(seed=seed)
        rng = np.random.RandomState(seed * 1000)
        ls_env = make_env("LS20"); ft_env = make_env("FT09"); vc_env = make_env("VC33")
        ls_total = 0; ft_total = 0; vc_total = 0

        for p in range(n_passes):
            run_cifar_steps(sub, cifar_imgs, cifar_lbls, rng, cifar_per_pass)
            ls_total += run_arc_steps(sub, ls_env, ls_env, 4, seed * 1000, steps_per_game)
            ft_total += run_arc_steps(sub, ft_env, ft_env, 68, seed * 1000, steps_per_game)
            vc_total += run_arc_steps(sub, vc_env, vc_env, 68, seed * 1000, steps_per_game)
            run_cifar_steps(sub, cifar_imgs, cifar_lbls, rng, cifar_per_pass)

        lsl.append(ls_total); ftl.append(ft_total); vcl.append(vc_total)
        print(f"  seed={seed}: LS20={ls_total:4d} FT09={ft_total:4d} VC33={vc_total:4d}"
              f"  alpha_conc={sub.alpha_conc():.1f}")
    return lsl, ftl, vcl


if __name__ == "__main__":
    import os
    print("=" * 70)
    print(f"STEP 986 — MULTI-PASS CHAIN ({N_PASSES} passes × {STEPS_PER_GAME_PER_PASS} steps/game)")
    print("=" * 70)
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), '?')
    vc33_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/vc33') if len(d) >= 8), '?')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}  VC33={vc33_hash}")
    print(f"965 exact mechanism. {N_PASSES} × {STEPS_PER_GAME_PER_PASS} = {N_PASSES*STEPS_PER_GAME_PER_PASS} steps/game total.")
    print(f"W_pred/alpha persist. h resets each game switch. CIFAR={CIFAR_PER_PASS}/pass.")
    print(f"Kill: LS20<67.0  Success: FT09>0 OR VC33>0")
    print()

    t0 = time.time()
    cifar_imgs, cifar_lbls = load_cifar()
    ls, ft, vc = run_multipass_chain(TEST_SEEDS, N_PASSES, STEPS_PER_GAME_PER_PASS,
                                      CIFAR_PER_PASS, cifar_imgs, cifar_lbls)

    print()
    print("=" * 70)
    print("STEP 986 RESULTS (965 chain: LS20=67.0):")
    print(f"  LS20: {np.mean(ls):.1f}/seed  nonzero={sum(1 for x in ls if x > 0)}/10  {ls}")
    print(f"  FT09: {np.mean(ft):.1f}/seed  nonzero={sum(1 for x in ft if x > 0)}/10  {ft}")
    print(f"  VC33: {np.mean(vc):.1f}/seed  nonzero={sum(1 for x in vc if x > 0)}/10  {vc}")
    ls_v = f"LS20 PASS ({np.mean(ls):.1f})" if np.mean(ls) >= 60.3 else f"LS20 DEGRADED ({np.mean(ls):.1f})"
    ft_v = f"FT09 SIGNAL ({sum(1 for x in ft if x>0)}/10)" if any(x>0 for x in ft) else "FT09 ZERO"
    vc_v = f"VC33 SIGNAL ({sum(1 for x in vc if x>0)}/10)" if any(x>0 for x in vc) else "VC33 ZERO"
    print(f"  {ls_v}  |  {ft_v}  |  {vc_v}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("STEP 986 DONE")
