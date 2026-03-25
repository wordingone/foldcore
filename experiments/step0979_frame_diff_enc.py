"""
step0979_frame_diff_enc.py -- Frame-diff augmented encoding on 916 base.

FAMILY: Encoding augmentation (916 base with h-reset from 965)
R3 HYPOTHESIS: Augment enc with frame difference — substrate sees WHAT CHANGED, not
just what's there. enc = cat[avgpool16(obs), avgpool16(|obs-prev_obs|)] (512 dims).
W_pred predicts 512-dim enc: predicts BOTH state and changes per action.
800b tracks which actions produce UNPREDICTED changes (high delta).

For FT09: clicking right tile → pixels change → frame diff is large in that region →
W_pred learns change patterns per action → 800b discovers which actions do SOMETHING.
For LS20: movement → frame diff = movement direction → W_pred predicts movement patterns.
For CIFAR: iid images → diff = random → adds noise (acceptable — same for current enc).

Changes from 965: ENC_DIM=512 (256 static + 256 diff). EXT_DIM=576. W_pred(576,576).
W_h(64,64), W_x(64,512). Alpha(576). enc_mean(512). Frame diff uses prev_obs_raw.
h resets on game switch (965 fix). prev_obs_raw resets too (don't carry frame across games).

Chain: CIFAR(1K) → LS20(10K) → FT09(10K) → VC33(10K) → CIFAR(1K). 10 seeds.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM_STATIC = 256   # avgpool16(obs)
ENC_DIM_DIFF   = 256   # avgpool16(|obs - prev_obs|)
ENC_DIM        = ENC_DIM_STATIC + ENC_DIM_DIFF  # 512
H_DIM          = 64
EXT_DIM        = ENC_DIM + H_DIM               # 576
ETA_W          = 0.01
ALPHA_EMA      = 0.10
INIT_DELTA     = 1.0
ALPHA_UPDATE_DELAY = 50
ALPHA_LO       = 0.10
ALPHA_HI       = 5.00
EPSILON        = 0.20
SOFTMAX_TEMP   = 0.10
TEST_SEEDS     = list(range(1, 11))
PHASE_STEPS    = 10_000
CIFAR_STEPS    = 1_000


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp
    x -= np.max(x)
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class FrameDiffEnc979:
    """916 base + frame-diff augmented encoding. ENC_DIM=512, EXT_DIM=576."""

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Fixed random reservoir (W_x now maps 512→64)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # W_pred: (576, 576) predicts ext_enc from prev_ext
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM), dtype=np.float32)

        # Alpha: persistent (576 dims)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)

        # Encoding running mean: persistent (512 dims)
        self._running_mean_enc = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

        # Game-specific (reset on set_game)
        self._n_actions = 4
        self.running_mean = None
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._prev_ext = None
        self._prev_action = None
        self._prev_obs_raw = None  # for frame diff

    def set_game(self, n_actions):
        # h-reset (from 965)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._n_actions = n_actions
        self.running_mean = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._prev_ext = None
        self._prev_action = None
        self._prev_obs_raw = None  # reset prev obs — don't carry frame diff across games
        # W_pred, alpha, enc_mean persist

    def _encode(self, obs):
        obs_arr = np.asarray(obs, dtype=np.float32)

        # Static encoding: avgpool16(obs) → 256 dims
        enc_static = _enc_frame(obs_arr)

        # Diff encoding: avgpool16(|obs - prev_obs|) → 256 dims
        if self._prev_obs_raw is not None:
            diff = np.abs(obs_arr - self._prev_obs_raw)
            enc_diff = _enc_frame(diff)
        else:
            enc_diff = np.zeros(ENC_DIM_DIFF, dtype=np.float32)

        self._prev_obs_raw = obs_arr.copy()

        # Full 512-dim encoding
        enc_raw = np.concatenate([enc_static, enc_diff]).astype(np.float32)

        # Running mean normalization (persistent, 512-dim)
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean_enc = (1 - a) * self._running_mean_enc + a * enc_raw
        enc = enc_raw - self._running_mean_enc

        # Recurrent state
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        ext_enc = np.concatenate([enc, self.h]).astype(np.float32)
        return ext_enc

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY:
            return
        mean_errors = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(mean_errors)) or np.any(np.isinf(mean_errors)):
            return
        raw_alpha = np.sqrt(np.clip(mean_errors, 0, 1e6) + 1e-8)
        mean_raw = np.mean(raw_alpha)
        if mean_raw < 1e-8 or np.isnan(mean_raw):
            return
        self.alpha = np.clip(raw_alpha / mean_raw, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        ext_enc = self._encode(obs)

        if self._prev_ext is not None and self._prev_action is not None:
            target = ext_enc * self.alpha
            pred = self.W_pred @ self._prev_ext
            error = target - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
                err_norm = 10.0
            if not np.any(np.isnan(error)):
                self.W_pred += ETA_W * np.outer(error, self._prev_ext)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            delta = err_norm
            a = self._prev_action
            self.running_mean[a] = (1 - ALPHA_EMA) * self.running_mean[a] + ALPHA_EMA * delta

        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_sel(self.running_mean, SOFTMAX_TEMP, self._rng)

        self._prev_ext = ext_enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None
        self._prev_action = None
        self._prev_obs_raw = None  # clear frame diff on level reset

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
        imgs = np.array([np.array(ds[i][0]).transpose(1, 2, 0) for i in range(len(ds))],
                        dtype=np.float32)
        lbls = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
        return imgs, lbls
    except Exception as e:
        print(f"  CIFAR load failed: {e}"); return None, None


def run_cifar(sub, imgs, lbls, seed, n_steps):
    if imgs is None:
        return 0.0
    sub.set_game(100)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(imgs))[:n_steps]
    correct = sum(1 for i in idx if sub.process(imgs[i]) % 100 == lbls[i])
    return correct / len(idx)


def run_arc(sub, game, n_actions, seed, n_steps):
    sub.set_game(n_actions)
    env = make_env(game)
    obs = env.reset(seed=seed)
    step = 0; completions = 0; level = 0
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


def run_chain(seeds, n_steps, cifar_steps, cifar_imgs, cifar_lbls):
    cifar1_list, ls20_list, ft09_list, vc33_list, cifar2_list = [], [], [], [], []
    for seed in seeds:
        sub = FrameDiffEnc979(seed=seed)
        c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, cifar_steps)
        l = run_arc(sub, "LS20", 4, seed * 1000, n_steps)
        f = run_arc(sub, "FT09", 68, seed * 1000, n_steps)
        v = run_arc(sub, "VC33", 68, seed * 1000, n_steps)
        c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, cifar_steps)
        cifar1_list.append(c1); ls20_list.append(l); ft09_list.append(f)
        vc33_list.append(v); cifar2_list.append(c2)
        print(f"  seed={seed}: CIFAR1={c1:.3f} LS20={l:4d} FT09={f:4d} VC33={v:4d} CIFAR2={c2:.3f}"
              f"  alpha_conc={sub.alpha_conc():.1f}")
    return cifar1_list, ls20_list, ft09_list, vc33_list, cifar2_list


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("STEP 979 — FRAME-DIFF AUGMENTED ENCODING (enc=512: static+diff)")
    print("=" * 70)
    t0 = time.time()
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), '?')
    vc33_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/vc33') if len(d) >= 8), '?')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}  VC33={vc33_hash}")
    print(f"Steps: ARC={PHASE_STEPS}  CIFAR={CIFAR_STEPS}  Seeds={TEST_SEEDS}")
    print(f"ENC_DIM={ENC_DIM} (static={ENC_DIM_STATIC} + diff={ENC_DIM_DIFF})")
    print(f"EXT_DIM={EXT_DIM}. W_pred({EXT_DIM},{EXT_DIM}). W_x({H_DIM},{ENC_DIM}).")
    print(f"h resets on game switch. prev_obs_raw resets. W_pred/alpha/enc_mean persist.")
    print()

    cifar_imgs, cifar_lbls = load_cifar()

    c1, ls, ft, vc, c2 = run_chain(TEST_SEEDS, PHASE_STEPS, CIFAR_STEPS,
                                    cifar_imgs, cifar_lbls)

    print()
    print("=" * 70)
    print(f"STEP 979 RESULTS (965 chain: LS20=67.0, standalone 916@10K=74.7):")
    print(f"  CIFAR-1: {np.mean(c1):.3f} (chance=0.010)")
    print(f"  LS20:    {np.mean(ls):.1f}/seed  nonzero={sum(1 for x in ls if x > 0)}/10  {ls}")
    print(f"  FT09:    {np.mean(ft):.1f}/seed  nonzero={sum(1 for x in ft if x > 0)}/10  {ft}")
    print(f"  VC33:    {np.mean(vc):.1f}/seed  nonzero={sum(1 for x in vc if x > 0)}/10  {vc}")
    print(f"  CIFAR-2: {np.mean(c2):.3f}")
    ls20_baseline = 67.0
    chain_verdict = (f"LS20 PASS ({np.mean(ls):.1f} ≥ {ls20_baseline*0.9:.1f})"
                     if np.mean(ls) >= ls20_baseline * 0.9
                     else f"LS20 DEGRADED ({np.mean(ls):.1f} < {ls20_baseline*0.9:.1f})")
    ft09_verdict = (f"FT09 SIGNAL ({sum(1 for x in ft if x > 0)}/10 nonzero)"
                    if any(x > 0 for x in ft) else "FT09 ZERO (0/10)")
    vc33_verdict = (f"VC33 SIGNAL ({sum(1 for x in vc if x > 0)}/10 nonzero)"
                    if any(x > 0 for x in vc) else "VC33 ZERO (0/10)")
    print(f"  Chain verdict: {chain_verdict}")
    print(f"  FT09: {ft09_verdict}  |  VC33: {vc33_verdict}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("STEP 979 DONE")
