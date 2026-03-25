"""
step0976_ensemble_disagreement.py -- K=3 W_pred ensemble on 916 base, disagreement explore.

FAMILY: Ensemble prediction disagreement (916 base with h-reset from 965)
R3 HYPOTHESIS: K=3 W_pred(320,320) models (different random init, persistent across games)
diverge on genuinely uncertain state transitions but converge on noise. Per-action
running_disagreement = EMA of inter-model delta variance when action a taken.
High disagreement = genuine uncertainty → explore. Low = noise or explored → don't.

Avoids noisy TV: noise → all models equally wrong → low variance → low disagreement.
Structured transitions (FT09 sequence, VC33 zones) → models diverge → high disagreement.

Changes from 965: W_pred is K=3 matrix array (different random init, same (320,320) shape).
running_disagreement[a] replaces running_mean[a]. Action = softmax(running_disagreement).
916's delta = ||ext_enc - W_pred @ prev_ext|| → disagreement = var(delta_k across K models).
W_pred persists across games (same as 965, no n_actions dependency). h resets on switch.

Chain: CIFAR(1K) → LS20(10K) → FT09(10K) → VC33(10K) → CIFAR(1K). 10 seeds.
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
K_ENSEMBLE = 3               # W_pred ensemble size
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
ALPHA_LO = 0.10
ALPHA_HI = 5.00
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
TEST_SEEDS = list(range(1, 11))
PHASE_STEPS = 10_000
CIFAR_STEPS = 1_000


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp
    x -= np.max(x)
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class EnsembleDisagreement976:
    """916 base + K=3 W_pred ensemble. Disagreement = var(delta_k). W_pred persists."""

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Fixed random reservoir (same as 916)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # K W_pred(320,320) models: sequential rs calls give different matrices
        # Small random init so models start diverse (not zeros)
        self.W_preds = [
            rs.randn(EXT_DIM, EXT_DIM).astype(np.float32) * 0.01
            for _ in range(K_ENSEMBLE)
        ]

        # Alpha: persistent (use model-0 errors for alpha update, same as 916)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)

        # Encoding running mean: persistent
        self._running_mean_enc = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

        # Game-specific (reset on set_game)
        self._n_actions = 4
        self.running_disagreement = None
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._prev_ext = None
        self._prev_action = None

    def set_game(self, n_actions):
        # h-reset (from 965)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._n_actions = n_actions
        self.running_disagreement = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._prev_ext = None
        self._prev_action = None
        # W_preds NEVER reset (no n_actions dependency, (320,320) always)

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean_enc = (1 - a) * self._running_mean_enc + a * enc_raw
        enc = enc_raw - self._running_mean_enc
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
            target = ext_enc * self.alpha  # alpha-weighted target (same as 916)
            delta_k = []
            for k, W in enumerate(self.W_preds):
                pred = W @ self._prev_ext
                error = target - pred
                err_norm = float(np.linalg.norm(error))
                if err_norm > 10.0:
                    error = error * (10.0 / err_norm)
                    err_norm = 10.0
                if not np.any(np.isnan(error)):
                    self.W_preds[k] += ETA_W * np.outer(error, self._prev_ext)
                    if k == 0:
                        self._pred_errors.append(np.abs(error))
                delta_k.append(err_norm)

            # Disagreement = variance of per-model deltas
            # Low var = all models equally bad (noise) or equally good (explored)
            # High var = genuine uncertainty (models learned different predictions)
            disagreement = float(np.var(delta_k))

            a = self._prev_action
            self.running_disagreement[a] = (
                (1 - ALPHA_EMA) * self.running_disagreement[a] + ALPHA_EMA * disagreement
            )
            self._update_alpha()

        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_sel(self.running_disagreement, SOFTMAX_TEMP, self._rng)

        self._prev_ext = ext_enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None
        self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def disagreement_spread(self):
        if self.running_disagreement is None:
            return 0.0
        return float(np.max(self.running_disagreement) - np.min(self.running_disagreement))


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
        sub = EnsembleDisagreement976(seed=seed)
        c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, cifar_steps)
        l = run_arc(sub, "LS20", 4, seed * 1000, n_steps)
        f = run_arc(sub, "FT09", 68, seed * 1000, n_steps)
        v = run_arc(sub, "VC33", 68, seed * 1000, n_steps)
        c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, cifar_steps)
        cifar1_list.append(c1); ls20_list.append(l); ft09_list.append(f)
        vc33_list.append(v); cifar2_list.append(c2)
        print(f"  seed={seed}: CIFAR1={c1:.3f} LS20={l:4d} FT09={f:4d} VC33={v:4d} CIFAR2={c2:.3f}"
              f"  alpha_conc={sub.alpha_conc():.1f}  disag_spread={sub.disagreement_spread():.4f}")
    return cifar1_list, ls20_list, ft09_list, vc33_list, cifar2_list


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("STEP 976 — ENSEMBLE DISAGREEMENT (K=3 W_pred(320,320), 916 base)")
    print("=" * 70)
    t0 = time.time()
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), '?')
    vc33_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/vc33') if len(d) >= 8), '?')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}  VC33={vc33_hash}")
    print(f"Steps: ARC={PHASE_STEPS}  CIFAR={CIFAR_STEPS}  Seeds={TEST_SEEDS}")
    print(f"K={K_ENSEMBLE} W_pred(320,320) models. All persist across games (no n_actions dep).")
    print(f"delta_k = ||ext_enc - W_pred_k @ prev_ext||. disagreement = var(delta_k).")
    print(f"running_disagreement[a] = EMA. Action = softmax(running_disagreement).")
    print(f"h resets on game switch. alpha/enc_mean persist. running_disagreement resets.")
    print()

    cifar_imgs, cifar_lbls = load_cifar()

    c1, ls, ft, vc, c2 = run_chain(TEST_SEEDS, PHASE_STEPS, CIFAR_STEPS,
                                    cifar_imgs, cifar_lbls)

    print()
    print("=" * 70)
    print(f"STEP 976 RESULTS (965 chain baseline: LS20=67.0, standalone 916@10K=74.7):")
    print(f"  CIFAR-1: {np.mean(c1):.3f} (chance=0.010)")
    print(f"  LS20:    {np.mean(ls):.1f}/seed  nonzero={sum(1 for x in ls if x > 0)}/10  {ls}")
    print(f"  FT09:    {np.mean(ft):.1f}/seed  nonzero={sum(1 for x in ft if x > 0)}/10  {ft}")
    print(f"  VC33:    {np.mean(vc):.1f}/seed  nonzero={sum(1 for x in vc if x > 0)}/10  {vc}")
    print(f"  CIFAR-2: {np.mean(c2):.3f}")
    ls20_baseline = 67.0  # 965 chain baseline
    chain_verdict = (f"LS20 PASS ({np.mean(ls):.1f} ≥ {ls20_baseline*0.9:.1f})"
                     if np.mean(ls) >= ls20_baseline * 0.9
                     else f"LS20 DEGRADED ({np.mean(ls):.1f} < {ls20_baseline*0.9:.1f})")
    ft09_verdict = (f"FT09 SIGNAL ({sum(1 for x in ft if x > 0)}/10 nonzero)"
                    if any(x > 0 for x in ft)
                    else "FT09 ZERO (0/10)")
    vc33_verdict = (f"VC33 SIGNAL ({sum(1 for x in vc if x > 0)}/10 nonzero)"
                    if any(x > 0 for x in vc)
                    else "VC33 ZERO (0/10)")
    print(f"  Chain verdict: {chain_verdict}")
    print(f"  FT09: {ft09_verdict}  |  VC33: {vc33_verdict}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("STEP 976 DONE")
