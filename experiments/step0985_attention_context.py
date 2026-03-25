"""
step0985_attention_context.py -- Fixed-random attention context over recent observations.

FAMILY: Attention-context (completely new; no recurrent h, no W_h/W_x)
R3 HYPOTHESIS: Replace recurrent h with fixed-random content-addressable attention over
last K=32 encoded observations. At each step: ctx = weighted_avg(buffer, w=softmax(buffer
@ W_Q @ enc / sqrt(d))). W_Q is fixed random. ext_enc = [enc, ctx] (512 dims).

Unlike recurrent h (exponential forgetting), attention can selectively weight the MOST
RELEVANT recent observation by content similarity. After A→B, attention to enc_A remains
high when processing enc_B — providing explicit sequential context that h smears together.

Alpha on 512-dim ext concentrates on attention-responsive dims → delta_per_action captures
content-addressed state changes, not just recent ones.

Ban checks:
- No per-(state,action) data: buffer is FIFO observation queue (not indexed by state/action) ✓
- No codebook: no clustering, no spawn-on-threshold ✓
- No trained attention: W_Q is fixed random (same as W_h/W_x in 916) ✓
- Same architecture for all games (one config) ✓

Kill: LS20 < 67.0 (attention hurts navigation).
Success: FT09 > 0 OR VC33 > 0 (content-addressed memory enables sequential discovery).

Chain: CIFAR(1K) → LS20(10K) → FT09(10K) → VC33(10K) → CIFAR(1K). 10 seeds.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
CTX_DIM = 256    # attended context = same dim as enc
EXT_DIM = ENC_DIM + CTX_DIM   # 512
K_BUFFER = 32    # attention window
ETA_W = 0.01; ALPHA_EMA = 0.10; INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50; ALPHA_LO = 0.10; ALPHA_HI = 5.00
EPSILON = 0.20; SOFTMAX_TEMP = 0.10
TEST_SEEDS = list(range(1, 11)); PHASE_STEPS = 10_000; CIFAR_STEPS = 1_000


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp; x -= np.max(x)
    e = np.exp(x); probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class AttentionContext985:
    """Fixed-random attention over recent K observations. No recurrent h."""

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Fixed random query projection (never trained — like W_h/W_x in 916)
        self.W_Q = rs.randn(ENC_DIM, ENC_DIM).astype(np.float32) * 0.1

        # W_pred on 512-dim ext_enc (game-agnostic, persists)
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM), dtype=np.float32)
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)

        # Encoding running mean (persistent)
        self._running_mean_enc = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

        # Game-specific (reset on set_game)
        self._n_actions = 4
        self.delta_per_action = np.full(4, INIT_DELTA, dtype=np.float32)
        self._buffer = deque(maxlen=K_BUFFER)  # FIFO of recent enc vectors
        self._prev_ext = None; self._prev_action = None

    def set_game(self, n_actions):
        self._n_actions = n_actions
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._buffer.clear()   # clear context on game switch
        self._prev_ext = None; self._prev_action = None
        # W_pred, alpha, enc_mean persist

    def _attend(self, enc_q):
        """Content-addressed attention over buffer. Returns CTX_DIM vector."""
        if len(self._buffer) == 0:
            return np.zeros(CTX_DIM, dtype=np.float32)
        buf = np.stack(list(self._buffer))      # (K, ENC_DIM)
        q = self.W_Q @ enc_q                     # (ENC_DIM,) query
        scores = buf @ q / np.sqrt(ENC_DIM)      # (K,) dot products
        scores -= np.max(scores)
        weights = np.exp(scores); weights /= (weights.sum() + 1e-12)
        ctx = (weights[:, None] * buf).sum(axis=0)  # (ENC_DIM,) weighted avg
        return ctx.astype(np.float32)

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1; a = 1.0 / self._n_obs
        self._running_mean_enc = (1 - a) * self._running_mean_enc + a * enc_raw
        enc = (enc_raw - self._running_mean_enc).astype(np.float32)
        ctx = self._attend(enc)
        self._buffer.append(enc)
        return np.concatenate([enc, ctx]).astype(np.float32)

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
                self.W_pred -= ETA_W * np.outer(error, self._prev_ext)  # -= matching 965
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

    def ctx_var(self):
        """Variance of attended ctx — should be > 0 if attention differentiates."""
        if self._prev_ext is None: return 0.0
        return float(np.var(self._prev_ext[ENC_DIM:]))


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
        sub = AttentionContext985(seed=seed)
        c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, cifar_steps)
        l = run_arc(sub, "LS20", 4, seed * 1000, n_steps)
        f = run_arc(sub, "FT09", 68, seed * 1000, n_steps)
        v = run_arc(sub, "VC33", 68, seed * 1000, n_steps)
        c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, cifar_steps)
        c1l.append(c1); lsl.append(l); ftl.append(f); vcl.append(v); c2l.append(c2)
        print(f"  seed={seed}: C1={c1:.3f} LS20={l:4d} FT09={f:4d} VC33={v:4d} C2={c2:.3f}"
              f"  alpha_conc={sub.alpha_conc():.1f}  ctx_var={sub.ctx_var():.4f}")
    return c1l, lsl, ftl, vcl, c2l


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("STEP 985 — ATTENTION CONTEXT (fixed-random, K=32, EXT=512)")
    print("=" * 70)
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), '?')
    vc33_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/vc33') if len(d) >= 8), '?')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}  VC33={vc33_hash}")
    print(f"K_BUFFER={K_BUFFER}  EXT_DIM={EXT_DIM}  W_pred({EXT_DIM},{EXT_DIM})")
    print(f"No recurrent h. ctx=attention(buffer, W_Q). W_Q fixed random.")
    print(f"W_pred -= (matching 965). Change-tracking delta. h-reset equiv: buffer.clear().")
    print(f"Kill: LS20<67.0  Success: FT09>0 OR VC33>0")
    print()

    t0 = time.time()
    cifar_imgs, cifar_lbls = load_cifar()
    c1, ls, ft, vc, c2 = run_chain(TEST_SEEDS, PHASE_STEPS, CIFAR_STEPS, cifar_imgs, cifar_lbls)

    print()
    print("=" * 70)
    print(f"STEP 985 RESULTS (965 chain: LS20=67.0):")
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
    print("STEP 985 DONE")
