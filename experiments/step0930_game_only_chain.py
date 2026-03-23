"""
step0930_game_only_chain.py -- Game-only chain: LS20→FT09→VC33 (no CIFAR).

R3 hypothesis: CIFAR disrupts the h/alpha state for subsequent games.
Without CIFAR, does 895h chain LS20 score match or exceed standalone?
If so: CIFAR was the confound, not the chain itself.

Also tests: does LS20-trained alpha carry useful information into FT09/VC33?
FT09/VC33 stay at L1=0, but alpha_conc patterns after LS20 may differ.

Architecture: 895h cold (clamped alpha + 800b). No recurrent h.
Alpha persists across games. W + delta reset per game.
Chain: LS20(4) → FT09(68) → VC33(68). 10K/game (5-min cap), 10 seeds.

Compare: 895h standalone LS20=268.0/seed (25K). Scaled to 10K: ~107/seed expected.
Also compare: 926 chain LS20=212.6/seed (25K, 5 seeds, with CIFAR, with h).
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00

TEST_SEEDS = list(range(1, 11))
PHASE_STEPS = 10_000   # 10K/game for 5-min cap


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp; x -= np.max(x); e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class Chain895h:
    """895h with persistent alpha, reset W+delta per game."""

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        self._n = None; self.W = None; self.delta_per_action = None
        self._prev_enc = None; self._prev_action = None

    def set_game(self, n_actions):
        self._n = n_actions
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._pred_errors.clear()
        self._prev_enc = None; self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
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
                                   one_hot(self._prev_action, self._n)])
            pred = self.W @ inp; error = (enc * self.alpha) - pred
            en = float(np.linalg.norm(error))
            if en > 10.0: error *= 10.0 / en
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()
            weighted_delta = (enc - self._prev_enc) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (1 - ALPHA_EMA) * self.delta_per_action[a] + ALPHA_EMA * change
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n))
        else:
            action = softmax_sel(self.delta_per_action, SOFTMAX_TEMP, self._rng)
        self._prev_enc = enc.copy(); self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def top_delta(self, k=5):
        if self.delta_per_action is None: return []
        return list(np.argsort(self.delta_per_action)[-k:])


def make_env(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


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


print("=" * 70)
print("STEP 930 — GAME-ONLY CHAIN: LS20→FT09→VC33 (no CIFAR, 895h)")
print("=" * 70)
print("Alpha persists across games. W+delta reset per game.")
print(f"10K/game, 10 seeds (5-min cap compliance).")
t0 = time.time()

ls20_results = []; ft09_results = []; vc33_results = []

for seed in TEST_SEEDS:
    sub = Chain895h(seed=seed)

    ls20 = run_arc(sub, "LS20", 4, seed * 1000, PHASE_STEPS)
    ls20_results.append(ls20)
    ls20_conc = sub.alpha_conc()

    ft09 = run_arc(sub, "FT09", 68, seed * 1000, PHASE_STEPS)
    ft09_results.append(ft09)
    ft09_conc = sub.alpha_conc()

    vc33 = run_arc(sub, "VC33", 68, seed * 1000, PHASE_STEPS)
    vc33_results.append(vc33)

    print(f"  seed={seed}: LS20={ls20:4d}(α={ls20_conc:.2f})  "
          f"FT09={ft09:4d}(α={ft09_conc:.2f})  VC33={vc33:4d}")

print(f"\n{'='*70}")
print(f"STEP 930 RESULTS (game-only chain, 895h, 10K/game, 10 seeds):")
print(f"  LS20: {np.mean(ls20_results):.1f}/seed  std={np.std(ls20_results):.1f}  "
      f"zero={sum(1 for x in ls20_results if x==0)}/10  {ls20_results}")
print(f"  FT09: {np.mean(ft09_results):.1f}/seed  zero={sum(1 for x in ft09_results if x==0)}/10  {ft09_results}")
print(f"  VC33: {np.mean(vc33_results):.1f}/seed  zero={sum(1 for x in vc33_results if x==0)}/10  {vc33_results}")
print(f"\nComparison (all at 10K unless noted):")
print(f"  895h standalone 10K (scaled): ~107/seed  (25K=268.0, linear approx)")
print(f"  926 chain LS20 (25K, w/ CIFAR, w/ h): 212.6/seed")
print(f"  927 baseline chain LS20 (10K): Random=26.2, ICM=44.8")
print(f"  930 game-only chain LS20 (10K, 895h): {np.mean(ls20_results):.1f}/seed")
print(f"\nCIFAR interference check: compare 930 LS20 vs 927 Count-based (32.6).")
print(f"If 930 >> 927: alpha provides signal above count exploration.")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 930 DONE")
