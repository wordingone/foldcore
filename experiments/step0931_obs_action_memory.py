"""
step0931_obs_action_memory.py -- Observation-Action Association Memory.

R3 hypothesis: Per-observation memory (obs_key → best_action) enables recall
of what worked from each encoded state. NOT per-(state,action) graph.
Graph ban check: keyed by observation encoding only, NOT (state,action).
Stores best_action per observation. No visit counts. No argmin over visits.
ALLOWED.

Why this is the one door:
- CIFAR: same image → same key → same action = self-organizing classifier
- LS20:  same grid cell → same key → same direction = local nav memory
- FT09:  same puzzle state → same key → same click = sequential progress
- VC33:  same game state → same key → same action = state-conditioned recall

Architecture: 895h (alpha + 800b) + obs-action memory (N_MAX=2000).
Action: 70% recall (if obs in memory), 20% 800b, 10% random.
       20% 800b, 80% standard (if obs not in memory).

Run: PRISM-light CIFAR→LS20→FT09→VC33→CIFAR, 25K/phase, 5 seeds.
"""
import sys, time, hashlib
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
ETA_W = 0.01
ALPHA_EMA = 0.05
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.10
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
N_MAX = 2000       # max memory entries
RECALL_PROB = 0.70 # use memory when hit
DELTA_PROB = 0.20  # 800b when hit (else random)
TEST_SEEDS = list(range(1, 6))
PHASE_STEPS = 25_000


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v

def softmax_sel(delta, temp, rng):
    x = np.array(delta)/temp; x -= np.max(x); e = np.exp(x)
    return int(rng.choice(len(delta), p=e/(e.sum()+1e-12)))


class ObsActionMemory931:
    """895h + observation-action association memory."""

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        # Game-specific (reset per game)
        self._n = None; self.W = None; self.delta_per_action = None
        self._prev_enc = None; self._prev_action = None; self._prev_key = None
        # Memory: obs_key → (best_action, score). Persistent across games.
        self.memory = {}
        self._hits = 0; self._lookups = 0

    def set_game(self, n_actions):
        self._n = n_actions
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._pred_errors.clear()
        self._prev_enc = None; self._prev_action = None; self._prev_key = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1; a = 1.0/self._n_obs
        self._running_mean = (1-a)*self._running_mean + a*enc_raw
        return enc_raw - self._running_mean

    def _quantize(self, weighted_enc):
        mn = weighted_enc.min(); rng = (weighted_enc.max() - mn) + 1e-8
        scaled = np.clip((weighted_enc - mn)/rng * 255, 0, 255).astype(np.uint8)
        return hashlib.md5(scaled.tobytes()).hexdigest()[:16]

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY: return
        me = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(me)) or np.any(np.isinf(me)): return
        ra = np.sqrt(np.clip(me, 0, 1e6)+1e-8); mr = np.mean(ra)
        if mr < 1e-8 or np.isnan(mr): return
        self.alpha = np.clip(ra/mr, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        enc = self._encode(obs)
        weighted = enc * self.alpha
        obs_key = self._quantize(weighted)

        if self._prev_enc is not None and self._prev_action is not None:
            # Standard 895h W + alpha update
            inp = np.concatenate([self._prev_enc*self.alpha, one_hot(self._prev_action, self._n)])
            pred = self.W @ inp; error = (enc*self.alpha) - pred
            en = float(np.linalg.norm(error))
            if en > 10.0: error *= 10.0/en
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error)); self._update_alpha()
            # 800b delta
            change = float(np.linalg.norm((enc-self._prev_enc)*self.alpha))
            self.delta_per_action[self._prev_action] = (
                (1-ALPHA_EMA)*self.delta_per_action[self._prev_action] + ALPHA_EMA*change)
            # Memory update: associate prev_obs → (prev_action, change)
            if self._prev_key is not None:
                if self._prev_key in self.memory:
                    _, old_score = self.memory[self._prev_key]
                    if change > old_score:
                        self.memory[self._prev_key] = (self._prev_action, change)
                elif len(self.memory) < N_MAX:
                    self.memory[self._prev_key] = (self._prev_action, change)

        # Action selection: memory → 800b → random
        self._lookups += 1
        if obs_key in self.memory:
            self._hits += 1
            remembered_action, _ = self.memory[obs_key]
            r = self._rng.random()
            if r < 0.10:
                action = int(self._rng.randint(0, self._n))
            elif r < 0.30:
                action = softmax_sel(self.delta_per_action, SOFTMAX_TEMP, self._rng)
            else:
                action = remembered_action % self._n
        else:
            if self._rng.random() < EPSILON:
                action = int(self._rng.randint(0, self._n))
            else:
                action = softmax_sel(self.delta_per_action, SOFTMAX_TEMP, self._rng)

        self._prev_enc = enc.copy(); self._prev_action = action; self._prev_key = obs_key
        return action

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None; self._prev_key = None

    def alpha_conc(self):
        return float(np.max(self.alpha)/(np.min(self.alpha)+1e-8))

    def hit_rate(self):
        return self._hits/max(self._lookups, 1)


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
        imgs = np.array([np.array(ds[i][0]).transpose(1,2,0) for i in range(len(ds))], dtype=np.float32)
        lbls = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
        return imgs, lbls
    except Exception as e:
        print(f"CIFAR load failed: {e}"); return None, None

def run_cifar(sub, imgs, lbls, seed, n_steps):
    if imgs is None: return 0.0
    sub.set_game(100); rng = np.random.RandomState(seed)
    idx = rng.permutation(len(imgs))[:n_steps]
    correct = sum(1 for i in idx if sub.process(imgs[i]) % 100 == lbls[i])
    return correct/len(idx)

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
            completions += cl-level; level = cl; sub.on_level_transition()
        if done:
            obs = env.reset(seed=seed); level = 0; sub.on_level_transition()
    return completions


print("="*70)
print("STEP 931 — OBSERVATION-ACTION MEMORY (per-obs recall + 895h)")
print("="*70)
print(f"Memory: obs_key→(best_action,score), N_MAX={N_MAX}. Recall 70%/800b 20%/rand 10%.")
print("Graph ban: keyed by obs encoding only. NOT (state,action). ALLOWED.")
t0 = time.time()

cifar_imgs, cifar_lbls = load_cifar()
cifar1=[]; ls20=[]; ft09=[]; vc33=[]; cifar2=[]

for seed in TEST_SEEDS:
    sub = ObsActionMemory931(seed=seed)
    print(f"\n  Seed {seed}:")

    c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed*1000, PHASE_STEPS)
    cifar1.append(c1)
    print(f"    CIFAR-1: acc={c1:.4f}  mem={len(sub.memory)}  hit={sub.hit_rate():.3f}  alpha_conc={sub.alpha_conc():.2f}")

    l = run_arc(sub, "LS20", 4, seed*1000, PHASE_STEPS)
    ls20.append(l)
    print(f"    LS20:    L1={l:4d}  mem={len(sub.memory)}  hit={sub.hit_rate():.3f}  alpha_conc={sub.alpha_conc():.2f}")

    f = run_arc(sub, "FT09", 68, seed*1000, PHASE_STEPS)
    ft09.append(f)
    print(f"    FT09:    L1={f:4d}  mem={len(sub.memory)}  hit={sub.hit_rate():.3f}  alpha_conc={sub.alpha_conc():.2f}")

    v = run_arc(sub, "VC33", 68, seed*1000, PHASE_STEPS)
    vc33.append(v)
    print(f"    VC33:    L1={v:4d}  mem={len(sub.memory)}  hit={sub.hit_rate():.3f}  alpha_conc={sub.alpha_conc():.2f}")

    c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed*1000+1, PHASE_STEPS)
    cifar2.append(c2)
    print(f"    CIFAR-2: acc={c2:.4f}  mem={len(sub.memory)}  hit={sub.hit_rate():.3f}  alpha_conc={sub.alpha_conc():.2f}")

print(f"\n{'='*70}")
print(f"STEP 931 RESULTS (obs-action memory, PRISM-light, 25K/phase, 5 seeds):")
print(f"  CIFAR-1 acc: {np.mean(cifar1):.4f}  {[f'{x:.3f}' for x in cifar1]}")
print(f"  LS20  L1:    {np.mean(ls20):.1f}/seed  std={np.std(ls20):.1f}  zero={sum(1 for x in ls20 if x==0)}/5  {ls20}")
print(f"  FT09  L1:    {np.mean(ft09):.1f}/seed  zero={sum(1 for x in ft09 if x==0)}/5  {ft09}")
print(f"  VC33  L1:    {np.mean(vc33):.1f}/seed  zero={sum(1 for x in vc33 if x==0)}/5  {vc33}")
print(f"  CIFAR-2 acc: {np.mean(cifar2):.4f}  {[f'{x:.3f}' for x in cifar2]}")
print(f"\nComparison (chain, 25K, 5 seeds):")
print(f"  914 895h chain:   LS20=248.6  FT09=0  VC33=0")
print(f"  926 916h chain:   LS20=212.6  FT09=0  VC33=0")
print(f"  931 obs-memory:   LS20={np.mean(ls20):.1f}  FT09={np.mean(ft09):.1f}  VC33={np.mean(vc33):.1f}")
print(f"\nKill criterion: FT09 or VC33 > 0 → BREAKTHROUGH. LS20 regression → review.")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 931 DONE")
