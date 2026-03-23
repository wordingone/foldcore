"""
step0926_chain_scalar_forward.py -- Full chain: CIFAR→LS20→FT09→VC33→CIFAR.

R3 hypothesis: Recurrent h encodes cross-game trajectory context. W_scalar
conditioned on h adapts action selection to game type without explicit game
identification. h "changes character" per game → W_scalar predicts per-game
change → mechanism adapts.

Kill criterion (Jun via Leo mail 2670): if 925 helps one game but hurts
another → KILL (per-game tuning, not game-adaptive).

Architecture: 916 recurrent h (fixed W_h/W_x) + W_scalar (h + action → scalar)
+ alpha + 800b delta. Compare to Step 914 (895h cold, no h, no W_scalar).

Chain: CIFAR (100 actions, 25K) → LS20 (4, 25K) → FT09 (68, 25K)
       → VC33 (68, 25K) → CIFAR (100, 25K). 5 seeds.

Persistent across games: alpha, h, running_mean, W_h, W_x (fixed).
Reset per game: W_pred, W_scalar, delta_per_action.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
ETA_W_PRED = 0.01
ETA_SCALAR = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
SCALAR_WEIGHT = 0.5

TEST_SEEDS = list(range(1, 6))
PHASE_STEPS = 25_000

GAME_SEQUENCE = [
    ("CIFAR", 100),
    ("LS20",  4),
    ("FT09",  68),
    ("VC33",  68),
    ("CIFAR", 100),
]

# Step 914 chain baseline (895h cold, 5 seeds)
BASELINE_914 = {
    "CIFAR-1": 0.0100,  # ~chance
    "LS20":    248.6,
    "FT09":    0.0,
    "VC33":    0.0,
    "CIFAR-2": 0.0110,
}


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


def softmax_action(scores, temp, rng):
    x = np.array(scores) / temp; x -= np.max(x); e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(scores), p=probs))


class ChainScalarForward926:
    """916 recurrent h + W_scalar chain substrate.

    Persistent across games: alpha, h, running_mean, W_h, W_x (fixed random).
    Reset per game: W_pred, W_scalar, delta_per_action (n_actions changes).
    """

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Fixed recurrent weights — PERSISTENT across games
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # Persistent state
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)

        # Game-specific (set via set_game)
        self._n = None
        self.W_pred = None
        self.W_scalar = None
        self.delta_per_action = None
        self._prev_enc = None
        self._prev_ext = None
        self._prev_h = None
        self._prev_action = None

    def set_game(self, n_actions):
        """Switch to new game. Reset game-specific weights, preserve h + alpha."""
        self._n = n_actions
        EXT = ENC_DIM + H_DIM
        self.W_pred = np.zeros((EXT, EXT + n_actions), dtype=np.float32)
        self.W_scalar = np.zeros(H_DIM + n_actions, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._pred_errors.clear()
        self._prev_enc = None
        self._prev_ext = None
        self._prev_h = None
        self._prev_action = None

    def reset_seed(self, seed):
        """Full reset for new seed (alpha, h, running_mean all reset)."""
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors.clear()
        self._n = None
        self.W_pred = None
        self.W_scalar = None
        self.delta_per_action = None
        self._prev_enc = None
        self._prev_ext = None
        self._prev_h = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        ext = np.concatenate([enc, self.h]).astype(np.float32)
        return enc, ext

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY: return
        me = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(me)) or np.any(np.isinf(me)): return
        ra = np.sqrt(np.clip(me, 0, 1e6) + 1e-8); mr = np.mean(ra)
        if mr < 1e-8 or np.isnan(mr): return
        self.alpha = np.clip(ra / mr, ALPHA_LO, ALPHA_HI)

    def process(self, obs):
        enc, ext = self._encode(obs)

        if self._prev_enc is not None and self._prev_action is not None:
            # Alpha signal via forward model on ext
            inp = np.concatenate([self._prev_ext * self.alpha,
                                   one_hot(self._prev_action, self._n)])
            pred = self.W_pred @ inp
            error = (ext * self.alpha) - pred
            en = float(np.linalg.norm(error))
            if en > 10.0: error *= 10.0 / en
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W_PRED * np.outer(error, inp)
                self._pred_errors.append(np.abs(error[:ENC_DIM]))  # alpha on enc dims only
                self._update_alpha()

            # 800b change tracking
            weighted_delta = (ext - self._prev_ext) * np.concatenate([self.alpha, np.ones(H_DIM)])
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (1 - ALPHA_EMA) * self.delta_per_action[a] + ALPHA_EMA * change

            # Scalar model update
            actual_change = float(np.linalg.norm(enc - self._prev_enc))
            if self._prev_h is not None:
                scalar_inp = np.concatenate([self._prev_h, one_hot(self._prev_action, self._n)])
                pred_change = float(self.W_scalar @ scalar_inp)
                scalar_error = actual_change - pred_change
                self.W_scalar += ETA_SCALAR * scalar_error * scalar_inp

        # Action selection
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n))
        else:
            h_now = self.h.copy()
            scalar_scores = np.array([
                float(self.W_scalar @ np.concatenate([h_now, one_hot(a, self._n)]))
                for a in range(self._n)
            ], dtype=np.float32)
            combined = SCALAR_WEIGHT * scalar_scores + (1 - SCALAR_WEIGHT) * self.delta_per_action
            action = softmax_action(combined, SOFTMAX_TEMP, self._rng)

        self._prev_enc = enc.copy()
        self._prev_ext = ext.copy()
        self._prev_h = self.h.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None; self._prev_ext = None
        self._prev_h = None; self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def scalar_top_k(self, k=5):
        if self.W_scalar is None or self._n is None: return []
        h_now = self.h.copy()
        scores = np.array([
            float(self.W_scalar @ np.concatenate([h_now, one_hot(a, self._n)]))
            for a in range(self._n)
        ])
        return list(np.argsort(scores)[-k:])


def make_env(game_name):
    try:
        import arcagi3; return arcagi3.make(game_name)
    except:
        import util_arcagi3; return util_arcagi3.make(game_name)


def load_cifar():
    try:
        import torchvision
        import torchvision.transforms as transforms
        data_dir = 'B:/M/the-search/data'
        ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True,
                                            transform=transforms.ToTensor())
        images = np.array([np.array(ds[i][0]).transpose(1, 2, 0) for i in range(len(ds))],
                          dtype=np.float32)
        labels = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
        return images, labels
    except Exception as e:
        print(f"  CIFAR-100 load failed: {e}"); return None, None


def run_arc_phase(sub, game_name, n_actions, env_seed, n_steps):
    sub.set_game(n_actions)
    env = make_env(game_name)
    obs = env.reset(seed=env_seed)
    step = 0; completions = 0; current_level = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            sub.on_level_transition()
        if done:
            obs = env.reset(seed=env_seed); current_level = 0
            sub.on_level_transition()
    return completions


def run_cifar_phase(sub, n_actions, images, labels, rng_seed, n_steps):
    if images is None: return 0, 0, 0.0
    sub.set_game(n_actions)
    rng = np.random.RandomState(rng_seed)
    idx = rng.permutation(len(images))[:n_steps]
    correct = 0; steps = 0
    for img_idx in idx:
        obs = images[img_idx]
        label = int(labels[img_idx])
        action = sub.process(obs) % n_actions
        if action == label: correct += 1
        steps += 1
    return correct, steps, correct / max(steps, 1)


print("=" * 70)
print("STEP 926 — FULL CHAIN: CIFAR→LS20→FT09→VC33→CIFAR (SCALAR FORWARD MODEL)")
print("=" * 70)
print("Architecture: 916 recurrent h (persistent) + W_scalar (game-reset) + 800b")
print("Kill criterion: scalar helps one game but hurts another → KILL")
t0 = time.time()
cifar_images, cifar_labels = load_cifar()
cifar_ok = cifar_images is not None
print(f"CIFAR-100: {'loaded' if cifar_ok else 'NOT AVAILABLE'}")

cifar1_results = []; cifar2_results = []
ls20_results = []; ft09_results = []; vc33_results = []

for ts in TEST_SEEDS:
    sub = ChainScalarForward926(seed=ts)
    sub.reset_seed(ts)
    print(f"\n  Seed {ts}:")

    c1, s1, acc1 = run_cifar_phase(sub, 100, cifar_images, cifar_labels, ts * 1000, PHASE_STEPS)
    cifar1_results.append(acc1)
    print(f"    CIFAR-1: acc={acc1:.4f} ({c1}/{s1})  alpha_conc={sub.alpha_conc():.2f}  h_norm={float(np.linalg.norm(sub.h)):.3f}")

    ls20_l1 = run_arc_phase(sub, "LS20", 4, ts * 1000, PHASE_STEPS)
    ls20_results.append(ls20_l1)
    print(f"    LS20:    L1={ls20_l1:4d}  alpha_conc={sub.alpha_conc():.2f}  scalar_top5={sub.scalar_top_k(5)}")

    ft09_l1 = run_arc_phase(sub, "FT09", 68, ts * 1000, PHASE_STEPS)
    ft09_results.append(ft09_l1)
    print(f"    FT09:    L1={ft09_l1:4d}  alpha_conc={sub.alpha_conc():.2f}  scalar_top5={sub.scalar_top_k(5)}")

    vc33_l1 = run_arc_phase(sub, "VC33", 68, ts * 1000, PHASE_STEPS)
    vc33_results.append(vc33_l1)
    print(f"    VC33:    L1={vc33_l1:4d}  alpha_conc={sub.alpha_conc():.2f}  scalar_top5={sub.scalar_top_k(5)}")

    c2, s2, acc2 = run_cifar_phase(sub, 100, cifar_images, cifar_labels, ts * 1000 + 1, PHASE_STEPS)
    cifar2_results.append(acc2)
    print(f"    CIFAR-2: acc={acc2:.4f} ({c2}/{s2})  alpha_conc={sub.alpha_conc():.2f}")

print(f"\n{'='*70}")
print(f"STEP 926 CHAIN RESULTS (5 seeds):")
print(f"  CIFAR-1 acc: {np.mean(cifar1_results):.4f}  {[f'{x:.3f}' for x in cifar1_results]}")
print(f"  LS20  L1:    {np.mean(ls20_results):.1f}/seed  std={np.std(ls20_results):.1f}  zero={sum(1 for x in ls20_results if x==0)}/5  {ls20_results}")
print(f"  FT09  L1:    {np.mean(ft09_results):.1f}/seed  std={np.std(ft09_results):.1f}  zero={sum(1 for x in ft09_results if x==0)}/5  {ft09_results}")
print(f"  VC33  L1:    {np.mean(vc33_results):.1f}/seed  std={np.std(vc33_results):.1f}  zero={sum(1 for x in vc33_results if x==0)}/5  {vc33_results}")
print(f"  CIFAR-2 acc: {np.mean(cifar2_results):.4f}  {[f'{x:.3f}' for x in cifar2_results]}")
print(f"\nComparison vs Step 914 (895h cold, 5 seeds):")
print(f"  914 LS20:  248.6/seed  (926: {np.mean(ls20_results):.1f})")
print(f"  914 FT09:    0.0/seed  (926: {np.mean(ft09_results):.1f})")
print(f"  914 VC33:    0.0/seed  (926: {np.mean(vc33_results):.1f})")
print(f"  CIFAR transfer delta: {np.mean(cifar2_results) - np.mean(cifar1_results):+.4f}")
print(f"\nKill criterion check (helps one, hurts another):")
ls20_delta = np.mean(ls20_results) - 248.6
print(f"  LS20 delta vs 914: {ls20_delta:+.1f}")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 926 DONE")
