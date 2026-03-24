"""Step 966: Action-embedded prediction — O(clusters) coverage, not O(n_actions).

FAMILY: Action-embedded prediction (new — addresses n_actions scaling)

R3 HYPOTHESIS: Learned action embeddings E[a] ∈ R^8 let W_pred generalize
across similar actions. Instead of needing to try all n actions to estimate
their values, W_pred sees [h, E[a]] and infers similar-embedding actions have
similar outcomes. Coverage scales as O(k) clusters not O(n) actions.

N_ACTIONS SCALING PROBLEM (from 948-965 kills):
- Hebbian: positive lock probability = (1 - epsilon)^step → worse with n_actions
- 800b: needs ~50 steps/action minimum → 68 actions × 50 = 3400 warmup steps → 0/10
- Both degrade linearly in n_actions

ACTION EMBEDDING FIX: W_pred @ [h, E[a]] → actions clustering in E space share
gradient → W_pred generalizes → FT09's 68 actions converge without full coverage.

E[a] update: E[prev_action] += lr_e * delta * error[:e_dim]  (Hebbian on enc error)
NOT codebook: no cosine matching, no attract update, no per-(state,action) lookup.
Selection remains 800b argmin on running_mean[a] scalar.

Kill: LS20 < 74.7 (916@10K). Success: FT09 > 0 at 10K.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
E_DIM = 8          # action embedding dimension
ETA_W = 0.01
ETA_E = 0.001      # action embedding lr
ALPHA_EMA = 0.10
ALPHA_UPDATE_DELAY = 50
ALPHA_LO = 0.10
ALPHA_HI = 5.00
INIT_DELTA = 1.0
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 10_000


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp
    x -= np.max(x)
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


class ActionEmbedding966:
    """916 + learned action embeddings. W_pred: (ENC_DIM, H_DIM + E_DIM)."""

    def __init__(self, n_actions, seed):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        # Fixed reservoir (same as 916)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        # Forward model: predicts enc from [h, E[action]] (H_DIM + E_DIM = 72)
        self.W_pred = np.zeros((ENC_DIM, H_DIM + E_DIM), dtype=np.float32)

        # Action embeddings: (n_actions, E_DIM) — learned
        self.E = rs.randn(n_actions, E_DIM).astype(np.float32) * 0.01

        # Alpha on enc (ENC_DIM)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._pred_errors = deque(maxlen=200)

        # 800b tracking
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)

        # State
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._prev_h = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        return enc

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
        enc = self._encode(obs)

        if self._prev_h is not None and self._prev_action is not None:
            # Predict enc from [prev_h, E[prev_action]]
            pred_input = np.concatenate([self._prev_h, self.E[self._prev_action]])
            pred = self.W_pred @ pred_input
            error = enc - pred  # ENC_DIM
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
                err_norm = 10.0
            if not np.any(np.isnan(error)):
                self.W_pred += ETA_W * np.outer(error, pred_input)
                # Update action embedding from prediction error (first E_DIM dims)
                self.E[self._prev_action] += ETA_E * err_norm * error[:E_DIM]
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

                # 800b: EMA of prediction error per action
                a = self._prev_action
                self.delta_per_action[a] = (1 - ALPHA_EMA) * self.delta_per_action[a] + ALPHA_EMA * err_norm

        # Softmax action selection (same as 916)
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            action = softmax_sel(self.delta_per_action, SOFTMAX_TEMP, self._rng)

        self._prev_h = self.h.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_h = None
        self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def e_spread(self):
        """Variance of action embeddings — should grow if embeddings differentiate."""
        return float(np.var(self.E))


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except Exception:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps):
    results = []
    for seed in seeds:
        sub = ActionEmbedding966(n_actions=n_actions, seed=seed)
        env = make_game(game_name)
        obs = env.reset(seed=seed * 1000)
        step = 0; completions = 0; current_level = 0
        while step < n_steps:
            if obs is None:
                obs = env.reset(seed=seed * 1000)
                sub.on_level_transition()
                continue
            action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
            obs, _, done, info = env.step(action)
            step += 1
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > current_level:
                completions += (cl - current_level)
                current_level = cl
                sub.on_level_transition()
            if done:
                obs = env.reset(seed=seed * 1000)
                current_level = 0
                sub.on_level_transition()
        results.append(completions)
        print(f"    seed={seed}: L1={completions:4d}  alpha_conc={sub.alpha_conc():.2f}  e_spread={sub.e_spread():.4f}")
    return results


if __name__ == "__main__":
    import os, time
    print("=" * 70)
    print("STEP 966 — ACTION EMBEDDING (O(clusters) coverage)")
    print("=" * 70)
    t0 = time.time()
    ls20_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ls20') if len(d) >= 8), '?')
    ft09_hash = next((d for d in os.listdir('B:/M/the-search/environment_files/ft09') if len(d) >= 8), '?')
    print(f"Game versions: LS20={ls20_hash}  FT09={ft09_hash}")
    print(f"E_DIM={E_DIM}  ETA_E={ETA_E}  ETA_W={ETA_W}  EPSILON={EPSILON}")
    print(f"W_pred: ({ENC_DIM}, {H_DIM+E_DIM})  E: (n_actions, {E_DIM})")
    print()

    print("--- LS20 (4 actions, 10K steps) ---")
    ls20_r = run_game("LS20", 4, TEST_SEEDS, TEST_STEPS)
    ls20_mean = np.mean(ls20_r)
    ls20_nz = sum(1 for x in ls20_r if x > 0)
    print(f"  LS20: L1={ls20_mean:.1f}/seed  nonzero={ls20_nz}/10  {ls20_r}")
    print(f"  916@10K baseline: 74.7/seed  8/10")

    print()
    print("--- FT09 (68 actions, 10K steps) ---")
    ft09_r = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS)
    ft09_mean = np.mean(ft09_r)
    ft09_nz = sum(1 for x in ft09_r if x > 0)
    print(f"  FT09: L1={ft09_mean:.1f}/seed  nonzero={ft09_nz}/10  {ft09_r}")

    print()
    print("=" * 70)
    ls20_pass = ls20_mean >= 74.7
    ft09_signal = ft09_nz > 0
    if ls20_pass and ft09_signal:
        verdict = f"SUCCESS — LS20={ls20_mean:.1f} (≥74.7) + FT09 signal"
    elif ls20_pass:
        verdict = f"PARTIAL — LS20={ls20_mean:.1f} (≥74.7) but FT09=0"
    elif ft09_signal:
        verdict = f"PARTIAL — FT09 signal but LS20={ls20_mean:.1f} (<74.7)"
    else:
        verdict = f"KILL — LS20={ls20_mean:.1f} (<74.7) and FT09=0"
    print(f"  VERDICT: {verdict}")
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("STEP 966 DONE")
