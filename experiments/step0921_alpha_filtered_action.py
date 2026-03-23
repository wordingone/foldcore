"""
step0921_alpha_filtered_action.py -- Alpha-Filtered Action Space + Sequence Memory.

R3 hypothesis: Alpha already narrows the OBSERVATION space (3 informative dims on FT09).
Can it also narrow the ACTION space? Filter to top-K actions by alpha-weighted delta,
then search sequences among filtered actions only.

FT09 bottleneck (from Step 920): 68^7 ≈ 10^12 action combinations — untractable.
With K=5: 5^7 = 78,125. With K=8: 8^7 ≈ 2M. Still large but within reach.
With SEQ_LEN=3: 5^3=125 unique sequences. At 20K steps → ~160 visits/sequence. Tractable.

Graph ban check:
- sequence_outcomes[tuple(K=3 actions)]: per-action-PATTERN, not per-(state,action). ALLOWED.
- top_K filter: selected once from delta_per_action (1D array, not per-state). ALLOWED.

Protocol:
- Phase 1 (0-5K steps): standard 895h on ALL n_actions. Build alpha + delta.
- Phase 2 (5K+ steps): filter to top-K, sequence memory among filtered actions only.

Variants:
- 921a: K=5 (5^3=125 sequences)
- 921b: K=8 (8^3=512 sequences)

Run: FT09 only (25K, 10 seeds, cold). LS20 not expected to benefit (4 actions, already tractable).
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque, defaultdict
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
PHASE1_STEPS = 5_000
SEQ_LEN = 3
SEQ_EMA = 0.10
TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


def softmax_select(scores, temp, rng):
    x = np.array(scores) / temp
    x = x - np.max(x)
    e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(scores), p=probs))


class AlphaFilteredSequence921:
    """Phase 1: 895h on all actions. Phase 2: sequence memory on top-K."""

    def __init__(self, n_actions, seed, K=5):
        self._n_actions = n_actions
        self._K = K
        self._rng = np.random.RandomState(seed)

        # Forward model (same as 895h)
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)

        # Running mean
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

        # History
        self._pred_errors = deque(maxlen=200)
        self._prev_enc = None
        self._prev_action = None

        # Phase 2 state
        self._step = 0
        self._filtered = False
        self._top_K = None  # list of top-K action indices

        # Sequence memory: sequence_tuple → EMA delta
        self._seq_outcomes = defaultdict(lambda: INIT_DELTA)
        self._action_history = deque(maxlen=SEQ_LEN - 1)  # last SEQ_LEN-1 actions

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

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
        self.alpha = raw_alpha / mean_raw
        self.alpha = np.clip(self.alpha, ALPHA_LO, ALPHA_HI)

    def _895h_action(self, enc):
        """Standard 895h action selection."""
        if self._rng.random() < EPSILON:
            return int(self._rng.randint(0, self._n_actions))
        x = self.delta_per_action / SOFTMAX_TEMP
        x = x - np.max(x)
        e = np.exp(x)
        probs = e / (e.sum() + 1e-12)
        return int(self._rng.choice(self._n_actions, p=probs))

    def _phase2_action(self):
        """Sequence memory among top-K actions."""
        if not self._filtered:
            # One-time filter: top K actions by delta_per_action
            self._top_K = list(np.argsort(self.delta_per_action)[-self._K:])
            self._filtered = True

        if self._rng.random() < EPSILON:
            return int(self._rng.choice(self._top_K))

        # For each candidate action, look up expected outcome of resulting sequence
        history = tuple(self._action_history)
        scores = []
        for a in self._top_K:
            potential_seq = history + (a,)
            score = self._seq_outcomes[potential_seq]
            scores.append(score)

        idx = softmax_select(scores, SOFTMAX_TEMP, self._rng)
        return self._top_K[idx]

    def process(self, obs):
        enc = self._encode(obs)
        self._step += 1

        if self._prev_enc is not None and self._prev_action is not None:
            # Forward model update (always active)
            inp = np.concatenate([self._prev_enc * self.alpha,
                                   one_hot(self._prev_action, self._n_actions)])
            pred = self.W @ inp
            error = (enc * self.alpha) - pred

            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            # 800b change-tracking
            weighted_delta = (enc - self._prev_enc) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (
                (1 - ALPHA_EMA) * self.delta_per_action[a] + ALPHA_EMA * change
            )

            # Update sequence outcome for the completed sequence
            if self._step > PHASE1_STEPS and len(self._action_history) >= SEQ_LEN - 1:
                completed_seq = tuple(self._action_history) + (a,)
                self._seq_outcomes[completed_seq] = (
                    (1 - SEQ_EMA) * self._seq_outcomes[completed_seq] + SEQ_EMA * change
                )

        # Action selection
        if self._step <= PHASE1_STEPS:
            action = self._895h_action(enc)
        else:
            action = self._phase2_action()

        self._prev_enc = enc.copy()
        self._prev_action = action
        self._action_history.append(action)
        return action

    def on_level_transition(self):
        self._prev_enc = None
        self._prev_action = None
        self._action_history.clear()

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def n_sequences(self):
        return len(self._seq_outcomes)

    def top_K_actions(self):
        if self._top_K is not None:
            return self._top_K
        return list(np.argsort(self.delta_per_action)[-self._K:])


def make_game(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


def run_game(game_name, n_actions, seeds, n_steps, K):
    results = []
    for seed in seeds:
        sub = AlphaFilteredSequence921(n_actions=n_actions, seed=seed, K=K)
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
        top_k = sub.top_K_actions()
        print(f"    seed={seed}: L1={completions:4d}  alpha_conc={sub.alpha_conc():.2f}  "
              f"n_seq={sub.n_sequences()}  top_K={top_k[:5]}")
    return results


print("=" * 70)
print("STEP 921 — ALPHA-FILTERED ACTION SPACE + SEQUENCE MEMORY (FT09)")
print("=" * 70)
print("Phase 1 (0-5K): 895h on all 68 actions. Phase 2 (5K+): top-K + seq memory.")
print("FT09 bottleneck: 68^7≈10^12. K=5: 5^3=125 seqs. K=8: 8^3=512 seqs.")
t0 = time.time()

# 921a: K=5
print("\n--- 921a: K=5, FT09, 25K, 10 seeds ---")
ft09_k5 = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS, K=5)
k5_mean = np.mean(ft09_k5)
k5_zeros = sum(1 for x in ft09_k5 if x == 0)
print(f"  K=5: L1={k5_mean:.1f}/seed  std={np.std(ft09_k5):.1f}  zero={k5_zeros}/10")
print(f"  {ft09_k5}")

# 921b: K=8
print("\n--- 921b: K=8, FT09, 25K, 10 seeds ---")
ft09_k8 = run_game("FT09", 68, TEST_SEEDS, TEST_STEPS, K=8)
k8_mean = np.mean(ft09_k8)
k8_zeros = sum(1 for x in ft09_k8 if x == 0)
print(f"  K=8: L1={k8_mean:.1f}/seed  std={np.std(ft09_k8):.1f}  zero={k8_zeros}/10")
print(f"  {ft09_k8}")

print(f"\n{'='*70}")
print(f"STEP 921 RESULTS (Alpha-Filtered Action Space + Sequence Memory):")
print(f"  921a K=5: L1={k5_mean:.1f}/seed  zero={k5_zeros}/10  (5^3=125 seqs, ~160 visits each)")
print(f"  921b K=8: L1={k8_mean:.1f}/seed  zero={k8_zeros}/10  (8^3=512 seqs, ~40 visits each)")
print(f"\nComparison:")
print(f"  895h cold (68 actions):  0.0/seed  10/10 zeros")
print(f"  915 (K=3 seqs, 68 act):  0.0/seed  10/10 zeros (8711 seqs/seed)")
print(f"  921a K=5: {k5_mean:.1f}/seed  {k5_zeros}/10 zeros")
print(f"  921b K=8: {k8_mean:.1f}/seed  {k8_zeros}/10 zeros")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 921 DONE")
