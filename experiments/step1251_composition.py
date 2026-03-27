"""
Step 1251 — Component Composition Experiment, Phase 1

R3 hypothesis: composing 7 cross-family validated components into a sequential
pipeline (Wiring A) produces at least one instrumentation stage (I1-I5, R3) that
argmin-alone (Control C) fails. This determines whether composition is worth
pursuing or whether the components fail to synergize.

Phase 1: Control C vs Wiring A. 3 games (ls20, vc33, sp80). 5 draws each. 30 runs.
Phase 2 (contingent): Add Wiring B if Phase 1 finds signal.
Phase 3 (contingent): Expand to 10 games if Phase 2 confirms.

Instrumentation (all 5 measured simultaneously on every run):
  I3 — Action discovery at step 200 (Spearman ρ vs game responsive-action profile)
  I1 — State representation at step 1000 (within vs between level cosine distance)
  I4 — Temporal structure at steps 100 + 5000 (entropy reduction)
  I5 — Cross-level transfer (gated on L1 reliability: ≥5/5 draws solve L1)
  R3 — Jacobian sketch at step 5000 (||J_exp - J_fresh||_F on internal encoding)

Components in 7-component Wiring A:
  C4  — Centered encoding (x - running_mean)
  C21/C26 — Recurrent h (tanh(W_h@h + W_x@enc), fixed random)
  C25 — Alpha attention (per-dim prediction error weighting, clamped 0.1-5.0)
  C15/C20 — Transition detection (inconsistent-successor signal)
  C1  — Novelty-triggered growth (novel obs → grow exploration space, track novelty rate)
  C14 — Argmin selection (global least-visited action)
  C22 — Self-observation (action freq distribution → feeds back into encoding)
"""
import sys, os, time, json, math, copy
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame

# ─── Phase 1 config ───
PHASE1_GAMES = ['ls20', 'vc33', 'sp80']
N_DRAWS = 5
MAX_STEPS = 10_000
MAX_SECONDS = 300  # 5-min cap

# Shared component hyperparameters (same for both substrates)
ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM   # 320
ETA_PRED = 0.01
ALPHA_EMA_RATE = 0.10
ALPHA_LO, ALPHA_HI = 0.10, 5.00
TRANSITION_SCALE = 0.30
SELF_OBS_SCALE = 0.05
NOVELTY_EXPLORE_THRESHOLD = 0.6  # EMA novelty rate above which → random action

# Instrumentation steps
I3_STEP = 200
I1_STEP = 1000
I4_STEPS = [100, 5000]
R3_STEP = 5000
R3_N_OBS = 100
R3_N_DIRS = 20
R3_EPSILON = 0.01
I4_WINDOW = 50

# Diagnostics path
DIAG_DIR = os.path.join(os.path.dirname(__file__), 'results', 'game_diagnostics')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'composition_experiment')

# ─── Hash function for obs ───
_HASH_H = np.random.RandomState(42).randn(12, ENC_DIM).astype(np.float32)

def hash_enc(x: np.ndarray) -> int:
    """Hash encoded obs to int via fixed LSH."""
    bits = (_HASH_H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits[:8], bitorder='big').tobytes().hex(), 16)


# ─── Game environment factory ───
def make_game(game_name: str):
    """Create game environment (uppercase name for API)."""
    try:
        import arcagi3
        return arcagi3.make(game_name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name.upper())


def get_n_actions(env, game_name: str) -> int:
    """Get action space size from env or fallback table."""
    n = getattr(env, 'n_actions', None)
    if n is not None:
        return int(n)
    _table = {'ls20': 7, 'vc33': 4103, 'sp80': 4103, 'ft09': 4103}
    return _table.get(game_name.lower(), 4103)


# ─── Game diagnostics ───
def load_game_diag(game_name: str) -> dict:
    """Load per-game diagnostic JSON."""
    path = os.path.join(DIAG_DIR, f'{game_name.lower()}_diagnostic.json')
    with open(path) as f:
        return json.load(f)


def build_kb_profile(diag: dict) -> tuple:
    """Return (kb_delta_mean, kb_responsive) arrays, indexed ACTION1-ACTION7 → 0-6."""
    kb = diag.get('kb_responsiveness', {})
    delta = np.zeros(7, np.float32)
    responsive = np.zeros(7, bool)
    for i, key in enumerate([f'ACTION{j}' for j in range(1, 8)]):
        if key in kb:
            delta[i] = kb[key].get('delta_mean', 0.0)
            responsive[i] = kb[key].get('responsive', False)
    return delta, responsive


# ─── Entropy computation ───
def action_entropy(action_seq: list, n_actions: int) -> float:
    """Shannon entropy of action sequence over n_actions."""
    if not action_seq:
        return 0.0
    counts = np.zeros(n_actions, np.float32)
    for a in action_seq:
        if 0 <= a < n_actions:
            counts[a] += 1
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


# ─── Cosine distance ───
def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - cosine similarity)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# ─── Permutation test for I1 ───
def permutation_test_i1(within_dists, between_dists, n_perms=500, seed=0):
    """One-sided permutation test: P(mean_between - mean_within >= observed)."""
    rng = np.random.RandomState(seed)
    within = np.array(within_dists, dtype=np.float32)
    between = np.array(between_dists, dtype=np.float32)
    if len(within) == 0 or len(between) == 0:
        return 1.0
    observed = float(np.mean(between) - np.mean(within))
    all_dists = np.concatenate([within, between])
    n_w = len(within)
    count = 0
    for _ in range(n_perms):
        perm = rng.permutation(len(all_dists))
        perm_w = all_dists[perm[:n_w]]
        perm_b = all_dists[perm[n_w:]]
        if float(np.mean(perm_b) - np.mean(perm_w)) >= observed:
            count += 1
    return count / n_perms


# ─── Spearman rho (manual) ───
def spearman_rho(x, y):
    """Spearman rank correlation."""
    if len(x) < 2 or len(y) < 2:
        return None
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(np.float32)
    ry = np.argsort(np.argsort(y)).astype(np.float32)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = (np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom < 1e-8:
        return None
    return float(np.dot(rx, ry) / denom)


# ─────────────────────────────────────────────────────────────
# SUBSTRATE IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────

class ControlC:
    """
    Control C — centered encoding + argmin only.

    C4: x - running_mean
    C14: argmin(action_counts) — global least-visited action
    No recurrence, no prediction error, no novelty, no self-obs.
    """

    def __init__(self, n_actions: int, seed: int):
        self.n_actions = n_actions
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        # C4: Running mean for centered encoding
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0

        # C14: Global action visit counts
        self.action_counts = np.zeros(n_actions, np.float32)

        # Last encoded obs (for I1 / R3)
        self._last_enc = None
        self.step = 0

    def _centered_encode(self, obs: np.ndarray) -> np.ndarray:
        """C4: avgpool16 + running mean centering."""
        x = _enc_frame(obs)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw: np.ndarray,
                                   frozen_running_mean: np.ndarray) -> np.ndarray:
        """Read-only encoding for Jacobian computation."""
        x = _enc_frame(np.asarray(obs_raw, dtype=np.float32))
        return x - frozen_running_mean  # 256D

    def process(self, obs_raw) -> int:
        obs = np.asarray(obs_raw, dtype=np.float32)
        enc = self._centered_encode(obs)
        self._last_enc = enc.copy()
        action = int(np.argmin(self.action_counts))
        self.action_counts[action] += 1
        self.step += 1
        return action

    def on_level_transition(self):
        pass  # argmin-alone doesn't reset on level transition

    def reset(self, seed: int):
        self.running_mean[:] = 0
        self.n_obs = 0
        self.action_counts[:] = 0
        self._last_enc = None
        self.step = 0
        self._rng = np.random.RandomState(seed)

    def get_state(self) -> dict:
        return {
            'running_mean': self.running_mean.copy(),
            'n_obs': self.n_obs,
            'action_counts': self.action_counts.copy(),
            'step': self.step,
        }

    def set_state(self, state: dict):
        self.running_mean = state['running_mean'].copy()
        self.n_obs = state['n_obs']
        self.action_counts = state['action_counts'].copy()
        self.step = state['step']


class WiringA:
    """
    Wiring A — all 7 components in sequential pipeline.

    Pipeline:
      obs → centered_enc [C4]
          → (self_obs feedback loops back here) [C22]
          → recurrent_h [C21/C26]
          → pred_error_attention [C25]
          → transition_detect [C15/C20]
          → novelty_growth [C1]
          → argmin [C14]
          → action

    C22 (self-observation) reads own action_counts distribution and feeds
    a small signal back into the encoding pathway before recurrence.
    """

    def __init__(self, n_actions: int, seed: int):
        self.n_actions = n_actions
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        rng_sub = np.random.RandomState(seed + 10000)  # substrate-fixed seed

        # C4: Running mean
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0

        # C21/C26: Recurrent h — fixed random weights, tanh nonlinearity
        self.W_h = rng_sub.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rng_sub.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)

        # C25: Alpha attention — prediction error weights on ext_enc
        self.alpha = np.ones(EXT_DIM, np.float32)
        self.pred_err_ema = np.zeros(EXT_DIM, np.float32)
        # Forward model for prediction error (delta rule)
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM + n_actions), np.float32)

        # C14: Global action visit counts (argmin)
        self.action_counts = np.zeros(n_actions, np.float32)

        # C1: Novelty tracking
        self.novelty_set = set()
        self.novelty_ema = 0.0  # EMA of novel obs rate

        # C15/C20: Transition detection
        self.transition_map = {}  # obs_hash → set of next_obs_hashes
        self.transition_inconsistency_ema = 0.0

        # C22: Self-observation projection (fixed random, never trained)
        self.W_self = rng_sub.randn(ENC_DIM, n_actions).astype(np.float32) * 0.01

        # History buffers
        self._prev_ext_enc = None
        self._prev_attended = None
        self._prev_action = None
        self._prev_obs_hash = None

        self.step = 0

    def _centered_encode(self, obs: np.ndarray) -> np.ndarray:
        """C4: avgpool16 + running mean centering."""
        x = _enc_frame(obs)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def _update_alpha(self, ext_enc: np.ndarray):
        """C25: Update alpha from prediction error on ext_enc."""
        if self._prev_ext_enc is None or self._prev_action is None:
            return
        a_oh = np.zeros(self.n_actions, np.float32)
        a_oh[self._prev_action] = 1.0
        inp = np.concatenate([self._prev_ext_enc * self.alpha, a_oh])
        pred = self.W_pred @ inp
        error = (ext_enc * self.alpha) - pred
        # Clip error gradient for stability
        err_norm = float(np.linalg.norm(error))
        if err_norm > 10.0:
            error = error * (10.0 / err_norm)
        if not np.any(np.isnan(error)):
            self.W_pred -= ETA_PRED * np.outer(error, inp)
            per_dim_err = np.abs(error)
            self.pred_err_ema = (1 - ALPHA_EMA_RATE) * self.pred_err_ema + ALPHA_EMA_RATE * per_dim_err
            # Recompute alpha from EMA prediction error
            raw_alpha = np.sqrt(self.pred_err_ema + 1e-8)
            mean_raw = raw_alpha.mean()
            if mean_raw > 1e-8 and not np.isnan(mean_raw):
                self.alpha = np.clip(raw_alpha / mean_raw, ALPHA_LO, ALPHA_HI)

    def _transition_signal(self, obs_hash: int, prev_hash: int) -> float:
        """C15/C20: Return 1.0 if this transition reveals inconsistency."""
        if prev_hash is None:
            return 0.0
        prev_nexts = self.transition_map.setdefault(prev_hash, set())
        was_inconsistent = len(prev_nexts) > 1
        prev_nexts.add(obs_hash)
        is_now_inconsistent = len(prev_nexts) > 1
        # Signal = 1.0 on first discovery of inconsistency
        signal = 1.0 if (is_now_inconsistent and not was_inconsistent) else (
            0.5 if is_now_inconsistent else 0.0
        )
        self.transition_inconsistency_ema = (
            0.95 * self.transition_inconsistency_ema + 0.05 * float(is_now_inconsistent)
        )
        return signal

    def get_internal_repr_readonly(self, obs_raw: np.ndarray,
                                   frozen_running_mean: np.ndarray,
                                   frozen_h: np.ndarray,
                                   frozen_alpha: np.ndarray,
                                   frozen_action_counts: np.ndarray) -> np.ndarray:
        """
        Read-only encoding for R3 Jacobian computation.
        Does NOT update any substrate state.
        """
        obs = np.asarray(obs_raw, dtype=np.float32)
        x = _enc_frame(obs)
        enc_raw = x - frozen_running_mean

        # C22: Self-obs feedback (frozen action_counts)
        self_obs = frozen_action_counts / (frozen_action_counts.sum() + 1e-8)
        self_obs_signal = self.W_self @ self_obs
        enc = enc_raw + SELF_OBS_SCALE * self_obs_signal

        # C21/C26: Recurrent (frozen h, do NOT update self.h)
        h_new = np.tanh(self.W_h @ frozen_h + self.W_x @ enc)
        ext_enc = np.concatenate([enc, h_new])

        # C25: Alpha attention (frozen alpha)
        attended = frozen_alpha * ext_enc
        return attended  # 320D

    def process(self, obs_raw) -> int:
        obs = np.asarray(obs_raw, dtype=np.float32)

        # C4: Centered encoding
        enc_raw = self._centered_encode(obs)

        # C22: Self-observation feeds back into encoding pathway
        self_obs = self.action_counts / (self.action_counts.sum() + 1e-8)
        self_obs_signal = self.W_self @ self_obs  # 256D projection
        enc = enc_raw + SELF_OBS_SCALE * self_obs_signal  # modified enc

        # Obs hash for novelty + transition detection
        obs_hash = hash_enc(enc_raw)  # use raw enc for hashing (stable)

        # C21/C26: Recurrent h
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        ext_enc = np.concatenate([enc, self.h])  # 320D

        # C25: Alpha attention (update W_pred and alpha)
        self._update_alpha(ext_enc)
        attended = self.alpha * ext_enc  # 320D

        # C15/C20: Transition detection
        trans_signal = self._transition_signal(obs_hash, self._prev_obs_hash)
        attended_mod = attended * (1 + TRANSITION_SCALE * trans_signal)

        # C1: Novelty-triggered growth
        novel = obs_hash not in self.novelty_set
        if novel:
            self.novelty_set.add(obs_hash)
            self.novelty_ema = 0.9 * self.novelty_ema + 0.1 * 1.0
        else:
            self.novelty_ema = 0.9 * self.novelty_ema + 0.1 * 0.0

        # C14: Argmin action selection
        # Novelty-triggered: when novelty rate is high, explore randomly
        if self.novelty_ema > NOVELTY_EXPLORE_THRESHOLD:
            action = self._rng.randint(0, self.n_actions)
        else:
            action = int(np.argmin(self.action_counts))

        self.action_counts[action] += 1

        # Store history
        self._prev_ext_enc = ext_enc.copy()
        self._prev_attended = attended_mod.copy()
        self._prev_obs_hash = obs_hash
        self._prev_action = action
        self.step += 1

        return action

    def on_level_transition(self):
        # Reset transition-sensitive state; keep running_mean and h (trajectory context)
        self._prev_ext_enc = None
        self._prev_attended = None
        self._prev_action = None
        self._prev_obs_hash = None

    def reset(self, seed: int):
        self.running_mean[:] = 0
        self.n_obs = 0
        self.h[:] = 0
        self.alpha[:] = 1
        self.pred_err_ema[:] = 0
        self.W_pred[:] = 0
        self.action_counts[:] = 0
        self.novelty_set.clear()
        self.novelty_ema = 0.0
        self.transition_map.clear()
        self.transition_inconsistency_ema = 0.0
        self._prev_ext_enc = None
        self._prev_attended = None
        self._prev_action = None
        self._prev_obs_hash = None
        self.step = 0
        self._rng = np.random.RandomState(seed)

    def get_state(self) -> dict:
        return {
            'running_mean': self.running_mean.copy(),
            'n_obs': self.n_obs,
            'h': self.h.copy(),
            'alpha': self.alpha.copy(),
            'pred_err_ema': self.pred_err_ema.copy(),
            'W_pred': self.W_pred.copy(),
            'action_counts': self.action_counts.copy(),
            'novelty_ema': self.novelty_ema,
            'transition_inconsistency_ema': self.transition_inconsistency_ema,
            'step': self.step,
        }

    def set_state(self, state: dict):
        self.running_mean = state['running_mean'].copy()
        self.n_obs = state['n_obs']
        self.h = state['h'].copy()
        self.alpha = state['alpha'].copy()
        self.pred_err_ema = state['pred_err_ema'].copy()
        self.W_pred = state['W_pred'].copy()
        self.action_counts = state['action_counts'].copy()
        self.novelty_ema = state['novelty_ema']
        self.transition_inconsistency_ema = state['transition_inconsistency_ema']
        self.step = state['step']


class WiringB:
    """
    Wiring B — parallel streams.

    Stream 1: obs → centered_enc [C4] → transition_detect [C15/C20] → novelty_growth [C1]
    Stream 2: obs → centered_enc [C4] → recurrent_h [C21/C26] → pred_error_attention [C25]
    C22 (self-observation): bridges both streams via shared encoding pathway.
    C14 (argmin): merges both stream outputs for action selection.
    Novelty-triggered exploration governed by stream 1 novelty_ema.

    Both streams receive the same enc input — parallel not sequential data flow.
    Internal representation for R3: concat(stream1_out 256D, stream2_out 320D) = 576D.
    """

    def __init__(self, n_actions: int, seed: int):
        self.n_actions = n_actions
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        rng_sub = np.random.RandomState(seed + 10000)

        # C4: Running mean (shared across both streams)
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0

        # Stream 2: C21/C26 — Recurrent h (same init as WiringA)
        self.W_h = rng_sub.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rng_sub.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)

        # Stream 2: C25 — Alpha attention
        self.alpha = np.ones(EXT_DIM, np.float32)
        self.pred_err_ema = np.zeros(EXT_DIM, np.float32)
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM + n_actions), np.float32)

        # C14: Global action visit counts (argmin)
        self.action_counts = np.zeros(n_actions, np.float32)

        # Stream 1: C1 — Novelty tracking
        self.novelty_set = set()
        self.novelty_ema = 0.0

        # Stream 1: C15/C20 — Transition detection
        self.transition_map = {}
        self.transition_inconsistency_ema = 0.0

        # C22: Self-observation (bridges both streams via encoding pathway)
        self.W_self = rng_sub.randn(ENC_DIM, n_actions).astype(np.float32) * 0.01

        # History buffers
        self._prev_ext_enc = None  # for C25 update (stream 2)
        self._prev_action = None
        self._prev_obs_hash = None
        self._last_repr = None  # 576D concat for I1 repr log

        self.step = 0

    def _centered_encode(self, obs: np.ndarray) -> np.ndarray:
        """C4: avgpool16 + running mean centering (shared)."""
        x = _enc_frame(obs)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def _update_alpha(self, ext_enc: np.ndarray):
        """C25: Update alpha from prediction error. Identical to WiringA."""
        if self._prev_ext_enc is None or self._prev_action is None:
            return
        a_oh = np.zeros(self.n_actions, np.float32)
        a_oh[self._prev_action] = 1.0
        inp = np.concatenate([self._prev_ext_enc * self.alpha, a_oh])
        pred = self.W_pred @ inp
        error = (ext_enc * self.alpha) - pred
        err_norm = float(np.linalg.norm(error))
        if err_norm > 10.0:
            error = error * (10.0 / err_norm)
        if not np.any(np.isnan(error)):
            self.W_pred -= ETA_PRED * np.outer(error, inp)
            per_dim_err = np.abs(error)
            self.pred_err_ema = (1 - ALPHA_EMA_RATE) * self.pred_err_ema + ALPHA_EMA_RATE * per_dim_err
            raw_alpha = np.sqrt(self.pred_err_ema + 1e-8)
            mean_raw = raw_alpha.mean()
            if mean_raw > 1e-8 and not np.isnan(mean_raw):
                self.alpha = np.clip(raw_alpha / mean_raw, ALPHA_LO, ALPHA_HI)

    def _transition_signal(self, obs_hash: int, prev_hash) -> float:
        """C15/C20: Return transition inconsistency signal. Identical to WiringA."""
        if prev_hash is None:
            return 0.0
        prev_nexts = self.transition_map.setdefault(prev_hash, set())
        was_inconsistent = len(prev_nexts) > 1
        prev_nexts.add(obs_hash)
        is_now_inconsistent = len(prev_nexts) > 1
        signal = 1.0 if (is_now_inconsistent and not was_inconsistent) else (
            0.5 if is_now_inconsistent else 0.0
        )
        self.transition_inconsistency_ema = (
            0.95 * self.transition_inconsistency_ema + 0.05 * float(is_now_inconsistent)
        )
        return signal

    def get_internal_repr_readonly(self, obs_raw: np.ndarray,
                                   frozen_running_mean: np.ndarray,
                                   frozen_h: np.ndarray,
                                   frozen_alpha: np.ndarray,
                                   frozen_action_counts: np.ndarray) -> np.ndarray:
        """
        Read-only 576D internal repr for R3 Jacobian: concat(stream1, stream2).
        Does NOT update any substrate state.
        Stream 1 output (256D): enc with self-obs (transition is state-history dependent,
        not obs-dependent, so omitted for Jacobian).
        Stream 2 output (320D): alpha * [enc, h_new].
        """
        obs = np.asarray(obs_raw, dtype=np.float32)
        x = _enc_frame(obs)
        enc_raw = x - frozen_running_mean

        # C22: Self-obs (frozen action_counts)
        self_obs = frozen_action_counts / (frozen_action_counts.sum() + 1e-8)
        self_obs_signal = self.W_self @ self_obs
        enc = enc_raw + SELF_OBS_SCALE * self_obs_signal

        # Stream 1 output: enc (transition signal is state-history dependent, not obs-dependent)
        stream1_out = enc  # 256D

        # Stream 2: recurrent + attention
        h_new = np.tanh(self.W_h @ frozen_h + self.W_x @ enc)
        ext_enc = np.concatenate([enc, h_new])
        stream2_out = frozen_alpha * ext_enc  # 320D

        return np.concatenate([stream1_out, stream2_out])  # 576D

    def process(self, obs_raw) -> int:
        obs = np.asarray(obs_raw, dtype=np.float32)

        # C4: Centered encoding (shared entry point for both streams)
        enc_raw = self._centered_encode(obs)

        # C22: Self-observation bridges both streams via encoding pathway
        self_obs = self.action_counts / (self.action_counts.sum() + 1e-8)
        self_obs_signal = self.W_self @ self_obs
        enc = enc_raw + SELF_OBS_SCALE * self_obs_signal

        obs_hash = hash_enc(enc_raw)

        # ── Stream 1: transition_detect → novelty_growth ──
        trans_signal = self._transition_signal(obs_hash, self._prev_obs_hash)
        stream1_out = enc * (1 + TRANSITION_SCALE * trans_signal)  # 256D

        # C1: Novelty tracking (stream 1)
        novel = obs_hash not in self.novelty_set
        if novel:
            self.novelty_set.add(obs_hash)
            self.novelty_ema = 0.9 * self.novelty_ema + 0.1 * 1.0
        else:
            self.novelty_ema = 0.9 * self.novelty_ema + 0.1 * 0.0

        # ── Stream 2: recurrent_h → pred_error_attention ──
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        ext_enc = np.concatenate([enc, self.h])  # 320D

        # C25: Alpha attention update (stream 2)
        self._update_alpha(ext_enc)
        stream2_out = self.alpha * ext_enc  # 320D

        # ── Merge: argmin action selection ──
        # Novelty-triggered exploration from stream 1
        if self.novelty_ema > NOVELTY_EXPLORE_THRESHOLD:
            action = self._rng.randint(0, self.n_actions)
        else:
            action = int(np.argmin(self.action_counts))

        self.action_counts[action] += 1

        # Store 576D repr for I1 log
        self._last_repr = np.concatenate([stream1_out, stream2_out])

        # History for C25 update
        self._prev_ext_enc = ext_enc.copy()
        self._prev_action = action
        self._prev_obs_hash = obs_hash
        self.step += 1

        return action

    def on_level_transition(self):
        self._prev_ext_enc = None
        self._prev_action = None
        self._prev_obs_hash = None

    def reset(self, seed: int):
        self.running_mean[:] = 0
        self.n_obs = 0
        self.h[:] = 0
        self.alpha[:] = 1
        self.pred_err_ema[:] = 0
        self.W_pred[:] = 0
        self.action_counts[:] = 0
        self.novelty_set.clear()
        self.novelty_ema = 0.0
        self.transition_map.clear()
        self.transition_inconsistency_ema = 0.0
        self._prev_ext_enc = None
        self._prev_action = None
        self._prev_obs_hash = None
        self._last_repr = None
        self.step = 0
        self._rng = np.random.RandomState(seed)

    def get_state(self) -> dict:
        return {
            'running_mean': self.running_mean.copy(),
            'n_obs': self.n_obs,
            'h': self.h.copy(),
            'alpha': self.alpha.copy(),
            'pred_err_ema': self.pred_err_ema.copy(),
            'W_pred': self.W_pred.copy(),
            'action_counts': self.action_counts.copy(),
            'novelty_ema': self.novelty_ema,
            'transition_inconsistency_ema': self.transition_inconsistency_ema,
            'step': self.step,
        }

    def set_state(self, state: dict):
        self.running_mean = state['running_mean'].copy()
        self.n_obs = state['n_obs']
        self.h = state['h'].copy()
        self.alpha = state['alpha'].copy()
        self.pred_err_ema = state['pred_err_ema'].copy()
        self.W_pred = state['W_pred'].copy()
        self.action_counts = state['action_counts'].copy()
        self.novelty_ema = state['novelty_ema']
        self.transition_inconsistency_ema = state['transition_inconsistency_ema']
        self.step = state['step']


# ─────────────────────────────────────────────────────────────
# INSTRUMENTATION
# ─────────────────────────────────────────────────────────────

def compute_i3(action_counts_200: np.ndarray, kb_delta: np.ndarray) -> tuple:
    """
    I3 — Action discovery (step 200).
    Spearman ρ between substrate action_counts[0:7] and kb_delta[0:7].
    PASS: ρ > 0.5
    Returns: (rho, pass_bool) or (None, None) if insufficient data.
    """
    n = min(len(action_counts_200), 7)
    if n < 2:
        return None, None
    freq = action_counts_200[:n].astype(np.float32)
    ref = kb_delta[:n].astype(np.float32)
    if ref.max() - ref.min() < 1e-6:
        return None, None
    rho = spearman_rho(freq, ref)
    if rho is None:
        return None, None
    return rho, bool(rho > 0.5)


def compute_i1(repr_level_log: list) -> dict:
    """
    I1 — State representation (step 1000).
    repr_level_log: list of (repr_vec: np.ndarray, level: int)
    Samples pairs from the log. Within-level pairs = same level.
    Between-level pairs = different levels.
    PASS: mean within-level cosine_dist < mean between-level, p < 0.05
    """
    if len(repr_level_log) < 4:
        return {'within': None, 'between': None, 'p_value': 1.0, 'pass': False}

    rng = np.random.RandomState(1)
    n = len(repr_level_log)

    within_dists = []
    between_dists = []

    # Sample up to 200 random pairs
    for _ in range(200):
        i, j = rng.choice(n, 2, replace=False)
        r1, l1 = repr_level_log[i]
        r2, l2 = repr_level_log[j]
        d = cosine_dist(r1, r2)
        if l1 == l2:
            within_dists.append(d)
        else:
            between_dists.append(d)

    if len(within_dists) < 2 or len(between_dists) < 2:
        return {'within': None, 'between': None, 'p_value': 1.0, 'pass': False}

    p_val = permutation_test_i1(within_dists, between_dists, n_perms=500)
    within_mean = float(np.mean(within_dists))
    between_mean = float(np.mean(between_dists))
    i1_pass = bool(within_mean < between_mean and p_val < 0.05)

    return {
        'within': round(within_mean, 4),
        'between': round(between_mean, 4),
        'p_value': round(p_val, 4),
        'pass': i1_pass,
    }


def compute_i4(action_log: list, n_actions: int) -> dict:
    """
    I4 — Temporal structure.
    action_log: full list of actions taken.
    Computes sliding-window entropy at step 100 and step 5000.
    PASS: entropy at 5000 < entropy at 100 by >10%
    """
    def entropy_at(step):
        if step > len(action_log):
            return None
        window = action_log[max(0, step - I4_WINDOW): step]
        if len(window) < 5:
            return None
        return action_entropy(window, n_actions)

    h100 = entropy_at(100)
    h5000 = entropy_at(5000)

    if h100 is None or h5000 is None or h100 < 1e-6:
        return {'entropy_100': h100, 'entropy_5000': h5000, 'reduction_pct': None, 'pass': False}

    reduction_pct = float((h100 - h5000) / h100 * 100)
    return {
        'entropy_100': round(h100, 4),
        'entropy_5000': round(h5000, 4) if h5000 is not None else None,
        'reduction_pct': round(reduction_pct, 2),
        'pass': bool(reduction_pct > 10.0),
    }


def compute_r3_jacobian(substrate, obs_sample: list,
                        frozen_state: dict, n_actions: int,
                        rng_seed: int = 42) -> dict:
    """
    R3 — Sketched Jacobian on internal representation.

    For each obs in sample:
      - Compute encoding with frozen experienced state
      - Compute encoding with fresh state (running_mean=0, h=0, alpha=1)
      - Estimate Jacobian via 20 random directional perturbations

    R3_jacobian_diff = mean Frobenius norm of (J_exp_sketch - J_fresh_sketch)
    PASS: diff > 0.05 (provisional; final threshold set from Control C baseline)

    Notes:
      - Control C internal repr = enc (256D), no h no alpha
      - Wiring A internal repr = alpha * [enc, h] (320D)

    Returns: {'jacobian_diff': float, 'pass': bool, 'n_obs_used': int}
    """
    rng = np.random.RandomState(rng_seed)
    if not obs_sample:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}

    # Fresh state for comparison
    fresh_running_mean = np.zeros(ENC_DIM, np.float32)
    frozen_rm = frozen_state['running_mean']

    # For WiringA/WiringB: also freeze h and alpha
    _is_stateful = isinstance(substrate, (WiringA, WiringB))
    if _is_stateful:
        frozen_h = frozen_state['h']
        frozen_alpha = frozen_state['alpha']
        frozen_ac = frozen_state['action_counts']
        fresh_h = np.zeros(H_DIM, np.float32)
        fresh_alpha = np.ones(EXT_DIM, np.float32)
        fresh_ac = np.zeros(n_actions, np.float32)
    else:
        frozen_h = fresh_h = frozen_alpha = fresh_alpha = None
        frozen_ac = fresh_ac = None

    diffs = []
    for obs_raw in obs_sample[:R3_N_OBS]:
        obs_flat = np.asarray(obs_raw, dtype=np.float32)

        # Experienced Jacobian sketch
        if _is_stateful:
            baseline_exp = substrate.get_internal_repr_readonly(
                obs_flat, frozen_rm, frozen_h, frozen_alpha, frozen_ac)
        else:
            baseline_exp = substrate.get_internal_repr_readonly(obs_flat, frozen_rm)

        J_exp_cols = []
        for _ in range(R3_N_DIRS):
            direction = rng.randn(*obs_flat.shape).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            perturbed = obs_flat + R3_EPSILON * direction
            if _is_stateful:
                pert_repr = substrate.get_internal_repr_readonly(
                    perturbed, frozen_rm, frozen_h, frozen_alpha, frozen_ac)
            else:
                pert_repr = substrate.get_internal_repr_readonly(perturbed, frozen_rm)
            J_exp_cols.append((pert_repr - baseline_exp) / R3_EPSILON)
        J_exp = np.stack(J_exp_cols, axis=0)  # [n_dirs, repr_dim]

        # Fresh Jacobian sketch
        if _is_stateful:
            baseline_fresh = substrate.get_internal_repr_readonly(
                obs_flat, fresh_running_mean, fresh_h, fresh_alpha, fresh_ac)
        else:
            baseline_fresh = substrate.get_internal_repr_readonly(obs_flat, fresh_running_mean)

        J_fresh_cols = []
        for _ in range(R3_N_DIRS):
            direction = rng.randn(*obs_flat.shape).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            perturbed = obs_flat + R3_EPSILON * direction
            if _is_stateful:
                pert_repr = substrate.get_internal_repr_readonly(
                    perturbed, fresh_running_mean, fresh_h, fresh_alpha, fresh_ac)
            else:
                pert_repr = substrate.get_internal_repr_readonly(perturbed, fresh_running_mean)
            J_fresh_cols.append((pert_repr - baseline_fresh) / R3_EPSILON)
        J_fresh = np.stack(J_fresh_cols, axis=0)

        diff = float(np.linalg.norm(J_exp - J_fresh))
        diffs.append(diff)

    if not diffs:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}

    jacobian_diff = float(np.mean(diffs))
    return {
        'jacobian_diff': round(jacobian_diff, 6),
        'pass': bool(jacobian_diff > 0.05),
        'n_obs_used': len(diffs),
    }


# ─────────────────────────────────────────────────────────────
# SINGLE RUN HARNESS
# ─────────────────────────────────────────────────────────────

def run_single(game_name: str, condition: str, draw: int, seed: int,
               n_actions: int, kb_delta: np.ndarray) -> dict:
    """
    Run one experiment: one game, one condition, one draw.
    Returns a results dict with all instrumentation metrics.
    """
    print(f"  Running {game_name.upper()} | {condition} | draw={draw} | seed={seed} ...", end='', flush=True)

    # Build substrate
    if condition == 'control_c':
        substrate = ControlC(n_actions=n_actions, seed=seed)
    elif condition == 'wiring_a':
        substrate = WiringA(n_actions=n_actions, seed=seed)
    elif condition == 'wiring_b':
        substrate = WiringB(n_actions=n_actions, seed=seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Create game env
    env = make_game(game_name)
    obs = env.reset(seed=seed)

    # Instrumentation accumulators
    action_log = []         # [int] full sequence
    repr_log = []           # [(repr, level)] for I1 — sampled up to I1_STEP
    obs_store = []          # [raw_obs] for R3 — up to 200 obs
    level_action_starts = {}  # level → step when level was reached
    level_actions = {}      # level → action count to solve

    i3_action_counts = None
    r3_snapshot = None
    r3_obs_sample = None

    steps = 0
    level = 0
    level_start_step = 0
    t_start = time.time()
    fresh_episode = True
    l1_step = l2_step = None

    while steps < MAX_STEPS:
        elapsed = time.time() - t_start
        if elapsed > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)

        # Store obs for R3 (rolling buffer of last 200)
        obs_store.append(obs_arr)
        if len(obs_store) > 200:
            obs_store.pop(0)

        # Store repr for I1 (up to step I1_STEP)
        if steps <= I1_STEP and steps % 20 == 0:
            if hasattr(substrate, '_last_enc') and substrate._last_enc is not None:
                repr_log.append((substrate._last_enc.copy(), level))
            elif hasattr(substrate, '_last_repr') and substrate._last_repr is not None:
                repr_log.append((substrate._last_repr.copy(), level))
            elif hasattr(substrate, '_prev_attended') and substrate._prev_attended is not None:
                repr_log.append((substrate._prev_attended.copy(), level))

        # I3 snapshot
        if steps == I3_STEP:
            i3_action_counts = substrate.action_counts[:min(7, n_actions)].copy()

        # R3 snapshot
        if steps == R3_STEP:
            r3_snapshot = substrate.get_state()
            r3_obs_sample = list(obs_store)  # copy current buffer

        action = substrate.process(obs_arr) % n_actions
        action_log.append(action)

        obs, reward, done, info = env.step(action)
        steps += 1

        if fresh_episode:
            fresh_episode = False
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1_step is None:
                l1_step = steps
            if cl == 2 and l2_step is None:
                l2_step = steps
            # I5: record action count for this level transition
            level_actions[level + 1] = steps - level_start_step
            level = cl
            level_start_step = steps
            substrate.on_level_transition()

        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            level = 0
            level_start_step = steps

    elapsed = time.time() - t_start

    # ── After run: compute instrumentation ──

    # Store final repr for I1 if not enough yet
    if hasattr(substrate, '_last_enc') and substrate._last_enc is not None:
        repr_log.append((substrate._last_enc.copy(), level))

    # I3
    if i3_action_counts is not None:
        i3_rho, i3_pass = compute_i3(i3_action_counts, kb_delta)
    else:
        i3_rho, i3_pass = None, None

    # I1
    i1_result = compute_i1(repr_log)

    # I4
    i4_result = compute_i4(action_log, n_actions)

    # I5 (gated on l1_step being not None = L1 reached)
    i5_pass = None
    i5_level_actions = None
    if l1_step is not None:
        i5_level_actions = {int(k): int(v) for k, v in level_actions.items() if k >= 1}
        # PASS: L2 action count < L1 action count (transfer)
        if 1 in i5_level_actions and 2 in i5_level_actions:
            i5_pass = bool(i5_level_actions[2] < i5_level_actions[1])
        # else: can't measure (L2 not reached)

    # R3
    if r3_snapshot is not None and r3_obs_sample:
        r3_result = compute_r3_jacobian(substrate, r3_obs_sample, r3_snapshot, n_actions)
    else:
        r3_result = {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}

    result = {
        'game': game_name.lower(),
        'condition': condition,
        'draw': draw,
        'seed': seed,
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': level,
        'L1_solved': bool(l1_step is not None),
        'L2_solved': bool(l2_step is not None),
        'l1_step': l1_step,
        'l2_step': l2_step,

        # I3
        'I3_spearman_rho': round(i3_rho, 4) if i3_rho is not None else None,
        'I3_pass': i3_pass,

        # I1
        'I1_within_dist': i1_result['within'],
        'I1_between_dist': i1_result['between'],
        'I1_p_value': i1_result['p_value'],
        'I1_pass': i1_result['pass'],

        # I4
        'I4_entropy_100': i4_result['entropy_100'],
        'I4_entropy_5000': i4_result['entropy_5000'],
        'I4_reduction_pct': i4_result['reduction_pct'],
        'I4_pass': i4_result['pass'],

        # I5
        'I5_level_actions': i5_level_actions,
        'I5_pass': i5_pass,

        # R3
        'R3_jacobian_diff': r3_result['jacobian_diff'],
        'R3_n_obs_used': r3_result['n_obs_used'],
        'R3_pass': r3_result['pass'],

        # Raw per-step log
        'action_sequence': action_log,
    }

    # Console summary
    l1_str = f"L1={l1_step}" if l1_step else "L1=None"
    i3_str = f"I3ρ={i3_rho:.2f}" if i3_rho is not None else "I3=null"
    r3_str = f"R3={r3_result['jacobian_diff']:.4f}" if r3_result['jacobian_diff'] else "R3=null"
    print(f" {l1_str} | {i3_str} | I1={'P' if i1_result['pass'] else 'F'} | I4={'P' if i4_result['pass'] else 'F'} | {r3_str} | {elapsed:.1f}s")

    return result


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    t_global = time.time()

    print("=" * 70)
    print("STEP 1251 — COMPONENT COMPOSITION EXPERIMENT, PHASE 1")
    print("=" * 70)
    print(f"Games: {PHASE1_GAMES}")
    print(f"Conditions: control_c, wiring_a")
    print(f"Draws: {N_DRAWS} per condition per game = 30 runs")
    print(f"Budget: {MAX_STEPS} steps / {MAX_SECONDS}s per run")
    print()

    for game_name in PHASE1_GAMES:
        # Load game diagnostics
        try:
            diag = load_game_diag(game_name)
            kb_delta, kb_responsive = build_kb_profile(diag)
            game_n_actions = diag.get('n_actions', 4103)
        except Exception as e:
            print(f"ERROR: Could not load game diagnostics for {game_name}: {e}")
            continue

        print(f"\n{'─'*60}")
        print(f"GAME: {game_name.upper()} | n_actions={game_n_actions}")
        print(f"KB responsive: {kb_responsive.sum()}/7 | delta_mean={kb_delta.tolist()}")
        print(f"{'─'*60}")

        for condition in ['control_c', 'wiring_a']:
            print(f"\n  Condition: {condition}")
            for draw in range(1, N_DRAWS + 1):
                seed = draw * 100 + (0 if condition == 'control_c' else 1)
                result = run_single(
                    game_name=game_name,
                    condition=condition,
                    draw=draw,
                    seed=seed,
                    n_actions=game_n_actions,
                    kb_delta=kb_delta,
                )
                all_results.append(result)

                # Save incrementally (without action_sequence for incremental file)
                compact = {k: v for k, v in result.items() if k != 'action_sequence'}
                incr_path = os.path.join(RESULTS_DIR, f"{game_name}_{condition}_draw{draw}.json")
                with open(incr_path, 'w') as f:
                    json.dump(result, f, indent=2)

    # ── Summary ──
    total_elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"PHASE 1 COMPLETE — {len(all_results)} runs in {total_elapsed:.1f}s")
    print(f"{'='*70}")
    print()

    # Per-stage summary: compare control_c vs wiring_a
    stages = ['I1_pass', 'I3_pass', 'I4_pass', 'I5_pass', 'R3_pass']
    print(f"{'Stage':<12} {'ControlC':>12} {'WiringA':>12}  Decision")
    print("-" * 50)

    phase2_trigger = False
    for stage in stages:
        ctrl = [r[stage] for r in all_results
                if r['condition'] == 'control_c' and r[stage] is not None]
        wira = [r[stage] for r in all_results
                if r['condition'] == 'wiring_a' and r[stage] is not None]
        ctrl_rate = sum(ctrl) / len(ctrl) if ctrl else None
        wira_rate = sum(wira) / len(wira) if wira else None
        cr = f"{ctrl_rate:.2f}" if ctrl_rate is not None else "null"
        wr = f"{wira_rate:.2f}" if wira_rate is not None else "null"
        sig = ""
        if ctrl_rate is not None and wira_rate is not None:
            if wira_rate > ctrl_rate + 0.1:
                sig = "→ PHASE 2"
                phase2_trigger = True
            elif wira_rate < ctrl_rate - 0.1:
                sig = "← C wins"
        print(f"  {stage:<12} {cr:>12} {wr:>12}  {sig}")

    print()
    if phase2_trigger:
        print("DECISION: Signal found — PHASE 2 triggered (run Wiring B)")
    else:
        print("DECISION: No signal — composition adds nothing. Report to Leo.")

    # Per-game summary
    print(f"\nPer-game L1 solve rate:")
    for game in PHASE1_GAMES:
        for cond in ['control_c', 'wiring_a']:
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            l1_rate = sum(r['L1_solved'] for r in runs) / len(runs) if runs else 0
            l2_rate = sum(r['L2_solved'] for r in runs) / len(runs) if runs else 0
            print(f"  {game.upper()}/{cond}: L1={l1_rate:.1%} L2={l2_rate:.1%}")

    # Save full results
    out_path = os.path.join(RESULTS_DIR, 'phase1_results.json')
    # Save without action_sequence for the summary file
    summary_results = [{k: v for k, v in r.items() if k != 'action_sequence'}
                       for r in all_results]
    with open(out_path, 'w') as f:
        json.dump({
            'step': 1251,
            'phase': 1,
            'conditions': ['control_c', 'wiring_a'],
            'games': PHASE1_GAMES,
            'n_draws': N_DRAWS,
            'total_runs': len(all_results),
            'phase2_triggered': phase2_trigger,
            'runs': summary_results,
        }, f, indent=2)

    print(f"\nResults saved: {out_path}")
    print(f"Total elapsed: {total_elapsed:.1f}s")

    if phase2_trigger:
        run_phase2(phase1_wira_results=all_results, phase1_ctrl_results=all_results)


def run_phase2(phase1_wira_results: list, phase1_ctrl_results: list):
    """
    Phase 2: Run Wiring B on same 3 games, 5 draws (15 runs).
    Compare Wiring A vs Wiring B on the triggered stage (R3).
    If both pass same stages → component-level finding → Phase 3.
    If only A passes → wiring is load-bearing → Phase 3 with both wirings.
    """
    t_p2 = time.time()
    print()
    print("=" * 70)
    print("PHASE 2 — WIRING B (parallel streams), 15 runs")
    print("=" * 70)
    print(f"Games: {PHASE1_GAMES}")
    print(f"Conditions: wiring_b")
    print(f"Draws: {N_DRAWS} per game = 15 runs")
    print()

    p2_results = []

    for game_name in PHASE1_GAMES:
        try:
            diag = load_game_diag(game_name)
            kb_delta, _ = build_kb_profile(diag)
            game_n_actions = diag.get('n_actions', 4103)
        except Exception as e:
            print(f"ERROR: Could not load diagnostics for {game_name}: {e}")
            continue

        print(f"\n{'─'*60}")
        print(f"GAME: {game_name.upper()} | n_actions={game_n_actions}")
        print(f"{'─'*60}")
        print(f"\n  Condition: wiring_b")

        for draw in range(1, N_DRAWS + 1):
            seed = draw * 100 + 2  # distinct from wiring_a (seed+1) and control_c (seed+0)
            result = run_single(
                game_name=game_name,
                condition='wiring_b',
                draw=draw,
                seed=seed,
                n_actions=game_n_actions,
                kb_delta=kb_delta,
            )
            p2_results.append(result)

            incr_path = os.path.join(RESULTS_DIR, f"{game_name}_wiring_b_draw{draw}.json")
            with open(incr_path, 'w') as f:
                json.dump(result, f, indent=2)

    # ── Phase 2 summary ──
    total_p2 = time.time() - t_p2
    print(f"\n{'='*70}")
    print(f"PHASE 2 COMPLETE — {len(p2_results)} runs in {total_p2:.1f}s")
    print(f"{'='*70}")
    print()

    stages = ['I1_pass', 'I3_pass', 'I4_pass', 'I5_pass', 'R3_pass']

    # Wiring A pass rates from phase 1
    wira_rates = {}
    ctrl_rates = {}
    for stage in stages:
        wa = [r[stage] for r in phase1_wira_results
              if r['condition'] == 'wiring_a' and r[stage] is not None]
        cc = [r[stage] for r in phase1_ctrl_results
              if r['condition'] == 'control_c' and r[stage] is not None]
        wira_rates[stage] = sum(wa) / len(wa) if wa else None
        ctrl_rates[stage] = sum(cc) / len(cc) if cc else None

    print(f"{'Stage':<12} {'ControlC':>10} {'WiringA':>10} {'WiringB':>10}  Interpretation")
    print("-" * 65)

    phase3_trigger = False
    phase3_both_wirings = False
    for stage in stages:
        wb = [r[stage] for r in p2_results if r[stage] is not None]
        wb_rate = sum(wb) / len(wb) if wb else None
        wa_rate = wira_rates.get(stage)
        cc_rate = ctrl_rates.get(stage)

        cc_s = f"{cc_rate:.2f}" if cc_rate is not None else "null"
        wa_s = f"{wa_rate:.2f}" if wa_rate is not None else "null"
        wb_s = f"{wb_rate:.2f}" if wb_rate is not None else "null"

        # Determine interpretation
        interp = ""
        if wa_rate is not None and wb_rate is not None and cc_rate is not None:
            wa_beats_cc = wa_rate > cc_rate + 0.1
            wb_beats_cc = wb_rate > cc_rate + 0.1
            if wa_beats_cc and wb_beats_cc:
                interp = "COMPONENT FINDING (both beat C)"
                phase3_trigger = True
            elif wa_beats_cc and not wb_beats_cc:
                interp = "WIRING FINDING (A>C, B=C)"
                phase3_trigger = True
                phase3_both_wirings = True
            elif wb_beats_cc and not wa_beats_cc:
                interp = "WIRING FINDING (B>C, A=C)"
                phase3_trigger = True
                phase3_both_wirings = True

        print(f"  {stage:<12} {cc_s:>10} {wa_s:>10} {wb_s:>10}  {interp}")

    print()
    if phase3_trigger:
        if phase3_both_wirings:
            print("DECISION: WIRING FINDING — Phase 3 with BOTH wirings (wiring_a + wiring_b)")
        else:
            print("DECISION: COMPONENT FINDING — Phase 3 with best wiring")
    else:
        print("DECISION: No confirmation — composition signal is wiring-specific but not robust.")

    # Per-game L1 for Wiring B
    print(f"\nPer-game L1 solve rate (Wiring B):")
    for game in PHASE1_GAMES:
        runs = [r for r in p2_results if r['game'] == game]
        l1_rate = sum(r['L1_solved'] for r in runs) / len(runs) if runs else 0
        print(f"  {game.upper()}/wiring_b: L1={l1_rate:.1%}")

    # Save Phase 2 results
    p2_path = os.path.join(RESULTS_DIR, 'phase2_results.json')
    with open(p2_path, 'w') as f:
        json.dump({
            'step': 1251,
            'phase': 2,
            'conditions': ['wiring_b'],
            'games': PHASE1_GAMES,
            'n_draws': N_DRAWS,
            'total_runs': len(p2_results),
            'phase3_triggered': phase3_trigger,
            'phase3_both_wirings': phase3_both_wirings,
            'runs': [{k: v for k, v in r.items() if k != 'action_sequence'} for r in p2_results],
        }, f, indent=2)

    print(f"\nResults saved: {p2_path}")
    print(f"Phase 2 elapsed: {total_p2:.1f}s")

    if phase3_trigger:
        run_phase3(p2_results=p2_results, phase1_results=phase1_wira_results,
                   both_wirings=phase3_both_wirings)


def run_phase3(p2_results: list, phase1_results: list, both_wirings: bool):
    """
    Phase 3: Expand to all 10 fully-solved games, 10 draws, best wiring (75 runs).
    Or both wirings if Phase 2 found wiring-dependent signal (150 runs).
    """
    SOLVED_GAMES = ['ft09', 'ls20', 'vc33', 'lp85', 'tr87', 'sb26', 'sp80', 'cd82', 'cn04', 'tu93']
    P3_DRAWS = 10

    if both_wirings:
        conditions = ['wiring_a', 'wiring_b']
        total_runs = len(SOLVED_GAMES) * P3_DRAWS * 2
    else:
        conditions = ['wiring_a']
        total_runs = len(SOLVED_GAMES) * P3_DRAWS

    t_p3 = time.time()
    print()
    print("=" * 70)
    print(f"PHASE 3 — 10-GAME EXPANSION, {total_runs} runs")
    print("=" * 70)
    print(f"Games: {SOLVED_GAMES}")
    print(f"Conditions: {conditions}")
    print(f"Draws: {P3_DRAWS} per game per condition = {total_runs} runs")
    print()

    p3_results = []

    for game_name in SOLVED_GAMES:
        try:
            diag = load_game_diag(game_name)
            kb_delta, _ = build_kb_profile(diag)
            game_n_actions = diag.get('n_actions', 4103)
        except Exception as e:
            print(f"WARNING: No diagnostics for {game_name}: {e} — skipping")
            continue

        print(f"\n{'─'*60}")
        print(f"GAME: {game_name.upper()} | n_actions={game_n_actions}")
        print(f"{'─'*60}")

        for cond in conditions:
            print(f"\n  Condition: {cond}")
            cond_offset = 0 if cond == 'wiring_a' else 2
            for draw in range(1, P3_DRAWS + 1):
                seed = draw * 100 + cond_offset
                result = run_single(
                    game_name=game_name,
                    condition=cond,
                    draw=draw,
                    seed=seed,
                    n_actions=game_n_actions,
                    kb_delta=kb_delta,
                )
                p3_results.append(result)

                incr_path = os.path.join(RESULTS_DIR, f"p3_{game_name}_{cond}_draw{draw}.json")
                with open(incr_path, 'w') as f:
                    json.dump(result, f, indent=2)

    # ── Phase 3 summary ──
    total_p3 = time.time() - t_p3
    print(f"\n{'='*70}")
    print(f"PHASE 3 COMPLETE — {len(p3_results)} runs in {total_p3:.1f}s")
    print(f"{'='*70}")
    print()

    stages = ['I1_pass', 'I3_pass', 'I4_pass', 'I5_pass', 'R3_pass']
    for cond in conditions:
        print(f"\n  [{cond.upper()}] per-stage pass rates across 10 games:")
        for stage in stages:
            vals = [r[stage] for r in p3_results
                    if r['condition'] == cond and r[stage] is not None]
            rate = sum(vals) / len(vals) if vals else None
            rate_str = f"{rate:.2f}" if rate is not None else "null"
            print(f"    {stage:<12} {rate_str}")

    # Per-game R3 breakdown
    print(f"\n  R3 pass rate by game:")
    for game in SOLVED_GAMES:
        for cond in conditions:
            runs = [r for r in p3_results if r['game'] == game and r['condition'] == cond]
            r3 = [r['R3_pass'] for r in runs if r['R3_pass'] is not None]
            rate = sum(r3) / len(r3) if r3 else None
            rate_str = f"{rate:.2f}" if rate is not None else "null"
            print(f"    {game.upper()}/{cond}: R3={rate_str}")

    # Save Phase 3 results
    p3_path = os.path.join(RESULTS_DIR, 'phase3_results.json')
    with open(p3_path, 'w') as f:
        json.dump({
            'step': 1251,
            'phase': 3,
            'conditions': conditions,
            'games': SOLVED_GAMES,
            'n_draws': P3_DRAWS,
            'total_runs': len(p3_results),
            'both_wirings': both_wirings,
            'runs': [{k: v for k, v in r.items() if k != 'action_sequence'} for r in p3_results],
        }, f, indent=2)

    print(f"\nResults saved: {p3_path}")
    print(f"Phase 3 elapsed: {total_p3:.1f}s")


if __name__ == '__main__':
    # Load API key
    _env_file = r'C:\Users\Admin\.secrets\.env'
    if os.path.exists(_env_file):
        with open(_env_file) as f:
            for line in f:
                if line.strip().startswith('ARC_API_KEY='):
                    os.environ['ARC_API_KEY'] = line.strip().split('=', 1)[1].strip()
                    break
    main()
