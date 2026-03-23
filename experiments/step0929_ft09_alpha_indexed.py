"""
step0929_ft09_alpha_indexed.py -- FT09 alpha-indexed action grouping.

R3 hypothesis: Alpha identifies informative spatial dims. Actions can be
grouped by which alpha dims they maximally affect. Cycling through alpha-
grouped action sets forces sequential tile visitation — state-conditioned
by encoding structure, not per-state memory.

Phase 1 (5K steps): 895h on all 68 actions. Track per-action alpha-dim change.
Phase 2 (20K steps): Group actions by top-affected alpha dim. Cycle groups
top_dim_A → top_dim_B → top_dim_C → repeat. Within group: pick max delta.

NOT graph (no per-state counting). NOT codebook. Pure alpha-dim grouping.
Alpha dims computed from prediction error — same mechanism as 895h.

Run: FT09, 25K, 10 seeds cold.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
N_ACTIONS = 68
ETA_W = 0.01
ALPHA_EMA = 0.10
DELTA_EMA = 0.10
EFFECT_EMA = 0.05
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
PHASE1_STEPS = 5_000
TOP_DIMS = 3       # how many alpha dims to group around
MIN_GROUP_SIZE = 1 # discard groups with fewer actions

TEST_SEEDS = list(range(1, 11))
TEST_STEPS = 25_000


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


def softmax_sel(scores, temp, rng):
    x = np.array(scores) / temp; x -= np.max(x); e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(scores), p=probs))


class AlphaIndexedFT09:
    """895h + alpha-dim action grouping for FT09."""

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        self.W = np.zeros((ENC_DIM, ENC_DIM + N_ACTIONS), dtype=np.float32)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self.delta_per_action = np.full(N_ACTIONS, INIT_DELTA, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        self._prev_enc = None; self._prev_action = None

        # Per-action alpha-dim effect tracker (ENC_DIM per action)
        self.action_alpha_effect = np.zeros((N_ACTIONS, ENC_DIM), dtype=np.float32)

        # Phase 2 state
        self._step = 0
        self._groups = None       # list of action lists, in cycle order
        self._group_idx = 0       # which group we're currently targeting
        self._group_log = None    # logged at phase transition

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

    def _build_groups(self):
        """After phase 1: identify top-K alpha dims, group actions by max effect on those dims."""
        # Top informative dims: highest alpha value (highest prediction error = most informative)
        top_dim_indices = list(np.argsort(self.alpha)[-TOP_DIMS:])

        # For each action, which top dim does it maximally affect?
        groups = {d: [] for d in top_dim_indices}
        unassigned = []
        for a in range(N_ACTIONS):
            effect_on_top = self.action_alpha_effect[a, top_dim_indices]
            if effect_on_top.max() < 1e-6:
                unassigned.append(a)  # no effect on any top dim
                continue
            best_top_dim = top_dim_indices[int(np.argmax(effect_on_top))]
            groups[best_top_dim].append(a)

        # Filter empty groups, sort by alpha value (most informative first)
        active_groups = []
        for d in sorted(top_dim_indices, key=lambda x: -self.alpha[x]):
            if len(groups[d]) >= MIN_GROUP_SIZE:
                active_groups.append(groups[d])

        # Fallback: if grouping fails, just use delta_per_action
        if not active_groups:
            active_groups = [list(range(N_ACTIONS))]

        self._groups = active_groups
        self._group_idx = 0
        return top_dim_indices, groups, unassigned

    def process(self, obs):
        enc = self._encode(obs)
        self._step += 1

        if self._prev_enc is not None and self._prev_action is not None:
            inp = np.concatenate([self._prev_enc * self.alpha,
                                   one_hot(self._prev_action, N_ACTIONS)])
            pred = self.W @ inp
            error = (enc * self.alpha) - pred
            en = float(np.linalg.norm(error))
            if en > 10.0: error *= 10.0 / en
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            weighted_delta = (enc - self._prev_enc) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = (1 - DELTA_EMA) * self.delta_per_action[a] + DELTA_EMA * change

            # Track per-action alpha-dim effect (phase 1 only)
            if self._step <= PHASE1_STEPS:
                per_dim_change = np.abs((enc - self._prev_enc) * self.alpha)
                self.action_alpha_effect[self._prev_action] = (
                    (1 - EFFECT_EMA) * self.action_alpha_effect[self._prev_action]
                    + EFFECT_EMA * per_dim_change
                )

        # Phase transition
        if self._step == PHASE1_STEPS and self._groups is None:
            top_dims, groups, unassigned = self._build_groups()
            self._group_log = (top_dims,
                               {d: groups[d] for d in top_dims},
                               len(unassigned))

        # Action selection
        if self._step <= PHASE1_STEPS or self._groups is None:
            # Phase 1: standard 895h
            if self._rng.random() < EPSILON:
                action = int(self._rng.randint(0, N_ACTIONS))
            else:
                action = softmax_sel(self.delta_per_action, SOFTMAX_TEMP, self._rng)
        else:
            # Phase 2: cycle through alpha-grouped action sets
            if self._rng.random() < EPSILON:
                action = int(self._rng.randint(0, N_ACTIONS))
            else:
                # Pick from current group, advance group on action
                group = self._groups[self._group_idx % len(self._groups)]
                group_deltas = self.delta_per_action[group]
                best_in_group = group[int(np.argmax(group_deltas))]
                action = best_in_group
                # Advance to next group (cycle)
                self._group_idx += 1

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))


def make_game():
    try:
        import arcagi3; return arcagi3.make("FT09")
    except:
        import util_arcagi3; return util_arcagi3.make("FT09")


BASELINE_ACTIONS = [17, 19, 15, 21, 65, 26]

print("=" * 70)
print("STEP 929 — FT09 ALPHA-INDEXED ACTION GROUPING")
print("=" * 70)
print("Phase 1 (5K): track per-action alpha-dim change. Phase 2 (20K): cycle groups.")
t0 = time.time()

results = []
for seed in TEST_SEEDS:
    sub = AlphaIndexedFT09(seed=seed)
    env = make_game()
    obs = env.reset(seed=seed * 1000)
    step = 0; completions = 0; level = 0

    while step < TEST_STEPS:
        if obs is None:
            obs = env.reset(seed=seed * 1000); sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            completions += cl - level; level = cl; sub.on_level_transition()
        if done:
            obs = env.reset(seed=seed * 1000); level = 0; sub.on_level_transition()

    results.append(completions)
    gl = sub._group_log
    if gl:
        top_dims, groups, n_unassigned = gl
        overlap_per_dim = {d: [a for a in groups[d] if a in BASELINE_ACTIONS]
                           for d in top_dims}
        print(f"  seed={seed}: L1={completions:4d}  alpha_conc={sub.alpha_conc():.2f}"
              f"  top_dims={top_dims}  groups={[len(groups[d]) for d in top_dims]}"
              f"  unassigned={n_unassigned}  baseline_overlap={overlap_per_dim}")
    else:
        print(f"  seed={seed}: L1={completions:4d}  alpha_conc={sub.alpha_conc():.2f}"
              f"  (phase 2 not reached)")

mean = np.mean(results); zeros = sum(1 for x in results if x == 0)
print(f"\n{'='*70}")
print(f"STEP 929 RESULTS (alpha-indexed grouping, FT09):")
print(f"  L1={mean:.1f}/seed  std={np.std(results):.1f}  zero={zeros}/10")
print(f"  {results}")
print(f"Comparison: 895h standalone FT09: 0.0/seed  0/10")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 929 DONE")
