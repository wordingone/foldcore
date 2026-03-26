"""
sub1125_defense_v42.py — Action sequence probing (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1125 --substrate experiments/sub1125_defense_v42.py

FAMILY: Action sequence probing (NEW defense family)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: 0% games may require ACTION SEQUENCES to produce any
observable change. Single actions are no-ops, but specific 2-3 action
sequences produce change. No substrate has ever tried multi-step probing.

Architecture:
- Phase 1 (steps 1-50): single-action explore (same as v30)
- Phase 2 (steps 51-200): if Phase 1 found NO responsive actions,
  probe 2-action SEQUENCES. For each pair (a,b), execute a then b,
  check if the 2-step sequence produced more change than either alone.
- Phase 3 (step 200+): reactive switching using discovered responsive
  actions OR sequences.

For games where single actions work: Phase 1 finds them, Phase 2 is
skipped. No overhead on solvable games.

KILL: ARC ≤ 0 (sequences don't help either).
SUCCESS: ARC > 0 on previously-0% games (sequence actions were needed).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
PHASE1_STEPS = 50
PHASE2_STEPS = 150  # 7*7=49 pairs × 3 steps each ≈ 147
MAX_PATIENCE = 20
CHANGE_THRESH = 0.5


def _obs_to_enc(obs):
    """avgpool4: 64x64 → 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class ActionSequenceSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None

        # Phase tracking
        self._phase = "single_explore"
        self._single_responsive = []  # actions that produced change alone
        self._single_change = np.zeros(N_KB, dtype=np.float32)

        # Sequence probing
        self._seq_pairs = []  # (a, b) pairs to try
        self._seq_idx = 0
        self._seq_step = 0  # 0=first action, 1=second action, 2=measure
        self._seq_pre_enc = None
        self._responsive_seqs = []  # (a, b) pairs that produced change

        # Reactive switching
        self._active_plan = []  # list of actions to cycle through
        self._plan_idx = 0
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._init_state()
        # Build sequence pairs
        self._seq_pairs = []
        for a in range(self._n_actions):
            for b in range(self._n_actions):
                if a != b:
                    self._seq_pairs.append((a, b))
        self._rng.shuffle(self._seq_pairs)

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_dist = 0.0
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        dist = self._dist_to_initial(enc)

        # === Phase 1: Single-action explore ===
        if self._phase == "single_explore":
            # Track per-action change
            if self._prev_enc is not None:
                change = float(np.sum(np.abs(enc - self._prev_enc)))
                prev_action = (self.step_count - 1) % self._n_actions
                self._single_change[prev_action] = max(
                    self._single_change[prev_action], change
                )

            if self.step_count <= PHASE1_STEPS:
                action = self.step_count % self._n_actions
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return action

            # Phase 1 done — check what responded
            for a in range(self._n_actions):
                if self._single_change[a] > CHANGE_THRESH:
                    self._single_responsive.append(a)

            if len(self._single_responsive) >= 2:
                # Found responsive single actions — skip to reactive
                self._phase = "reactive"
                self._active_plan = self._single_responsive[:]
                self._current_action = 0
            else:
                # No responsive single actions — try sequences
                self._phase = "sequence_probe"
                self._seq_idx = 0
                self._seq_step = 0
                self._seq_pre_enc = enc.copy()

        # === Phase 2: Sequence probing ===
        if self._phase == "sequence_probe":
            if self._seq_idx >= len(self._seq_pairs) or self.step_count > PHASE1_STEPS + PHASE2_STEPS:
                # Probing done
                if self._responsive_seqs:
                    # Build plan from responsive sequences
                    self._active_plan = []
                    for a, b in self._responsive_seqs[:5]:  # top 5 sequences
                        self._active_plan.extend([a, b])
                    self._phase = "reactive"
                elif self._single_responsive:
                    self._active_plan = self._single_responsive[:]
                    self._phase = "reactive"
                else:
                    # Nothing works — fall back to round-robin
                    self._active_plan = list(range(self._n_actions))
                    self._phase = "reactive"
                self._current_action = 0
                self._steps_on_action = 0
            else:
                pair = self._seq_pairs[self._seq_idx]
                if self._seq_step == 0:
                    # Record pre-sequence encoding
                    self._seq_pre_enc = enc.copy()
                    self._prev_enc = enc.copy()
                    self._prev_dist = dist
                    self._seq_step = 1
                    return pair[0]  # first action of pair
                elif self._seq_step == 1:
                    self._seq_step = 2
                    self._prev_enc = enc.copy()
                    self._prev_dist = dist
                    return pair[1]  # second action of pair
                else:
                    # Measure change from pre-sequence
                    seq_change = float(np.sum(np.abs(enc - self._seq_pre_enc)))
                    if seq_change > CHANGE_THRESH:
                        self._responsive_seqs.append(pair)
                    self._seq_idx += 1
                    self._seq_step = 0
                    self._prev_enc = enc.copy()
                    self._prev_dist = dist
                    # Start next pair immediately
                    if self._seq_idx < len(self._seq_pairs):
                        next_pair = self._seq_pairs[self._seq_idx]
                        self._seq_pre_enc = enc.copy()
                        self._seq_step = 1
                        return next_pair[0]
                    return self._rng.randint(self._n_actions)

        # === Phase 3: Reactive switching ===
        if self._phase == "reactive":
            n_active = len(self._active_plan)
            if n_active == 0:
                self._active_plan = list(range(self._n_actions))
                n_active = self._n_actions

            progress = (self._prev_dist - dist) > 1e-4 if self._prev_dist is not None else False
            no_change = abs(self._prev_dist - dist) < 1e-6 if self._prev_dist is not None else True

            self._steps_on_action += 1

            if progress:
                self._consecutive_progress += 1
                self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
                self._actions_tried_this_round = 0
            else:
                self._consecutive_progress = 0
                if self._steps_on_action >= self._patience or no_change:
                    self._actions_tried_this_round += 1
                    self._steps_on_action = 0
                    self._patience = 3

                    if self._actions_tried_this_round >= n_active:
                        self._current_action = self._rng.randint(n_active)
                        self._actions_tried_this_round = 0
                    else:
                        self._current_action = (self._current_action + 1) % n_active

            action = self._active_plan[self._current_action % n_active]
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # Fallback
        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return int(self._rng.randint(0, self._n_actions))

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Keep phase and active_plan — same game type across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "phase1_steps": PHASE1_STEPS,
    "phase2_steps": PHASE2_STEPS,
    "max_patience": MAX_PATIENCE,
    "change_thresh": CHANGE_THRESH,
    "family": "action sequence probing",
    "tag": "defense v42 (ℓ₁ 2-action sequence probing for games where singles are no-ops)",
}

SUBSTRATE_CLASS = ActionSequenceSubstrate
