"""Synthetic keyboard-sequence game for childhood pretraining (Step 1300+).

ARC-compatible interface. 64×64 frame, 7-action keyboard. One 2-key sequence
is the target. Press target sequence → level advances, new frame + new target.
Wrong or partial press → small visual change, sequence resets.

Action space: 7 (same as LS20/TR87 KB games).
Observation: (64, 64, 3) float32 in [0, 1]. _enc_frame uses first channel.
Level: increments on each correct sequence. Infinite levels. Never done=True.

Progress hint in frame: top 4 rows show progress bar (brightens per correct key).
The substrate doesn't see the target sequence — must discover it by exploration.

Spec: Leo mail 3658, 2026-03-28.
"""
import numpy as np

N_ACTIONS = 7
FRAME = 64
SEQ_LEN = 2


class KBSequenceGame:

    @property
    def n_actions(self):
        return N_ACTIONS

    def reset(self, seed=None):
        self._rng = np.random.RandomState(seed if seed is not None else 0)
        self._level = 0
        self._pressed = []
        self._new_frame()
        return self._obs()

    def _new_frame(self):
        self._target = list(self._rng.randint(0, N_ACTIONS, SEQ_LEN))
        # Base pattern: random colored blocks — state, not target
        self._base = (self._rng.rand(FRAME, FRAME, 3) * 0.6).astype(np.float32)
        self._frame = self._base.copy()
        self._pressed = []

    def _obs(self):
        return self._frame.copy()

    def _draw_progress(self):
        n = len(self._pressed)
        px = int(FRAME * n / SEQ_LEN)
        self._frame[:4, :, :] = 0.1
        if px > 0:
            self._frame[:4, :px, :] = 0.9

    def step(self, action):
        action = int(action) % N_ACTIONS
        self._pressed.append(action)

        if self._pressed == self._target:
            self._level += 1
            self._new_frame()
        elif len(self._pressed) >= SEQ_LEN:
            # Wrong sequence: reset progress, small noise flash
            self._pressed = []
            self._frame = self._base.copy()
            y = self._rng.randint(8, FRAME - 8)
            self._frame[y:y+4, :] = self._rng.rand(4, FRAME, 3).astype(np.float32) * 0.4
        else:
            # Partial — update progress bar
            self._draw_progress()

        return self._obs(), 0.0, False, {'level': self._level}


def make(name='kb_sequence'):
    return KBSequenceGame()
