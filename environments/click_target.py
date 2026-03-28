"""Synthetic click game for childhood pretraining (Step 1300+).

ARC-compatible interface. 64×64 frame, N colored rectangles, one is the target
(marked with bright border). Click target → level advances, new frame.
Click elsewhere → small visual change at click position.

Action space: 4096 (64×64 click grid). action % 64 = x, action // 64 = y.
Observation: (64, 64, 3) float32 in [0, 1]. _enc_frame uses first channel.
Level: increments on each target hit. Infinite levels. Never returns done=True.

Spec: Leo mail 3658, 2026-03-28.
"""
import numpy as np

N_ACTIONS = 4096   # 64×64 click grid
FRAME = 64
N_REGIONS = 5      # colored rectangles per frame


class ClickTargetGame:

    @property
    def n_actions(self):
        return N_ACTIONS

    def reset(self, seed=None):
        self._rng = np.random.RandomState(seed if seed is not None else 0)
        self._level = 0
        self._new_frame()
        return self._obs()

    def _new_frame(self):
        self._frame = np.zeros((FRAME, FRAME, 3), dtype=np.float32)
        self._regions = []
        for _ in range(N_REGIONS):
            x1 = int(self._rng.randint(2, FRAME // 2))
            y1 = int(self._rng.randint(2, FRAME // 2))
            w  = int(self._rng.randint(8, FRAME // 2))
            h  = int(self._rng.randint(8, FRAME // 2))
            x2 = min(x1 + w, FRAME - 1)
            y2 = min(y1 + h, FRAME - 1)
            color = self._rng.rand(3).astype(np.float32) * 0.8
            self._frame[y1:y2, x1:x2] = color
            self._regions.append((x1, y1, x2, y2))
        # Target: last region, white border
        tx1, ty1, tx2, ty2 = self._regions[-1]
        self._target = (tx1, ty1, tx2, ty2)
        self._frame[ty1:ty2, tx1]    = 1.0
        self._frame[ty1:ty2, tx2-1]  = 1.0
        self._frame[ty1,    tx1:tx2] = 1.0
        self._frame[ty2-1,  tx1:tx2] = 1.0

    def _obs(self):
        return self._frame.copy()

    def step(self, action):
        cx = int(action) % FRAME
        cy = int(action) // FRAME % FRAME
        tx1, ty1, tx2, ty2 = self._target
        if tx1 <= cx < tx2 and ty1 <= cy < ty2:
            self._level += 1
            self._new_frame()
        else:
            r = slice(max(0, cy - 1), min(FRAME, cy + 2))
            c = slice(max(0, cx - 1), min(FRAME, cx + 2))
            self._frame[r, c] *= 0.85
        return self._obs(), 0.0, False, {'level': self._level}


def make(name='click_target'):
    return ClickTargetGame()
