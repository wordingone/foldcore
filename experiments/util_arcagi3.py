"""
arcagi3 — wrapper around arc_agi 0.9.4 exposing a gym-style interface.

arcagi3.make("LS20") -> env
env.reset(seed=None) -> frame   (frame[0] is 64x64 grid, values 0-15)
env.step(action_int) -> (frame, reward, done, info)  info={'level': int}
env.n_actions -> int   (total keyboard + click actions)

Action encoding (PRISM-compatible):
  action 0-6:    keyboard ACTION1-ACTION7 (7 actions, skip RESET)
  action 7+:     click at pixel (x, y) via ACTION6 with data={"x": x, "y": y}
                 click_idx = action - 7, x = click_idx % 64, y = click_idx // 64
  n_actions:     7 (keyboard-only games) or 7 + 4096 = 4103 (click games)

Click support detected from env.action_space: if ACTION6 is listed, the game
accepts click with coordinates.

BUG FIX (2026-03-26): ACTION6 is overloaded — it's both keyboard action index 5
AND the click action. Click games crash with KeyError: 'x' when ACTION6 is sent
without data. Fix: for click games, keyboard action 5 sends ACTION6 with
data={"x": 0, "y": 0}. Additionally, game errors (obs=None) no longer cause
game resets — the last valid frame is returned instead.
"""
import arc_agi
from arcengine import GameAction, GameState

N_KB_ACTIONS = 7   # ACTION1-ACTION7, always available
N_CLICK_TARGETS = 64 * 64  # 4096 pixel-level click targets

_GA_KB = list(GameAction)[1:]  # [ACTION1..ACTION7], skip RESET


class _Env:
    def __init__(self, game_name):
        self._arc = arc_agi.Arcade()
        games = self._arc.get_environments()
        key = game_name.lower().split('-')[0]
        info = next(g for g in games if key in g.game_id.lower())
        self._env = self._arc.make(info.game_id)

        # Detect click support from the game's declared action_space.
        # action_space is a list of GameAction values the game primarily uses.
        # If ACTION6 is present, game supports click with x,y coordinates.
        self._supports_click = any(
            ga == GameAction.ACTION6 for ga in self._env.action_space
        )
        self.n_actions = N_KB_ACTIONS + (N_CLICK_TARGETS if self._supports_click else 0)

        self._last_obs = None
        self._last_frame = None  # cache last valid frame for error recovery
        self._game_id = info.game_id
        self._levels_offset = 0
        self._error_count = 0  # track consecutive errors

    def reset(self, seed=None):
        # arc_agi doesn't support seeding — seed param is ignored.
        self._last_obs = self._env.reset()
        if self._last_obs is not None and self._last_obs.levels_completed > 0:
            # arc_agi bug: game stuck at wrong level after GAME_OVER. Recreate.
            self._env = self._arc.make(self._game_id)
            self._last_obs = self._env.reset()
        self._levels_offset = self._last_obs.levels_completed if self._last_obs is not None else 0
        self._error_count = 0
        frame = self._frame()
        self._last_frame = frame
        return frame

    def step(self, action_int):
        if action_int < N_KB_ACTIONS:
            # Keyboard action: 0-6 → ACTION1-ACTION7
            ga = _GA_KB[action_int % len(_GA_KB)]
            if ga == GameAction.ACTION6 and self._supports_click:
                # ACTION6 is the click action for this game — sending without
                # data causes KeyError: 'x'. Provide default click at (0,0).
                obs = self._env.step(GameAction.ACTION6, data={"x": 0, "y": 0})
            else:
                obs = self._env.step(ga)
        else:
            # Click action: 7+ → ACTION6 with pixel coordinates
            click_idx = action_int - N_KB_ACTIONS
            x = click_idx % 64
            y = click_idx // 64
            obs = self._env.step(GameAction.ACTION6, data={"x": x, "y": y})

        if obs is None:
            # Game error (e.g. internal KeyError). Don't reset — return last
            # valid frame so the substrate can continue playing.
            self._error_count += 1
            if self._error_count > 100:
                # Too many consecutive errors — game is truly broken, signal done
                return self._last_frame, 0.0, True, {'level': 0}
            level = 0
            if self._last_obs is not None:
                level = self._last_obs.levels_completed - self._levels_offset
            return self._last_frame, 0.0, False, {'level': level}

        self._last_obs = obs
        self._error_count = 0  # reset on success
        done = obs.state in (GameState.GAME_OVER, GameState.WIN)
        info = {'level': obs.levels_completed - self._levels_offset}
        frame = obs.frame if obs.frame else None
        if frame is not None:
            self._last_frame = frame
        return frame if frame is not None else self._last_frame, 0.0, done, info

    def _frame(self):
        if self._last_obs is None or not self._last_obs.frame:
            return None
        return self._last_obs.frame


def make(game_name):
    return _Env(game_name)
