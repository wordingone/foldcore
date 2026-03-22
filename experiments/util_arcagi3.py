"""
arcagi3 — thin wrapper around arc_agi 0.9.4 exposing a gym-style interface.

arcagi3.make("LS20") -> env
env.reset(seed=None) -> frame   (frame[0] is 64x64 grid, values 0-15)
env.step(action_int) -> (frame, reward, done, info)  info={'level': int}
"""
import arc_agi
from arcengine import GameState

_GAME_IDS = {
    'LS20': 'ls20',
    'FT09': 'ft09',
    'VC33': 'vc33',
}


class _Env:
    def __init__(self, game_name):
        self._arc = arc_agi.Arcade()
        games = self._arc.get_environments()
        key = _GAME_IDS.get(game_name, game_name.lower())
        info = next(g for g in games if key in g.game_id.lower())
        self._env = self._arc.make(info.game_id)
        self._action_space = self._env.action_space
        self._last_obs = None

    def reset(self, seed=None):
        # arc_agi doesn't support seeding — seed param is ignored
        self._last_obs = self._env.reset()
        return self._frame()

    def step(self, action_int):
        obs = self._env.step(self._action_space[action_int])
        self._last_obs = obs
        if obs is None:
            return None, 0.0, True, {'level': 0}
        done = obs.state in (GameState.GAME_OVER, GameState.WIN)
        info = {'level': obs.levels_completed}
        frame = obs.frame if obs.frame else None
        return frame, 0.0, done, info

    def _frame(self):
        if self._last_obs is None or not self._last_obs.frame:
            return None
        return self._last_obs.frame


def make(game_name):
    return _Env(game_name)
