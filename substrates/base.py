"""
BaseSubstrate — abstract interface for all substrates.

Every new substrate: inherit this, implement 5 methods.
That's all a researcher needs to do. The judge handles the rest.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseSubstrate(ABC):
    """Minimal substrate interface: (f, g, F).

    f = encode (observation -> internal representation)
    g = select (internal representation -> action)
    F = update (state evolution rule)

    All three happen inside process(). The split is analytic, not structural.
    """

    @abstractmethod
    def process(self, observation: np.ndarray) -> int:
        """Take raw observation, return action index.
        All internal state updates happen here (F applied).
        observation: numpy array, any shape (game-dependent).
        """

    @abstractmethod
    def get_state(self) -> dict:
        """Return complete internal state for auditing.
        Must include every mutable data structure.
        Used by R2 (adaptation check) and R6 (ablation).
        Keys should be stable names, values snapshots.
        """

    @abstractmethod
    def frozen_elements(self) -> list:
        """Return list of dicts describing every design element.

        Each dict: {name: str, class: 'M'|'I'|'U', justification: str}
          M = Modified by system dynamics (R3 requires this or I)
          I = Irreducible: removing destroys all capability
          U = Unjustified: could be different, system doesn't choose

        R3 PASSES if and only if U_count == 0.
        Be honest. U elements are the research finding.
        """

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Reset to fresh state with given seed.
        Called: at episode start, after level transitions.
        Must set all mutable state to initial conditions.
        """

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Number of possible actions for current game."""

    def ablate(self, component_name: str) -> None:
        """Nullify a named component for R6 ablation testing.
        Optional. If not implemented, R6 reports UNTESTABLE.
        Should set the component to a no-op or zero state.
        """
        raise NotImplementedError(f"ablate('{component_name}') not implemented")

    def on_level_transition(self) -> None:
        """Called when the game signals a level transition.
        Optional hook. Default: call reset(seed=0) — override if needed.
        The substrate may choose to preserve some state across levels.
        """
        pass
