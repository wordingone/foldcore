"""BaseSubstrate adapter for ExprSubstrate — self-modifying expression trees.

Killed: Step 440-445. R3 partial (tree structure IS fully self-modifying but
mutation rate, pop_size, scoring formula are frozen). Navigation: 0/5 on LS20.
94% action collapse (ExprSubstrate with U20 variant). Kept as evidence.
"""
import copy
import numpy as np
import torch
from substrates.base import BaseSubstrate, Observation
from substrates.expr.expr import ExprSubstrate


class ExprSubstrateAdapter(BaseSubstrate):
    """Wraps ExprSubstrate into BaseSubstrate protocol."""

    def __init__(self, d=256, n_act=4, pop_size=4):
        self._d = d
        self._n_act = n_act
        self._pop_size = pop_size
        self._sub = ExprSubstrate(d, n_act, pop_size)

    def process(self, observation):
        if isinstance(observation, Observation):
            obs = observation.data
        else:
            obs = observation
        flat = obs.flatten().astype(np.float32)[:self._d]
        if len(flat) < self._d:
            flat = np.pad(flat, (0, self._d - len(flat)))
        x = torch.from_numpy(flat)
        return self._sub.step(x, self._n_act)

    def get_state(self):
        return {
            "pop": copy.deepcopy(self._sub.pop),
            "scores": list(self._sub.scores),
            "best": self._sub.best,
            "history": [(h[0].clone().cpu().numpy() if torch.is_tensor(h[0]) else h[0], h[1])
                        for h in self._sub.history],
            "steps": self._sub.steps,
        }

    def set_state(self, state):
        self._sub.pop = copy.deepcopy(state["pop"])
        self._sub.scores = list(state["scores"])
        self._sub.best = state["best"]
        self._sub.history = [(torch.from_numpy(h[0]) if isinstance(h[0], np.ndarray) else h[0], h[1])
                             for h in state["history"]]
        self._sub.steps = state["steps"]

    def frozen_elements(self):
        return [
            {"name": "pop_trees", "class": "M", "justification": "Expression trees mutated every window steps. Structure, thresholds, features all change."},
            {"name": "scores", "class": "M", "justification": "Population scores updated from action diversity × consistency"},
            {"name": "best_index", "class": "M", "justification": "Best tree index updated by argmax of scores"},
            {"name": "evaluate_recursive", "class": "I", "justification": "Recursive descent evaluation of tree. Removing = no action."},
            {"name": "mutate_operator", "class": "I", "justification": "Tree mutation is the self-modification mechanism. Removing = no learning."},
            {"name": "pop_size_4", "class": "U", "justification": "Population of 4 trees. Arbitrary. Could be 1, 8, or 100."},
            {"name": "max_depth_4", "class": "U", "justification": "Max tree depth 4. Arbitrary. Limits expressiveness."},
            {"name": "window_32", "class": "U", "justification": "Scoring window of 32 steps. Arbitrary evaluation period."},
            {"name": "mutation_rate_0.15", "class": "U", "justification": "15% mutation rate. Designer-chosen. Not adaptive."},
            {"name": "scoring_diversity_x_consistency", "class": "U", "justification": "diversity × consistency formula. Designer-chosen. Could use any fitness."},
            {"name": "replace_worst_with_mutated_best", "class": "U", "justification": "Evolutionary strategy. Could use crossover, tournament, etc."},
            {"name": "degenerate_filter", "class": "U", "justification": "Reject constant-action candidates. Designer-imposed constraint."},
        ]

    def reset(self, seed: int):
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        self._sub = ExprSubstrate(self._d, self._n_act, self._pop_size)

    @property
    def n_actions(self):
        return self._n_act

    def on_level_transition(self):
        pass
