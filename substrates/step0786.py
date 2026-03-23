"""
step0786.py — PopulationSubstrate786: Price equation / evolutionary selection.

R3 hypothesis: population-level selection produces R3 dynamics without individual
self-modification. N=10 independent substrates with random W params. Each runs
1K steps. Every 1K steps: copy params from substrate with most unique obs hashes.
Mutation: Gaussian perturbation (sigma=0.01).

R1 check: selection uses unique obs count (environmental signal), not external
objective. The unique-obs metric is derived from the substrate's own dynamics.
D(s) = {W_population (all N), selected_generation}. L(s) = ∅ (no per-action data).

Note: R3_cf protocol still applies — pretrain N steps, cold vs warm on test seeds.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

D = 64
ETA = 0.01
N_POP = 10
SELECTION_EVERY = 1_000
MUTATION_SIGMA = 0.01


def _hash_vec(v: np.ndarray) -> int:
    return int(np.packbits((v > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)


class _IndividualForwardModel:
    """One population member: W + running_mean."""
    def __init__(self, n_actions: int, rng):
        d_in = D + n_actions
        self.W = rng.randn(D, d_in).astype(np.float32) * 0.01
        self.running_mean = np.zeros(D, np.float32)
        self._n_obs = 0
        self.unique_hashes: set = set()

    def encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)[:D]
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        x_c = x - self.running_mean
        self.unique_hashes.add(_hash_vec(x_c))
        return x_c

    def hebbian_update(self, prev_enc: np.ndarray, curr_enc: np.ndarray, action: int, n_actions: int):
        a_oh = np.zeros(n_actions, np.float32)
        a_oh[action] = 1.0
        inp = np.concatenate([prev_enc, a_oh])
        self.W += ETA * np.outer(curr_enc, inp)

    def copy_from(self, other: '_IndividualForwardModel', sigma: float, rng):
        self.W = other.W.copy() + rng.randn(*other.W.shape).astype(np.float32) * sigma
        self.running_mean = other.running_mean.copy()
        self._n_obs = other._n_obs
        self.unique_hashes = set()  # reset fitness counter


class PopulationSubstrate786(BaseSubstrate):
    """N=10 forward models. Every 1K steps: select best (most unique obs), mutate rest.

    Active individual = index 0 (champion). Others run in parallel tracking.
    D(s) = {all W_i, running_means}. L(s) = ∅.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        self._step = 0
        rng = np.random.RandomState(seed)
        self.population = [_IndividualForwardModel(n_actions, np.random.RandomState(seed * 1000 + i))
                           for i in range(N_POP)]
        self._prev_encs = [None] * N_POP
        self._prev_action = None
        self._rng = np.random.RandomState(seed + 99)

    def process(self, observation) -> int:
        import numpy as np
        observation = np.asarray(observation, dtype=np.float32)

        # All individuals encode current obs
        encs = [ind.encode(observation) for ind in self.population]

        # Hebbian updates for all
        for i, ind in enumerate(self.population):
            if self._prev_encs[i] is not None and self._prev_action is not None:
                ind.hebbian_update(self._prev_encs[i], encs[i], self._prev_action, self._n_actions)

        # Action from champion (individual 0): prediction-contrast
        x = encs[0]
        best_a, best_score = 0, -1.0
        for a in range(self._n_actions):
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[a] = 1.0
            inp = np.concatenate([x, a_oh])
            pred = self.population[0].W @ inp
            score = float(np.sum((pred - x) ** 2))
            if score > best_score:
                best_score = score
                best_a = a

        self._prev_encs = [e.copy() for e in encs]
        self._prev_action = best_a
        self._step += 1

        # Selection every SELECTION_EVERY steps
        if self._step % SELECTION_EVERY == 0:
            self._select_and_mutate()

        return best_a

    def _select_and_mutate(self):
        """Copy best individual (most unique hashes) to all others, with mutation."""
        scores = [len(ind.unique_hashes) for ind in self.population]
        best_idx = int(np.argmax(scores))
        best = self.population[best_idx]
        new_pop = [_IndividualForwardModel(self._n_actions, np.random.RandomState(0))
                   for _ in range(N_POP)]
        new_pop[0].W = best.W.copy()
        new_pop[0].running_mean = best.running_mean.copy()
        new_pop[0]._n_obs = best._n_obs
        for i in range(1, N_POP):
            new_pop[i].copy_from(best, MUTATION_SIGMA, self._rng)
        self.population = new_pop
        self._prev_encs = [None] * N_POP

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        return {
            "Ws": [ind.W.copy() for ind in self.population],
            "running_means": [ind.running_mean.copy() for ind in self.population],
            "_n_obs_list": [ind._n_obs for ind in self.population],
            "_step": self._step,
            "_prev_action": self._prev_action,
        }

    def set_state(self, state: dict) -> None:
        for i, ind in enumerate(self.population):
            ind.W = state["Ws"][i].copy()
            ind.running_mean = state["running_means"][i].copy()
            ind._n_obs = state["_n_obs_list"][i]
        self._step = state["_step"]
        self._prev_action = state["_prev_action"]
        self._prev_encs = [None] * N_POP

    def reset(self, seed: int) -> None:
        self._prev_encs = [None] * N_POP
        self._prev_action = None

    def on_level_transition(self) -> None:
        self._prev_encs = [None] * N_POP
        self._prev_action = None

    def frozen_elements(self) -> list:
        return [
            {"name": "W_population_hebbian", "class": "M",
             "justification": "All N W matrices updated Hebbianly each step. System-driven."},
            {"name": "running_means", "class": "M",
             "justification": "All N running means adapt to obs distribution. System-driven."},
            {"name": "selection_by_unique_obs", "class": "M",
             "justification": "Selection mechanism adapts population based on unique obs count. System-driven."},
            {"name": "n_pop_size", "class": "U",
             "justification": "N=10. Could be 5 or 20. System doesn't choose."},
            {"name": "selection_every", "class": "U",
             "justification": "Every 1K steps. System doesn't choose."},
            {"name": "mutation_sigma", "class": "U",
             "justification": "sigma=0.01. System doesn't choose."},
            {"name": "enc_truncate_64", "class": "U",
             "justification": "64-dim. System doesn't choose."},
            {"name": "eta_learning_rate", "class": "U",
             "justification": "eta=0.01. System doesn't choose."},
        ]
