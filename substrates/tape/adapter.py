"""BaseSubstrate adapter for TapeMachine — integer tape, not vectors.

Killed: Step 430-435. R3 partial (structure is fully self-modifying but
the hash function and write formula are frozen). Navigation: 0/5 on LS20.
Discrimination: 35% on synthetic (below threshold). Kept as evidence.
"""
import copy
import numpy as np
import torch
from substrates.base import BaseSubstrate, Observation
from substrates.tape.tape import TapeMachine


class TapeMachineAdapter(BaseSubstrate):
    """Wraps TapeMachine into BaseSubstrate protocol."""

    def __init__(self, K=256, addr_bits=8, n_act=4):
        self._K = K
        self._addr_bits = addr_bits
        self._n_act = n_act
        self._sub = TapeMachine(K, addr_bits)

    def process(self, observation):
        if isinstance(observation, Observation):
            obs = observation.data
        else:
            obs = observation
        x = torch.from_numpy(obs.flatten().astype(np.float32))
        return self._sub.step(x, self._n_act)

    def get_state(self):
        return {
            "tape": copy.deepcopy(self._sub.tape),
            "K": self._sub.K,
            "mask": self._sub.mask,
        }

    def set_state(self, state):
        self._sub.tape = copy.deepcopy(state["tape"])
        self._sub.K = state["K"]
        self._sub.mask = state["mask"]

    def frozen_elements(self):
        return [
            {"name": "tape_contents", "class": "M", "justification": "All cells modified by write on every step"},
            {"name": "hash_function", "class": "U", "justification": "top-3 indices hash. Designer-chosen discretization. Could be any hash."},
            {"name": "read_write", "class": "I", "justification": "Tape read/write is the storage operation. Removing = no memory."},
            {"name": "chain_pointer", "class": "I", "justification": "key+symbol+1 chain. Removing = no self-reference."},
            {"name": "K_alphabet", "class": "U", "justification": "K=256 alphabet size. Arbitrary. Could be 64 or 1024."},
            {"name": "addr_bits", "class": "U", "justification": "8-bit address space (256 cells). Arbitrary."},
            {"name": "initial_tape", "class": "U", "justification": "i*7+13 initialization. Designer-chosen. Could be zeros or random."},
            {"name": "write_formula", "class": "U", "justification": "symbol + (key&0xFF) + 1. Designer-chosen update rule."},
        ]

    def reset(self, seed: int):
        self._sub = TapeMachine(self._K, self._addr_bits)

    @property
    def n_actions(self):
        return self._n_act

    def on_level_transition(self):
        pass
