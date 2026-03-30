# Substrate Interface Spec

## Problem

Experiment scripts are decoupled. A new architecture takes ~500 lines of custom harness code. It should take <30.

## Solution

Two components:

### 1. BaseSubstrate (abstract class)

```python
from abc import ABC, abstractmethod
import numpy as np

class BaseSubstrate(ABC):
    """Minimal interface. The substrate is (f, g, F)."""

    @abstractmethod
    def process(self, observation: np.ndarray) -> int:
        """f + g: Take raw observation (64x64x3 uint8), return action index.
        All internal state updates happen here."""

    @abstractmethod
    def get_state(self) -> dict:
        """Return complete internal state for auditing.
        Must include every mutable data structure."""

    @abstractmethod
    def frozen_elements(self) -> list:
        """Return [{name, class: M|I|U, justification}, ...].
        M = modified by system dynamics.
        I = irreducible (removing kills all capability).
        U = unjustified (could be different, system doesn't choose).
        R3 requires: zero U elements."""

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Reset to initial state with given seed."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Number of possible actions."""
```

A new substrate = just implement these 5 methods. Example:

```python
class MySubstrate(BaseSubstrate):
    def __init__(self):
        self.graph = {}
        self.edge_counts = {}

    def process(self, obs):
        x = self._encode(obs)       # frozen or adaptive?
        cell = self._hash(x)        # frozen or adaptive?
        action = self._select(cell)  # frozen or adaptive?
        self._update(cell, action)   # frozen or adaptive?
        return action

    def frozen_elements(self):
        return [
            {"name": "_encode", "class": "U", "justification": "avgpool16, fixed"},
            {"name": "_hash", "class": "U", "justification": "LSH, fixed planes"},
            {"name": "_select", "class": "I", "justification": "argmin, removing = random"},
            {"name": "_update", "class": "M", "justification": "edge counts modified by dynamics"},
        ]
    # ... ~20 more lines
```

### 2. ConstitutionalJudge

Automated R1-R6 evaluation. Takes a BaseSubstrate, runs checks:

```python
class ConstitutionalJudge:
    def audit(self, substrate_cls: type, prism: list) -> dict:
        """Full R1-R6 audit on the PRISM benchmark."""
        results = {}
        results['R1'] = self._check_r1(substrate_cls)
        results['R2'] = self._check_r2(substrate_cls)
        results['R3'] = self._check_r3(substrate_cls)
        results['R4'] = self._check_r4(substrate_cls, prism)
        results['R5'] = self._check_r5(substrate_cls)
        results['R6'] = self._check_r6(substrate_cls, prism)
        results['prism'] = self._run_prism(substrate_cls, prism)
        return results
```

#### R1 Check (no external objectives)
- Static: grep for loss/reward/target/label imports/usage
- Runtime: wrap substrate, intercept any external signal access
- Binary: PASS if no external objectives detected

#### R2 Check (adaptation from computation)
- Compare `get_state()` at t=0 vs t=N
- If state changed: identify what changed and what drove the change
- PASS if adaptation signal is computed by same dynamics as input processing

#### R3 Check (minimal frozen frame)
- Call `frozen_elements()`
- Count U elements
- PASS if U_count == 0
- Report: list every element with its classification

#### R4 Check (modifications tested against prior)
- Save state at t=N, continue to t=2N
- Compare performance on novel tasks before/after modification
- PASS if no degradation on untrained tasks

#### R5 Check (one fixed ground truth)
- Ground truth = environmental (death, level transitions)
- PASS if substrate uses only game-provided signals

#### R6 Check (no deletable parts)
- For each component: ablate it, rerun PRISM
- PASS if every component's removal causes measurable degradation

#### PRISM Benchmark
```python
prism = [
    ("CIFAR-100", cifar_env, 10_000),
    ("LS20", ls20_env, 10_000),
    ("FT09", ft09_env, 10_000),
    ("VC33", vc33_env, 10_000),
    ("CIFAR-100", cifar_env, 10_000),  # transfer test
]
```
- 10 seeds per game
- 5-minute cap per seed
- Report: L1 reached (Y/N), coverage, steps to L1

## File Structure

```
substrates/
  base.py          # BaseSubstrate ABC
  judge.py         # ConstitutionalJudge
  prism.py         # PRISM benchmark runner
  step0674.py      # 674 rewritten as BaseSubstrate (reference impl)
```

## Constraints

- 5-minute cap per seed (existing rule)
- Environment files in environment_files/ at repo root
- No codebook mechanisms (existing ban)
- Judge output = structured dict, not prose
