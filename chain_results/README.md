# PRISM — Progressive Recursive Intelligence Sequence Metric

One system. One config. No reward. Sequential diverse tasks.

The parts of the chain are one problem seen from different angles — classification, navigation, sequential puzzles, cross-domain transfer. A substrate that solves one solves all.

## The Chain

```
Split-CIFAR-100 → LS20 → FT09 → VC33 → Split-CIFAR-100
  (classify)    (navigate) (puzzle) (puzzle) (classify again)
```

- **Budget:** n_steps per phase (default 25K)
- **Seeds:** minimum 10 per run
- **Constraint:** R1 (no reward/labels passed to substrate)
- **Persistence:** ONE substrate instance per seed, state carries across all phases

## How to Run

```python
from substrates.chain import ChainRunner, make_prism

chain = make_prism(n_steps=25000)
runner = ChainRunner(chain, n_seeds=10)
results = runner.run(MySubstrate, {"n_actions": 4})
```

## Results Format

Each run saves a JSON to `runs/`. Schema:

```json
{
  "substrate": "name",
  "timestamp": "ISO-8601",
  "chain": ["Split-CIFAR-100", "LS20", "FT09", "VC33", "Split-CIFAR-100"],
  "budget_per_phase": 25000,
  "n_seeds": 10,
  "results": { "LS20": { "l1_rate": 0.8, "mean_l1_per_seed": 268.0, ... }, ... },
  "chain_score": { "phases_passed": 1, "chain_complete": false }
}
```

## Current Best

| Substrate | CIFAR-1 | LS20 | FT09 | VC33 | CIFAR-2 | Chain |
|-----------|---------|------|------|------|---------|-------|
| 895h cold (clamped alpha + 800b) | 1% | 248.6/seed | 0 | 0 | 1% | 1/5 |
| 916 (recurrent h + alpha + 800b) | 1% | 212.6/seed (chain) | 0 | 0 | 1% | 1/5 |
| Random | 1% | 26.2/seed | 0 | 0 | 1% | 0/5 |
| ICM (Pathak 2017) | 1% | 44.8/seed | 0 | 0 | 1% | 0/5 |
| RND (Burda 2018) | 1% | 38.2/seed | 0 | 0 | 1% | 0/5 |

**No substrate has completed the chain.** Exit condition: 5/5 phases passed.
