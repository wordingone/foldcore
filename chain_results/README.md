# PRISM — Progressive Recursive Intelligence Sequence Metric

One system. One config. No reward. Sequential diverse tasks.

The parts of the chain are one problem seen from different angles — classification, navigation, sequential puzzles, cross-domain transfer. A substrate that solves one solves all.

## Modes

**PRISM-light** (current, through March 25):
```
Split-CIFAR-100 → LS20 → FT09 → VC33 → Split-CIFAR-100
  (classify)    (navigate) (puzzle) (puzzle) (classify again)
```

**PRISM** (post March 25, under design):
```
Split-CIFAR-100 → [150+ ARC-AGI-3 games] → Terminal-Bench 2.0 → BrowseComp → HLE → Split-CIFAR-100
  (classify)       (navigate/puzzle)         (CLI interaction)   (web browse)  (Q&A)  (classify again)
```

## Order Modes

- **Fixed:** phases run in declared order. Tests specific transfer directions. Use for controlled comparison.
- **Randomized:** phases shuffled per seed (deterministic from seed RNG). Tests order-independence. A substrate that passes in ANY order is genuinely adaptive. If performance varies by order, the substrate is fragile.

## Rules

- **Budget:** n_steps per phase (default 25K)
- **Seeds:** minimum 10 per run
- **Constraint:** R1 (no reward/labels passed to substrate)
- **Persistence:** ONE substrate instance per seed, state carries across all phases
- **Kill criterion:** any mechanism that improves one phase at the cost of another is per-game tuning and must be killed

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

**No substrate has completed the chain.** Exit condition: all phases passed.

## Future Phases (under evaluation)

| Benchmark | Type | Integration | Status |
|-----------|------|-------------|--------|
| Terminal-Bench 2.0 | CLI interaction (89 Docker tasks) | Render terminal as screenshot → action = keystroke | Researched, not integrated |
| BrowseComp | Web browsing (1,266 questions) | Render page as screenshot → action = click/scroll | Researched, not integrated |
| Humanity's Last Exam | Expert Q&A | Question image → action = answer index | Compatible (static, like CIFAR) |

Adding these would test: can the substrate handle text-based environments (terminal, web) alongside visual ones (games, images)? The one-door hypothesis predicts yes — the mechanism that adapts to unknown visual games should adapt to unknown text environments. If it can't, that's data about what's missing.
