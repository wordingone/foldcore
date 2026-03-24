# The Search

Can a system improve itself by criteria it generates?

1009+ experiments across 16 architecture families testing substrates for recursive self-improvement on published benchmarks (Split-CIFAR-100, Atari 100K) and interactive games (ARC-AGI-3). R3 encoding self-modification achieved (Step 895). Two bans — codebook (Step 416) and graph (Step 777, permanent) — remove both known-working mechanisms. PRISM infrastructure overhaul (Step 1006): randomized game order, chain-based kills, all three games from step 1. Chain SOTA: 994 adaptive eta (LS20=286.2@25K, 9/10 seeds). FT09/VC33=0 under all mechanisms — formally characterized as the temporal credit wall (Theorem 4, Proposition 31). Extraction sprint (Steps 1008+): testing CC zone discovery and attention mechanisms to break the FT09/VC33 wall.

## Results (honest)

| Benchmark | Substrate | Result | Compare |
|-----------|-----------|--------|---------|
| Split-CIFAR-100 (20 tasks, no labels) | 674 | 20.2% avg accuracy | Chance=20%. DER++=29.6% (with labels) |
| Split-CIFAR-100 BWT | 674 | +5.6% | Zero forgetting by construction |
| LS20 navigation (20 seeds, 10K steps) | 674+running-mean | 20/20 L1 | Argmin over visit counts |
| FT09 navigation | 674+running-mean | 20/20 L1 | 7-action space |
| Cross-domain transfer (LS20→CIFAR) | 674 | 0% improvement | Zero transfer in both directions |
| R3 (self-modification) | 674 | FAIL (8 U elements) | 0/777 substrates pass R3 |
| R3 counterfactual | 674 | FAIL (cold > warm, p<0.0001) | Pretraining hurts new environments |
| Atari 100K (no reward) | 674 | 6/26 above random | RoadRunner 11x, most games at/below random |

**Post-ban results (Steps 778-920):**

| Benchmark | Substrate | Result | Compare |
|-----------|-----------|--------|---------|
| LS20 nav (10 seeds, 25K) | 895h cold (clamped alpha + 800b) | 268.0/seed, 0/10 zeros | +32% over L2-norm baseline (203.9) |
| LS20 nav (10 seeds, 25K) | 868d raw L2 baseline | 203.9/seed, 1/10 zeros | True post-ban baseline |
| FT09 nav | ALL post-ban | 0/seed | Sequential ordering unsolved (Prop 23) |
| VC33 nav | 895h cold (chain) | 0/seed | First post-ban VC33 result |
| Full chain (914) | 895h cold | CIFAR=chance, LS20=237.6, FT09=0, VC33=0 | Chain: 1/4 |
| R3 encoding (Step 895) | Prediction-error attention | alpha=[60,51,52] on FT09, UNIVERSAL | First post-ban ℓ_π |
| D(s) pred transfer | Forward model W (delta rule) | 5/7 PASS | First positive R3_cf |

**Key findings:**
- **R3 encoding self-modification achieved** (Prop 22). Alpha discovers game-informative dims from prediction error alone. Universal on FT09 (dims [60,51,52] = puzzle tiles, all seeds).
- **Navigation: +32% with clamped alpha.** Change-tracking (800b) + prediction-error attention = best post-ban mechanism. 0/10 zero-seeds.
- **FT09/VC33 unsolved post-ban.** Sequential games need action ordering, not change detection (Prop 23). The graph ban makes sequential puzzles exponentially hard.
- **Warm alpha transfer FAILED** (n_eff=10). Alpha is per-episode adaptation, not cross-seed transfer. Cold > warm.
- **800b "10× random" retracted.** True mean = 203.9/seed (L2 norm, n_eff=10). Prior claim was seed artifact.
- Compression progress dead across 5 variants. Novelty-based action selection dead on LS20. Only change-tracking navigates.

## Structure

```
paper/           Research paper (PDF + source)
constraints/     Constraint map (CONSTRAINTS.md), constitution (R1-R6), experiment log
experiments/     1009+ experiment scripts (chronological, Steps 1-1009+)
substrates/      Active substrate code:
  base.py          BaseSubstrate interface (5 methods)
  step0674.py      Reference substrate (transition-triggered dual-hash)
  plain_lsh.py     PlainLSH baseline (no refinement)
  chain.py         Chain benchmark runner (pluggable — any gym env)
  judge.py         ConstitutionalJudge (R1-R6 audit + R3 counterfactual)
  calibration_agents.py   R3 calibration baselines
  archive/         Killed families (12 architectures, preserved)
viz/             Search space visualization
```

## Quick Start

```bash
# Setup
bash setup.sh
source .venv/Scripts/activate   # Windows
# or: source .venv/bin/activate  # Linux/Mac

# Run Split-CIFAR-100 benchmark (no API key needed)
PYTHONPATH=. python -c "
from substrates.chain import SplitCIFAR100Wrapper, ChainRunner
from substrates.step0674 import Substrate674
wrapper = SplitCIFAR100Wrapper(500)
result = wrapper.run_seed(Substrate674(), seed=0)
print(f'CIFAR acc={result[\"avg_accuracy\"]:.1%}, BWT={result[\"backward_transfer\"]:+.1%}')
"

# Run R1-R6 audit
PYTHONPATH=. python -c "
from substrates.judge import ConstitutionalJudge
from substrates.step0674 import Substrate674
results = ConstitutionalJudge().audit(Substrate674)
for rule in ['R1','R2','R3','R4','R5','R6']:
    r = results.get(rule, {})
    print(f'{rule}: {\"PASS\" if r.get(\"pass\") else \"FAIL\"} — {r.get(\"detail\",\"\")}')
"

# Run reference substrate on ARC game (requires ARC-AGI-3 API key)
PYTHONPATH=. python substrates/step0674.py
```

## The Framework

Six rules for recursive self-improvement (R1-R6):

| Rule | Requirement | What it means |
|------|------------|---------------|
| R1 | No external objectives | No reward, no labels, no loss function |
| R2 | Adaptation from computation | Parameters ⊆ state. No optimizer. |
| R3 | Minimal frozen frame | Every design choice either self-modified or irreducible |
| R4 | Modifications tested against prior state | Changes must not degrade |
| R5 | One fixed ground truth | Only environmental signals (death, level transitions) |
| R6 | No deletable parts | Every component behaviorally load-bearing |

**R3 is the wall.** 0/777 substrates pass. The constraint map characterizes why. Two bans (codebook + graph) remove both tested mechanisms that satisfy other rules, forcing the search toward genuinely self-modifying architectures.

## Citation

```bibtex
@article{han2026search,
  title={Characterizing the Feasible Region for Self-Modifying Substrates in Interactive Environments},
  author={Han, Hyun Jun},
  year={2026}
}
```

## License

CC-BY-NC 4.0
