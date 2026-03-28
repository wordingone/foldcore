# The Search

Can a system improve itself by criteria it generates?

1252+ experiments across 16 architecture families testing substrates for recursive self-improvement on published benchmarks (Split-CIFAR-100, Atari 100K) and interactive games (ARC-AGI-3, 150+ games, scoring = action efficiency squared). R3 (self-modification) achieved by component composition (Step 1251, 100/100 across 10 games). Current bottleneck: I1 (encoding doesn't distinguish game states).

## Results (honest)

**Phase 1  - Reference substrate (Steps 1-416):**

| Benchmark | Substrate | Result | Compare |
|-----------|-----------|--------|---------|
| Split-CIFAR-100 (20 tasks, no labels) | 674 | 20.2% avg accuracy | Chance=20%. DER++=29.6% (with labels) |
| Split-CIFAR-100 BWT | 674 | +5.6% | Zero forgetting by construction |
| LS20 navigation (20 seeds, 10K steps) | 674+running-mean | 20/20 L1 | Argmin over visit counts |
| FT09 navigation | 674+running-mean | 20/20 L1 | 7-action space |
| Cross-domain transfer (LS20→CIFAR) | 674 | 0% improvement | Zero transfer in both directions |
| R3 (self-modification) | 674 | FAIL (8 U elements) | Solved by composition (Step 1251, 100/100) |
| Atari 100K (no reward) | 674 | 6/26 above random | RoadRunner 11x, most at/below random |

**Phase 2  - Post-ban exploration (Steps 417-1081):**

| Benchmark | Substrate | Result | Compare |
|-----------|-----------|--------|---------|
| LS20 nav (10 seeds, 25K) | 916 recurrent h | 290.7/seed SOTA, 0/10 zeros | 2-2.5x over ICM/RND/Count baselines |
| LS20 nav (10 seeds, 25K) | 868d raw L2 baseline | 203.9/seed | True post-ban baseline |
| FT09 nav | ALL post-ban substrates | 0/seed | Generic exploration fails (Step 1017) |
| VC33 nav | ALL post-ban substrates | 0/seed | Same gap as FT09 |
| R3 encoding | 895 prediction-error attention | alpha=[60,51,52] UNIVERSAL on FT09 | First post-ban R3 encoding (Prop 22) |
| Bans lifted (Step 1017) | Full graph + all mechanisms | FT09/VC33 still 0% | Bans are NOT the cause |

**Debate v3  - ARC-AGI-3 sprint (Steps 1082-1097, 15 experiments):**

| Metric | Defense (ℓ₁ reactive) | Prosecution (ℓ_π forward model) |
|--------|----------------------|-------------------------------|
| Architecture | Zero-param reactive switching | W_fwd action-conditioned prediction |
| Best single-draw L1 | 100% (10/10 seeds) | 100% (10/10 seeds) |
| Best single-draw ARC | 0.2973 | 0.0045 |
| Draw-robustness (new draw) | ARC = 0.0000 | ARC = 0.0000 |
| Modifications tested | 5 (all degraded) | 3 (all degraded) |

**Key findings:**
- **ℓ_π ≈ ℓ₁ at L1 (PB26 CONFIRMED).** Draw-robustness falsified both sides' ARC claims. All non-zero ARC scores were game-draw artifacts.
- **Simplicity is load-bearing (PB30).** Adding complexity to either architecture degrades performance (n=5+). No optimizer (R2) means extra parameters accumulate noise. Prop 32.
- **0% wall = ~2/3 random ARC games.** TWO failure modes: Mode 1 (near-inert, noise) and Mode 2 (responsive, oscillating). Neither architecture solves Mode 2.
- **FT09/VC33 unsolved  - bans are NOT the cause (Step 1017).** Full graph + all bans lifted = still 0%.
- **R3 encoding achieved (Prop 22).** Alpha discovers game-informative dims from prediction error alone.
- **L2+ = 0 across entire search.** No substrate has ever autonomously solved Level 2 of any game.

**Phase 3 - Component composition (Steps 1251+):**

| Step | What | Result |
|------|------|--------|
| 1251 | 7 cross-family components composed | R3 = 100/100 (10 games, both wirings). I1 = 0/100. I3 = 0.67 (same as argmin-alone). |
| 1252 | Allosteric substrate (shared W for encoding + action, LPL update) | I4 = 1.00 (temporal structure). I3 = 0.40 (argmax locks). L1 = 0/8 games. |
| 1253 | Allosteric + adaptive softmax | R3 = 4/5, I3 = 5/5 on VC33 (coexist). I1 = 0/15. L1 = 0 except VC33 control. |

Bottleneck: I1 (encoding doesn't distinguish game states). R3 works. Action selection reads from encoding (allosteric). The encoding lacks state-distinguishing capacity.

## Structure

```
paper/                    Research paper (PDF + source)
constraints/              Constraint map, constitution (R1-R6), composition loop state
  CONSTITUTION.md           R1-R6 rules
  MAP.md                    Constraint map (1252+ experiments)
  RESEARCH_STATE.md         Composition loop state (current composition, stages, bottleneck)
  COMPONENT_CATALOG.md      Parts bin (C1-C33+)
  FAMILY_KILLS.md           Killed family register
experiments/
  compositions/             Composition-era experiments (Step 1251+)
  components/               Validated component implementations
  archive/                  Pre-composition experiments (Steps 1-1250, 1300+ scripts)
  results/                  Raw experiment data (never deleted)
  solvers/                  Analytical game solvers (gitignored)
substrates/               Active substrate code
viz/                      Search space visualization
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
    print(f'{rule}: {\"PASS\" if r.get(\"pass\") else \"FAIL\"}  - {r.get(\"detail\",\"\")}')
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

R3 passes when 7 cross-family components compose (Step 1251, 100/100). Current bottleneck: I1 (state-distinguishing encoding).

## Citation

```bibtex
@article{han2026search,
  title={Self-Modification by Composition: R3 Solved, the Bridge Remains},
  author={Han, Hyun Jun},
  year={2026}
}
```

## License

CC-BY-NC 4.0

