# R2-Performance Tradeoff (Audit Finding 6)

*2026-03-18. Documenting the empirical relationship between frozen frame size and performance.*

## The Pareto Frontier

| Substrate | U count | P-MNIST AA (supervised) | Navigates LS20? | R2 status |
|---|---|---|---|---|
| TemporalPrediction (reduced) | 1 | not tested | No (U22: convergence) | Near-pass |
| ReadIsWrite (tau=0.01) | 2 | 91.84% | Not tested | R2 by construction |
| process_novelty + softmax | 5-7 | **94.48%** | **Yes (26K)** | R2 partial |
| ExprSubstrate | 8 | 46% (chance) | No | R2 by construction |
| SelfRef | 10 | 94% disc (not CL protocol) | No | Fails R2 |
| TapeMachine | 10 | 35% disc | No | Fails R2 |

## The Tradeoff

**Step 425 vs ReadIsWrite:** process_novelty + softmax voting (5-7U) beats ReadIsWrite (2U) by 2.64pp on classification. The system with MORE frozen elements performs BETTER.

**Step 426 sharpens this:** The same softmax voting that improves classification by +3.3pp KILLS navigation entirely (dom collapses to 41-45%). Performance on one benchmark trades against the other.

**Step 432 further sharpens:** The entire 94.48% depends on external labels (84.68pp gap without). The classification capability IS a frozen frame — external labels are a frozen element not counted in the R3 audit.

## Interpretation

The frozen frame and capability are positively correlated in this architecture family. Removing frozen elements (moving toward R3 compliance) costs performance. This is the R2-performance tradeoff:

- R2 by construction (ReadIsWrite): fewer U, lower accuracy
- R2 partial (process_novelty): more U, higher accuracy, navigates
- R2 by construction + softmax (Step 425): mixed — R2 update + non-R2 scoring = best classification but kills navigation

## What This Means for the Search

The feasible region (all R1-R6 simultaneously) may require accepting lower performance than the current best. A substrate with 0U that classifies at 80% and navigates at 30K would be more constitutionally significant than one with 7U at 94.48%.

The external audit's Finding 11 (two papers) addresses this: the constitutional paper cares about R1-R6 compliance, not accuracy. The empirical paper cares about accuracy. They serve different claims.
