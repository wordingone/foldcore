# Kill: Multi-horizon Prediction
Steps: 1004-1005 | Trigger: 2 consecutive kills, numerical instability

## What worked
- K-step prediction concept: different horizons DO produce different error profiles per action
- Step 1005 partial: 11.6/seed (3/10 nonzero) after normalization — signal exists but overwhelmed

## What failed
- **1004:** Gradient ascent on K-step predictor creates unbounded errors (long_spread=3557). Delta_long overflow from accumulated gradient updates.
- **1005:** Normalization (z-score per horizon) reduces overflow but long_spread still 6917. The fundamental issue: gradient ascent on prediction creates divergent dynamics. Normalization fights but doesn't fix the instability.

## What next family needs
- Long-horizon temporal credit WITHOUT gradient ascent on predictors
- State-conditioned retrieval (not global average) for temporal credit
- Numerically stable mechanism for multi-step lookahead

## Return condition
If a non-gradient mechanism for multi-step prediction is found (e.g., attention over trajectory buffer), the multi-horizon concept could be revisited with stable dynamics.
