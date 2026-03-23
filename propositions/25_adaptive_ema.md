# Proposition 25: Adaptive Memory Timescale (Adaptive EMA)
Status: OPEN
Steps: 868b, 937 (running)

## Statement
Replace fixed EMA decay lambda with adaptive rate lambda_t = f(sigma_t^2), where sigma_t^2 is running variance of recent observation changes. High variance (fast-changing like FT09 resets) -> low lambda (short memory). Low variance (slow-changing like LS20 navigation) -> high lambda (long memory). Self-adjusts without per-game tuning.

## Evidence
Motivated by bacterial chemotaxis literature: optimal memory timescale matches environment's fluctuation timescale (Nature Communications, 2020). 800b EMA decay (lambda=0.9, ~10 step memory) matches LS20 slow spatial dynamics but too slow for FT09 1-step position residence.

Step 937 (adaptive EMA) was running at last known state. Results pending.

## Implications
If confirmed, provides a game-adaptive mechanism that adjusts "tumble frequency" to match gradient structure. Would dissolve the LS20-specific nature of 800b without per-game tuning. Still bounded by Prop 23b on sequential games (position-blindness unchanged).

## Supersedes / Superseded by
N/A. Addresses game-specificity of 800b.
