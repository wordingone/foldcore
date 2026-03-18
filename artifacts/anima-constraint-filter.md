# ANIMA Through the Constraint Map

*2026-03-18. Re-evaluating ANIMA (Sessions 18-23, killed "Stage 2 vacuous") through U1-U26.*

## Why Re-examine

ANIMA was killed under the old stage-climbing framework (superseded by R1-R6). Constraints U20-U26 didn't exist. The separation theorem (reactive vs state-conditioned) predicts ANIMA-style architectures should be exponentially more state-efficient. Worth checking if the constraint map rules it in or out.

## Constraint Filter

| # | Constraint | Codebook | ANIMA | Notes |
|---|---|---|---|---|
| U1 | Read=write one operation | PASS | UNCLEAR | W,I,T have separate update rules. Not one operation. |
| U2 | One data structure | PASS (codebook) | FAIL | W⊕I⊕T = three data structures |
| U3 | Zero forgetting | PASS (append) | UNCLEAR | I accumulates (good). W is "reversible" (forgetting?) |
| U4 | Minimal | PASS (22 lines) | FAIL | W+I+T + coupling = much larger |
| U5 | Sparse selection | PASS (top-K) | UNCLEAR | Depends on routing mechanism |
| U8 | Hard selection | PASS (WTA) | LIKELY FAIL | Original used soft coupling. U8 predicts this is why Stage 2 was vacuous. |
| U13 | Additions hurt | CONFIRMED | UNTESTED | ANIMA is architecturally "additive" vs codebook |
| U15 | Robust to perturbation | FAIL (brittle) | UNKNOWN | |
| U20 | Local continuity | PASS (cosine) | PASS | Neural projections continuous |
| U22 | Convergence kills | PASS (growth) | PARTIAL | I accumulates. W might converge. |
| U24 | argmin≠argmax | CONFIRMED | MAY DISSOLVE | State-conditioned selection is neither argmin nor argmax |
| U25 | Score/bias coupling | CONFIRMED | N/A | No top-K scoring |
| U26 | Self-label compounding | CONFIRMED | N/A | No label-dependent voting |

## Verdict

ANIMA avoids the codebook-SPECIFIC constraints (U25, U26) but fails UNIVERSAL ones:
- **U1**: Separate operations for W, I, T updates
- **U2**: Three data structures, not one
- **U4**: Much larger than minimal
- **U8**: Soft coupling (the original kill reason)

These are the constraints ANIMA was designed before. They're universal, not codebook-specific.

## The Shape the Constraints Carve

If ANIMA is too heavy (U2, U4) and process_novelty is too frozen (R3), the answer is between them:

**A codebook with a state-conditioned metric.**

- One data structure (U2): codebook V, satisfies U1 (read=write=attract)
- Minimal (U4): similar to process_novelty's 22 lines
- Hard selection (U8): winner-take-all, not soft coupling
- Growth (U22): spawn prevents convergence
- Local continuity (U20): metric must be continuous
- **State-conditioned**: the METRIC (currently frozen cosine) depends on the codebook's own state

This is ReadIsWrite taken one step further. ReadIsWrite made the UPDATE state-conditioned (softmax weights from V). The next step: make the COMPARISON state-conditioned. Not "always cosine" but "similarity measure derived from V."

## What This Would Look Like

```
# Current (reactive): metric is frozen
sim = F.cosine_similarity(V, x)  # always cosine, regardless of V's state

# State-conditioned: metric depends on V
weights = derive_weights(V)  # V determines what dimensions matter
sim = weighted_cosine(V, x, weights)  # comparison adapts to codebook state
```

The weights would be derived from V's internal structure (e.g., variance per dimension, inter-entry correlation). The metric adapts as the codebook learns. When the codebook discovers that certain dimensions are noise, the metric downweights them.

This is exactly the encoding discovery problem (Frontier B) solved at the metric level instead of the preprocessing level.

## Open Questions

1. Does weighted cosine maintain U20 (local continuity)?
2. Is `derive_weights(V)` a frozen operation (another U element) or can it be self-referential?
3. Can the weights be derived with zero additional state (V already exists)?
4. Does this dissolve U24? If weights change with codebook maturity, early weights favor exploration, late weights favor discrimination.
