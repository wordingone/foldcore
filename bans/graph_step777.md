# Ban: Graph (per-state-action data)
Step: 777 | Status: ACTIVE

## What is banned
Any data structure keyed by (state, action). The mechanism, not the name:
- Edge dicts: G[(cell, action)] → {successor: count}
- Per-cell visit counters
- Transition tables or matrices
- Action frequency tracking per state
- ANY structure enabling "which action least from this state?"

## Why
Proposition 19 (R3 counterfactual): graph produces negative transfer — pretrained state hurts on new environments (Step 776, p<0.0001). Per-(state,action) tracking IS navigation capability but PREVENTS R3. The wall.

## What survives
- Global state (total step count, running statistics)
- Per-observation state (hash tables, encodings) — NOT per-(observation, action)
- Prediction models (W matrices, trees) — if they don't track visit frequencies
- Population-level mechanisms

## Lifting condition
Permanent. No lift condition.
