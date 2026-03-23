# Proposition 21: Global-Local Gap
Status: CONFIRMED
Steps: 780-806v2, 800

## Statement
Navigation requires per-state action selection (productive action depends on current state). Forward model W learns GLOBAL dynamics. Action selection g(s) = argmax_a f(W,s,a) depends on both W and current state. The gap: W trained on one environment learns transferable dynamics (Prop 20b confirmed), but g(s) at a new state in a new environment produces the wrong action because which action is productive depends on LOCAL state.

## Evidence
- Prediction-contrast (argmax_a ||W(s,a)-s||): L1=0 on LS20 (novelty-seeking avoids navigable paths). L1=0 on FT09 (static background = uniform predictions).
- Per-action change tracking: uniform delta ~0.008 for all 68 FT09 actions (position signal masked by global averaging).
- Step 806v2: L1 PASS at sub_seed=0 RETRACTED (seeds 1-3: 0 or negative).
- Random baseline: 36.4 L1/seed on LS20, better than any D(s)-guided mechanism.

## Implications
Post-ban substrates face fundamental tradeoff: D(s) captures transferable global dynamics (prediction R3_cf PASS), but converting to local navigation requires per-state info -- which graph ban removes. Feasible region for PREDICTION transfer is non-empty. Feasible region for NAVIGATION transfer remains empty post-ban.

## Supersedes / Superseded by
Extends Proposition 20. The structural gap remains the open frontier.
