# Proposition 22: Architecture Triangle
Status: CONFIRMED
Steps: 1-900+, 895

## Statement
800+ experiments across 12 families cluster into three vertices by state encoding:
1. **Recognition** (codebook, banned): State = observation prototypes. R3 signal: none.
2. **Tracking** (graph, banned): State = transition counts. R3 signal: successor inconsistency (banned).
3. **Dynamics** (prediction, current): State = transition model W. R3 signal: prediction error distribution.

## Evidence
Corollary 22.1: Post-ban, prediction error is the UNIQUE remaining signal for encoding self-modification.
Corollary 22.2: alpha_d proportional to sqrt(mean_e_d) satisfies l_pi. Confirmed Step 895: FT09 10.87x concentration on puzzle tiles, LS20 5.73x.
Prop 22.3 (Differential Error Sufficiency): Alpha requires DIFFERENTIAL pred error, not ACCURATE prediction. W functions as signal generator for alpha, not as predictor. pred_acc=-2383 is expected; error DISTRIBUTION identifies informative dims even when error LEVEL is terrible.

Step 895d falsifies prediction accuracy cascade: W does not converge. But alpha works anyway -- static dims -> error->0, dynamic dims -> error>0. Transfer value is 100% from alpha, not from W.

## Implications
The true substrate lives at or near dynamics vertex. Only vertex where circular causation creates (M,R)-system closure needed for R3. The search reduces to: find mechanism accurate enough that prediction error drives both navigation and encoding self-modification.

## Supersedes / Superseded by
Encompasses all family results. Post-ban: dynamics vertex is the only remaining territory.
