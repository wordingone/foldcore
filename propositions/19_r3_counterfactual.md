# Proposition 19: R3 Counterfactual Requirement
Status: CONFIRMED (prediction transfer); FAIL (navigation transfer)
Steps: 776, 780v5, 855b, 809b

## Statement
R3 requires: E[P(s_N, tau)] > E[P(s_0, tau)] -- self-modification must produce state outperforming initial state on average across task distribution, not just encountered tasks. R3_dynamic (state change) is necessary but not sufficient. R3_counterfactual is the discriminator.

## Evidence
- Step 776 (n=20, p<0.0001): 674 substrate R3_cf FAIL. E[P(s_N)]=2899 < E[P(s_0)]=4054. Graph stores WHERE substrate has been, producing negative transfer on new seeds.
- Prediction transfer: 5/7 PASS. Cold 11.5-50.2%, warm 19.9-54.6% across three D-only substrates. Forward model captures transferable dynamics.
- Navigation transfer: ZERO confirmed. Step 806v2 retracted (sub_seed=0 artifact). No post-ban mechanism produces consistent L1 improvement.

R3_cf is metric-dependent: PASSES for prediction accuracy, INCONCLUSIVE for level completion on LS20.

## Implications
1. Graph ban is consequence: per-(state,action) tracking produces negative counterfactual.
2. Forward model storing dynamics (not counts) could produce positive counterfactual.
3. The R3_cf protocol is implemented in ConstitutionalJudge.
4. Whether positive counterfactual is achievable for navigation under ANY (T,N,P) remains open.

## Supersedes / Superseded by
Motivated Proposition 20 (state decomposition).
