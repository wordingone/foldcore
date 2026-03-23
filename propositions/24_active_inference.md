# Proposition 24: Active Inference as Action Selection Framework
Status: KILLED (Steps 934-936)
Steps: 934, 934b, 935, 936

## Statement
Define epistemic value G_t(a) = ||x_hat(a) - x_t|| / (conf_a + epsilon), where the numerator captures predicted novelty and denominator captures model uncertainty. This should dissolve Prop 23b: at each position, the correct action predicts novel outcomes with low confidence, while wrong actions predict familiar outcomes with high confidence.

## Evidence
Theoretical: G_t(a) is a tractable approximation of expected free energy (Da Costa et al. 2020). Substrate already implements active inference components: alpha=precision, W=generative model, h=temporal context. Only action selector missing.

**KILLED experimentally:**
- Step 934: Raw novelty = uniform noise (all actions ~20K +/- 400, <2% spread)
- Step 934b: Confidence equalizes across actions
- Step 935: h-delta too small to discriminate
- Step 936: Input-distance flat
When W never converges, G_t(a) ~ 1/conf_a ~ pred_error_a = ICM. The 800b-variant family is EXHAUSTED after 160 experiments.

## Implications
The forward model is too inaccurate for action discrimination in games. Active inference reduces to noisy ICM in practice. Next direction must be a genuinely different family (see FAMILY_KILLS.md).

## Supersedes / Superseded by
Attempted to dissolve Proposition 23b. Failed. FT09 remains structurally unsolved.
