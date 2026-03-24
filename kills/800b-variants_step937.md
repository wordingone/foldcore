# Kill: 800b-variants
Steps: 800-990 (200+ experiments) | Trigger: ALL modifications degrade LS20; position-blind (Prop 23b); frozen (27 kills, Steps 966-990)

**Theoretical explanation:** Theorem 4 (PAPER.md §4.17). Global running mean SNR ≤ 1/|N_a| for state-dependent actions. 800b's delta_per_action averages over ALL states → signal for the correct action at state s* is diluted by 1/|N_a|. LS20 (4 actions, state-independent movement): SNR ≈ 1. FT09 (68 actions, state-dependent sequences): SNR → 0.007 at 10K. This is why 800b works on LS20 and provably cannot work on FT09/VC33.

## What worked
- ACTUAL observation change magnitude is the only working action signal
- Alpha-weighted encoding self-modification (R3 confirmed universally)
- Per-action GLOBAL statistics (not per-state)
- Softmax T=0.1 + epsilon=0.20
- Echo-state h improves standalone LS20 by 8.5% (hurts chain by 14%)
- Clamped alpha [0.1, 5.0]

## What failed
- ANY prediction-based action selector (W pred_acc=-2383)
- ANY addition to 800b selector degrades LS20
- Multi-resolution encoding dilutes alpha
- h-delta compressed by tanh
- Confidence metrics equalize fast (LS20: 4 actions)
- Adaptive lambda loses signal without exact 916 formula
- Warm alpha transfer unreliable (cold > warm at n_eff=10)

## What next family needs
- Use ACTUAL observation changes (proven) — NOT per-action delta EMA
- Preserve alpha-weighted encoding
- Handle sequential ordering WITHOUT per-state memory
- Handle near-static observations (VC33 var≈0.000)
- Structurally different in action selection, state representation, or learning rule

## Return condition
Predict a result this kill register says is IMPOSSIBLE: position-aware without per-state memory, or action selection without prediction errors overwhelmed by W noise.
