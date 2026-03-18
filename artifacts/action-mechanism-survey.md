# Action Mechanism Survey: Steps 354-416

*Built by autonomous loop (iteration 10). Classifies each post-Step-353 experiment by what action mechanism it used and whether it navigated faster than the 26K random-walk baseline.*

## The Question

Step 353 found LS20 Level 1 at ~26K steps via stochastic coverage (random walk with argmin novelty-seeking). 63 experiments followed. Did ANY make navigation purposeful (faster than 26K)?

## Classification

### A. Action Mechanism Unchanged (pure argmin baseline)

| Step | What | Result | Steps to Level |
|------|------|--------|---------------|
| 354 | Extended pure novelty (200K) | Baseline ceiling | ~26K when found |
| 358 | All 3 games at 16x16 | Baselines established | ~26K (LS20) |
| 359 | LS20 deep (50K) | Level 1 sometimes | ~26K when found |
| 363 | Reliability (5 trials) | P(Level 1) = 60% in 50K | ~26K avg |
| 372 | LS20 all levels (100K) | Multi-level measurement | ~26K per level |
| 373 | FT09 all levels (100K) | Level 1 at step 82 | 82 (encoding-driven) |

### B. Action Selection Modified

| Step | Mechanism | Result | Faster? |
|------|-----------|--------|---------|
| 355 | Boltzmann (softmax with thresh as temperature) | Structurally worked (thresh modulated entropy) | **NO** — same or worse |
| 356 | Local-sim epsilon-greedy (sim → exploration probability) | Structurally worked | **NO** — same or worse |
| 357 | 356 + no-overcompress | Restored dynamics | **NO** |
| 369 | 2-action restriction (reduce branching factor) | Scaling test | Halves steps mechanically, not purposeful |

### C. Encoding Modified (action mechanism unchanged)

| Step | Encoding Change | Result | Faster? |
|------|----------------|--------|---------|
| 361 | Click-space (69 classes for FT09) | Level 1 at step 82 | **YES** — 300x faster (different game) |
| 364 | Windowed [t, t-1, t-2] | More unique windows | **NO** — killed |
| 365 | Variance-weighted dims | Correct signal isolation | **NO** — cosine HIGHER = hurts |
| 366 | Diff-variance weighting | Timer suppressed | **NO** — killed |
| 367 | Timer mask + windowed | Timer removed | **NO** — degraded |
| 368 | Windowed efficiency A/B test | Head-to-head vs baseline | **NO** — killed |
| 375 | 3-zone mapping for VC33 | Level 1 at step 283 | **YES** — prescribed game knowledge |

### D. Game Signal Used

| Step | Signal | Result | Faster? |
|------|--------|--------|---------|
| 370 | Custom timing game (click-when-flash) | Temporal pattern test | Test was whether substrate detects timing |
| 376 | Codebook reset on level transition | Uses level-up signal | Addresses stale codebook, not speed |

### E. 64x64 Resolution Attempts (Steps 377-416)

| Step Range | What | Result |
|-----------|------|--------|
| 377-389 | Raw 64x64 | All killed — noise dominates signal |
| 390-393 | Auto-select resolution | All killed — no signal before playing |
| 394-395 | Self-feeding / cascade | Killed — U13 (additions hurt) |
| 396-399 | Two-codebook | Killed — bootstrap failure, votes uniform |
| 400-403 | Timer/mask/kernel at 64x64 | All killed or marginal |
| 404-410 | Recursive / chunk / temperature | All killed |
| 411-412 | Evolving metric / no-normalize | Killed |
| 413-413b | Multi-resolution persist | Killed |
| 414 | Sequential dedication (64→32→16→8) | **LEVEL 1 at 16x16, step 26218** |
| 415-416 | Click grid / probability tuning | Killed |

## Summary

**Action modifications (B): 0/4 improved navigation speed.** Every directed action mechanism (Boltzmann, epsilon-greedy, 2-action) either matched argmin or degraded. The random walk IS the exploration engine. Making it directed doesn't help.

**Encoding modifications (C): 2 succeeded, 5 killed.** FT09 click-space (82 steps) and VC33 zones (283 steps) both succeeded by matching the encoding to the game's action structure. All other encoding modifications (windowed, variance, timer mask) were killed.

**Resolution discovery (E): 1 success in 40 experiments.** Step 414 discovered 16x16 by sequential dedication — trying all resolutions and letting the game select. 39 other attempts to work at 64x64 or auto-select resolution all failed.

## The Pattern

Navigation speed is determined by **encoding-game structural match**, not by action selection mechanism. When the encoding matches (FT09 click-space, VC33 zones, LS20 16x16): fast. When it doesn't: 26K random walk or never.

No experiment in Steps 354-416 made the SAME encoding navigate FASTER. The only speedups came from BETTER encodings. This confirms the session's encoding compilation finding: the 300x lives in the encoding, not the substrate.

## Implication for Phase 2

The substrate search should shift from "better action mechanisms" to "encoding discovery." Step 414 proved resolution is discoverable. The meta-protocol (try encodings, monitor codebook health, keep what works) is the path. The action mechanism (argmin or softmax or whatever) is secondary — the encoding IS the intelligence.
