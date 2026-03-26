# ARC-AGI-3 Competitor Intelligence

Collected 2026-03-26. Sources: GitHub repos, ARC Prize leaderboard, preview results.

## Preview Top 3

### 1st: StochasticGoose (Tufa Labs) — 12.58%, 18 levels
- **Author:** Dries Smit (Tufa Labs)
- **Code:** github.com/DriesSmit/ARC3-solution
- **Architecture:** CNN-based action learning (SUPERVISED)
  - 4-layer CNN backbone: 32→64→128→256 channels
  - Input: 16-channel one-hot encoded 64x64 frames
  - Dual head: action head (ACTION1-5 probs) + coordinate head (64x64 for ACTION6 clicks)
  - Coordinate head uses CONV layers (preserves 2D spatial bias, not flattened)
- **Training:** Supervised on (state, action) → frame_changed binary labels
  - 200K unique state-action pairs, hash-deduplicated
  - Binary cross-entropy + entropy regularization
  - Dynamic reset on new levels
- **Exploration:** Stochastic sampling from sigmoid probs, biased toward change-causing actions
- **ℓ classification:** ~90% ℓ_π (learned CNN model drives action selection)
- **R1 compliance:** NO — uses BCE loss (external objective). But the PRINCIPLE is R1-adjacent: "did the frame change?" is observable without reward.

### 2nd: Blind Squirrel (Will Dick) — 6.71%, 13 levels
- **Architecture:** Frame-to-graph + ResNet18 value model
- **Approach:** Online learning, back-labels on score improvement
- **ℓ classification:** ~60% ℓ_π / 40% ℓ₁ hybrid

### 3rd: Graph Exploration (Rudakov et al.) — 30/52 levels median, TRAINING-FREE
- **Code:** github.com/dolphin-in-a-coma/arc-agi-3-just-explore
- **Paper:** arxiv.org/abs/2512.24156, AAAI 2026 Workshop
- **Architecture:** Pure heuristic graph exploration, ZERO learning
  - Frame segmentation: single-color connected components
  - Status bar masking
  - 5 action-priority tiers (button likelihood: size + salient color)
  - State graph: nodes=frames (hashed), edges=actions, frontier=untested (state,action) pairs
  - Exploration: highest-priority untested actions first, shortest path to frontier
- **Post-fix note:** After fixing graph-reset bug, solves median 17 levels (1 below 1st place!)
- **ℓ classification:** 100% ℓ₁ — zero learned params
- **Key insight:** Structured exploration with good frame processing nearly matches CNN learning

## Frontier LLM Scores on ARC-AGI-3
ALL show N/A on main leaderboard. Estimated < 1% from preview data.

## Key Implications for Our Search

1. **Supervised learning on "did frame change" wins.** StochasticGoose's core signal is identical to our v80's change-rate detection — but learned via CNN instead of EMA statistics.

2. **Spatial click encoding is critical.** 64x64 coordinate head with conv layers, NOT flattened. Our block-center clicks miss this.

3. **Graph exploration is surprisingly strong.** Training-free 3rd place (17 levels post-fix) nearly matches CNN 1st place (18 levels). Validates defense's core thesis.

4. **R1-compliant version possible.** Replace BCE loss with outer-product update on frame-change signal. The SIGNAL is R1-compliant ("did frame change?"), the TRAINING METHOD is not (SGD+BCE). Our forward model family was the right idea.

5. **State deduplication via hashing is load-bearing.** Both 2nd and 3rd place use it. Our LSH approach was on the right track.

6. **200K experience buffer.** 1st place stores 200K unique (state,action) pairs. Within 10K-step budget per game, this means retaining experiences across episodes/resets.
