# Ban: Codebook
Step: 416 | Status: LIFTED (Jun directive, 2026-03-25)

## What is banned
The codebook/LVQ mechanism and anything sharing its spatial engine:
- Cosine similarity matching (V @ x) for observation→entry lookup
- Attract update (entry += lr * (x - entry)) for learning
- Spawn-on-similarity-threshold for growth
- F.normalize to unit sphere
- Winner-take-all nearest-prototype selection
- Class/label tagging with argmin/argmax scoring
- ANY combination of the above

## Why
416 experiments proved process() IS LVQ (1988). The codebook family is fully characterized — no remaining degrees of freedom. Continuing is reinventing known work.

## What survives
- Genuinely new spatial mechanisms (LSH, quantization, L2 centroids — not on unit sphere with cosine)
- Reservoir family (recurrent dynamics, Hebbian learning)
- Graph family (if nodes are NOT cosine-matched entries)
- U20 (local continuity) satisfied through a DIFFERENT mechanism

## Lifting condition
See [Ban Policy](POLICY.md).
