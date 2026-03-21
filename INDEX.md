# The Search — Index

*596+ experiments across 12 families. These are the ones that matter.*

---

## Tier 1 — Changed the research direction

| Step | What happened | Why it matters |
|------|--------------|----------------|
| [442b](experiments/run_step442_graph_substrate.py) | First non-codebook navigation | Graph + edge-count argmin navigates LS20 without codebook machinery. Proved the mechanism is the graph, not the codebook. Opened Phase 2. |
| [432](experiments/run_step432_labeled_vs_self.py) | Classification requires external labels | Self-generated labels: 9.8% (below chance). The 94.48% P-MNIST result depends entirely on external supervision. 84.68pp gap. R1 violation. |
| [418](experiments/run_step418_readiswrite.py) | argmin ≠ argmax | Navigation needs argmin (least-visited). Classification needs argmax (most-similar). One mechanism cannot serve both. The substrate must be general enough for both — but not by switching between them. |

## Tier 2 — Established key empirical facts

| Step | What happened | Why it matters |
|------|--------------|----------------|
| [460](experiments/run_step460_reservoir_lsh.py) | Reservoir+LSH hybrid navigates | Composed mapping (reservoir dynamics + LSH projection) works. Mappings are composable. The observation-to-cell function is a degree of freedom. |
| [481](experiments/run_step481_prediction_error.py) | Prediction error 0/10 | Smart exploration kills navigation. Every targeted action strategy performs worse than uniform argmin. Coverage, not signal-chasing, is what works. |
| 484/[485](experiments/run_step485_ls20_9of10.py) | 6/10 was budget artifact; 9/10 at 120K | The apparent 6/10 ceiling was step budget, not mechanism. Hard seeds navigate at 35K-115K. No architecture change needed — just more time. |
| 489–[493](experiments/run_step493_kmeans_level2.py) | Level 2 closed across two families | LSH (259 cells) and k-means (286 cells) both plateau. The reachable state space is bounded regardless of mapping architecture. Level 2 requires purposeful exploration, not a better map. |

## Tier 3 — Filled in the picture

| Step | What happened | Why it matters |
|------|--------------|----------------|
| [453](experiments/run_step453_lsh_graph.py) | LSH navigates | Fixed random hyperplanes, zero learning, zero parameters. The mapping doesn't need to learn — it needs to be locally continuous and persistent. |
| 503/[505](experiments/run_step505_vc33_zones.py) | FT09 and VC33 solved | Action decomposition is the variable. FT09: 69 actions (64 grid + 5 simple). VC33: 3 zones (2 magic pixels). All 3 ARC games Level 1 solved. |
| 446–[452](experiments/run_step452_kdtree_graph.py) | Grid, kd-tree, CA failures | Negative evidence for what mappings need: local continuity (grid fails), persistence (kd-tree splits destroy edges), non-degeneracy (CA collapses). |
| [428](experiments/run_step428_score_diagnostic.py) | Score convergence after ~5K steps | Action scores converge 150x by 30K steps. The codebook becomes a random walk. Navigation happens in a pre-convergence window. Scoring is not the mechanism. |
| [512](experiments/run_step512_cifar_threshold.py) | Chain threshold incompatibility | CIFAR needs threshold ≥ 3.0 for meaningful clustering. ARC needs ≤ 0.5 for navigation. One fixed threshold cannot serve both domains. |
| [515](experiments/run_step515_kmeans_neg_transfer.py)/[522](experiments/run_step522_kmeans_crossgame.py) | Negative transfer universal; frozen centroids collapse | K-means confirms codebook's chain failure (Step 506). Centroid-based families need online adaptation. LSH avoids this entirely. |

## Tier 4 — Important context

| Step | What happened | Why it matters |
|------|--------------|----------------|
| 65 | P-MNIST zero forgetting | First real-data continual learning result. The codebook partitions by geometry, preventing cross-task interference. Foundation for everything after. |
| [494](experiments/run_step494_bloom_filter.py) | Bloom filter control | 1/10 (random walk luck). Without the graph's edge memory, observation→cell mapping alone is insufficient. The graph mechanism is required. |
| [465](experiments/run_step465_reservoir_kill.py) | Reservoir killed at 20 experiments | Temporal memory hurts navigation. The reservoir's history-dependent mapping violates local continuity. Memory of the past interferes with the present. |
| 181/235/[250](experiments/run_step250_complete_substrate.py) | Computation from primitives | Rule 110, OOD addition, algorithm synthesis — all from iterated k-NN. The substrate CAN compute. The question is whether it discovers WHAT to compute. |
| 506–[508](experiments/run_step508_full_chain.py) | Chain passes mechanically | CIFAR→LS20→FT09→VC33→CIFAR. 3/3 games, zero forgetting, 10K centroids, 1% CIFAR accuracy. The chain works. The efficiency doesn't. |
| [516](experiments/run_step516_lsh_chain.py) | LSH chain via action-scope isolation | WIN@1116 (10x faster than codebook). Each game queries only its own action slice. Second independent chain mechanism — no encoding-space separation needed. |

---

## Reference

| Document | What it contains |
|----------|-----------------|
| [CONSTITUTION.md](CONSTITUTION.md) | R1-R6: the six rules any substrate must satisfy |
| [CONSTRAINTS.md](CONSTRAINTS.md) | Full constraint map with cross-family validation |
| [RESEARCH_STATE.md](RESEARCH_STATE.md) | Complete experiment log (Steps 1–525+) |
| [PAPER.md](PAPER.md) | Publication draft — formal framework, theorems, results |
| [R3_AUDIT.md](R3_AUDIT.md) | Frozen frame analysis per substrate |

---

## Indexing Process

Every experiment passes through four gates. First match wins.

| Gate | Question | Tier |
|------|----------|------|
| 1 | **Did this change what we pursue next?** Before/after this result, the research direction is fundamentally different. | Tier 1 |
| 2 | **Will 5+ future experiments reference this as a given?** The result becomes a premise, not a finding. | Tier 2 |
| 3 | **Does this confirm cross-family, provide key negative evidence, or quantify a boundary?** | Tier 3 |
| 4 | **Is this foundational context or a necessary control?** | Tier 4 |
| — | None of the above. | Not indexed. |

Most experiments are not indexed. That's correct. The index is a curated signal, not an exhaustive log. RESEARCH_STATE.md is the exhaustive log.

### Replication status

Each indexed entry should track:

| Status | Meaning |
|--------|---------|
| **Replicated** | Confirmed across 2+ architecturally distinct families |
| **Single-family** | Result from one family only — hypothesis, not law |
| **Superseded** | A later experiment replaced this finding |

Current replication status of indexed experiments:

| Entry | Families confirmed | Status |
|-------|-------------------|--------|
| 442b (graph navigates) | Codebook, LSH, k-means | Replicated |
| 432 (classification requires labels) | Codebook only | Single-family |
| 418 (argmin ≠ argmax) | Codebook, LSH, k-means | Replicated |
| 481 (smart exploration kills) | LSH, k-means | Replicated |
| 485 (9/10 budget artifact) | LSH (9/10), k-means (9/10) | Replicated |
| 489-493 (Level 2 closed) | LSH, k-means | Replicated |
| 453 (LSH navigates) | LSH only (by definition) | Single-family |
| 503/505 (action decomposition) | Codebook, k-means | Replicated |
| 428 (score convergence) | Codebook only | Single-family |
| 512 (threshold tension) | Codebook chain only | Single-family |
| 506/515 (negative transfer) | Codebook, k-means | Replicated |
| 516 (action-scope isolation) | LSH only | Single-family |

**Re-benchmark trigger:** When 3+ entries are "Single-family," run the single-family experiments on the strongest non-codebook family (currently LSH). Priority: entries where cross-family replication would strengthen or invalidate a paper claim.

Current re-benchmark candidates:
- **432** (classification requires labels) — test on LSH/k-means. Does argmax over LSH edge counts classify? Prediction: no, but data needed.
- **428** (score convergence) — does LSH edge-count argmin also converge? Prediction: yes (counts converge by definition), but the mechanism is different.
- **512** (threshold tension) — test LSH on chain with CIFAR. Already partially answered by Step 516 (LSH chain passes via action-scope isolation, bypassing the threshold issue entirely).

---

*The 16 experiments above cover what 525 found. The other 509 are the search space that made these visible.*
