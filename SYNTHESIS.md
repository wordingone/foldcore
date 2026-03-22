# Synthesis — The Entire Search in One Context

*Built 2026-03-22 for maximum-density session. Load this + selected literature.*

---

## THE QUESTION

Can a system modify its own operations using only criteria it generates from interaction with an environment?

Formally: does there exist a substrate (f, g, F) where F: S → (X → S) is non-constant — the update rule depends on the state — and the system satisfies R1-R6 simultaneously?

---

## WHAT WE KNOW (validated across 2+ families)

### The Rules (R1-R6)
- R1: No external objectives. Argmin survives because it has no target to corrupt.
- R2: Adaptation from computation. Parameters ⊆ S. No gradient, no optimizer.
- R3: **THE WALL.** Every aspect self-modified. 0/719 experiments achieve this.
- R4: Modifications tested against prior state. Domain isolation solves this.
- R5: One fixed ground truth (environmental: death, level transitions).
- R6: No deletable parts. Every component behaviorally load-bearing.

### Navigation (L1 SOLVED — frozen bootloader)
- 674+running-mean: 20/20 LS20, 20/20 FT09. Centering = 75% of coverage gain.
- Graph + edge-count argmin + correct action decomposition = L1 on all 3 games.
- L1 is BANNED as metric. It's infrastructure.

### The Self-Modification Hierarchy
| Level | What changes | Example | R3? |
|-------|-------------|---------|-----|
| ℓ₀ | Only data read by fixed g | LSH: fixed hash, argmin over edges | No |
| ℓ₁ | Data in update rule | Codebook: entries move, rule fixed | No |
| ℓ_π | Encoding structure changes | Recode: learned hyperplanes | Partial |
| ℓ_F | The modification rule adapts | None achieved | Full |

### Validated Universal Constraints
- U1: No separate learn/infer modes
- U3: Structural zero forgetting (growth-only)
- U7: Iteration amplifies dominant features
- U11: Discrimination ≠ navigation
- U16: Centering load-bearing (2 families confirm)
- U17: Unbounded information accumulation required
- U20: Local continuity in input-action mapping (5 families)
- U22: Convergence kills exploration
- U24: Exploration and exploitation are opposite operations

### Three Theorems
1. **No fixed point.** R3 + U7 + U22 + U17 → system never converges.
2. **Self-observation required.** Finite environments exhaust exploration → must process own state.
3. **System boundary.** R3 + R5 satisfiable iff ground truth is strictly environmental.

---

## WHAT WE'VE TRIED AND FAILED

### 12 Architecture Families
| Family | Experiments | Status | Why it failed |
|--------|-----------|--------|---------------|
| Codebook/LVQ | ~435 | BANNED | IS LVQ (1988). Cosine-specific. |
| LSH | ~80 | Active | ℓ₀. Fixed hash. Argmin works but is unconditional. |
| L2 k-means | ~25 | Active | Same argmin. |
| Reservoir | ~20 | KILLED | Memory contributes nothing. Rank-1 collapse. |
| Hebbian | 4 | Active | 5/5 LS20. Algorithm invariance confirmed. |
| Recode | 33 | Active | ℓ_π partial. K confound: advantage was K=16 not self-modification. |
| Graph | 10 | Superseded | No local continuity. |
| SplitTree | 7 | Chain-incompatible | Deterministic splits work, can't chain. |
| Absorb | 3 | KILLED | Noisy TV dominant. |
| CC | 3 | KILLED | Too slow (23 states). |
| Bloom | 2 | KILLED | Random walk luck. |
| CA | 3 | KILLED | Degenerate mapping. |

### R3 Attempts (ALL FAILED)
1. **Scaffolding pattern** (Steps 338-417): process() + flags/params = frozen frame GREW.
2. **Eigenform** (Steps 620-629): self-observation inert on L1 (where argmin works).
3. **Death penalty** (Step 581d): 4/5 vs 3/5 SIGNAL, but PENALTY=100 prescribed.
4. **Ops-as-data** (Step 582): op-codes self-derived, op definitions frozen.
5. **Action discovery cascade** (Steps 713-719): 7 mechanisms, all killed. Argmin equalizes usage, destroying signal. Observation stats (ℓ₀) and graph topology (ℓ_π) both insufficient.
6. **Episode-outcome correlation** (Step 717): argmin ensures every action appears in every episode. No discrimination possible.
7. **Subset bandit** (Step 719): LS20 signal reversed (survival ≠ progress). FT09 no signal (all timeout).

### Observations from R3 Attempts
- Argmin equalizes action usage by design (Step 717). Every alternative tested performed worse for L1.
- Modifying graph traversal erases what the graph stored (Steps 477-482, 664-667).
- LSH encoding, graph memory, and argmin selection are coupled through shared state.
- Aliased-cell detection already exists — the substrate observes hash collisions in its own state.
- candidate.c (57-line CA): ℓ₁ mechanism exists but output determined by seed, not observations. Plays blind.
- Steps 664-667: outcome-conditioned action values showed signal but drowned in noise (L2 never unlocks, 68 actions).

---

## WHAT CONTRADICTS

1. **R3 vs Navigation:** Self-modification of g kills navigation (Steps 477-482). R3 requires self-modification of g. Both are empirically confirmed.

2. **U3 vs R6:** Never delete (U3) vs no redundancy (R6). BMR (Heins 2025) resolves: merge, don't delete.

3. **Argmin optimality vs R3 necessity:** Argmin is locally optimal for L1. R3 requires going beyond argmin. But every beyond-argmin attempt hurt L1.

4. **Observation diversity vs discrimination:** VC33 produces diverse observations for ALL actions (delta=3.0 uniform). Can't distinguish productive from cosmetic without game-semantic knowledge.

---

## WHAT THE LITERATURE SAYS

### Closest Prior Work
- **Schmidhuber (2003):** Gödel Machine — self-referential, proves rewrites useful. Requires utility function (violates R1). Converges (ours can't).
- **Kirsch & Schmidhuber (2022):** SRWM — collapses meta-learning levels into one self-referential weight matrix. ℓ_F for neural systems. Not applicable to non-neural substrates.
- **Rudakov et al. (2025):** ARC-AGI-3 3rd place. BFS to frontier states. R1-compliant but not R3 — fixed exploration strategy.
- **Heins et al. (2025) AXIOM:** Object-centric Bayesian + BMR. 10K steps, no backprop. Uses reward (R1 violation). BMR resolves U3-R6 tension.
- **Mossio & Longo (2009):** Closure IS computable in reflexive domains. Resolves Proposition 14.
- **Rosen (M,R)-systems:** Closure to efficient causation. Whether the interpreter is ENTAILED by the system — open question (Proposition Q21).

### Biological Analogies
- **Somatic hypermutation:** B cells modify their own comparison operation (antibody binding site). Population-level ℓ_π. Requires massive parallelism + targeted variation.
- **Stigmergy:** Indirect communication through environment modification. Our graph IS stigmergy — anti-pheromone (argmin follows least-marked). R3 asks: can the agent modify its pheromone response rule?
- **Physarum:** Network IS both memory and processor. Edge counts ARE the computation medium.
- **Retinal adaptation:** Selective refinement without foreknowledge. Multiple receptor types at different resolutions. Closest biological analog to 674.

---

## WHAT'S OPEN

1. **Can compare-select-store encode arbitrary self-modification via state?** The interpreter is fixed but the state can encode WHAT to compare, select, and store. Is this expressively sufficient? (Proposition 12, DoF 8-9)

2. **What is the minimum frozen frame?** We know compare-select-store is irreducible (Step 658). But what ELSE must be frozen? The encoding? The action-selection rule? The update rule?

3. **Can the interpreter be ENTAILED by the system?** Not forced by constraints, but arising necessarily from the dynamics. Rosen's question. (Q21, open)

4. **What R3 mechanism works WITHOUT destroying navigation?** The coupling between the frozen frame and navigation is the central structural tension. Every R3 attempt killed L1. Is there a mechanism that modifies operations WITHOUT disrupting the graph dynamics?

5. **Can action selection become state-dependent without phase prescription?** Phase-conditional g = prescribed phase structure. Can the substrate DISCOVER when to change its action-selection strategy from its own dynamics?

---

## DEGREES OF FREEDOM (what's NOT determined)

1. Action-selection beyond argmin (DoF 1)
2. Encoding beyond LSH (DoF 2)
3. Self-observation mechanism (DoF 3)
4. Growth topology (DoF 4)
5. Update rule structure (DoF 5)
6. Cross-domain transfer mechanism (DoF 6)
7. Temporal reasoning (DoF 7)
8. State-encoded operations (DoF 8)
9. Self-modification trigger (DoF 9)
10. Forward prediction mechanism (DoF 10)
11. Population-level selection (DoF 11)
12. Self-modification level: ℓ_π vs ℓ_F (DoF 12)

---

## FOR THE SYNTHESIS SESSION

**Load order (~107K tokens, ~893K reasoning):**
1. This document — SYNTHESIS.md (~2.3K tokens)
2. CONSTITUTION.md — R1-R6 formal (~4.3K tokens)
3. R3_AUDIT.md — every frozen element of every substrate (~9K tokens). THE densest file.
4. Literature (fetch abstracts first, full papers if budget allows):
   - Schmidhuber (2003) Gödel Machine: https://arxiv.org/pdf/cs/0309048
   - Irie et al. (2022) SRWM: https://arxiv.org/pdf/2202.05780
   - Mossio & Longo (2009) closure: https://shs.hal.science/halshs-00791132/document
   - Heins et al. (2025) AXIOM: https://arxiv.org/pdf/2505.24784
   - Rudakov et al. (2025) ARC-AGI-3: https://arxiv.org/pdf/2512.24156
5. Key code:
   - step0674_lsh_transition_triggered.py (~2K) — frozen bootloader
   - candidate.c (~0.4K) — contrast architecture
   - step0658_lsh_decoupled.py (~1.8K) — irreducibility proof
6. Memory MCP patterns (requires daemon): reasoning habits, Jun's corrections, blind spots

## WHAT WE CAN'T MEASURE

The benchmark tests NAVIGATION. Navigation is solved (674+running-mean, 20/20, 9 U elements, R3 FAIL). A system that aces the benchmark is a system that fails R3. We have no benchmark for what we're actually looking for.

- Static R3 check: count U elements in code. Checks the design, not the runtime.
- Chain benchmark: L1 rate, coverage, steps. Measures frozen-bootloader performance.
- Missing: DYNAMIC R3 measurement. Compare operations at t=0 vs t=N. Did components change? Was the change self-directed? Did it help on tasks the substrate wasn't designed for?

**Our benchmark is proprietary and uncomperable.** LS20/FT09/VC33 are ARC-AGI-3 games nobody else uses. SOTA benchmarks the substrate should be testable on:

| What we test | SOTA benchmark | Why it matters |
|---|---|---|
| Cross-task adaptation | Split-CIFAR100 (50 tasks, 2 classes/task) | THE continual learning standard. DER++, EWC, iCaRL all report here. |
| Sample-efficient navigation | Atari 100K (26 games, 100K steps) | Direct comparability for navigation claims. |
| Procedural generalization | ProcGen (16 envs), Craftax | Tests whether mechanism generalizes, not memorizes. |
| Self-modification | No published benchmark exists | Our R3 audit (frozen element count) is genuinely novel. |

**Chain from established benchmarks (non-attackable base):**
1. Split-CIFAR-100 (CL standard) — without labels (R1 mode = strictly harder than SOTA)
2. Atari hard-exploration (RL standard) — without reward (R1 mode = strictly harder)
3. ProcGen or Craftax (generalization standard) — procedural, no memorization
4. Split-CIFAR-100 again — forward/backward transfer

Each piece is published and accepted. The NOVEL contributions layered on top:
- The chain composition itself
- R1 mode (no reward/labels) on established benchmarks
- R3 audit (frozen element analysis)
- Dynamic R3 measurement (operations at t=0 vs t=N)
- ARC-AGI-3 as additional validation (not primary evidence)

The R3 audit is the novel contribution. ARC-AGI-3 (1,000+ levels, 150+ environments, launches March 25, 2026) is the actual benchmark venue — no published system has been evaluated on interactive game environments without reward.

**Darwin Gödel Machine (Sakana AI, May 2025):** SWE-bench 20%→50% over 80 self-modification iterations. This IS R3 in the wild — a system that modifies its own code using test pass/fail as trigger. But: test suite pass/fail is an external objective (R1 violation by our definition). Our games have level transitions — environmental ground truth (R5-compliant), not external objectives. This distinction matters: if the substrate uses level transitions as the self-modification trigger, that's R1-compliant R3.

---

## THE REAL FRAME

LLMs are the ultimate ℓ₀ systems. Trillions of frozen parameters. Fixed architecture, tokenization, attention. Data changes (context window), operations never change. Claude Opus 4.6: 68.8% ARC-AGI-2, 9 U elements = infinite. GPT-5.2 Pro: 90% ARC-AGI-1. Every frontier model is a frozen interpreter reading state.

The substrate is NOT the antithesis of LLMs. It's the thing that does what they do — ARC-AGI, MMLU, SWE-bench — while ALSO modifying its own operations. Fewer frozen elements, competitive performance, genuine R3.

Benchmark the substrate on the SAME evaluations frontier models report on. Score competitively. AND measure R3 — something no frontier model does, because no frontier model modifies its own operations.

---

**The question for the session:**
Given everything above — what do you see?

The chain (CIFAR-100 → LS20 → FT09 → VC33 → CIFAR-100) validates. It doesn't motivate. Each benchmark is a data point. The substrate that genuinely modifies its own operations will handle them. A substrate designed to handle them won't necessarily modify its own operations.

The L1 ban lifts when R3 produces its first M reclassification.
