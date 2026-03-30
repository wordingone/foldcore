# Component Extraction Catalog

*Created 2026-03-24 compression. Reinterprets all 14+ killed families as component libraries per the Extraction Protocol (bans/POLICY.md).*

**Purpose:** Each killed family was killed AS A SYSTEM. Individual components were never tested in isolation. This catalog lists every extractable component, identifies which were tested in isolation, and specifies the extraction experiment required under the protocol.

**Protocol reminder:** (1) State killing finding. (2) Isolate one component. (3) Integrate into post-ban substrate. (4) Test against killing finding. One variable. (5) Failure reproduces → component banned. (6) Failure doesn't reproduce → component allowed.

---

## Era 1: Codebook (Steps 1-416, ~435 experiments)

**Family:** LVQ spatial engine (Kohonen 1988, Fritzke GNG 1995)
**Killing finding:** Fully characterized as LVQ — no remaining degrees of freedom (Step 416).
**What this means:** The SYSTEM is LVQ. Individual COMPONENTS may not be.

### Components

| # | Component | Tested in isolation? | What it does | Extraction test |
|---|-----------|---------------------|-------------|-----------------|
| C1 | **Novelty-triggered spawning** | NO (always with cosine+attract) | Grow memory when input is sufficiently novel (sim < threshold). Provides U3 (zero forgetting) and U17 (unbounded growth). | Spawn-on-novelty WITHOUT cosine metric, WITHOUT attract. Use L2 distance or other metric. Does spawning alone provide growth-only memory? |
| C2 | **Top-K voting** | YES (Step 425: 94.48% P-MNIST) | Classification via per-class sum of top-K similarities. | Already tested. Works for classification with external labels. R1-violating (needs labels). |
| C3 | **Softmax scoring** | YES (Step 425) | Softmax temperature on similarity scores for voting. | Already tested. +3.3pp over baseline. |
| C4 | **Centered encoding** | YES (Steps 414, 453) | x - mean(x) before processing. Prevents saturation/hash concentration. | Already validated as U16 across 2 families. CARRY FORWARD — not extraction, just use it. |
| C5 | **F.normalize** | Codebook-only | Unit-norm projection. Creates Goldilocks noise zone for cosine. | Only tested with cosine. Irrelevant for non-cosine substrates. |
| C6 | **Self-generated targets** | YES (Step 432) | Prediction as label (use softmax output as training signal). | Tested: 9.8% = chance. FAILS without external labels. Dead for classification. |
| C7 | **Change-rate detection** | YES (Step 400) | Track temporal change rate per pixel/dim to find active regions. | Works for detection (finds sprite in LS20, Steps 400-401). Detection ≠ encoding (Step 401). Could be COMBINED with alpha attention. |
| C8 | **Sequential resolution discovery** | PARTIALLY (Step 414) | Try resolutions sequentially, keep the one that works. | Tested with full codebook system. The CONCEPT (try multiple encodings) is valid. Could extract as "encoding search" component. |
| C9 | **Phi equation** (period detection + extrapolation) | YES (Steps 291-319) | Detect periodic structure in codebook, extrapolate. 95.2% OOD. | Works brilliantly for periodic classification (a%b). Never tested on navigation. Game-specific? |
| C10 | **Attract dynamics** | Only with cosine | Move stored vectors toward matching inputs. | Core of LVQ. This IS the banned mechanism. Unlikely to survive extraction — attract IS cosine learning. |
| C11 | **Avgpool spatial encoding** | YES (many steps) | Spatial downsampling (64x64 → 16x16). Noise reduction. | Already widely used. Not family-specific. CARRY FORWARD. |

### Era 1 extraction priorities (for FT09/VC33):
1. **C7 (change-rate detection)** — found active game regions autonomously. Never combined with post-ban substrates.
2. **C1 (novelty spawning) without cosine** — growing memory is U3/U17 compliant. Never tested with non-cosine metric.
3. **C8 (resolution search)** — multi-resolution could help VC33's near-static observations.

---

## Era 2: LSH / Graph / Recode (Steps 417-777, ~300 experiments)

### LSH Family (~80 experiments)
**Killing finding:** Not killed per se — absorbed into graph family. LSH cells are permanent (U3), random hyperplanes (no learning). Limited by fixed hash granularity.

| # | Component | Tested in isolation? | What it does | Extraction test |
|---|-----------|---------------------|-------------|-----------------|
| C12 | **Random hyperplane hashing** | YES (Steps 453+) | Project to binary code via random planes. Locality-preserving. | Works for navigation (U20 satisfied). Fixed — no adaptation. CARRY FORWARD as encoding baseline. |
| C13 | **Self-label classification** | YES (Step 573) | LSH k=16 achieves 36.2% self-labels on P-MNIST. | Best R1-compliant classification ever achieved. Component: random partition → majority vote. |

### Graph Family (~80 experiments)
**Killing finding:** Negative transfer — cold > warm, p<0.0001 (Step 776). Per-(state,action) data structures banned.
**What this means:** Per-(state,action) STORAGE is banned. Components that don't store per-(state,action) data may survive.

| # | Component | Tested in isolation? | What it does | Extraction test |
|---|-----------|---------------------|-------------|-----------------|
| C14 | **Argmin action selection** | Always with graph | Select least-visited action. Provides systematic coverage. | Argmin requires SOME form of per-action counting. Global per-action counting (800b) works and isn't graph-banned. Argmin over GLOBAL counts = 800b. Already extracted. |
| C15 | **Transition detection** | Always with graph | Detect when the same cell leads to different successors → mark as aliased. | The DETECTION doesn't require storage. Step 674 uses this for refinement. Could extract: detect inconsistent transitions WITHOUT storing the transition table. Use it as a signal, not a data structure. |
| C16 | **Edge-count accumulation** | Always with graph | Grow edge counts monotonically (U3, U17). | This IS the graph. Can't extract without being a graph. |
| C17 | **Visit-frequency signal** | Always with graph | Use visit counts to identify under-explored states. | Global version = 800b delta tracking. Per-state version = banned. |
| C18 | **Domain separation** (per-domain centering) | YES (Step 546) | Reset running mean on domain switch. Prevents cross-domain contamination. | Already extracted and used in chains. CARRY FORWARD. |
| C19 | **Death avoidance** (sparse signal) | YES (Step 581d) | Sparse environmental signal (<5%) for argmin perturbation. | Works at n=20 (p=0.63, marginal). The concept: use RARE game events as signals. Not graph-specific. |

### Recode Family (~33 experiments)
**Killing finding:** Not killed — subsumed. K confound invalidated results. Clean test never done.

| # | Component | Tested in isolation? | What it does | Extraction test |
|---|-----------|---------------------|-------------|-----------------|
| C20 | **Transition-triggered splitting** | YES (Step 542, Step 674) | When cell has inconsistent transitions, split it (learn a hyperplane). | Step 674 = this exact component. 9/10 L1 → 20/20 with running mean. Already extracted for LS20. Never tested on FT09/VC33 post-ban (because graph needed for navigation). |

### Phase 2 Candidates (Steps 437-453)
**Killed families:** SelfRef, TapeMachine, ExprSubstrate, TemporalPrediction, Reservoir, CA

| # | Component | Tested in isolation? | What it does | Extraction test |
|---|-----------|---------------------|-------------|-----------------|
| C21 | **Reservoir dynamics** (ESN) | YES (Steps 438-439, 787) | Random recurrent network. Fixed W_h, read out via W_out. | Rank-1 collapse via Hebbian (Step 439). BUT: echo-state h WORKS in 916 (Step 916). Already extracted — recurrent h is a component of 916. |
| C22 | **Eigenform self-observation** | YES (Steps 620-629) | Substrate reads its own edge-count distribution, computes thresholds. | Produces non-zero output (94-99% NEUTRAL, 0-10% AVOID). No performance effect on L1. Untested on L2. |

### Mode Map + Isolated CC (Step 576)
| # | Component | Tested in isolation? | What it does | Extraction test |
|---|-----------|---------------------|-------------|-----------------|
| C23 | **Mode map + CC zone discovery** | YES (Step 576: VC33 5/5) | Frame differencing → pixel mode map → connected component isolation → zone identification → burst navigation. | **HIGHEST EXTRACTION PRIORITY.** Only autonomous multi-game discovery mechanism. Used graph for navigation AFTER discovery, but the DISCOVERY itself (mode map → CC isolation) doesn't require per-(state,action) storage. Test: mode map + CC discovery + 800b navigation (no graph). |

---

## Era 3: Post-Ban (Steps 778-1007, ~230 experiments)

### 800b / 916 Family (~200 experiments)
**Killing finding:** 27+ modifications all degrade LS20. Frozen fixed point. Theorem 4 proves SNR → 0 for FT09/VC33.

| # | Component | Tested in isolation? | What it does | Extraction test |
|---|-----------|---------------------|-------------|-----------------|
| C24 | **Delta-per-action EMA** | Core of 800b | Track EMA of observation change magnitude per action. Softmax for selection. | Already the primary post-ban mechanism. Works on LS20 (state-independent), fails on FT09/VC33 (state-dependent). |
| C25 | **Alpha attention** (prediction-error weighting) | YES (Steps 895-895h) | Weight encoding dimensions by prediction error. Self-modifying encoding (R3). | Confirmed R3. alpha_conc discovers game-relevant dims (FT09 tiles at [60,51,209]). CARRY FORWARD to all future substrates. |
| C26 | **Recurrent h state** (echo-state) | YES (Step 916 vs 955) | Fixed random W_h, tanh nonlinearity. Trajectory discrimination. | +8.5% LS20 standalone, -14% in PRISM (CIFAR interference). Architecture irrelevant (Prop 29: ESN = Hebbian RNN). |
| C27 | **h-novelty transition detection** | YES (Step 994) | Detect game transitions via h-state novelty. Fast adaptation (500 steps). | Best PRISM mechanism: LS20=83.8 at 10K (+25%). Specific to level transitions. |
| C28 | **Forward model W** (linear, delta rule) | YES (Steps 780v5+) | Linear prediction of next observation. D(s) transfer confirmed (cold→warm +73% pred acc). | Prediction transfers. But prediction-based ACTION selection fails (Steps 934-936). W is useful as SIGNAL GENERATOR for alpha, not for action. |

### Hebbian Family (~21 experiments)
**Killing finding:** W_a unreliable (1/10 seeds navigate, Step 953). Positive lock (Prop 30).

| # | Component | Tested in isolation? | What it does | Extraction test |
|---|-----------|---------------------|-------------|-----------------|
| C29 | **Hebbian W_a** | YES (Steps 948-962) | Outer-product learning for action selection from h state. | 1/10 seeds (seed 8 = 96 L1). Unreliable. Positive lock (sigmoid → all positive dots → winner-take-all). Fix: relu(h-0.5) gating (Step 959). |
| C30 | **Sparse gating** (relu threshold) | YES (Step 959) | relu(h - 0.5) makes representations state-dependent. Dissolves positive lock. | Theoretical fix for Prop 30. Never fully tested in a complete substrate. |

### Oscillatory Family (3 experiments)
**Killing finding:** Phase credit can't replace argmax. Random W_in encoding inferior to learned h.

| # | Component | Tested in isolation? | What it does | Extraction test |
|---|-----------|---------------------|-------------|-----------------|
| C31 | **Stuart-Landau dynamics** | NO (always with specific credit mechanisms) | Oscillatory encoding from input-driven oscillators. Phase diversity 1.5-1.8. | 800b + oscillatory features: 9.5/seed vs 72.7 baseline. 87% gap from random W_in. Could work if W_in is LEARNED. |

### Attention-Trajectory Family (1 experiment)
**Status:** ALIVE (1/20). Step 1007 = 0/10 all games. Bootstrap problem.

| # | Component | Tested in isolation? | What it does | Extraction test |
|---|-----------|---------------------|-------------|-----------------|
| C32 | **Append-only trajectory buffer** | NO (first test) | Store (enc, action, delta) triples. Time-indexed, not state-indexed. | Graph-ban safe. Buffer grows with time, not with state space. U3/U17 compliant. |
| C33 | **Attention retrieval** (softmax q@K^T @ V) | NO (first test) | State-conditioned information retrieval. Bypasses Theorem 4. | The mechanism that should provide temporal credit for FT09/VC33. Failed on first test — bootstrap problem. Needs fixing, not killing. |

---

## Cross-Cutting Analysis

### Components that appear in 2+ families (extraction candidates):

| Component | Families | Status |
|-----------|---------|--------|
| Centered encoding (x - mean) | Codebook, LSH, 800b/916 | U16 validated. CARRY FORWARD. |
| Novelty-triggered growth | Codebook (spawn), LSH (edge growth), Recode (split) | U3/U17 mechanism. Different implementations, same principle. |
| Transition detection | Recode (split trigger), 674 (refinement trigger), Step 576 (CC discovery) | Detection without storage is graph-ban safe. |
| Argmin / least-visited selection | Graph (per-cell), 800b (per-action global) | Global version extracted. Per-state version banned. |
| Prediction error as attention signal | 916 (alpha), Codebook (spawn threshold) | Alpha extracted and confirmed R3. |
| Recurrent state | Hebbian RNN, ESN (916), Oscillatory | Architecture irrelevant (Prop 29). h state WORKS. |
| Self-observation | Eigenform (Step 620), Recode (transition stats) | Signal present but unexploited for L1. |

### Components that address Problem 2 (FT09/VC33 action-space discovery):

| Priority | Component | Source | Why |
|----------|-----------|--------|-----|
| **1** | **C23: Mode map + CC zone discovery** | Step 576 | Only autonomous VC33 solve. Discovery mechanism may be graph-ban safe. |
| **2** | **C33: Attention retrieval** | Step 1007 | Bypasses Theorem 4. State-conditioned credit for sequences. Broken — needs fixing. |
| **3** | **C7: Change-rate detection** | Step 400 | Found LS20 sprite region autonomously. Never combined with post-ban navigation. |
| **4** | **C15: Transition detection without graph** | Step 674 | Detection signal exists without storing transitions. Signal for encoding refinement. |
| **5** | **C30: Sparse gating** (relu threshold) | Step 959 | Dissolves positive lock → state-dependent representations → state-conditioned actions without per-state storage. |

---

## Anti-Collapse Mechanisms (reference, not components)

| Mechanism | Source | How it prevents collapse | Relevance |
|-----------|--------|------------------------|-----------|
| SIGReg | LeWorldModel (LeCun 2026) | Gaussian-distributed embeddings via covariance + mean loss | Prevents representation collapse in JEPA |
| Turrigiano homeostatic scaling | Biology (Turrigiano 2008) | Neuron adjusts all synaptic weights to maintain target firing rate | Prevents runaway excitation/silence |
| BCM sliding threshold | Biology (BCM 1982) | Plasticity threshold slides with postsynaptic activity history | Prevents winner-take-all, enables selectivity |
| FSQ (Finite Scalar Quantization) | Genesis project (Feb 2026) | Bounded discrete latents, no codebook, collapse impossible by construction | Architectural anti-collapse (vs regularizer-based). Straight-through estimator for gradients. |

All address the same problem: unconstrained dynamics drift to degenerate fixed points. Our Hebbian positive lock (Prop 30) and SSM action-blind attractor are instances. Biology uses all three biological mechanisms simultaneously.

---

## Budget Audit Per Banned Family

| Family | Experiments | Budget (total steps run) | Key result | Under-explored? |
|--------|-----------|-------------------------|------------|-----------------|
| **Codebook** | ~435 | Millions | LVQ characterized. 94.48% P-MNIST. All 3 games L1. | NO — exhaustively mapped |
| **Graph** | ~80 | ~500K+ | Negative transfer. Visit counts. Edge enrichment. | Moderate — 80 exp but narrow focus |
| **LSH** | ~80 | ~500K+ | 20/20 LS20. 36.2% self-labels. | Moderate |
| **Recode** | ~33 | ~200K | 5/5 L1 with self-refinement. K confound. | YES — clean test never done |
| **Reservoir** | ~20 | ~100K | Rank-1 collapse. Hebbian diverges. | YES — only 20 exp, killed early |
| **Hebbian RNN** | ~6 | ~60K | 1/10 seeds navigate. Positive lock. | **YES — severely under-explored** |
| **Hebbian W_a** | ~15 | ~150K | Unreliable. 800b confirmed unique. | Moderate |
| **916-augmentation** | ~25 | ~250K | ALL modifications degrade. Frozen. | NO — exhaustively mapped |
| **Prediction selectors** | ~10 | ~100K | ALL prediction-based selectors fail. | Moderate |
| **Adaptive-eta** | ~8 | ~80K | Only safe DoF in 916. | Moderate |
| **GFS** | ~3 | ~30K | Dynamic dims break alpha. | **YES — killed after 3 exp** |
| **Obs preprocessing** | ~2 | ~20K | Replacement destroys position info. | **YES — killed after 2 exp** |
| **Oscillatory** | ~3 | ~30K | Phase credit fails. Random W_in bad. | **YES — killed after 3 exp** |
| **Multi-horizon** | ~2 | ~20K | Gradient overflow. | **YES — killed after 2 exp** |
| **Attention-trajectory** | ~1 | ~10K | Bootstrap failure. | **YES — only 1 experiment** |

**Severely under-explored families:** Hebbian RNN (6), GFS (3), obs preprocessing (2), oscillatory (3), multi-horizon (2), attention-trajectory (1). These got killed fast — possibly too fast for the extraction protocol to evaluate individual components.

---

## Benchmark System Changes

| Step range | Game versions | PRISM? | Notes |
|-----------|--------------|--------|-------|
| 1-416 | Original (pre-change) | No | Codebook era. All prescribed encodings. |
| 417-689 | Original | No | LSH/Graph era. |
| 690-712 | LS20/9607627b, FT09/0d8bbf25 | No | Current versions verified. 674 characterization. |
| 713-777 | Current | No | Action discovery, graph ban, Table 1+2. |
| 778-963 | Current | No (should have been) | Post-ban. PRISM existed but wasn't used. |
| 964-1005 | Current | Partial (some through PRISM runner) | PRISM tests but not all through run_experiment.py. |
| 1006-1007 | Current | **YES** (run_experiment.py enforced) | First proper PRISM experiments. |

**Key gap:** Steps 778-963 (~186 experiments) were supposed to go through PRISM but didn't. Results are valid per-game but PRISM interactions weren't tested. The infrastructure overhaul (2026-03-24) fixed this going forward.

