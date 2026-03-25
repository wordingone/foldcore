# Extraction Experiment Specs

*Generated from COMPONENT_CATALOG.md compression, 2026-03-24. All experiments through PRISM (run_experiment.py). All three games. One variable per experiment.*

**Design principle:** Every spec targets FT09/VC33 signal. LS20 is monitored for regression only (kill if LS20 < baseline chain_score). The 230 post-ban experiments over-indexed on LS20. These specs correct the skew.

---

## Extraction 1: CC Zone Discovery + 800b (Component C23)

**Source:** Step 576 (VC33 5/5, autonomous zone discovery)
**Killing finding to test against:** Graph ban — negative transfer (cold > warm, Step 776)
**Hypothesis:** The CC zone discovery mechanism (mode map → connected components → zone identification) is an ENCODING component, not an action-selection component. It should survive the graph ban because it doesn't create per-(state,action) data — it discovers ACTION SPACES from frame analysis.

**Substrate:**
```python
# Phase 1: Random exploration (1000 steps)
# Accumulate frame differences → pixel mode map
# Isolated CC analysis on mode map → identify zones/click regions
# Phase 2: 800b navigation with discovered action space
# Actions = one per zone/region (not 68 raw actions)
```

**R3 hypothesis:** The zone discovery modifies the ACTION SPACE (how many actions, what they mean) from game interaction. This is R3 on the action representation — the substrate discovers which actions exist.

**Kill criteria:**
- FT09 or VC33 L1 > 0 on ANY seed = ALIVE (unprecedented post-ban)
- LS20 L1 < baseline chain_score = REGRESSION KILL
- Chain kill criterion applies

**Budget:** 10K steps per game. 5 min cap.
**Seeds:** 10
**Family:** extraction-cc (new)

---

## Extraction 2: Change-Rate Encoding + Alpha + 800b (Component C7)

**Source:** Step 400 (change-rate detection finds LS20 sprite region)
**Killing finding to test against:** Codebook ban — LVQ characterization (Step 416)
**Hypothesis:** Change-rate detection is NOT LVQ. It's temporal statistics on raw pixels — no cosine, no attract, no codebook. It could provide game-adaptive encoding that helps alpha concentrate on action-relevant regions.

**Substrate:**
```python
# Encoding: raw obs → change_rate per dim (EMA of |obs_t - obs_{t-1}|)
# Alpha: prediction-error weighting on change_rate-MASKED encoding
# Action: 800b delta-per-action on alpha-weighted change-rate encoding
# change_rate identifies which dims are dynamic (game-relevant)
# alpha identifies which dynamic dims are PREDICTIVE
# Two-layer attention: temporal (change_rate) × predictive (alpha)
```

**R3 hypothesis:** Change-rate + alpha provides TWO levels of self-modifying encoding: (1) which dims change (temporal), (2) which changing dims matter (predictive). Deeper R3 than alpha alone.

**Kill criteria:**
- FT09 alpha_conc on change-rate dims > alpha_conc on raw dims = signal improvement
- Any game L1 > baseline chain_score = ALIVE
- LS20 < baseline = regression check

**Budget:** 10K steps per game. 5 min cap.
**Seeds:** 10
**Family:** extraction-changerate (new)

---

## Extraction 3: Transition Detection Without Graph (Component C15)

**Source:** Step 674 (transition-triggered refinement, 20/20 LS20)
**Killing finding to test against:** Graph ban — negative transfer (Step 776)
**Hypothesis:** Transition detection (observing that the same action from similar states leads to different outcomes) is INFORMATION, not STORAGE. The graph ban prohibits storing per-(state,action) data. But detecting inconsistency doesn't require permanent storage — a sliding window of recent (state, action, outcome) triples can detect inconsistency without building a persistent graph.

**Substrate:**
```python
# Rolling buffer of last K (enc, action, next_enc) triples
# For each new triple: check if similar enc + same action → different next_enc
# If yes: mark this encoding region as "aliased" (needs finer discrimination)
# Use aliased signal to sharpen alpha (increase attention weight on dims that differ)
# Action: 800b on sharpened encoding
# Buffer is FIFO — no persistent per-state storage
```

**R3 hypothesis:** Transition inconsistency drives encoding refinement. The substrate learns WHERE in observation space it needs finer discrimination — R3 on the encoding resolution.

**Kill criteria:**
- Aliased detection fires on FT09/VC33 = mechanism works on hard games
- FT09 or VC33 L1 > 0 = ALIVE
- Buffer memory exceeds 5MB = efficiency kill

**Budget:** 10K steps per game. 5 min cap.
**Seeds:** 10
**Family:** extraction-transition (new)

---

## Extraction 4: Sparse Gating for State-Dependent Actions (Component C30)

**Source:** Step 959 (relu threshold dissolves positive lock, Prop 30)
**Killing finding to test against:** Hebbian W_a unreliability (1/10 seeds, Step 953)
**Hypothesis:** The positive lock (Prop 30) is THE reason Hebbian action selection fails — sigmoid h ∈ [0,1] makes all dot products positive → first action always wins. Sparse gating (relu(h - threshold)) creates representations where most dimensions are zero → dot products can be negative → state-dependent action selection becomes possible.

**Substrate:**
```python
# Encoding: 916 base (alpha + running-mean centering)
# State: recurrent h (ESN, 64D)
# Gating: h_gated = relu(h - 0.5)  # sparse, ~10% active
# Action: W_a @ h_gated (Hebbian outer-product learning)
# Fallback: if max(W_a @ h_gated) < threshold → 800b action
# Hebbian update: W_a += eta * outer(h_gated, action_onehot) * reward_proxy
```

**R3 hypothesis:** Sparse gating makes h state-dependent → W_a learns state-conditional action preferences → position-aware action selection without per-state memory. This predicts an IMPOSSIBLE result per the 800b kill register: "position-aware without per-state memory."

**Kill criteria:**
- FT09 action selection non-uniform (not all 68 actions equally selected) = mechanism differentiates
- More than 1/10 seeds navigate LS20 = robust (Hebbian was 1/10)
- FT09 or VC33 L1 > 0 = ALIVE

**Budget:** 10K steps per game. 5 min cap.
**Seeds:** 10
**Family:** extraction-sparsegate (new)

---

## Extraction 5: Attention-Trajectory v2 — Fixed Bootstrap (Component C33)

**Source:** Step 1007 (attention-trajectory, 0/10 all games, bootstrap failure)
**Killing finding to test against:** Step 1007 bootstrap problem
**Hypothesis:** Step 1007 failed because: (1) running_mean alpha=0.01 too slow, (2) buffer accumulates noise early, (3) O(buffer) attention too slow. Fix: (1) alpha=0.1 for faster encoding adaptation, (2) buffer_size=200 (not 2000) to keep early noise minimal, (3) warm-start with 100 random steps before attention kicks in.

**Substrate:**
```python
# Same as Step 1007 but:
# RUNNING_MEAN_ALPHA = 0.1 (10x faster)
# MAX_BUFFER = 200 (10x smaller)
# WARMUP = 100 (pure random exploration to populate buffer)
# TEMP = 0.1 (sharper attention, not 1.0)
# Optional: 800b fallback when buffer < 50 entries
```

**R3 hypothesis:** Same as 1007 — state-conditioned retrieval provides temporal credit for sequential actions. The fix addresses implementation, not theory.

**Kill criteria:**
- LS20 L1 > 0 on any seed = bootstrap fixed
- FT09 or VC33 L1 > 0 = unprecedented
- Runtime < 5 min per game = efficiency check

**Budget:** 10K steps per game. 5 min cap.
**Seeds:** 10
**Family:** attention-trajectory (experiment 2/20)

---

## Unconstrained Diagnostic

**Purpose:** Calibration ceiling. All bans lifted, all constraints suspended. Best known techniques.

**Substrate:**
```python
# Graph: per-(cell, action) edge counts. Visit-frequency argmin.
# Encoding: avgpool16 + centered + running-mean
# LS20: 4 directional actions. Graph + 674 refinement.
# FT09: 69 actions (64 grid + 5 simple). Graph + argmin.
# VC33: Mode map + CC discovery (Step 576) → 3-zone graph + argmin.
# CIFAR: LSH k=16 self-labels.
# Domain separation: per-domain centering.
# Action space: auto-detect via mode map (click game vs direction game).
#   If frame barely changes between actions → click game → expand action space
#   If frame changes significantly → direction game → keep base actions
```

**Expected:** LS20 20/20, FT09 20/20, VC33 5/5, CIFAR ~36%
**Through PRISM:** Randomized game order. Chain score as metric.
**Kill criteria:** None — this is calibration, not search.
**Budget:** 120K steps per game (uncapped for ceiling measurement).
**Seeds:** 20

---

## Experiment Priority Order

1. **Extraction 1 (CC zone discovery)** — highest expected information gain. If CC discovery + 800b solves VC33 or FT09 post-ban, it changes everything.
2. **Extraction 5 (Attention v2)** — fixes known bug in Step 1007. Fast iteration.
3. **Extraction 4 (Sparse gating)** — predicts impossible result per kill register. Theoretically strongest.
4. **Extraction 2 (Change-rate encoding)** — two-layer attention. Novel combination.
5. **Extraction 3 (Transition detection)** — graph mechanism without graph storage. Conceptually clean.
6. **Unconstrained diagnostic** — calibration. Run after extractions establish baseline.
