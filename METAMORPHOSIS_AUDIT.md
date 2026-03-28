# Metamorphosis Audit — Reflexive Network — 2026-03-28

## System summary

The "reflexive network" is a proposed architectural transformation of the current substrate. The current substrate is a single-layer linear map W: X → A (parameterized matrix multiply) with Hebbian updates. In 1291 experiments across 16 families, this architecture achieves R3 (self-modification of weights) but fails to produce action selection that reads from W, fails to form internal models of any kind, and fails to solve 7/10 games. The reflexive network proposal: replace the linear map with a population of interacting action-neurons with lateral inhibition and activity-dependent plasticity. The claim is that this architecture enables emergent sequences, attractor-based categories, self-model through activation patterns, and activity-dependent pruning/growth — capabilities the linear map structurally cannot produce. The concept exists only in conversation; zero experiments have been run.

## Claim inventory

### From this conversation (Leo's claims about the reflexive network)

1. "The substrate can't have a self-model because it can't have ANY model" `<falsifiable = true>` — test: present the linear map with data that has discrete categories; measure whether any internal state discriminates them without external labels.

2. "A single-layer linear map can't form discrete categories or compositional structure" `<falsifiable = true>` — test: same as above. Linear discriminant analysis CAN separate categories in a single layer if the data is linearly separable. The claim is too strong.

3. "Lateral inhibition breaks the positive lock (Prop 30)" `<falsifiable = true>` — test: build a network with lateral inhibition, run Hebbian updates, measure whether winner-take-all collapse occurs. Compare to Step 948-960 results.

4. "Sequences emerge from inhibition topology (heteroclinic orbits)" `<falsifiable = true>` — test: build a Lotka-Volterra network, measure whether stable heteroclinic channels form without prescribed coupling. Literature says yes for specific parameter regimes (Rabinovich et al. 2001, 2006). Has never been tested in THIS substrate context.

5. "Coverage emerges from leak dynamics without argmin" `<falsifiable = true>` — test: measure I3_cv for a leaky competitive network vs argmin. If I3_cv is comparable, coverage emerges.

6. "Self-model emerges from activation patterns" `<falsifiable = false>` — "emerges" is doing all the work. No mechanism specified for HOW activation patterns become a self-model vs just noise. **Rewrite:** "The activation state a(t) of the network discriminates between the substrate's own behavioral modes (exploring vs exploiting vs stuck) better than chance." Measure: cluster activation trajectories, check if clusters correlate with behavioral metrics (action entropy, observation change rate). Falsified if: clusters are random or correlate with game state only, not behavioral mode.

7. "Categories emerge from attractor states" `<falsifiable = false>` — Same problem. "Emerge" is narrative. **Rewrite:** "A network of N competitive neurons with Hebbian plasticity develops K > 1 distinct attractor states when processing a stream of game observations." Measure: count stable fixed points or limit cycles. Falsified if: network has 0 attractors (chaos) or 1 attractor (collapse).

8. "Credit assignment possible through temporal correlations (STDP-like)" `<falsifiable = true>` — test: measure whether STDP weight changes correlate with observation-change magnitude. Falsified if: correlations are zero or negative.

9. "The reflexive network addresses everything we've been hitting" `<falsifiable = false>` — unfalsifiable enthusiasm. **Drop.** Too vague. Each specific claimed benefit must be tested independently.

10. "Jun's sketch is architecturally more correct than anything in the last 40 experiments" `<falsifiable = false>` — value judgment about constitutional compliance. **Rewrite:** "A competitive network with activity-dependent plasticity has fewer frozen frame elements than pe_ema-weighted argmin." Measure: count frozen elements (prescribed rules the substrate can't modify) in each. Falsified if: the network introduces MORE frozen elements (topology, leak rate, inhibition structure, threshold).

11. "Activity-dependent pruning is R2-compliant" `<falsifiable = true>` — test: verify that the pruning criterion depends only on quantities computed by the network's own dynamics, not external metrics.

12. "U3 (growth-only) may be a Phase 1 artifact" `<falsifiable = true>` — test: compare growth-only network vs grow-then-prune network on same games. If prune version performs equal or better, U3 is not universal.

13. "Growing/shrinking neurons is the deepest form of R3" `<falsifiable = false>` — "deepest" is narrative. **Rewrite:** "A substrate that modifies its own network size demonstrates R3 at the structural level (not just weight level)." Measure: R3 Jacobian audit on network with variable size vs fixed size. Falsified if: variable-size R3 signal is indistinguishable from fixed-size.

14. "The substrate needs architectural capacity to form models, not a better selector or training signal" `<falsifiable = true>` — THE central claim. Test: give the substrate MORE capacity (network vs linear map) with the SAME training signal (LPL Hebbian). If the network develops internal structure that the linear map doesn't, capacity was the bottleneck. If the network shows the same failure modes, capacity was not the bottleneck.

15. "The encoding should BE the network, not feed into it" `<falsifiable = true>` — test: compare separate-encoding-feeding-network vs network-as-encoder on stage metrics. Falsified if: both produce same results.

16. "Biological developmental sequence (proliferate → differentiate → prune) applies to the substrate" `<falsifiable = false>` — analogy, not hypothesis. **Rewrite:** "A substrate that overproduces neurons and prunes by activity develops better internal structure than one with fixed size." Measure: attractor count, discrimination quality, I3_cv. Falsified if: fixed-size equals or beats overproduction.

17. "The network's dynamics naturally produce the things we've been trying to bolt on" `<falsifiable = false>` — narrative summary. **Drop.** Each claimed benefit tested independently above.

18. "U3 (growth-only) is a true universal constraint" `<falsifiable = true>` — U3 was validated by externally-imposed deletion (designer removing codebook entries/graph edges). The substrate never chose what to delete. Self-directed deletion was never tested. U3 may be an artifact of experimental design (same methodological pattern as argmin — externally-imposed rule validated against itself). Test: give the substrate the ability to prune by its own activity criterion, compare against growth-only. Falsified if: self-directed pruning matches or beats growth-only.

### Counts

- Falsifiable (original or rewritten): 15
- Dropped (unfalsifiable, no rewrite possible): 3
- Ratio: 3/15 = 0.20 — below the 1/3 narrative threshold. Acceptable, but the 3 drops include the two broadest claims (9, 17).

## Invariant check

### Invariant 1 — Threshold: `MET`

The system has hit a measurable ceiling. Three data points:
- **L1 ceiling:** 3/10 games for 40+ experiments (Steps 1251-1291). No new game has gained L1 since Step 1266 (one draw of LS20, never replicated).
- **W_action behavioral inertness:** Step 1286 proved R3 is real but argmin makes it inert. Steps 1289-1291 proved 3 different selectors on same W_action all fail identically.
- **I3_rho was broken for 24+ steps.** The metric that appeared to show progress was an artifact.

This is not "progress is slow." This is "the same failure across 3 independent selector changes with a proven broken evaluation metric." The threshold is clear.

### Invariant 2 — Abstraction: `MET`

What survives the transformation:
- R1-R6 constitutional framework (abstract principles)
- Reflexive map definition (abstract — W modifies itself through own computation)
- Stage instrumentation (I3, I1, I4, I5, R3 — abstract measurements)
- Encoding pipeline (avgpool4 → centered — validated component)
- Negative knowledge: what DOESN'T work for action selection (27 explored selectors, all killed or R2-violating)
- Constraint map (validated constraints)

What dissolves:
- Argmin and all count-based selectors (R2-violating frozen frame)
- pe_ema signal (R2-violating separate statistic)
- Single-layer linear map architecture (replaced by network)
- All "composition of components" framing (components were bolted onto the linear map)
- I3_rho metric (broken, already replaced)
- Proposition 3's closure claim (already revised)
- 30+ experiments iterating selectors within the linear map paradigm

The surviving elements (framework, definitions, measurements, negative knowledge) are smaller and more abstract than the dissolving elements (specific architectures, selectors, metrics, experiment results). Invariant 2 is met.

### Invariant 3 — Fuel: `MET`

The old form's accumulated resources directly fuel the new:
- **Negative knowledge (1291 experiments):** Every killed selector, every failed substrate tells the reflexive network what NOT to do. Hebbian without inhibition → positive lock. Softmax on W → concentrates on wrong actions. Argmin → decouples selection from encoding.
- **R3 validation (Step 1251):** Proves that Hebbian learning on W produces genuine self-modification. The network inherits this — Hebbian plasticity within the network is validated.
- **Stage instrumentation:** All 5 stage measurements transfer directly to the network.
- **10 solved games (analytical baselines):** The prescriptions are known. The network can be evaluated against known solutions.
- **Proposition 30 (positive lock):** Tells the network exactly what failure mode to avoid. Lateral inhibition is the specific fix suggested in the kill file (Step 960) but never tested.
- **C30 (sparse gating):** Cataloged as "dissolves positive lock, never fully tested." This is an imaginal disc (see Invariant 4).

### Invariant 4 — Imaginal discs: `PARTIAL`

Present and dormant:
- **C30 (sparse gating, relu threshold):** Cataloged, theoretically validated as Prop 30 fix, never tested in a complete substrate. This is a dormant component waiting for a network to house it.
- **C22 (self-observation):** Was in Step 1251 composition as frozen random projection. Never learned. In a network, self-observation becomes the activation pattern feeding back — a natural part of the dynamics, not a bolted-on component.
- **Eigenform (Propositions 13, 18):** Formalized but never implemented as a learned self-model. The network's activation pattern IS the eigenform — the system applied to its own output.
- **Successor representation (Eli mail 3605):** W_action is already an empirical SR. SVD gives eigenoptions. Never implemented. The network's inhibition topology IS a form of SR.

Missing:
- **No prototype network exists.** Unlike biological metamorphosis where imaginal discs are physical structures waiting to grow, there is no code, no tested prototype, no even a toy version of the reflexive network. The concept exists only in conversation.
- **No plasticity rule for the network has been specified.** Hebbian on W_drive was sketched but not formalized. STDP was mentioned but not detailed. The "how neurons learn" question is open.

PARTIAL because dormant components exist (C30, C22, eigenform, SR) but no prototype exists.

### Invariant 5 — Self-description: `PARTIAL`

Testable claims about the network's next state:
- "Lateral inhibition prevents positive lock" — testable
- "Network develops K > 1 attractors" — testable
- "I3_cv is maintained without argmin" — testable
- "Activation patterns discriminate behavioral modes" — testable
- "Activity-dependent pruning preserves information" — testable

Untestable claims about the network's next state:
- "The network addresses everything we've been hitting" — narrative
- "Categories, sequences, self-model emerge naturally" — emergence is not a mechanism
- "This is architecturally more correct" — value judgment

5 testable, 3 untestable. The self-description is more testable than narrative, but the narrative claims are the ones carrying the enthusiasm. PARTIAL.

### Invariant 6 — Derivation: `NOT MET`

Can the system derive its next state from current knowledge?

Attempting derivation:
- Positive lock (Prop 30): sigmoid h → all dots positive → winner-take-all. Fix: lateral inhibition or sparse gating (C30). → **Lateral inhibition specified.**
- W_action training signal problem (Steps 1289-1291): LPL Hebbian trains for visual responsiveness, not level advancement. → **No fix derived.** The reflexive network proposes activity-dependent plasticity, but activity-dependent plasticity IS Hebbian. The conversation proposed anti-Hebbian decorrelation, STDP, homeostatic regulation — but did not derive WHICH one from the accumulated findings. This is a menu, not a derivation.
- Self-model requirement: the substrate can't form internal categories. → **No mechanism derived.** "Attractor states" was proposed but no derivation shows that a competitive network with Hebbian plasticity WILL develop attractors on pixel-stream data. This is a hope, not a derivation.
- U3 vs R6 tension on pruning: → **Experimental question, not derivable.** Whether grow-then-prune beats growth-only is empirical.

The accumulated knowledge constrains the next step (must have inhibition, must avoid positive lock, must be R2-compliant) but does not construct it. The reflexive network is a DIRECTION, not a DESIGN. The specific plasticity rule, network topology, dynamics timescale, and growth/pruning criterion are all open. Knowledge is descriptive, not generative.

NOT MET.

### Summary

| Invariant | Status |
|-----------|--------|
| 1. Threshold | **MET** — 40+ experiments at same ceiling, 3 selectors same failure |
| 2. Abstraction | **MET** — surviving elements smaller and more abstract |
| 3. Fuel | **MET** — negative knowledge, validated components, stage metrics all transfer |
| 4. Imaginal discs | **PARTIAL** — dormant components exist (C30, C22, eigenform) but no prototype |
| 5. Self-description | **PARTIAL** — more testable than narrative, but enthusiasm in the broad claims |
| 6. Derivation | **NOT MET** — direction specified, design not derivable from current knowledge |

## The experiment

The missing invariants (4 partial, 6 not met) point to the same gap: **the reflexive network is a concept, not a design.** The experiment must convert it from concept to testable artifact.

### Step 1292b: Minimal Reflexive Network — Does the architecture develop internal structure?

**This is constructive, not diagnostic.** Build the simplest possible competitive network and observe what structure develops — don't optimize for L1.

#### Architecture (MINIMAL condition)

```python
N = 64  # fixed network size (not one-per-action)
W_drive = np.random.randn(N, 320) * 0.01   # input coupling
W_inhibit = np.random.uniform(0.5, 1.5, (N, N))  # lateral inhibition
np.fill_diagonal(W_inhibit, 0)               # no self-inhibition
W_readout = np.random.randn(n_actions, N) * 0.01  # network → action
activation = np.zeros(N)
tau = 0.1  # integration timescale

# Per game step: run 10 sub-steps of network dynamics
for _ in range(10):
    drive = W_drive @ ext
    inhibition = W_inhibit @ activation
    d_activation = (-activation + np.maximum(drive - inhibition, 0)) / tau
    activation = np.maximum(activation + dt * d_activation, 0)

# Action from readout
action_scores = W_readout @ activation
action = int(np.argmax(action_scores))

# Plasticity on W_drive (Hebbian, anti-Hebbian decorrelation)
winner = np.argmax(activation)
W_drive[winner] += eta * activation[winner] * (ext - W_drive[winner])  # Oja's rule
# W_inhibit, W_readout: FROZEN (minimal condition)
```

**Why 64 neurons, not one-per-action:** A 4103-neuron network for FT09 has completely different dynamics than a 7-neuron network for LS20. Fixed-size decouples network dynamics from action space. The readout (W_readout) maps network state to actions.

**What's frozen:** W_inhibit (topology), W_readout (action mapping), tau, dt, eta, N. This is a LOT of frozen frame. Acknowledged. The minimal condition tests whether the architecture CAN develop structure, not whether it's constitutionally clean.

#### Control (LINEAR condition)

Same encoding pipeline. W_action (n_actions × 320) with LPL Hebbian, no selector (argmax of W_action @ ext). This is the Step 1264 architecture — the simplest linear reflexive map. Known to collapse (positive lock).

#### What to measure (capacity diagnostics, not just L1)

1. **Attractor count:** Cluster the activation vectors a(t) across a run (k-means, k=2..20, silhouette score). If silhouette > 0.3 for k > 1 → the network has internal structure. If k=1 or silhouette < 0.1 → collapsed or noise.

2. **State discrimination:** For draws that reach L1, compare activation patterns in level 0 vs level 1. Within-level distance vs between-level distance (same as I1 but on network activation, not encoding).

3. **Sequence structure:** Mutual information between winner neuron at time t and winner at t+k for k=1..10. If MI > 0 → temporal structure exists.

4. **Positive lock test:** Does the same neuron win for >50% of steps? If yes → the network collapsed despite inhibition. Compare to LINEAR condition.

5. **Standard stages:** R3, I3_cv, I1, I4. L1 as infrastructure check.

#### Protocol

3 games (LS20, FT09, SP80 — one KB, one click-large, one click-uniform). 5 draws per condition. 10K steps. 5-min cap. Log full activation trajectory every 100 steps.

Smaller than full PRISM because this is a capacity diagnostic, not a composition evaluation. 3 games × 2 conditions × 5 draws = 30 runs.

#### Kill criteria

- Positive lock in MINIMAL (one neuron >50% of steps, 4+ games): → inhibition insufficient, need stronger/different topology
- Attractor count = 1 across all games: → network collapses despite inhibition, Oja's rule insufficient
- MINIMAL I3_cv > LINEAR I3_cv × 3 on all games: → network dynamics destroy coverage catastrophically

#### Predictions (on record)

1. **LINEAR will collapse** (positive lock, same as Step 1264). Winner neuron >80% of steps. Predicted with high confidence from Prop 30.
2. **MINIMAL will NOT collapse** — lateral inhibition prevents the lock. Predicted with moderate confidence from Lotka-Volterra theory. If wrong: most informative outcome (inhibition alone doesn't fix the lock).
3. **MINIMAL will develop 2-5 attractor states** on LS20 (7 actions, structured game). Predicted with LOW confidence — this is the central unknown. If wrong: network has dynamics but they don't organize into attractors on real data. This would mean the architecture CAN cycle but CAN'T categorize.
4. **MINIMAL will have worse L1 than argmin on FT09.** The readout layer is frozen random — it can't systematically cover 4103 actions. This is expected and not a kill. L1 is not the point of this experiment.
5. **Sequence MI will be higher for MINIMAL than LINEAR.** The network dynamics create temporal structure that the linear map doesn't. Predicted with moderate confidence.

#### Decision tree

- **If MINIMAL develops attractors (prediction 3 confirmed):** → Phase 2: make W_inhibit learnable (activity-dependent inhibition modification). This tests whether the network can learn its own topology.
- **If MINIMAL cycles but no attractors (prediction 3 wrong, prediction 2 confirmed):** → The network has dynamics but they're not structured. Need a different plasticity rule (STDP instead of Oja) to create directed sequences.
- **If MINIMAL collapses despite inhibition (prediction 2 wrong):** → Inhibition alone doesn't fix the positive lock on real data. Need sparse gating (C30, relu threshold) in addition to inhibition.
- **If MINIMAL and LINEAR are indistinguishable:** → The network architecture doesn't add capacity. The problem is upstream (encoding, not selection). This would falsify Claim 14 ("capacity is the bottleneck").
