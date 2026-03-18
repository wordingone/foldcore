# ANIMA/Search Connection — Finding 8 Analysis

*2026-03-18. Post-audit analysis mapping ANIMA's separation theorem onto The Search's constraint map.*

## ANIMA's Core Distinction

- **Reactive:** θ_t = G(x_t) — computational parameters depend only on current input
- **State-conditioned:** θ_t = G(φ(s_{t-1}), x_t) — parameters depend on accumulated state + input

**Theorem:** For k-way routing, reactive requires Ω(k) state, state-conditioned requires O(log k) state. Exponential separation.

## Mapping The Search's Substrates

### process_novelty() — REACTIVE

The operations applied at each step are fixed code:
1. Cosine similarity: V @ x (fixed operation)
2. Winner selection: argmin class scores (fixed operation)
3. Attract: V[w] += alpha * (x - V[w]) (fixed operation)
4. Spawn: if sim < thresh, append (fixed operation)

State V is consulted as a lookup table, but the RULES never change based on state. There is no "if the codebook looks like X, use operation A; if Y, use operation B." This is reactive under ANIMA's definition.

### ReadIsWrite — CLOSER TO STATE-CONDITIONED

- Softmax weights over codebook entries depend on the ENTIRE state V (through Gram similarities)
- Update distributes to ALL entries proportionally to their match
- The weighting scheme is a function of global state structure, not just current input match

This is closer to state-conditioned: the effective parameters (which entries get how much update) depend on φ(V), not just x_t.

## Evidence Supporting the Mapping

**Codebook size:** ReadIsWrite achieves comparable exploration with HALF the codebook (4929 vs 8054, Step 418f). ANIMA predicts reactive systems need more state for the same routing capability. CONSISTENT.

**Classification:** ReadIsWrite 91.84% with fewer U elements (2U vs 5-7U) while process_novelty needs 94.48% with more frozen structure. The reactive system compensates with larger state (more codebook entries) and more frozen operations (class scoring, argmin).

## The Deep Connection: R3 = Beyond State-Conditioned

ANIMA distinguishes reactive vs state-conditioned: do operations DEPEND on state?

R3 is STRONGER: do operations get MODIFIED by state?

| Level | Operations | State relationship | Example |
|-------|-----------|-------------------|---------|
| Reactive | Fixed | Don't depend on state | process_novelty: always cosine, always argmin |
| State-conditioned | Parameterized | Depend on state | ReadIsWrite: softmax weights from V |
| R3-passing | Self-modifying | ARE the state | ??? |

The frozen frame in process_novelty IS the "reactive" constraint. Cosine, argmin, top-K, attract, spawn are fixed operations that don't change based on what the substrate has learned. An R3-passing substrate would need operation selection to be state-conditioned — choosing WHICH similarity metric, WHICH selection rule, WHICH growth policy based on accumulated state.

## Predictions

If ANIMA's theorem applies to The Search:

1. **Reactive substrates (process_novelty, SelfRef) need large codebooks** because they encode routing information in state. The ~20K codebook for LS20 is the reactive system's Ω(k) cost for navigating a game with effective k routing decisions.

2. **State-conditioned substrates should need exponentially smaller state** for the same routing. ReadIsWrite's half-codebook result is early evidence. A fully state-conditioned substrate might navigate with O(log k) codebook entries.

3. **R3 requires the level BEYOND state-conditioning.** Not just "parameters depend on state" but "the operation space itself is state-derived." This might be why R3 is the binding constraint — it demands a qualitative jump past what ANIMA formalizes.

4. **U24 (argmin ≠ argmax) is a REACTIVE constraint.** In a reactive system, the action selection rule is fixed — you must choose argmin OR argmax. In a state-conditioned system, the action selection could depend on state: explore when the codebook is young (small, high novelty), exploit when mature (large, saturated). U24 might dissolve at the state-conditioned level.

## Implication for the Search Direction

The audit's Observation C (U24 as false axis) connects here. If we build a substrate where action selection IS state-conditioned — choosing explore vs exploit based on codebook maturity — then U24 stops being a wall. The substrate navigates early (argmin-like when codebook is small) and classifies late (argmax-like when codebook is large).

The meta-protocol for encoding discovery (Step 414) is ALREADY state-conditioned: "monitor codebook health → decide whether to keep encoding → switch." The question is whether this meta-protocol can be internalized into the substrate's step-level operation.

## Open Questions

1. Can we formalize The Search's routing requirements as k-way routing to apply the theorem directly?
2. Does ReadIsWrite's half-codebook result hold for navigation (not just classification)?
3. What would a substrate look like that selects between cosine/L1/dot product based on its own state?
