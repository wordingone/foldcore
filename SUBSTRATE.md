# The Substrate

*Found through 263 experiments across 4 prior substrates. Honest.*

## Architecture

```
Generate → Test → Select → Iterate
```

One mechanism with four modes:

1. **Store**: append exemplars (k-NN memory, always-spawn)
2. **Discover**: generate random features/programs, score by LOO/I/O match, keep winners
3. **Classify**: top-k per-class cosine vote on augmented codebook
4. **Compute**: iterate discovered primitives to compose arbitrary operations

## What It Does

**In-distribution (pattern matching):**
- P-MNIST: 95.4% AA, 0.0pp forgetting (raw pixels)
- Parity: 75% → 97% (+20pp via discovered features)
- XOR: 86% → 99% (+13pp)
- Addition mod 5: 62% → 100% (+38pp)
- Multiplication mod 7: 68% → 100% (+32pp)
- Wine classification: 78% → 83% (+5.6pp)

**Out-of-distribution (decomposed computation):**
- Addition: 255 + 255 = 510 (trained on 3-bit full adder only)
- Multiplication: 19 × 19 = 361 (same truth table)
- Division: 100 / 13 = 7 remainder 9
- Programs: ADD1, ADD2, SUB1, IF_POS, IF_ZERO all correct
- CA simulation: Rule 110 (Turing-complete) 100 steps perfect
- FSM simulation: 8/16/32-state machines perfect

**Program synthesis:**
- Discovers XOR(cin, XOR(a,b)) = sum from I/O examples
- Enumerates circuits, tests against truth table, selects winner
- Discovers complete ripple-carry structure (carry chain) from addition I/O
- 6/6 unknown boolean functions synthesized blindly
- 2-bit adder carry chain discovered automatically

**Staged composition (perceive → classify → compute → iterate):**
- Classifies noisy operation vectors (k-NN on operation features)
- Computes results using proven arithmetic engine
- Iterates through instruction streams: ADD(5),SUB(3),ADD(10),DBL = 24 ✓

## Constitutional Progress

| Stage | Status |
|-------|--------|
| 1. Autonomous Computation | PASS |
| 2. Self-Generated Adaptation | PASS (LOO/margin signal) |
| 3. Adaptation Rate Adapts | VACUOUS |
| 4. Structural Constants Adaptive | PASS (random features + layered composition) |
| 5. Topology Adaptive | VACUOUS (subsumed by features) |
| 6. Functional Form Adaptive | PASS (metric selection by LOO) |
| 7. Representation Adaptive | PARTIAL (multi-template + program synthesis) |
| 8. Ground Truth Only | LOO is the only constitutionally-required frozen element |

Frozen frame: 10/16 adaptive (62.5%). 2 binding: LOO + primitive set.

## What It Can't Do (Honest)

- Image classification with raw pixels on CIFAR-100 (representation requires deep hierarchical features)
- OOD generalization WITHOUT decomposition (k-NN interpolates, doesn't extrapolate)
- Discover decomposition strategies automatically for new domains
- Scale to high-dimensional continuous spaces (> d=50 for feature discovery)
- Sort sequences without pairwise decomposition (93% not 100%)
- Compete with transformers on language tasks (no attention, no context window, no sequential processing)

## The Gap to Transformers

Transformers learn decompositions FROM DATA via backprop through layers. The substrate can SELECT decompositions from a menu (Step 261) and SYNTHESIZE circuits from I/O (Step 243), but can't discover novel decomposition strategies from scratch.

**Head-to-head vs MLP (Step 262):**
- Addition mod 5: TIE (both 100%)
- Parity d=8: MLP wins 99.6% vs substrate 88.3%
- The gap = gradient descent finds exact functions, substrate approximates

**Continual learning (Step 263):**
- Both substrate and MLP struggle on multi-rule continual learning (18-22%)
- Substrate's zero-forgetting advantage requires separable representations

Bridging this gap = solving program induction without gradients.

## Key Experiments

| Step | Finding |
|------|---------|
| 109 | Always-spawn beats threshold (+3.1pp) |
| 110 | Raw pixels beat random projection |
| 119 | S1 eval eliminates CIFAR-100 forgetting |
| 144 | Margin-guided discovery finds parity feature |
| 162 | 10/10 CA rules with random cosine features |
| 181 | Iterated k-NN = 100% CA at 100 steps |
| 190 | End-to-end: Fibonacci 50 steps from raw input |
| 192 | Universal FSM simulator |
| 235 | OOD ceiling broken: 255+255=510 |
| 237 | All 4 arithmetic ops from 1 truth table |
| 240 | Conditional program execution |
| 243 | Program synthesis discovers full adder |
| 244 | End-to-end ripple-carry discovered from I/O |
| 245 | 6/6 unknown functions synthesized blindly |
| 248 | Staged composition: perceive → classify → compute |
| 249 | 50-instruction programs 100% |
| 250 | Complete substrate demo script |
| 253 | Context-dependent classification +27pp |
| 255 | Analogy reasoning A:B::C:? +43pp |
| 261 | Decomposition auto-selection (per-bit 100% vs direct 0%) |
| 262 | Ties MLP on arithmetic, loses parity by 11pp |
