# The Search

Experimental record of recursive self-improvement research on ARC-AGI-3 and MBPP. 1341 experiments, 4 phases, 16 architecture families tested.

---

## Question

Can a system improve itself by criteria it generates?

## Feasibility

One known system satisfies all seven constitutional constraints simultaneously: biology. Cells self-modify (R3) through metabolic dynamics that ARE the computation (R2), without external objectives (R1). Environmental selection tests modifications against prior fitness (R4). DNA provides the fixed ground truth (R5). Every organelle is essential (R6). Identical genomes in different environments produce different organisms (R0).

This is not a theoretical exercise. The constitution is extracted from observing what already works. The open question is whether computation (discrete, finite precision) can implement what chemistry (continuous, molecular) does. A single computational substrate satisfying R0-R6 with RHAE > 0 would resolve this.

## Metric

**RHAE(try2) = mean(efficiency²)** across all games, measured on try2 (with weights from try1).

ARC Prize scoring. External judgment. The substrate plays each game twice: try1 (fresh weights), try2 (carrying try1's weights). MBPP (text) always in the game pool. efficiency = optimal_actions / actual_actions. efficiency = 0 when progress never reached.

**Current best RHAE(try2) from R2-compliant substrate: 0.0** (1341 experiments, zero non-zero results).

**Reference (R2-violating):** CNN+Adam RHAE=2.4e-5, speedup=10.5x on sp80 (Steps 1305-1324).

## Constitution

Seven simultaneous constraints (R0-R6). Full definitions, falsification conditions, and evidence in `constraints/CONSTITUTION.md`.

| Rule | Summary | Current status |
|------|---------|---------------|
| R0 | Dynamics dominate initial conditions | Partial — prediction consistent, task performance untested |
| R1 | No external objectives | Holds for prediction/navigation |
| R2 | Adaptation IS the computation | Adam violates. LPL/TP comply. Organism interpretation adopted, not falsification-tested |
| R3 | Everything self-modified | Holds for weights. Open for structure |
| R4 | Tested against prior state | **Not satisfied.** Zero genuine transfer in 1341 experiments |
| R5 | One fixed ground truth | Holds by construction |
| R6 | No deletable parts | Untested on current architecture |

## Test environment

**ARC-AGI-3:** 150+ game environments. 64x64 pixel grid, 16 colors, multiple levels. 7 keyboard + 4096 click actions. Rules, goals, and action effects unknown to the substrate.

**MBPP:** Text/code generation. 128 ASCII actions. Predicting next character IS selecting next action.

## Confirmed findings

Each finding cites the experiments that support it and states what would falsify it.

| # | Finding | Evidence | Falsified if |
|---|---------|----------|-------------|
| 1 | R2-compliant prediction compresses weakly (3-9%) | LPL cr=0.93 (Steps 1310-1313). K=50 iterations = K=5 (Step 1322). LPL normalized cr=0.91 (Step 1328). | An R2-compliant local update rule achieves cr < 0.5 without target propagation. |
| 2 | Target propagation compression is game-dependent (54-92%) and R2-compliant | TP cr=0.08 on easy games (Steps 1329-1343), cr=0.36-0.46 on harder games (Step 1344). "92% compression" was game-selection artifact. Local targets from forward computation, no global backward pass. | TP compression shown to require a global signal. |
| 3 | CNN+Adam compresses 98% and reaches task progress, violating R2 | cr=0.003 (Steps 1305-1307). RHAE=2.4e-5 (Step 1306). Adam optimizer separable from forward pass. | Adam shown to be non-separable from forward computation, or an R2-compliant method matches RHAE. |
| 4 | Prediction-based action selection ≈ random | 10 action mechanisms tested across ~30 experiments: argmax delta, REINFORCE dreaming, inverse model, eigenoptions, MI, allosteric, model-based novelty (1340), curiosity (1341). All ≈ entropy on ARC. (Steps 1306-1341) | A prediction-based action selector consistently outperforms entropy on masked PRISM chain. |
| 5 | All spatial representations are episode-specific | TP anti-speedup (Steps 1330, 1335). Mode map zones: try1 [3,4,4,3] → try2 [0,0,0,0] (Step 1338). CNN try2 diverges cr=20.76 (Step 1337). | A spatial representation produces speedup > 1 confirmed by seed swap control. |
| 6 | MLP processes text; CNN cannot | MLP MBPP cr=0.08 (Step 1337). CNN MBPP cr=null, wdrift=0. | CNN shown to compress MBPP text, or MLP shown to fail on text. |
| 7 | Single-layer Hebbian degrades prediction | cr=1.44 — prediction gets worse (Step 1309). | Single-layer Hebbian achieves cr < 1.0 on any game. |
| 8 | Seeds are unnecessary | Deterministic orthogonal init produces consistent results (Step 1313). | Deterministic init produces >10% metric variance across runs. |

## What doesn't work

| Mechanism | Experiments | Result |
|-----------|-----------|--------|
| Prediction-based action selection (10 variants) | Steps 1306-1341, ~30 experiments | All ≈ entropy. Model-based novelty (1340) and curiosity (1341) also fail because forward model is action-blind. |
| R2-compliant update rules for task progress | Steps 1309-1341, 14+ experiments | Zero RHAE > 0. TP compresses 92% but no level advancement. |
| Multi-episode training for transfer | Step 1336 | Worse than single-episode. Diverse episodes produce interference, not invariance. |
| Overfitting detection (internal R4) | Step 1334 | Detection triggers but LR reduction too aggressive. Calibration issue, not concept failure. |
| Meta-learned plasticity | Steps 1325, 1339 | Theta found correct direction (1325) but credit signal too weak. Theta frozen at init (1339) — implementation bugs identified, normalized credit fix available. |
| Mode map / zone discovery | Step 1338 | Zones are spatial (episode-specific), not functional. try1 zones don't persist to try2. |

## Compression spectrum (empirically mapped)

| Update rule | Compression | R2 status | RHAE(try2) | Key step |
|---|---|---|---|---|
| LPL Hebbian | 5% | Compliant | 0 | 1310 |
| LPL normalized | 9% | Compliant | 0 | 1328 |
| DFA (random backward) | 34% | Compliant | 0 | 1326 |
| Target propagation | 92% | Compliant | 0 | 1329 |
| Adam (full gradient) | 99.7% | Violating | 2.4e-5 | 1324 |

## Open directions (from catalog, 40 items)

Tested this session and killed: mode map (#16, Step 1338), meta-plasticity (#14, Step 1339).

Top untested:
- **Action-conditional forward model** — predict next_obs given (obs, action). Current model is action-blind. Step 1342 in progress.
- **Equilibrium propagation + learned halting** — R2-compliant deliberation. Substrate decides when to act. Not in literature.
- **#32/#33: Self-directed pruning / activity-dependent growth** — architecture emerges from dynamics.
- **#36: "Does the substrate understand what a game is?"** — no experiment has measured internal task-structure representation.

Full catalog: `docs/UNDEREXPLORED_CATALOG.md`

## Repository structure

- `constraints/` — Constitution (R0-R6), research state, component catalog
- `experiments/compositions/` — All experiment scripts and results (Steps 1334+)
- `experiments/compositions/prism_masked.py` — PRISM infrastructure (masked game selection, RHAE computation)
- `docs/` — Phase records, underexplored catalog
