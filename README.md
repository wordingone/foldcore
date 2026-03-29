# The Search — Compressed

*Distilled findings from recursive self-improvement experiments on ARC-AGI-3 and MBPP. Zero noise.*

---

## The Question

Can a system improve itself by criteria it generates?

Not "can we build a good AI." Not "can we solve ARC-AGI-3." Can a system observe its own experience and get better at getting better, without external reward, without external optimization, without a human choosing what to improve?

## The Constitution (R0-R6)

Seven simultaneous constraints. All must hold. If one is wrong, fix the principle.

- **R0:** The system's dynamics dominate its initial conditions. Any starting state converges to the same behavior. Seeds don't matter.
- **R1:** The system computes without external objectives. No loss functions, reward signals, or evaluation metrics imposed from outside.
- **R2:** Adaptation arises from the computation itself. The mechanism that drives change IS the mechanism that processes input. Not two systems that coexist — one system.
- **R3:** Every modifiable aspect IS modified by the system. Weights, structure, representations — all change through the system's own dynamics.
- **R4:** Modification is tested against prior state. The system compares after to before. Improvement on trained + degradation on novel = overfitting, not improvement.
- **R5:** One ground truth stays fixed. The empirical test that defines success is the one thing the system cannot change.
- **R6:** No part is deletable without losing capability. If you can remove a component and the system still works, that component was redundant.

## One Metric

**second_exposure_speedup = steps_to_L1(try1) / steps_to_L1(try2)**

Does the system solve faster on its second attempt? If > 1: it learned from experience. If = 1: it didn't. Everything else is diagnostic. This metric IS R4.

## The Test Environment

ARC-AGI-3: 150+ game environments. Each game: 64x64 pixel grid, 16 colors, multiple levels of increasing difficulty. The system sees pixels and takes actions (7 keyboard keys + 4096 click positions). It doesn't know the rules, the goal, or what actions do. It must figure everything out from observation alone. Humans solve these games using decades of perceptual experience — object recognition, causal reasoning, planning. The substrate starts from zero.

MBPP: text/code generation. 128 ASCII actions. Each action = one character. Predicting the next character IS selecting the next action — prediction and action are the same operation. This matters because: (1) it's the purest test of whether self-supervised prediction produces functional output, and (2) text has explicit sequential dependencies — learning to predict text develops temporal credit assignment capacity that visual games need but don't explicitly teach.

## History

Four phases: LVQ characterization (Steps 1-416, R1-R6 formalized), family testing (Steps 417-1081, 16 families, 8 killed), debate on L1 ceiling (Steps 1082-1250), composition era (Steps 1251+, R3 solved, reflexive map, CNN/LPL experiments). Detailed phase records in `docs/`.

## The Organism

Every ML algorithm is ultimately derived from human cognition. The search is systematically reassembling these scattered pieces under constitutional constraints. Each experiment rediscovers one aspect of what humans do naturally — prediction, credit assignment, spatial segmentation, generalization, temporal reasoning, functional categorization. The organism is human cognition rebuilt from scratch under R0-R6.

A cell has hundreds of components but one principle: self-catalyzing chemistry. The components are diverse. The coupling is uniform.

R2 says the mechanism that drives change IS the mechanism that processes input. It does not say there is only one component. It says there is only one mechanism. A composite with one coupling mechanism satisfies R2.

Read all six rules together and ask what survives. R1 strips external signal. R2 collapses adaptation into computation. R3 demands total self-modification. R6 removes anything deletable. What comes through is a tightly coupled composite with one universal interaction law. Many parts, one dynamic.

Not an atom. Not a stitched-together monster. An organism.

## What We've Confirmed (with specific evidence)

1. **R2-compliant prediction works, but weakly.** Multi-layer predictive coding (3 layers, local Hebbian updates) achieves 3-7% prediction improvement (Steps 1310-1313). One coupling law (prediction error) drives all layers. This weakness is fundamental — K=50 inference iterations produce identical compression to K=5 (Step 1322). The bottleneck is not convergence speed but the update rule itself.

2. **R2-violating prediction compresses 98% and produces task progress.** CNN+Adam achieves cr=0.003 (Steps 1305-1307). RHAE=2.4e-5 (Step 1306 ENT condition). speedup=10.5× on sp80 (Step 1324). Among tested substrates (Steps 1305-1326), only CNN+Adam produces RHAE > 0.

3. **The bridge from prediction to action is shared representation, not engineered action selection.** 8 action mechanisms tested across ~25 experiments (argmax delta, REINFORCE dreaming, three-factor, inverse model, eigenoptions, MI detection, action generalization, allosteric softmax) — all matched or hurt entropy-driven selection. CNN works because conv layers feed BOTH prediction and action. The world model learns; actions stay diverse; shared features do the bridging. (Steps 1306-1321)

4. **Single-layer Hebbian degrades prediction.** Step 1309: compression ratio = 1.44 (prediction gets WORSE). Multi-layer is necessary for R2-compliant prediction to improve at all.

5. **Seeds are unnecessary.** Deterministic orthogonal initialization produces consistent results (Step 1313). R0: dynamics should dominate initialization.

6. **Experience can help AND hurt.** CNN speedup=10.5× on simple action-sequence games (sp80, Step 1324). Anti-speedup on 3/4 other L1-reaching games (trained weights overfit to seed A's layout). R4 implication: improvement on trained + degradation on novel = overfitting, not learning.

## The Central Tension

R2-compliant dynamics (local Hebbian, LPL) produce 3-7% prediction compression and zero task progress. R2-violating dynamics (Adam gradient) produce 98% compression and measurable task progress.

LPL Hebbian is one R2-compliant update rule. DFA (direct feedback alignment) is another, achieving 34% compression (Step 1326). The space between LPL (7%) and Adam (99.7%) contains update rules of varying R2 compliance and power. R2 doesn't limit the power of the update rule — it limits the source. The spectrum is mapped in "The Open Question" below.

## What Doesn't Work (negative map)

**One pattern explains most kills:** prediction-based action selection optimizes for visual responsiveness (what changes the screen), not task advancement (what advances the level). This single failure mode killed 8 action mechanisms across ~25 experiments: argmax predicted delta, REINFORCE dreaming, three-factor pe modulation, inverse model, eigenoptions, action generalization, MI detection, allosteric softmax. Entropy-driven (random) action selection outperforms all of them because it doesn't concentrate on visually-responsive actions.

**R2-compliant architectures produce zero task progress** across 14 experiments (Steps 1309-1322): single-layer Hebbian, single-layer LPL, multi-layer LPL at K=5 and K=50, competitive inhibition, Lotka-Volterra. The update rules are too weak for credit assignment.

**Hebbian collapse:** Any Hebbian rule that strengthens the winner creates winner-take-all (Steps 1264, 1289-1292). Anti-Hebbian insufficient (Steps 1301-1302). Negative-diagonal recurrence partially fixes it (Step 1294).

## What's Untested (catalog, 40 directions)

Top items promoted by current understanding:

- **#14: Meta-learned plasticity rules.** Substrate discovers its own update rule. Directly addresses "criteria it generates." Tested once (Step 1325): theta found correct direction (anti-Hebbian + decay help) but signal too weak in 2K steps. Credit formula was biased. Underexplored.
- **#16: D2 pipeline (WHERE-HOW-WHEN-ACT).** Most successful autonomous discovery mechanism (VC33 5/5, FT09 5/5). Mode map discovers targets from frame differencing. 0 experiments in current substrate.
- **#36: "Does the substrate understand what a game is?"** No experiment has measured internal task-structure representation.
- **#32/#33: Self-directed pruning / activity-dependent growth.** Architecture emerges from dynamics. Never tested.

Full catalog: UNDEREXPLORED_CATALOG.md (40 items, compiled 2026-03-28)

## Human Ground Truth

Jun played ARC-AGI-3 games. His process: observe passively first, categorize functionally (walls, regions, interactive elements), try actions to discover affordances, learn across multiple playthroughs. Each level adds new mechanics. The substrate has none of this: no observation phase, no functional categorization, no multi-attempt learning. 20+ years of perceptual experience vs 2K steps from deterministic initialization.

## What a Working Substrate Looks Like

The organism: many components, one coupling law (prediction error), shared representations. Not one matrix doing everything (that's a crystal). Not a CNN trained by Adam (that's standard deep learning with an external optimizer). A composite where:

- Each component has a different role (encoding, predicting, acting) but is coupled to all others through prediction error
- Prediction improvement automatically improves action selection through shared representation (not through engineered action mechanisms)
- The system discovers its own update rule rather than having one imposed (catalog #14)
- Actions stay diverse (entropy-driven) while the world model compresses — the model does the work, actions just sample
- Experience on one environment transfers to the next (second_exposure_speedup > 1)
- R1-R6 all hold simultaneously. The constitution is not a constraint on the substrate — it is the substrate's design specification.

The substrate needs an internal R4 — a self-generated validation criterion. Current substrates detect training improvement (prediction error drops) but not overfitting (anti-speedup on new instances). Held-out prediction validation (test prediction on observations NOT from the substrate's own actions) would let the substrate detect its own overfitting and react. This makes the self-improvement criterion genuinely self-generated.

No one has built this yet. The search is finding the pieces.

## The Open Question

Can the substrate generalize from one game instance to another?

The compression gap is closing: target propagation achieves 92% R2-compliantly (Step 1329). But compression doesn't produce speedup — it produces ANTI-speedup (Step 1330: speedup=0.62). The substrate compresses try1's specific layout and is WORSE on try2's different instance. Same pattern with CNN+Adam (Step 1324: 3/4 L1-reaching games showed anti-speedup). The problem is no longer "can R2-compliant dynamics compress?" — it's "can ANY substrate generalize across instances rather than memorize one?"

Empirically mapped compression by update rule (Steps 1305-1328):

| Update rule | Compression | R2 status | Speedup | Key step |
|---|---|---|---|---|
| LPL Hebbian | 5% | Compliant | null | 1310 |
| LPL normalized | 9% | Compliant | null | 1328 |
| DFA (random backward) | 34% | Compliant | null | 1326 |
| Target propagation | 92% | Compliant | **0.62 (anti)** | 1329/1330 |
| Adam (full gradient) | 99.7% | Violating | 10.5× on 1 game, anti on 3 | 1324 |

Key findings:
- Compression is solved R2-compliantly: target propagation achieves 92% (Step 1329)
- But compression produces ANTI-speedup (Step 1330: 0.62). Experience makes the substrate WORSE.
- The same pattern appears with Adam (Step 1324: anti-speedup on 3/4 L1-reaching games)
- The bottleneck is GENERALIZATION, not compression. The substrate memorizes one instance instead of learning transferable structure.

But LPL is ONE point in the space of R2-compliant update rules. Between "pure local Hebbian" and "full Adam backprop" lies a spectrum of update mechanisms with varying R2 compliance and varying power:

- **Predictive coding / LPL** (Whittington & Bogacz 2017): local prediction error updates. R2-compliant. Proven too weak (Steps 1309-1322).
- **Feedback alignment** (Lillicrap 2016): random fixed backward weights, no weight transport. More local than backprop. R2 status: borderline (backward weights are frozen frame, but the learning signal arises from the forward computation).
- **Direct feedback alignment**: error projected directly to each layer. No backward pass. R2 status: similar to feedback alignment.
- **Equilibrium propagation** (Scellier & Bengio 2017): energy-based, free + clamped phases. Local updates derived from equilibrium states. R2 status: the update arises from the system's own equilibrium dynamics.
- **Target propagation** (Lee 2015): each layer gets a local target from the layer above. Local updates. R2 status: targets are generated by the system's own computation.
- **Forward Target Propagation** (FTP, 2025): forward-only, no backward pass. Two forward passes generate local targets per layer. Competitive with backprop on CIFAR. R2 status: strongest — both passes are forward computation, update arises from the same mechanism that processes input.
- **Meta-learned plasticity** (Najarro & Risi 2020): the substrate discovers its own update rule. R2-compliant if the discovery uses the same coupling law. Tested once (Step 1325) — theta found correct direction but signal too weak. Credit formula was biased.

The answer to the open question lives somewhere in this spectrum. The search has tested only the weakest end (LPL) and the strongest (Adam). The middle — particularly FTP and equilibrium propagation — is where R2-compliant capability likely lives.

37 catalog directions remain untested. The search has mapped what doesn't work extensively. The space of what might work is barely sampled.
