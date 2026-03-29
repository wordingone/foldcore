# The Search — Compressed

*1322 experiments distilled. Zero noise. Any LLM can pick this up and continue.*

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

## How We Got Here (1322 experiments in 4 phases)

**Phase 1 (Steps 1-416):** The substrate was LVQ (1988). 416 experiments characterized its limits. R1-R6 formalized as simultaneous constraints. Finding: sequential stage-climbing doesn't work.

**Phase 2 (Steps 417-1081):** 16 substrate families tested, 8 killed. Bans on graph memory and codebook imposed then lifted. Finding: the constraint map is path-dependent. Kill criteria are ~70% family-dependent.

**Phase 3 / Debate (Steps 1082-1250):** Prosecution vs defense on whether the substrate can surpass L1. 170 experiments. Finding: both sides hit the same wall. 200+ bridge mechanisms killed.

**Phase 4 / Composition era (Steps 1251+):** Components composed from catalog. R3 solved (100/100, Step 1251). Reflexive map defined. Then: CNN experiments (1305-1308), multi-layer LPL (1309-1322), inverse model, mode map, organism framing. Current phase.

## The Organism

A cell has hundreds of components but one principle: self-catalyzing chemistry. The components are diverse. The coupling is uniform.

R2 says the mechanism that drives change IS the mechanism that processes input. It does not say there is only one component. It says there is only one mechanism. A composite with one coupling mechanism satisfies R2.

Read all six rules together and ask what survives. R1 strips external signal. R2 collapses adaptation into computation. R3 demands total self-modification. R6 removes anything deletable. What comes through is a tightly coupled composite with one universal interaction law. Many parts, one dynamic.

Not an atom. Not a stitched-together monster. An organism.

## What We've Confirmed (with specific evidence)

1. **R2-compliant prediction works, but weakly.** Multi-layer predictive coding (3 layers, local Hebbian updates) achieves 3-7% prediction improvement (Steps 1310-1313). One coupling law (prediction error) drives all layers. This weakness is fundamental — K=50 inference iterations produce identical compression to K=5 (Step 1322). The bottleneck is not convergence speed but the update rule itself.

2. **R2-violating prediction is 14× stronger and the only path to task progress.** CNN+Adam achieves 98% compression (Steps 1305-1307) and is the only architecture producing RHAE > 0 (RHAE=2.4e-5, Step 1306; speedup=10.5× on sp80, Step 1324). The capability gap between R2-compliant and R2-violating is the central tension.

3. **The bridge from prediction to action is shared representation, not engineered action selection.** 8 action mechanisms tested across ~25 experiments (argmax delta, REINFORCE dreaming, three-factor, inverse model, eigenoptions, MI detection, action generalization, allosteric softmax) — all matched or hurt entropy-driven selection. CNN works because conv layers feed BOTH prediction and action. The world model learns; actions stay diverse; shared features do the bridging. (Steps 1306-1321)

4. **Single-layer Hebbian degrades prediction.** Step 1309: compression ratio = 1.44 (prediction gets WORSE). Multi-layer is necessary for R2-compliant prediction to improve at all.

5. **Seeds are unnecessary.** Deterministic orthogonal initialization produces consistent results (Step 1313). R0: dynamics should dominate initialization.

6. **Experience can help AND hurt.** CNN speedup=10.5× on simple action-sequence games (sp80, Step 1324). Anti-speedup on 3/4 other L1-reaching games (trained weights overfit to seed A's layout). R4 implication: improvement on trained + degradation on novel = overfitting, not learning.

## The Central Tension

R2-compliant dynamics (local Hebbian, LPL) produce 3-7% prediction compression and zero task progress. R2-violating dynamics (Adam gradient) produce 98% compression and measurable task progress.

This is NOT "R2 is wrong." LPL Hebbian is ONE R2-compliant update rule. We tested ONE rule and found it too weak. The space of R2-compliant rules is vast. The substrate could DISCOVER its own update rule through R2-compliant dynamics (catalog #14). The discovered rule could be as powerful as gradient. R2 doesn't limit the POWER of the update rule. It limits the SOURCE.

## What Doesn't Work (negative map)

**Action selection on W3 (8 mechanisms, ~25 experiments, all killed):** argmax predicted delta, REINFORCE dreaming, three-factor pe modulation, inverse model, eigenoptions, action generalization, MI detection, allosteric softmax. Common failure: prediction selects visually-responsive actions, not task-advancing ones.

**R2-compliant architectures (all produce RHAE=0):** single-layer linear + Hebbian, single-layer + LPL, multi-layer LPL (K=5 and K=50), competitive inhibition network, Lotka-Volterra.

**Collapse:** Any Hebbian rule that strengthens the winner creates winner-take-all (Steps 1264, 1289-1292). Anti-Hebbian decorrelation insufficient (Steps 1301-1302). Negative-diagonal recurrence partially fixes inference-time lock (Step 1294).

## What's Untested (catalog, 40 directions)

Top items promoted by current understanding:

- **#14: Meta-learned plasticity rules.** Substrate discovers its own update rule. Directly addresses "criteria it generates." R2-compliant. Never explored.
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

No one has built this yet. The search is finding the pieces.

## The Open Question

Can R2-compliant dynamics produce the same capability as gradient?

An R2-violating substrate (CNN+Adam) achieves second_exposure_speedup=10.5× on one game type (Step 1324). LPL (R2-compliant local prediction error) is fundamentally too weak (Step 1322). The gap is 14× in compression and qualitative in task progress.

But LPL is ONE point in the space of R2-compliant update rules. Between "pure local Hebbian" and "full Adam backprop" lies a spectrum of update mechanisms with varying R2 compliance and varying power:

- **Predictive coding / LPL** (Whittington & Bogacz 2017): local prediction error updates. R2-compliant. Proven too weak (Steps 1309-1322).
- **Feedback alignment** (Lillicrap 2016): random fixed backward weights, no weight transport. More local than backprop. R2 status: borderline (backward weights are frozen frame, but the learning signal arises from the forward computation).
- **Direct feedback alignment**: error projected directly to each layer. No backward pass. R2 status: similar to feedback alignment.
- **Equilibrium propagation** (Scellier & Bengio 2017): energy-based, free + clamped phases. Local updates derived from equilibrium states. R2 status: the update arises from the system's own equilibrium dynamics.
- **Target propagation** (Lee 2015): each layer gets a local target from the layer above. Local updates. R2 status: targets are generated by the system's own computation.
- **Meta-learned plasticity** (Najarro & Risi 2020): the substrate discovers its own update rule. R2-compliant if the discovery uses the same coupling law. Tested once (Step 1325) — theta barely moved but found correct direction. Credit formula was biased.

The answer to the open question lives somewhere in this spectrum. The search has tested only the weakest end (LPL) and the strongest (Adam). The middle is unexplored.

37 catalog directions remain untested. The search has mapped what doesn't work extensively. The space of what might work is barely sampled.
