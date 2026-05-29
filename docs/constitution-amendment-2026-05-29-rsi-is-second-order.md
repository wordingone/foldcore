# Constitutional Amendment — 2026-05-29: RSI is Second-Order

**Status:** ADOPTED. Decision delegated to Leo per user directive 2026-05-29 ("it's all your call after ultrathink"). Reversible; user retains override.

## The decision

For the question "does true RSI require a minimally-viable abstracted capability core folded into the substrate (a), or must it contain none (b)?" — the answer is **(a)**.

## The principled reason

RSI = Recursive Self-**Improvement**. "Improvement" presupposes a baseline to improve from. A system with zero capability is not "recursively self-improving toward capability" — it is attempting to bootstrap capability from nothing, which is the **origin-of-cognition** problem, not RSI. These are different problems.

The prior constitution (R1-R6) assumed (b) by omission: it required capability to EMERGE from local self-modifying rules with no borrowed capability, and banned (via R1/R2) the only known means of acquiring baseline capability (learned models). This conflated RSI (second-order: a capable system improving its own operations) with the origin question (first-order: capability from nothing).

Evidence the conflation was load-bearing:
- **792 ARC experiments**: no learning-based approach produced non-zero RHAE under any constraint set.
- **K1** (2026-05-29, the-search#3, commit `bac727fd`): on the 395 out-of-closure tasks, the capability-memory signal loop is broken at SOURCE — reprioritization needs accumulated success; success accumulates only from solves; the seed basis cannot solve out-of-closure tasks (vocabulary gap) → zero signal → nothing to reprioritize with. The self-modification layer is not inert; it is UNFED. The direct empirical instance of the ordering error.
- **ARC-AGI-3** (fact-checked 2026-05-29): the post-LS20 collapse below 1% hit ALL approaches — frontier LLMs AND the non-LLM symbolic/CNN substrates (StochasticGoose/Tufa Labs CNN+RL, rule-graph, frame-graph). "Transformers fail / violate R2" never justified the (b) ban; the (b)-flavored substrates fail too.

## The rescoping (R0-R6 under (a))

The constitution governs the **self-modification meta-layer** — the second-order loop where a capable system improves its own operations. It does NOT govern the origin of the base capability.

- **R0 (new):** The system = [frozen, minimal, abstracted capability core] + [self-modification meta-layer]. The constitution governs the meta-layer. The base capability core is a frozen substrate; its origin (pretraining/backprop) is out of scope. "Minimal" is binding — the core is the smallest capability that gives the meta-layer something to operate on, not a maximal pretrained agent.
- **R1 (rescoped):** No external loss/reward/metrics drive the META-LAYER. The meta-layer modifies the system's operations from self-generated criteria. (The frozen core's pretraining used external loss; that is the substrate, not the studied process.)
- **R2 (rescoped):** The meta-layer's update signal IS the computation — self-modification fused into operation, from the system's own trace, not via a separate external optimizer on the meta-layer.
- **R3:** Meta-layer modification changes behavior.
- **R4:** Modification tested against the pre-modification state (second-exposure / structural transfer).
- **R5:** One fixed ground truth (the task).
- **R6 (the honesty gate — load-bearing):** Delete the meta-layer; if novel-task performance does NOT degrade, the meta-layer is decorative and (a) has collapsed to "just an agent" — KILL. The self-modification must be deletion-load-bearing ON NOVEL TASKS for the result to be RSI rather than a measurement of the frozen core.

## The risk, named

(a)'s real risk: every instantiation may have a decorative meta-layer (the core does all the work). R6 is the defense — and even a decorative-verdict is an INFORMATIVE negative (self-modification adds nothing on a capable substrate), cleaner than (b)'s ambiguous nulls ("can't emerge" vs "haven't found the right rule"). The bet is sound regardless of whether (a) succeeds, because it is falsifiable.

## The origin question is NOT abandoned — it is distinguished

The (b) emergence question (can capability arise from local self-modifying rules with no borrowed capability?) is real and fundamental. It is the origin-of-cognition problem. It is NOT what "RSI" means and is NOT pursued under this constitution. Resurrecting it is a deliberate, separately-named track, not the default.

## Consequence for the active direction

E2-on-seed-basis (LGG over the 12 primitives) was a (b)-path experiment (symbolic primitives, no capability core). SUPERSEDED as the RSI experiment. The new direction (E2'): [frozen minimal capability core] + [self-modification meta-layer], with anti-unification/LGG re-entering as the meta-layer's abstraction-accumulation mechanism, tested on the 395 with the R6-load-bearing kill. The minimal-capability-core design (what is tractable + minimal on the available hardware, within the memory-budget serialization rule) is the next research-design crux.
