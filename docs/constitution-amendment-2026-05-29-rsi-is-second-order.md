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

## Refinement (2026-05-29): minimality is measured via the CORE_ONLY baseline, not assumed via core size

E2.2' (= the E2' named above) resolves the core-design crux: the frozen capability core is a **local LLM at :9876 used as a generative program-proposer** (Option-3), conditional on each task's I/O examples. Chosen because K3 (the-search#3, commit `3b7c13c2`) proved the dead selection axis failed *structurally*: the LGG meta-layer learned the MARGINAL P(d2), but novel synthesis requires the CONDITIONAL P(program | examples). An LLM proposes conditionally — that is the whole point — and it is the direct C6 operationalization (a learned generative program-prior at scale).

This appears to violate R0's "minimal — not a maximal pretrained agent." It does not, once minimality's PURPOSE is made explicit: R0 required minimality as a PROXY for R6-honesty (a maximal core that solves everything alone makes the meta-layer decorative). E2.2' measures R6-honesty DIRECTLY instead of assuming it via core size:

- **Stage 0 (CORE_ONLY):** frozen LLM-alone solve-rate on the 395, vs the 0.7% oracle bound + the seed-enumeration baseline. This IS the direct measurement of how much the (maximal) core does alone. KILL if it does not expand coverage above seed/oracle.
- **Stage 1 (CORE_META):** add the abstraction-accumulation library (anti-unification/LGG-with-holes from the system's own solved traces, fed back to the proposer as macro-ops/exemplars). R6 kill: CORE_META vs CORE_ONLY on accumulation-held-out novel tasks. Load-bearing = RSI (R4 firing); decorative = (a) collapsed to "just an agent" on this core, reported honestly.

R0 refined: the EXPERIMENTAL meta-layer must be minimal and isolable; the frozen core may be maximal in parameters **provided its standalone contribution is measured (CORE_ONLY)** so R6-honesty is established empirically rather than presumed. The "just an agent" risk R0 guarded against is not banned by construction — it is directly testable, and an informative negative if it fires.

## Refinement 2 (2026-05-29): Option-3 (frozen LLM) killed at Stage 0; the core is code-synthesis; the horizon is the any-to-any convergence

Refinement 1 named the frozen core as "a local LLM at :9876 used as a generative program-proposer (Option-3)." Stage-0 design review (the-search#3) killed it before launch:

- **The execution-ceiling realization.** Option-3 had the LLM propose in the 12-op DSL "+ allow novel primitives it names," executed via SUBSTRATE.py. But SUBSTRATE.py executes ONLY the 12 ops — a novel-named op is unexecutable. So Stage-0-as-specced was a SELECTION experiment in disguise (K1–K3 redux): a proposer over a fixed executable basis can only ever compose that basis. The EXECUTION layer — not the proposer — is the true coverage ceiling.
- **The fix: code-synthesis.** The core proposes ARBITRARY Python (grid→grid, sandboxed), executed directly with generate-and-test on the I/O examples. This breaks the 12-op execution ceiling — the executable layer's coverage becomes "any computable grid→grid transform," not "compositions of 12 primitives." Designer-DSL-expansion (hand-adding primitives) is rejected as non-RSI: the designer, not the system, would be expanding capability. Stage 0' (CORE_ONLY) measures whether code-synthesis expands coverage above the 0.7% oracle bound at all — the load-bearing feasibility test.
- **The inference-mode introspection (user's LeCun pointer).** Anchoring the core on an autoregressive LLM conflated "conditional generation" with "the right inference mode." An autoregressive model generates token-by-token; it does not SEARCH the program space. Generate-and-test with execution feedback moves toward inference-by-optimization (LeCun H-JEPA / energy-based), which is more R2-native than autoregressive proposal.
- **The horizon: the any-to-any convergence (user, 2026-05-29).** The eventual core is not a text LLM but an ANY-TO-ANY entity (AnyGPT / Chameleon / 4M-21): grid-native (no text-verbalization indirection — the indirection that motivated the Stage-0 kill), operating on representations rather than tokens (LeCun-aligned), on a discrete substrate (BitNet ternary — and any-to-any models already discretize every modality, so ternary is native), with PROGRAM/TRANSFORM as one output modality (preserving the executable-abstraction substrate the RSI library needs). Self-modification = compiling new any-to-any transforms onto the shared representational medium ("fixed rules on a shared medium, evolving state"). The irreducibility wall is unchanged (the interpreter core cannot rewrite itself from within); the achievable layer is richer (arbitrary cross-modal transforms, not just text-code). BitNet + LeCun + any-to-any + self-compilation are four faces of ONE architecture, not four interests.

This supersedes the "frozen LLM at :9876" framing of Refinement 1 for the CORE DESIGN. Refinement 1's principle — minimality measured via CORE_ONLY, not assumed via core size — is UNCHANGED and applies to code-synthesis and the any-to-any horizon alike. Forward-discipline: Stage 0' (code-synthesis CORE_ONLY) is the concrete build; the any-to-any convergence is the horizon the ladder reaches toward, gated on Stage 0'/1 feasibility — not a jump-ahead.

Refs: the-search#3 (live tracker); AnyGPT (arXiv:2402.12226); Chameleon (arXiv:2405.09818); 4M-21 (arXiv:2406.09406); BitNet b1.58 (arXiv:2402.17764).
