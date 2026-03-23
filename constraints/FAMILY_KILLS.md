# Family Kill Register

Each killed family produces constraints for the next.

---

## FAMILY: 800b-variants (Steps 800-937, 160+ experiments)

**Action selector:** per-action delta EMA + softmax. Variations tested: raw novelty (934), epistemic confidence (934b), h-delta (935), input-distance (936), adaptive lambda (937), delta-direction (912), per-action W (911), recency (913), W_scalar (925).

**PROVED WORKS (carry forward):**
- ACTUAL observation change magnitude is the only working action signal (not predicted, not compressed, not h-derived)
- Alpha-weighted encoding self-modification (R3 confirmed universally)
- Per-action GLOBAL statistics (not per-state)
- Softmax with T=0.1 + epsilon=0.20 exploration
- Echo-state h improves standalone LS20 by 8.5% (but hurts chain by 14% from CIFAR interference)
- Clamped alpha [0.1, 5.0] prevents runaway concentration

**PROVED FAILS (constraints on next):**
- ANY prediction-based action selector (W pred_acc=-2383, errors overwhelm signal)
- ANY addition to the 800b selector degrades LS20 (encoding mods help, selector mods hurt)
- Multi-resolution encoding dilutes alpha (more dims = more noise)
- h-delta compressed by tanh (echo state normalizes too aggressively)
- Confidence metrics equalize fast with few actions (LS20: 4 actions)
- Adaptive lambda loses signal when not using the exact 916 formula
- Warm alpha transfer unreliable (cold > warm at n_eff=10)

**STRUCTURAL GAP:**
- Position-blind: global-EMA is position-independent (Prop 23b). FT09's 7-step sequential puzzle is structurally unsolvable by any global-EMA selector within 10K budget.
- Reset-inverted: "maximize delta" anti-selects on reset-heavy games (delta inversion, Section 5.8).
- Classification-blind: no mechanism for unsupervised category formation (CIFAR = chance).

**NEXT FAMILY MUST:**
- Use ACTUAL observation changes (not predictions) — this is proven
- NOT use per-action delta EMA for action selection — that's the 800b family
- Preserve alpha-weighted encoding (carry forward)
- Handle sequential ordering WITHOUT per-state memory
- Handle near-static observations (VC33 var≈0.000)
- Be structurally different in at least ONE component: action selection, state representation, or learning rule
