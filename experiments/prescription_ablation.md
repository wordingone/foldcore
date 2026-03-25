# Prescription Ablation Analysis

*Jun directive 2026-03-24. For each game, strip the prescription to its minimum viable form. What's essential vs removable scaffolding? The minimum prescription = what the substrate must autonomously discover.*

## Ablation Hierarchy (most → least purgable)

1. **Pixel coordinates** — exact (x,y) positions. Replaceable by CC zone discovery.
2. **Pre-computed paths** — BFS/shortest-path. Replaceable by interaction-based exploration.
3. **Zone identity** — knowing which regions are clickable/important. Discoverable by random probing + frame diff.
4. **Action ordering** — which zone/action WHEN. The temporal credit problem. Likely essential.
5. **Mechanics understanding** — how game elements interact (colors cycle, water levels change, shapes match). Likely essential.

---

## FT09 — Full prescription: 75 clicks, 6 levels (ABLATION COMPLETE)

**Prescription components:**
- C1: Exact pixel coordinates per click (2-pixel precision required)
- C2: Which level you're on (level detection)
- C3: Per-level click SET (L1: 4 walls, L2: 7 walls, L3: 14 walls, L4: 16 clicks on 11 walls, L5: 21 Lights-Out, L6: 13 Lights-Out)
- C4: ~~Total ordering within each level~~ **ORDER IRRELEVANT (Step 1019b)**
- C5: Game mechanics: L1-L4 = single/multi-click wall toggling, L5/L6 = Lights-Out (+spread / up-spread, solved via GF(2) Gaussian elimination)

**Ablation results:**
| Step | Ablation | Result | Finding |
|------|----------|--------|---------|
| 1019a-A | Exact coords (baseline) | 6/6 PASS | 75 clicks (4+7+14+16+21+13) |
| 1019a-B | 8px zone snap | 1/6 (L1 only) | Walls at odd grid, snap at even → off-by-1 |
| 1019a-C | 6-cluster k-means | 0/6 | Clusters collapse position info |
| 1019b-A | Original order | 6/6 PASS | Baseline |
| 1019b-B | Reversed order | 6/6 PASS | Order irrelevant |
| 1019b-C | Random permutation (20×) | 6/6 PASS all | Order irrelevant for ALL levels including Lights-Out |

**CONFIRMED minimum prescription:** {Unordered set of exact (x,y) per level}. 75 clicks total, all necessary, 2-pixel precision. No ordering required. No redundancy possible.

**What the substrate must discover autonomously:**
1. Exact wall positions (not zones — 2px precision)
2. How many times to click each wall (L4: some walls need 2 for 3-color cycle)
3. For L5/L6: which subset of walls to toggle (Lights-Out solution = linear algebra over GF(2))
4. Does NOT need to learn click ordering

---

## VC33 — Full prescription: 176 clicks, 7 levels (ABLATION COMPLETE)

**Prescription components:**
- C1: Exact click coordinates (2-pixel precision required)
- C2: Canal lock mechanics (incremental height adjustment, interleaving)
- C3: Ordered click sequence (L4-L7 STRICTLY order-dependent)
- C4: Per-level click count: L1=3(1 distinct), L2=7(2), L3=23(4), L4=23(6), L5=49(10), L6=22(7), L7=49(9)

**Ablation results:**
| Step | Ablation | Result | Finding |
|------|----------|--------|---------|
| 1020-zone-8 | 8px zone snap | 0/7 FAIL | Same odd-grid problem as FT09 |
| 1020-zone-6 | 6-cluster k-means | 0/7 FAIL | Clusters collapse position info |
| 1020-order L1 | Random permutation | 20/20 PASS | Order-free (3 identical clicks) |
| 1020-order L2 | Random permutation | 13/20 PASS | Partially order-dependent |
| 1020-order L3 | Random permutation | 3/20 PASS | Mostly order-dependent |
| 1020-order L4-L7 | Random permutation | 0/20 PASS | Strictly order-dependent |

**CONFIRMED minimum prescription:** {Ordered sequence of exact (x,y) per level}. 176 clicks total. Exact coords AND correct ordering required for L4+.

**VC33 vs FT09 — fundamentally different:**
- FT09: unordered SET (clicks are independent toggles, order never matters)
- VC33: ordered SEQUENCE (canal locks require precise interleaving, L4+ strictly sequential)

**What the substrate must discover autonomously:**
1. Exact click positions (2px precision)
2. Correct ordering of clicks (especially L4-L7)
3. Canal lock mechanics (which lock to adjust when)
4. Progressive complexity: L1 trivial, L7 requires 49 precisely ordered clicks

---

## LS20 — Full prescription: 311 actions (Step 1018e confirmed), 7 levels [13,45,41,43,44,72,53]

**Prescription components (from source analysis):**
- C1: Wall layout per level (ihdgageizm positions)
- C2: Goal positions per level (rjlbuycveu positions)
- C3: Shape matching requirements (GoalColor, GoalRotation per level)
- C4: Collectible positions (npxgalaybz — extend step budget)
- C5: BFS path planning (start → shape match → goal)
- C6: Shape cycling mechanics (color/rotation/shape cycling via tagged sprites)
- C7: Step budget management (42 steps, 3 lives)

**Ablation results (Step 1021):**
| Test | Result | Finding |
|------|--------|---------|
| BFS minimum verification | All 7 levels confirmed minimum | No shorter paths exist |
| Order (shuffle) | 0/10 ALL levels | Strictly order-dependent |
| Move criticality | 311/311 critical (0 robust) | Every single move is load-bearing |

**Level sequences (BFS-optimal):**
- L1: 13 moves (LLLUUUURRRUUU)
- L2: 45 moves (shape matching + navigation)
- L3: 41 moves (shape matching + navigation)
- L4: 43 moves (shape matching + navigation)
- L5: 44 moves (moving triggers, cycle_lcm=8)
- L6: 72 moves (moving triggers, cycle_lcm=8)
- L7: 53 moves (fog + navigation)

**CONFIRMED minimum prescription:** {Exact ordered sequence of discrete actions per level}. 311 moves, all critical, no redundancy, no purgable components. LS20 is the most constrained: every move matters, and the exact ordering is the only viable path.

---

## Cross-Game Minimum (UPDATED with ablation results)

**REVISED — CC discovery FAILS (Steps 1019a, 1020a):**

| Component | FT09 | VC33 | LS20 |
|-----------|------|------|------|
| Position precision | Exact 2px (zones KILL) | Exact 2px (CC KILL) | Discrete actions (N/A) |
| Ordering | ORDER-FREE (set) | L4-L7 STRICT (sequence) | Sequential (navigation) |
| Mechanics | L1-L4 toggle, L5/L6 Lights-Out (GF(2)) | Canal lock interleaving | Shape matching + budget |
| Prescription type | Unordered set of (x,y) | Ordered sequence of (x,y) | Ordered action sequence |

**What's essential across ALL games (proven by ablation):**
1. **Exact position discovery** — 2px precision for click games. Zone/CC discovery FAILS universally. The substrate must discover precise targets from interaction.
2. **Game-specific ordering** — FT09 needs no ordering. VC33 L4+ needs strict ordering. LS20 is sequential by nature. The substrate must discover whether and how ordering matters PER GAME.
3. **Mechanics inference** — Lights-Out (FT09), canal locks (VC33), shape matching (LS20). Each game has unique mechanics discoverable only by interaction.

**What's purgable (confirmed):**
- Click ordering for FT09 (walls are independent toggles)
- L1-L3 ordering for VC33 (simple enough to tolerate shuffling)
- Pre-computed paths (if budget allows trial-and-error)

**What the substrate must discover autonomously (Direction 1 requirements):**
1. Where to act — exact positions, not approximate zones
2. How to act — per-game mechanics (toggle? cycle? interleave? navigate?)
3. When to act — game-specific ordering (sometimes irrelevant, sometimes critical)
4. How to REPLAY efficiently — RHAE requires action efficiency, not just eventual completion

---

## RHAE Scoring Constraint (ARC-AGI-3)

Score = (human_baseline / agent_actions)^2 per level. The substrate must not just SOLVE levels but solve them EFFICIENTLY.

| Game | Level | Human Baseline | BFS Solution | RHAE Ceiling | Max Budget (25%) |
|------|-------|---------------|-------------|-------------|-----------------|
| FT09 | L1 | 17 | 7 | 100% | 34 |
| FT09 | L2 | 19 | 7 | 100% | 38 |
| FT09 | L3 | 15 | 7 | 100% | 30 |
| FT09 | L4 | 21 | 7 | 100% | 42 |
| FT09 | L5 | 65 | 40 | 100% | 130 |
| FT09 | L6 | 26 | 7 | 100% | 52 |
| VC33 | L1 | 6 | 3 | 100% | 12 |
| VC33 | L2 | 13 | 7 | 100% | 26 |
| VC33 | L3 | 31 | 23 | 100% | 62 |
| VC33 | L4 | 59 | 23 | 100% | 118 |
| VC33 | L5 | 92 | 49 | 100% | 184 |
| VC33 | L6 | 24 | 22 | 100% | 48 |
| VC33 | L7 | 82 | 49 | 100% | 164 |
| LS20 | L1 | 21 | 13 | 100% | 42 |
| LS20 | L2 | 123 | 45 | 100% | 246 |
| LS20 | L3 | 39 | 41 | 95% | 78 |
| LS20 | L4 | 92 | 43 | 100% | 184 |
| LS20 | L5 | 54 | 44 | 100% | 108 |
| LS20 | L6 | 108 | 72 | 100% | 216 |
| LS20 | L7 | 109 | 53 | 100% | 218 |

**Implication for ablation:** A mechanism that discovers the prescription in 10K steps but EXECUTES it in 20 actions scores well. A mechanism that needs 10K steps to stumble on the solution and has no way to replay it scores ~0%. The substrate needs LEARNING (discover once) + EXECUTION (replay efficiently), not just exploration.
