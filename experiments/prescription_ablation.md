# Prescription Ablation Analysis

*Jun directive 2026-03-24. For each game, strip the prescription to its minimum viable form. What's essential vs removable scaffolding? The minimum prescription = what the substrate must autonomously discover.*

## Ablation Hierarchy (most → least purgable)

1. **Pixel coordinates** — exact (x,y) positions. Replaceable by CC zone discovery.
2. **Pre-computed paths** — BFS/shortest-path. Replaceable by interaction-based exploration.
3. **Zone identity** — knowing which regions are clickable/important. Discoverable by random probing + frame diff.
4. **Action ordering** — which zone/action WHEN. The temporal credit problem. Likely essential.
5. **Mechanics understanding** — how game elements interact (colors cycle, water levels change, shapes match). Likely essential.

---

## FT09 — Full prescription: 75 hardcoded clicks, 6 levels

**Prescription components:**
- C1: Exact pixel coordinates per click (e.g., click at (32, 48))
- C2: Which level you're on (level detection)
- C3: Per-level click sequence (L1: clicks A→B→C, L2: clicks D→E→F...)
- C4: Total ordering within each level (strict 7-step sequence)
- C5: That wrong clicks cause reset (mechanics)

**Ablation experiments:**
| Step | Remove | Replace with | Expected | Tests |
|------|--------|-------------|----------|-------|
| A1 | C1 (pixel coords) | CC-discovered zones | Still 6/6 if zones = correct areas | Zone identity sufficient? |
| A2 | C1+C2 (coords+level detect) | CC zones + auto level detect (frame change on level transition) | Probably works | Can substrate detect levels? |
| A3 | C1+C4 (coords+ordering) | CC zones + random ordering | 0/6 (random 7-step = 1/6^7) | Ordering is essential |
| A4 | C1+C3 (coords+per-level seq) | CC zones + ONE shared sequence | 0-1/6 (levels differ) | Per-level customization essential? |
| A5 | ALL except C5 | Pure CC zones + random clicking | 0/6 (proved by Step 1017) | Confirms minimum |

**Predicted minimum prescription:** Zone identity (C1→CC) + ordering (C4) + per-level sequence (C3). Level detection (C2) probably auto-detectable. Reset mechanics (C5) observable.

---

## VC33 — Full prescription: 176 analytical BFS clicks, 7 levels

**Prescription components:**
- C1: Zone coordinates (canal gates, water sources, targets)
- C2: Canal mechanics model (water flows downhill, gates block/allow)
- C3: BFS path planner (optimal gate sequence)
- C4: Camera offset correction
- C5: Per-level state estimation (water level tracking)

**Ablation experiments:**
| Step | Remove | Replace with | Expected | Tests |
|------|--------|-------------|----------|-------|
| B1 | C1 (zone coords) | CC-discovered zones | Probably works | Same as FT09 A1 |
| B2 | C1+C3 (coords+BFS) | CC zones + greedy (click zone that changes most) | Partial — may work for easy levels | Greedy vs planned |
| B3 | C1+C2+C3 (coords+mechanics+BFS) | CC zones + random | 0/7 (proved by Step 1017) | Confirms minimum |
| B4 | C4 (camera offset) | Auto-detect from frame analysis | Probably works | Minor scaffolding |

**Predicted minimum prescription:** Zone identity + mechanics model (C2) + some planning (C3, possibly simplified). Camera offset auto-detectable.

---

## LS20 — Full prescription: TBD (Step 1018 pending)

**Prescription components (from source analysis):**
- C1: Wall layout per level (ihdgageizm positions)
- C2: Goal positions per level (rjlbuycveu positions)
- C3: Shape matching requirements (GoalColor, GoalRotation per level)
- C4: Collectible positions (npxgalaybz — extend step budget)
- C5: BFS path planning (start → shape match → goal)
- C6: Shape cycling mechanics (color/rotation/shape cycling via tagged sprites)
- C7: Step budget management (42 steps, 3 lives)

**Ablation experiments (after Step 1018):**
| Step | Remove | Replace with | Expected | Tests |
|------|--------|-------------|----------|-------|
| C1 | Walls (C1) | Collision detection by trying to move | Works (discovery by interaction) | Wall knowledge purgable |
| C2 | Goal position (C2) | Explore until frame matches goal indicator | Likely works if goal visible | Goal detection purgable? |
| C3 | Shape matching (C3) | Trial-and-error cycling at goal | Works if budget allows ~12 tries (4 shapes × 4 rotations × 4 colors... too many?) | Shape matching essential? |
| C4 | Collectibles (C4) | Ignore collectibles | Fails on tight-budget levels | Budget-dependent |
| C5 | Path planning (C5) | Random walk within budget | Fails (21 moves too few for random, proved by Step 1015) | Planning essential |

**Predicted minimum prescription:** Goal detection (simplified C2) + shape matching (C3, possibly trial-and-error) + some path planning (C5, even simple). Wall knowledge purgable. Collectible knowledge depends on budget tightness.

---

## Cross-Game Minimum

**What ALL games share that's essential:**
1. **Zone/target discovery** — knowing where to interact (CC discovery handles this)
2. **Effect observation** — seeing what each action does (frame diff handles this)
3. **Ordering/sequencing** — knowing which action WHEN (the universal gap)
4. **Mechanics inference** — learning game rules from interaction (unsolved)

**What's purgable across all games:**
- Exact coordinates (CC discovery)
- Pre-computed paths (interaction-based exploration)
- Level detection (frame change monitoring)
- Camera correction (auto-detectable)

**The substrate must discover:** action-effect mappings + ordering + game mechanics. From interaction, not from source analysis.

---

## RHAE Scoring Constraint (ARC-AGI-3)

Score = (human_baseline / agent_actions)^2 per level. The substrate must not just SOLVE levels but solve them EFFICIENTLY.

| Game | Level | Human Baseline | Max Budget (score>25%) | Max Budget (score>1%) |
|------|-------|---------------|----------------------|---------------------|
| FT09 | L1 | 17 | 34 | 170 |
| FT09 | L2 | 19 | 38 | 190 |
| LS20 | L1 | 21 | 42 | 210 |
| LS20 | L2 | 123 | 246 | 1230 |
| VC33 | L1 | 6 | 12 | 60 |

**Implication for ablation:** A mechanism that discovers the prescription in 10K steps but EXECUTES it in 20 actions scores well. A mechanism that needs 10K steps to stumble on the solution and has no way to replay it scores ~0%. The substrate needs LEARNING (discover once) + EXECUTION (replay efficiently), not just exploration.
