# Prescription Studies — Status as of 2026-03-27

Ground truth: fullchain JSON files in this directory. Each file = complete action sequence verified by engine chain replay.

## Summary

- **10/25 games fully solved** (all levels cleared)
- **12/25 games partially solved** (at least L1, some L2+)
- **3/25 games unsolved** (0 levels)

---

## COMPLETE (10 games — all levels solved)

| Game | Levels | Actions | Notes |
|------|--------|---------|-------|
| ft09 | 6/6 | 75 | Original preview game |
| ls20 | 7/7 | 311 | Original preview game |
| vc33 | 7/7 | 176 | Original preview game |
| lp85 | 8/8 | 79 | Abstract BFS (button permutations) |
| tr87 | 6/6 | 123 | — |
| sb26 | 8/8 | 124 | — |
| sp80 | 6/6 | 107 | — |
| cd82 | 6/6 | 140 | — |
| cn04 | 5/5 | 107 | — |
| tu93 | 9/9 | 185 | — |

---

## PARTIAL (12 games — some levels solved)

| Game | Solved | Total | Actions | Blocking issue |
|------|--------|-------|---------|----------------|
| re86 | 5 | 8 | 210 | L6-L8 unsolved |
| r11l | 2 | 6 | 65 | L3+ unsolved |
| s5i5 | 2 | 8 | 39 | L3+ unsolved |
| m0r0 | 2 | 6 | 15 | L3+ unsolved |
| su15 | 2 | 9 | 25 | L3+ unsolved |
| ar25 | 2 | 8 | 26 | L3+ unsolved |
| dc22 | 1 | 6 | 20 | L2+ — v4 BFS solver written, correct buttons identified |
| sc25 | 1 | 6 | 17 | L2+ unsolved |
| g50t | 1 | 7 | 17 | L2+ — multiple BFS attempts (v1-v4) |
| wa30 | 1 | 9 | 77 | L2+ unsolved |
| bp35 | 1 | 9 | 29 | L2+ unsolved |
| lf52 | 1 | 10 | 8 | L2+ unsolved |

---

## UNSOLVED (3 games — 0 levels)

| Game | Levels | Notes |
|------|--------|-------|
| ka59 | 7 | analytical.json + prescription.json exist, no BFS solution |
| sk48 | 8 | full_seq.json + prescription.json exist, no verified chain |
| tn36 | 7 | prescription.json only |

---

## Key Files

- **Fullchain JSONs**: `{game}_fullchain.json` — ground truth action sequences, chain-verified
- **Analytical**: `{game}_analytical.json` — intermediate analysis artifacts
- **Prescription**: `{game}_prescription.json` — level-by-level notes
- **Solver scripts**: `B:/M/the-search/solve_*.py`

## DC22 Specific Notes (for resumption)

DC22 is ready to run with `solve_dc22_v4.py`:
- L1: 20 actions (already solved)
- L2 solution path KNOWN from manual analysis:
  1. Click 'b' button (enc=2488, display 49,38) — opens vertical corridor
  2. Navigate to zbhi trigger at (16,46) — reveals 'a' button
  3. Click 'c' button (enc=1336, display 49,20) — toggle hhxv-upry1↔2
  4. Click 'a' button (zbhi-revealed) — toggle hhxv1↔2
  5. Navigate to goal at (22,4)
- v4 uses per-toggle-state button caching, full all-sprite state comparison
- Scans dx=38-63 only (buttons always on right side of display)

## Solver Inventory

| Script | Target | Status |
|--------|--------|--------|
| solve_dc22_v4.py | DC22 L1-L6 | Ready to run |
| solve_lp85_v6.py | LP85 L1-L8 | DONE |
| solve_g50t_l2_v4.py | G50T L2+ | Incomplete |
| solve_su15_*.py | SU15 L3+ | Incomplete |
| solve_sk48_bfs.py | SK48 L1+ | Incomplete |
| solve_tu93_bfs.py | TU93 L1-9 | DONE |
| solve_sb26_fullchain.py | SB26 L1-8 | DONE |

---

*Pause directive: Jun, 2026-03-27. Resume when directed.*
