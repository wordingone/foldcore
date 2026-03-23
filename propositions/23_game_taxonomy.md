# Proposition 23: Game Taxonomy by Progress Structure
Status: CONFIRMED
Steps: 800b, 895f, 868b

## Statement
Interactive environments partition into two classes by progress structure:
(a) **Observation-sufficient (monotonic):** Progress produces observable change. Change-tracking sufficient. LS20: movement changes observation -> change leads to unexplored territory.
(b) **History-dependent (sequential):** Progress requires specific action sequences. Change-tracking identifies action VOCABULARY but not GRAMMAR. FT09: any click produces change, but only a specific 7-click sequence solves the puzzle.

## Evidence
Alpha-weighted 800b achieves +32% over baseline on LS20 (observation-sufficient) and L1=0 on FT09 (sequential), despite alpha identifying informative dimensions on BOTH games. The encoding problem is solved on both; action selection diverges by game class.

Connection to formal language theory: Monotonic games are "regular" (memoryless automaton navigates). Sequential games are "context-free" or higher (require stack-like memory). Graph ban removes the stack.

## Implications
Pre-ban graph solved BOTH classes. Post-ban, change-tracking only solves monotonic class. Sequential games require mechanism for action SEQUENCES without per-state memory. Whether prediction can substitute for stack memory is open.

## Supersedes / Superseded by
Extended by Proposition 23b (combinatorial barrier).
