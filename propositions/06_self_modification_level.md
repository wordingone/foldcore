# Proposition 6: Self-Modification Level Determines Speed vs Reachability
Status: PARTIALLY FALSIFIED (Step 589)
Steps: 542, 584, 587, 589

## Statement
l_1 self-modification (data-driven placement of prescribed operations) accelerates exploration but does not expand the reachable set. Only l_pi self-modification (data-driven encoding change) expands reachability.

## Evidence
- Step 584 (20 seeds, 50K): SP and AM converge to 13/20 at 50K, but SP leads 9/6 at 10K. Penalty accelerates without expanding graph.
- Step 542 (Recode, 5/5): Self-modification of pi expands graph 440->1267 cells, improves 3/3->5/5.
- Step 587: Death-count penalty solves at 2338 vs 7749 -- faster, same graph.
- **FALSIFIED (Step 589):** Recode(K=16) 18/20 = LSH(K=16) 18/20 > LSH(K=12) 13/20. Advantage from K=16, not adaptive splitting. l_pi adds nothing on top of the K it uses. Hierarchy descriptively useful but not operationally predictive. Speed-vs-reachability holds at mid-budget (p<0.05) but collapses at 50K.

## Implications
The self-modification hierarchy categorizes mechanisms by what they modify but is not predictive of outcomes. More hash bits (frozen l_0 parameter) achieves same reachability as adaptive splitting (l_pi).

## Supersedes / Superseded by
N/A
