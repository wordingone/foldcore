# Proposition 10: The Feature-Ground Truth Coupling Barrier
Status: CONFIRMED
Steps: 577d, 581d, 589

## Statement
The gap from l_1 to l_F in the self-modification hierarchy is the gap between RECORDING ground truth events and PREDICTING them.

## Evidence
- Step 577d: Pixel statistics navigate to WRONG cells (0/5 L1). Visual statistics not predictive of exit-relevant features.
- Step 581d: Death events as post-hoc soft penalties achieve 4/5 vs argmin 3/5. Ground truth feedback improves navigation when applied as l_1 (retrospective marking).
- Step 589: Encoding self-modification (l_pi, Recode) provides no advantage over fixed encoding at same K. System modifies WHERE hyperplanes split but not WHAT splits are FOR.

At l_1, ground truth predictor p is retrospective (marks states after events). At l_F, p is prospective (predicts which states WILL produce events). Prospective prediction requires features correlating with ground truth, but R1 prohibits optimizing for that correlation.

## Implications
Population-level R3 (GRN architecture) may bridge the gap: if multiple encodings compete and environmental ground truth selects the winner, the surviving encoding correlates with ground truth via selection, not optimization.

## Supersedes / Superseded by
N/A
