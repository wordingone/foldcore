# Kill: GFS (Growing Feature Space)
Steps: 939, 939b | Trigger: LS20 < 250 both variants

## What worked
- PCA discovers observation structure (10-16 features added across games)
- Feature trigger (eigenvalue ratio > 2.0) fires reliably
- Warm-up exclusion partially helped (seed 2: 0→35)

## What failed
- Zero-init W_pred rows create bootstrap failure (939: alpha_conc=50 from step 1)
- Even with warm-up, more dims → harder W_pred training → faster alpha concentration in BASE dims
- Cross-game alpha contamination: extra features carry miscalibrated alpha across game transitions
- Degradation worsens with chain length (more features = more contamination)

## What next family needs
- Fixed dimensionality (alpha/W_pred assume fixed dims)
- Change encoding INPUT, not encoding ARCHITECTURE
- Must not introduce new W_pred training burden
- Must not create cross-game state that accumulates errors

## Return condition
If alpha/W_pred system is replaced with a mechanism that handles dynamic dimensionality, GFS could be revisited. Current 916 base is incompatible.
