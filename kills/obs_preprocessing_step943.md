# Kill: Observation Preprocessing (Replacement)
Steps: 942, 943 | Trigger: LS20 < 250 both variants

## What worked
- Nothing. Both variants scored LS20=0 across all seeds.

## What failed
- Frame diff (942): consecutive game frames mostly identical → diff is mostly zeros → enc collapses → alpha_conc=50
- Pixel variance (943): variance map converges to static within ~100 steps → enc near-constant → delta_spr→0.005
- Both destroy positional scene structure that 916's navigation depends on
- CIFAR partially works (alpha_conc=15 not 50) because consecutive CIFAR images differ meaningfully

## What next family needs
- Must PRESERVE raw frame signal (position = information)
- Additive preprocessing (augment, not replace) — but this requires growing dims → GFS territory → killed
- OR: entirely new base architecture (not 916)

## Return condition
If a base architecture is found that doesn't depend on raw positional info for navigation, replacement preprocessing could be revisited. Under 916, raw obs is load-bearing.
