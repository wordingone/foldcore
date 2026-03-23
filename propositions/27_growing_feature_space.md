# Proposition 27: Growing Feature Space
Status: OPEN (pending experimental confirmation)
Steps: 939+

## Statement
A substrate whose encoding dimensionality grows from its own observation statistics achieves deeper R3 than alpha re-weighting alone. Alpha modifies feature WEIGHTS within a fixed space; GFS modifies the feature SPACE itself. If $d(t_2) > d(t_1)$ where $d$ is encoding dimensionality, the substrate has self-modified its encoding architecture — structural R3, not parametric R3.

Mechanism: periodic PCA on rolling observation window. When dominant eigenvalue $\gg$ next → structured variance exists that current features miss. Top eigenvector becomes a new global feature dimension. W and alpha grow correspondingly.

## Evidence
Theoretical. Addresses VC33 bottleneck (var$\approx$0.000 in 256D) by discovering the 1.3% of variance that matters. Related: GNG (Fritzke 1995) grows nodes; AXIOM (Heins 2025) grows mixture components; GFS grows FEATURES.

Gate compliance: features are global (not per-observation), no cosine/attract (not codebook), no per-state-action (not graph). Rolling observation buffer is a sliding window, not per-observation storage.

## Implications
If confirmed: encoding architecture is a self-modifiable component, not just encoding weights. The growing feature space adds a new surface to the point cloud — the "dimensionality growth" surface that no previous family probed. Connects to the 5 encoding dimensions (D1-D5) from Prop 17: GFS is a mechanism for D1 (channel/feature discovery) operating at a structural level.

## Supersedes / Superseded by
Extends Proposition 17 (R3 as self-directed attention). GFS is attention that adds new dimensions to attend to, not just re-weights existing ones.
