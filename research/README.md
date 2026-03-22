# FluxCore

Every technology follows birth → scale → compression. Vacuum tubes became integrated circuits. The current AI stack — transformers, KV cache, frozen weights, bolted-on tools, separate optimizer and inference loops — is in late-scale. The compression is coming. FluxCore is an attempt to derive the compressed form: a single operation where memory, learning, inference, and perception are the same thing. The fold equation is what remains after recursive compression of the full stack.

This is a research artifact, not a product claim. Every behavioral claim below has a test and exact numbers.

---

## The Fold

```
alr = baseLr × (1 + k × surprise)
u[i] = s[i] + alr × r[i] + (alr × 0.5) × |s[i] - r[i]| × grad[i] + memW × m[i]
s = normalize(u)
```

Where:
- `s` ∈ S^(d-1): self-state on the unit hypersphere
- `r` ∈ S^(d-1): reality (input)
- `m` ∈ S^(d-1): active memory (selected from attractor pool)
- `surprise` = mean(|s - r|)
- `grad[i]` = s[(i+1) % d] - s[(i-1) % d] (central difference, ring topology)

One equation. Four things happening simultaneously: tracking reality, adaptive learning rate, spatial gradient, memory pull. State lives on the unit sphere throughout — no renormalization required between operations.

---

## What It Is Not

- Not consciousness. Not AGI. Not a transformer replacement.
- The "active inference" in the current codebase is prediction-error-driven memory-directed steering. It shows measurable advantage in controlled conditions. It is not free-energy minimization in the Friston sense.
- The hierarchy is three stacked instances with a normalization-preserving inter-level signal. It tracks distribution epochs. It is not rule extraction in a symbolic sense.
- Call things what they are.

---

## Verified Behaviors

**Convergence** (Claims 1, 4)
The fold tracks arbitrary unit-vector distributions. Surprise drops from ~0.14 to <0.001 within 250 ticks at DIM=64; from ~0.05 to <0.001 at DIM=512. Verified at DIM=64, DIM=512, DIM=8192 in JavaScript; at DIM=64 and DIM=512 in CUDA (RTX 4090, sm_89).

The fold is not reducible to EMA: velocity-based anticipation (avg gain 0.00758 at DIM=64), distributional shift detection (1882σ spike at DIM=64, 6044σ at DIM=8192), and surprise-adaptive learning rate each require mechanisms absent from EMA.

**Self-organizing memory** (Claim 2)
The dynamic attractor pool grows one memory per novel distribution encountered. After an A→B→C→D→A sequence, the pool contains exactly 4 memories with perfect distribution alignment (dot=1.0000 for each). Verified at DIM=64 in JavaScript and CUDA.

**Accelerated reacquisition** (Claim 3)
After learning a distribution and returning to it after an interlude, convergence is 5.5–6.1× faster than first encounter. Verified at DIM=64 (6.1×), DIM=512 (5.5×), DIM=8192 (5.7×).

**Hierarchical epoch tracking** (Claim 6)
Three stacked instances, each receiving the lower level's EMA-smoothed velocity × surprise as input. After a normalization bug fix (scalar surprise was destroyed under L2 normalization), each level forms one memory per distribution epoch. L2 attractors show dot=-0.852 alignment with L0 distribution vectors — meta-level encoding references the raw distribution space.

**GPU scaling** (Claim 7)
CUDA implementation compiles and runs at DIM=64, DIM=512, and DIM=4096 on an RTX 4090 (sm_89), producing convergence behavior matching the JavaScript reference (steady-state surprise 0.0003 at all three dimensions).

**Real-world data** (Claim 8)
Fed 1920 CSI construction specification text embeddings (MiniLM-L6-v2, DIM=384, L2-normalized) through the entity version. 357/359 attractor memories aligned with recognizable CSI division centers (dot > 0.3), with mean best similarity 0.5710. All 33 divisions represented. Surprise plateau at ~0.038 (versus ~0.0003 on synthetic data) — real text embeddings have genuine within-division semantic variance that synthetic orthogonal vectors don't have. Some memories span division boundaries (e.g., div01/div26 cross-alignment at 0.8777/0.8331) — reflects genuine semantic overlap in construction specifications, not a failure mode.

---

## In Progress

**Action mechanism** (Claim 5)
The system generates a prediction-error-driven action vector: `normalize(activeMem) × predictionErrorMagnitude × actionGain`. In a controllable synthetic environment with blend ratio α=0.1 (10% external reality, 90% action), the active agent shows 24.7% lower surprise than a passive agent during post-perturbation recovery. Advantage scales with control authority (α). This is not validated on real-world data and the mechanism has not been verified against a formal active inference specification.

---

## Running It

```bash
# Dual-memory baseline — T1-T4 test suite
node fluxcore_true.mjs

# Dynamic attractor genesis, hierarchy, action mechanism
node fluxcore_entity.mjs
```

Requires Node.js. No dependencies.

CUDA (requires NVIDIA GPU, CUDA toolkit, WSL2 on Windows):
```bash
# Compile
wsl bash -c 'bash /path/to/fluxcore/compile.sh'

# Run at DIM=64 (default) or higher
wsl bash -c './fluxcore_entity 64'
wsl bash -c './fluxcore_entity 512'
wsl bash -c './fluxcore_entity 4096'
```

---

## Results

All test outputs are in `results/`:

| File | Contents |
|------|----------|
| `baseline_true.txt` | T1-T4 at DIM=64/512/8192 |
| `baseline_entity.txt` | TEST 1-3 baseline (pre-fix) |
| `hierarchy_fixed.txt` | Hierarchy post-normalization fix |
| `hierarchy_audit.txt` | L2 attractor fingerprints, pairwise dots |
| `entity_reacquisition.txt` | A→B→A reacquisition test |
| `entity_multidist.txt` | A→B→C→D→A, 4-distribution test |
| `cuda_baseline.txt` | CUDA DIM=64 run |
| `cuda_dim512.txt` | CUDA DIM=512 run |
| `active_exp2.txt` | Action mechanism experiment |
| `active_alpha_sweep.txt` | α sweep, actionGain=10.0 |
| `realdata_first.txt` | FluxCore on CSI corpus, DIM=384 |
| `claims_summary.md` | Evidence-gated claims, full detail |

---

## License

TBD — [director] to decide.

---

*FluxCore is not the finished product. It is the closest approximation that can be built on current hardware without lying about what it is.*
