# Research State — Live Document
*Read by /checkpoint skill. Source of truth for active work.*

---

## Phase 1 Conclusion (416 experiments)

```
CURRENT SYSTEM: process(x, label=None). ~22 lines. LVQ + growing codebook.
  - Competitive learning with cosine similarity (LVQ, Kohonen 1988)
  - Growing codebook with novelty-triggered spawning (cf. Growing Neural Gas, Fritzke 1995)
  - Top-K per-class vote for classification
  - Self-generated targets (prediction as label)
  - State-derived threshold (median NN distance)
  - 94.48% P-MNIST AA, 0pp forgetting (softmax voting, Step 425)
  - Previous: 91.20% with top-K scoring (not competitive with replay-based CL methods)

HONEST ASSESSMENT (per external review):
  - Mechanisms are NOT novel (LVQ + GNG from 1988/1995)
  - ARC-AGI-3 results are biased random walk, not intelligence
  - Stage progression was self-assessed and circularly validated
  - "22 lines" obscures: avgpool, centered_enc, F.normalize, random projection, evaluation code
  - The system is brittle: 5 implementation details each independently fatal

WHAT WAS FOUND (genuine contributions):
  - The constitution: testable framework for recursive self-improvement (architecture-independent)
  - The constraint map: 26 universal (7 provisional) + 9 intent constraints from experiments (see CONSTRAINTS.md)
  - The noise insight: stochastic coverage via cosine saturation IS the exploration engine
  - Dynamics ≠ features: healthy codebook dynamics achievable at any dim (p=0.75), but features require encoding
  - Fixed point: the research procedure IS structurally identical to the algorithm it found

WHAT WAS NOT FOUND:
  - The atomic substrate. LVQ is not it.
  - Self-modifying metric (R3 — open, central problem)
  - Representation discovery from raw observations (I1 — open)
  - Purposeful exploration (current system is biased random walk)
  - Temporal reasoning, transfer, richer output

SCALING HYPOTHESIS (not law — 2 data points, 1 trivial):
  steps ≈ branching_factor × path_length. Needs validation on more games.

ARC-AGI-3 (Steps 343-416): 3/3 preview games Level 1 with PRESCRIBED encodings.
  LS20: 16x16 avgpool + centered_enc. Level at ~26K steps. 60% reliable.
  FT09: 69-class click-space. Level at step 82.
  VC33: 3-zone mapping (PRESCRIBED — looked behind scenes. Not autonomous discovery).
  Sequential resolution trial (Step 414): substrate discovers 16x16 from raw input via gameplay.
  Encoding still prescribed per game type (I1 open — representation discovery).

PHASE 2 DIRECTION: See CONSTRAINTS.md. The next substrate is specified by U1-U24 + I1-I9.
```
Step 377: Raw 64x64 bootstraps codebook (1736 entries). PASSES mechanically.
Step 378: Raw 64x64 50K steps. 0 levels. Codebook builds but sim too uniform (0.984±0.009).
  Timer isn't the issue (0.05% of dims). Static background IS (63% of pixels). Signal = 0.3%.
Step 379: Centering at 64x64 — no effect. Same sim stats.
  The gap: V @ x at 4096 dims washes 0.3% signal in 99.7% noise.
  16x16 avgpool worked by accidentally doing feature selection (12 pixels → 4.7% of encoding).
  I1 = learned projection. The substrate discovers which pixels matter from its own state (R3).
  Chollet: "brute-force dense sampling is benchmark hacking, not intelligence."
  The substrate explores but doesn't reason. The gap = encoding self-discovery = intelligence.
CURRENT STEP: **Steps 1104-1151: 51 experiments. 0% wall = DISCOVERY GAP (PB19).** v30 ONLY draw-robust result (ARC=0.33). 51 experiments across every axis: encoding (4), goal (6), action space (4), patience (3-200 steps), forward models (3), novelty (3), state-conditioning (2), action sequences (1), periodic alternation (1). ALL same wall. PB19 confirmed: "Generic exploration = 0 even unconstrained." Bottleneck = DISCOVERY. 1151+ experiments.
  **Step 1133 FAIL (defense v47, multi-strategy meta-reactive, seed 54182):** 5 sequential strategies (keyboard→saliency click→ultra-patient→random click→brute force), 2000-step budget each. ARC=0.0000, 2/5 L1. Systematic fallback adds overhead but no signal.
  **Step 1134 FAIL (defense v48, full block scan + lagged detection, seed 71526):** 256-block scan, 10-step lag window. ARC=0.0000, 2/5 L1. Lagged change detection finds nothing immediate detection missed.
  **Step 1135 FAIL (prosecution v34, random feature forward model, seed 38521):** cos(W_random@enc+b) nonlinear feature map → per-action W_fwd (256×256). ARC=0.0000, 2/5 L1. Forward model only modeled keyboard actions — irrelevant for click-only 0% games. Nonlinearity is not the bottleneck.
  **Step 1136 FAIL (defense v49, change-maximizing reactive, seed 92417):** Maximize |enc_t - enc_{t-1}| instead of minimize |enc_t - enc_0|. ARC=0.0000, 2/5 L1. WORSE than v30 — consecutive change chases oscillatory noise. Distance-to-initial confirmed as better goal.
  **Step 1137 FAIL (defense v50, cycle-breaking reactive, seed 85213):** Detect state return at lag 5/10/20, exclude cycling action. ARC=0.0000, 2/5 L1. Cycle detection adds nothing.
  **Step 1138 FAIL (prosecution v35, spatial click forward, seed 64739):** 16 salient click regions + EMA per-action forward model. ARC=0.0000, 2/5 L1. Forward model can't discover click targets in opaque games.
  **Step 1139 FAIL (defense v51, LSH raw novelty, seed 37284):** 8-bit LSH hash → 256 buckets, visit count novelty. ARC=0.0000, 2/5 L1. Novelty seeking same wall as distance-to-initial.
  **Step 1140 FAIL (prosecution v36, alpha-weighted novelty, seed 19283):** W_pred → alpha → hash(alpha*enc) → visit counts. ARC=0.0000, 2/5 L1. Alpha-weighted novelty = same wall as raw novelty. Controlled comparison inconclusive (different seed/draw).
  **Step 1141 FAIL (prosecution v37, pixel scan diagnostic):** Systematic 64x64=4096 pixel scan. ARC=0.0000, 2/5 L1. GAME_4 regressed 100%→0% (scan consumed budget). 0% games do NOT respond to any single click at any pixel position. Most exhaustive action-targeting test possible.
  **Step 1142 FAIL (defense v52, Sobel edge encoding):** Gradient magnitude per block instead of avgpool4. ARC=0.0000, 2/5 L1. GAME_2 regressed 100%→0%. Edge encoding LOSES brightness info. Encoding type is not the bottleneck.
  **Step 1143 FAIL (defense v53, temporal diff encoding):** Progress = |enc_t - enc_{t-1}| > threshold. ARC=0.0000, 2/5 L1. Frame-to-frame changes zero on 0% games regardless of action.
  **Step 1144 FAIL (defense v54, raw 4096D pixel encoding):** Full resolution obs.ravel() (no averaging). ARC=0.0000, 2/5 L1. 0% games produce ZERO pixel change. Encoding resolution definitively ruled out as bottleneck.
  **Step 1145 FAIL (defense v55, ultra-patient 200-step holds):** Hold each action for 200 steps to detect delayed responses. ARC=0.0000, 2/5 L1. No delayed responses. Patience ruled out as variable.
  **Step 1146 SIGNAL (defense v56, state-conditioned action, graph ban LIFTED):** LSH→256 states, per-(state,action) EMA effect. ARC=0.0001, **3/5 L1**. GAME_2 L1=100% SOLVED=100% max_lvl=1. First 3/5 from non-v30 architecture.
  **Step 1147 FAIL (v56 robustness):** Different draw. ARC=0.0000, 2/5 L1. Back to standard pattern on this draw.
  **Step 1148 PARTIAL (v56 robustness):** GAME_3 L1=30% SOLVED=30%. ARC=0.0001. Marginal signal on 2/3 draws. State-conditioning helps on Mode 2 (responsive) games, not Mode 1 (inert). NOT pure draw luck but draw-dependent.
  **Step 1149 FAIL (prosecution v38, state-conditioned forward model):** Per-(state,action) 256D direction EMA + diversity tiebreaker. ARC=0.0000, 2/5 L1. Direction vectors don't add value over scalar magnitude when games produce zero pixel change.
  **Step 1150 FAIL (defense v57, 2-action sequence scan):** Systematic 7×7=49 keyboard pairs + 7×8 keyboard→click pairs. Compare sequence change vs sum of individual changes → no synergistic sequences found. ARC=0.0000, 2/5 L1. Games don't require 2-action sequences to trigger responses. Action SEQUENCES ruled out as bottleneck.
  **Step 1151 FAIL (defense v58, periodic alternation scan):** Rapid a,b,a,b... alternation for 30 steps per pair. All 21 keyboard pairs + 7 same-action repeats + 28 keyboard→click pairs. ARC=0.0000, 2/5 L1. Rhythmic/periodic interaction produces no response on 0% games. Frequency-based mechanics ruled out.
  **Step 1152 SIGNAL (defense v59, empowerment-based action selection):** Per-action outcome distributions via Welford mean+var, pairwise distinguishability (Cohen's d analog). ARC=0.0040, **3/5 L1**. GAME_4: L1=100%, SOLVED=100%, max_lvl=1. First substrate to use CONTRAST signal (distinguishability of outcomes) instead of activation signal (change magnitude). From information theory.
  **Step 1153 FAIL (v59 robustness):** Different draw. ARC=0.0000, 2/5 L1. Back to baseline. Draw-dependent signal, same pattern as v56. Empowerment helps on Mode 2 games (some draws) but not Mode 1.
  **Step 1154 FAIL (defense v60, combination set scan):** All 2/3/4/5/6/7-element keyboard subsets tested with reset steps between. ARC=0.0000, 2/5 L1. No action combination triggers response on Mode 1 games. Unordered sets ruled out (ordered sequences already ruled out by v57).
  **Step 1155 FAIL (defense v61, empowerment-filtered reactive):** Empowerment estimation → top-3 controllable actions → v30-style distance-to-initial reactive. ARC=0.0000, 2/5 L1. Filtering by empowerment doesn't improve v30's reactive approach. On this draw, empowerment filtering may exclude useful actions.
  **Step 1156 FAIL (defense v62, online empowerment reactive):** No estimation phase — act+learn from step 1. Empowerment+exploration+reactive improvement rate. ARC=0.0000, 2/5 L1. Eliminating estimation phase doesn't help. Online empowerment = same wall.
  **Step 1157 SIGNAL (defense v63, action-space-adaptive reactive):** Branch on n_actions: keyboard→reactive argmin, click→empowerment scan. ARC=0.0015, **3/5 L1**. GAME_3: L1=100%, max_lvl=1. Action-space branching picks up responsive games. Draw-dependent like v59.
  **Step 1158 FAIL (defense v64, interleaved dual-strategy):** Even=keyboard reactive, odd=click empowerment on every game. ARC=0.0000, 2/5 L1. GAME_3 regressed 100%→0% (CHAIN KILL FAIL). Interleaving halves effective budget per strategy — WORSE than branching.
  **Step 1159 FAIL (defense v65, bidirectional reactive):** Approach first (minimize dist-to-initial), auto-switch to retreat (maximize dist) after 2000 steps. ARC=0.0000, 2/5 L1. Direction of distance goal is not the bottleneck.
  **Step 1160 FAIL (defense v66, sustained-hold reactive):** v30 with 10-step holds before evaluating (tests delayed-effect games). ARC=0.0000, 2/5 L1. Sustained holds don't unlock new games. Rapid switching (v30) ≈ sustained holds.
  **Step 1161 SIGNAL (defense v67, MI-detected reactive):** MI detection (fixed formula, ℓ₁) + reactive cycling. ARC=0.2000, **3/5 L1**. GAME_2: SOLVED=100%, max_lvl=1, ARC=1.0000 (perfect efficiency). NO L2 — v16's L2 likely requires ℓ_π attention update, not just MI detection. MI detection is ℓ₁-compatible but insufficient for L2 alone.
  **Step 1162 FAIL (defense v68, adaptive MI reactive):** Start v30-fast, fallback to MI estimation after 200 steps of no progress. ARC=0.0000, 2/5 L1. Responsive games solve in <200 steps (avg_t=0.0s), MI fallback triggers on inert games but can't unlock them. Action selection method is not the bottleneck — inert games don't respond to ANY interaction.
  **Step 1163 MARGINAL (defense v69, coarse-to-fine click search):** Systematic 4×4→8×8→16×16 click grid search (no saliency assumption) + v30 keyboard reactive. ARC=0.0003, **3/5 L1**. GAME_3: SOLVED=100%, max_lvl=1. Found responsive click target that saliency missed. But poor efficiency (search phase burns budget). 2 inert games still at 0%.
  **Step 1164 FAIL (defense v70, transition-triggered action memory):** Record last 50 actions before level solve, replay on next level. ARC=0.0000, 2/5 L1. Responsive games solve too fast (avg_t≤0.1s) to build useful memory. Inert games never trigger transitions → memory stays empty. Success signal is available but not actionable at ℓ₁.
  **Step 1165 FAIL (defense v71, entropy-reactive):** Minimize Shannon entropy of observation (progress = organization). ARC=0.0000, 2/5 L1. Entropy metric ≈ distance metric on these games. Progress metric is not the bottleneck — inert games produce zero change on ALL metrics.
  **Step 1166 FAIL (defense v72, modal goal reactive):** Goal = most common observation per block (frequency counting). ARC=0.0000, 2/5 L1, 181.8s (slow). Modal goal ≈ initial goal on these games. Frequency-based goal estimation doesn't help — the modal observation IS the initial observation when games are inert.
  **Step 1167 FAIL (prosecution v39, MI + alpha attention ℓ_π, draw 1):** MI detection + alpha attention update + W_pred (256×256). ARC=0.0000, 2/5 L1. NO L2. Controlled comparison vs v67 (defense MI ℓ₁): same result. Kill criterion: 1/3 draws tested.
  **Step 1168 FAIL (prosecution v39, draw 2):** ARC=0.0000, 2/5 L1. NO L2. Kill criterion: 2/3 draws tested.
  **Step 1169 FAIL (prosecution v39, draw 3):** ARC=0.0000, 2/5 L1. NO L2. CHAIN KILL (GAME_4: 100%→0%). **PROSECUTION v39 KILLED: 3/3 draws, zero L2, attention update actively hurts responsive games. v16's L2 (Step 1074) was draw-dependent, not mechanism-dependent.**
  **Step 1170 CRITICAL (random baseline, draw 1):** Pure random action selection. ARC=0.0024, **3/5 L1**. GAME_3: SOLVED=100%, max_lvl=1. Random matches or exceeds engineered substrates on L1 count.
  **Step 1171 (random baseline, draw 2):** ARC=0.0002, **3/5 L1**. GAME_2: L1=90%, SOLVED=90%, max_lvl=1.
  **Step 1172 (random baseline, draw 3):** ARC=0.0000, **3/5 L1**. GAME_3: L1=10%, max_lvl=1.
  **RANDOM BASELINE CONCLUSION (3 draws): 3/5, 3/5, 3/5.** Random consistently gets 3/5 L1 while engineered substrates typically get 2/5. Random explores the FULL action space uniformly (including all click positions). Engineered substrates limit themselves to 7 KB + 8-16 click regions, MISSING responsive click targets. v30 is more ACTION-EFFICIENT (ARC=0.33 vs random's 0.002) but finds FEWER games. The bottleneck is ACTION SPACE COVERAGE, not mechanism intelligence.
  **Step 1173 FAIL (defense v73, wide-search reactive):** ARC=0.0000, 2/5 L1. Random exploration (200 steps) → reactive exploitation of discovered responsive actions. WORSE than pure random (3/5). Coverage requires CONTINUOUS exploration across all 10K steps, not explore-then-exploit. 200 samples from thousands of possible actions is insufficient to discover responsive targets reliably.
  **Step 1174 FAIL (defense v74, sequence-reactive):** ARC=0.0008, 3/5 L1. Tests if 0% games need action PAIRS (A then B). Matches random on L1 count but 0% wall persists. Sequences are not the missing ingredient. GAME_2: 10% L1, max_lvl=1 (1 seed found something).
  **Step 1175 FAIL (defense v75, epsilon-greedy reactive):** ARC=0.0011, 3/5 L1. Continuous 30% random exploration + 70% v30 reactive cycling. Matches random on L1 count. GAME_4: 20% L1, max_lvl=1. 0% wall persists — action space coverage helps find responsive games but can't make inert games respond.
  **ACTION SPACE COVERAGE CONCLUSION (v73-v75):** 3 experiments testing the random baseline insight. v73 (explore-then-exploit): 2/5, WORSE. v74 (sequences): 3/5. v75 (epsilon-greedy): 3/5. Coverage matches random's 3/5 but NEVER exceeds it. The 0% wall is NOT an action space coverage problem — inert games don't respond to ANY action (single, paired, random, systematic). The wall is structural.
  **Step 1065 DEBATE FAIL (defense v15):** Sustained probing + per-action tracking + two-pass cascade + ℓ₁ SPSA. GAME_2=20%, GAME_3=0%, GAME_4=0%. Chain 3/5 PASS. Sustained probing hypothesis FALSIFIED: holding KB actions 50 steps, thresholds 0.01/0.003, evo sequences max 15 — none crack the 0% wall. All probe parameter variations exhausted.
  **Step 1066 DEBATE FAIL (prosecution v14):** Statistical micro-change detection (z-score normalization per dimension, ℓ_π). GAME_2=80%, GAME_3=70%, GAME_4=0%. Chain 4/5. Only 1 game at 0% (vs defense v15's 2). Z-score detection finds responsive games that absolute thresholds miss. 0% wall persists on 1/3 games. fully_solved bug fixed (done flag → level>0 check). API key loaded, pool still 25 games.
  **Step 1067 DEBATE FAIL (defense v16):** Z-score + SPSA (ℓ₁) — same cascade and z-score detection as v14, only fitness function differs. GAME_2=0%, GAME_3=0%, GAME_4=0%. ALL THREE ZERO. Different game draw from v14 (caveat: not controlled for games), but pattern is stark: z-score detection + ℓ_π attention = 80%/70%/0% (v14) vs z-score detection + ℓ₁ SPSA = 0%/0%/0% (v16). Defense concession #7: ℓ_π attention is the active ingredient, not z-scores alone.
  **Step 1068 DEBATE FAIL (prosecution v15):** Multivariate Mahalanobis + ℓ_π attention. 5 games drawn (PRISM config change?). GAME_1=100%, GAME_2=80%/SOLVED=80%, GAME_3=0%, GAME_4=100%, GAME_5=100%. Chain 4/5. GAME_2 first non-zero SOLVED rate. 0% wall persists on 1/5 games. 170.9s.
  **Step 1069 DEBATE FAIL (defense v17):** Same Mahalanobis detection + block-reduced SPSA (ℓ₁, 64 dims). GAME_1=100%, GAME_2=80%/SOLVED=80%, GAME_3=10%/SOLVED=10%, GAME_4=90%/SOLVED=90%, GAME_5=100%. Chain 5/5 — NO ZERO GAMES. Chain kill (GAME_2 regression). 149.5s. **Initially appeared to challenge concession #7. Steps 1070-1071 showed NO ZEROS was draw luck.**
  **Step 1070 DEBATE FAIL (prosecution v15, seed 9999):** Mahalanobis + ℓ_π. GAME_4=0%. Chain 4/5. 1 zero game.
  **Step 1071 DEBATE FAIL (defense v17, seed 31415):** Mahalanobis + block-reduced SPSA (ℓ₁, 64D). GAME_3=0%, GAME_4=0%. Chain 3/5. 2 zero games. **Step 1069's NO ZEROS was draw luck. Concession #7 stands. Both substrates hit 0% wall on ~1/3 of random games. Wall is structural — neither ℓ-level nor detection method breaks it.**
  **Step 1072 DEBATE FAIL (prosecution v15, seed 54321):** Mahalanobis + ℓ_π. GAME_2=0%, GAME_3=0%, GAME_4=100%/SOLVED=100%. Chain 3/5. GAME_4 fully solved — first 100% SOLVED in sprint. 2 zero games.
  **Step 1073 FAILED:** Defense v17, seed 54321 — seed consumed by Step 1072 (seed reuse prevention). No results.
  **PB26 aggregate (v15 vs v17):** Prosecution v15 (ℓ_π): 11/15 responsive (73%). Defense v17 (ℓ₁ 64D): 8/10 responsive (80%). No statistically significant difference. PB26 = PROVISIONAL, approximate parity. Bottleneck = opaque games, not goal function.
  **Step 1074 (prosecution v16, MI-based, seed 77777):** MI action informativeness + ℓ_π attention. GAME_1=100%, GAME_2=100%/SOLVED=100%/**max_lvl=2**, GAME_3=0%, GAME_4=0%, GAME_5=100%. Chain 3/5. DEBATE FAIL (2 zeros). **FIRST L2 IN DEBATE SPRINT (30+ experiments).** MI-weighted attention maintained useful structure across level transition. 0% wall persists (2/5 games). 212.7s.
  **Step 1075 (defense v18, MI + block SPSA ℓ₁, seed 88888):** Same MI detection, block-reduced SPSA goal. GAME_1=100%, GAME_2=0%, GAME_3=90%/SOLVED=90%/max_lvl=1, GAME_4=0%, GAME_5=100%. Chain 3/5. DEBATE FAIL (2 zeros). No L2. 180.9s.
  **Step 1076 (prosecution v16 reproduction, MI+ℓ_π, seed 11111):** GAME_1=100%, GAME_2=30%/SOLVED=30%/max_lvl=1, GAME_3=0%, GAME_4=0%, GAME_5=100%. Chain 3/5. DEBATE FAIL. **No L2. Step 1074's L2 was draw luck.** 263.8s.
  **Step 1077 (prosecution v17, paired-action MI, seed 22222):** GAME_2=0%, GAME_3=0%, GAME_4=80%/SOLVED=80%/max_lvl=1. Chain 3/5 (2 zeros). DEBATE FAIL. Paired-action probing does NOT break 0% wall. 160s.
  **Step 1078 (defense v19, temporal bisection, seed 33333):** GAME_2=0%, GAME_3=100%/SOLVED=100%/max_lvl=1, GAME_4=0%. Chain 3/5. DEBATE FAIL. 200-step random walks produce no response on opaque games. 257s.
  **Step 1079 DIAGNOSTIC (seed 44444):** TWO distinct 0% failure modes. Mode 1 (near-inert): pixel_var=0.001, detection problem. Mode 2 (responsive-unsolved): pixel_var=0.37, mechanism problem. First separation of failure modes.
  **Step 1080 (defense v20, goal-contrast, seed 55555):** ALL 3 games 0%. Chain 2/5. Goal-contrast fitness = zero improvement. Mode-as-goal fails. 237s.
  **Step 1081 (prosecution v18, sub-threshold accumulation, seed 66666):** ALL 3 games 0%. Chain 2/5. 1000x lower thresholds + 50-frame accumulation = zero. **Mode 1 signal is noise, not real game response.** 329s.
  **Step 1082 DIAGNOSTIC (Mode 2, seed 77778):** Chain 4/5, 136s. KEY: (1) MI near-zero on KB actions — signal is directional (mean), not variable (variance). (2) Per-action progress detects what MI misses: kb1=+0.37, kb3=+0.49 TOWARD initial; kb6=-0.50 AWAY. (3) **OSCILLATION: 47/74 sign changes under best-action repetition.** Game bounces back and forth. (4) Distance from initial monotonically increases — 0/576 steps return within 10%. State drifts, never returns. (5) Mean obs stable (4.67→4.70) — changes are structural/local, not global. **DIAGNOSIS: Mode 2 bottleneck = oscillation. Actions produce directional effects but don't accumulate. Substrate needs state-dependent action selection ("if X, do A; if Y, do B"). Sequence evolution is the wrong architecture.**
  **Step 1083 DEBATE FAIL (prosecution v19, seed 99199):** GAME_2=0%, GAME_3=0%, GAME_4=0%. Chain 2/5 (CIFAR only). ARC=0.0000. State-dependent action-value with progress encoding in alpha-weighted space. GLOBAL per-action EMA deltas average out oscillation signal — mean progress per action ≈ 0 in oscillating games. **Same result as cascade evolution. State-dependent selection needs STATE-LOCAL values, not global averages.** Jun directives applied: ARC Prize scoring live, distinct architectures mandated.
  **Step 1084 DEBATE FAIL but SIGNAL (defense v21, seed 41277):** GAME_2=0%, GAME_3=0%, GAME_4=100%/SOLVED=100%/max_lvl=1/ARC=0.2973. Chain 3/5. ARC Prize total: 0.0595. **FIRST ARC GAME SOLVE IN DEBATE (Steps 1077-1084).** Reactive switching (zero learned params, pure immediate comparison) = defense architecture. 2/3 ARC games still 0%.
  **Step 1085 DEBATE FAIL but SIGNAL (prosecution v20, seed 58342):** GAME_2=0%, GAME_3=0%, GAME_4=90%/SOLVED=90%/max_lvl=1/ARC=0.0037. Chain 3/5. ARC Prize total: 0.0007. Attention-trajectory (buffer-based state-local retrieval + alpha encoding). **Prosecution also solves 1/3 ARC games, but 80x less action-efficient than defense (ARC 0.0037 vs 0.2973).** Different draws — not controlled. Both sides have signal, 0% wall persists on 2/3 games.
  **Step 1086 CHAIN KILL (defense v22, seed 73019):** GAME_2=0% (regressed from 100%), GAME_3=0%, GAME_4=0%. Chain 2/5 FAIL. Multi-scale reactive (3 pooling scales + action memory) strictly worse than v21's single-scale. **Added complexity hurts. Defense v21 (zero params, single scale) = peak defense. "Simpler is better" confirmed.**
  **Step 1087 FAIL (prosecution v21, seed 62801):** GAME_2=0%, GAME_3=0%, GAME_4=0% (regressed from 90%). Chain 2/5. ARC=0.0000. Alpha-projected attention (top-K dim selection, adaptive K 64→16) strictly worse than v20's full-dim attention. **Same pattern as defense: added complexity degrades.** Prosecution v20 (simple attention-trajectory) = peak prosecution. Trend: v19=0/3, v20=1/3, v21=0/3.
  **Step 1088 FAIL (defense v23, seed 84521):** GAME_2=0%, GAME_3=0% (regressed from 100%), GAME_4=0%. Chain 2/5. ARC=0.0000. Anti-oscillation reactive (ring buffer K=30, OSC_THRESH=0.5) interfered with working games. Oscillation flags force switches even when current action is the only one making progress. **3rd consecutive defense complexity-kill. PB30 n=3.**
  **Step 1089 marginal signal (defense v24, seed 39174):** GAME_2=0%, GAME_3=0%, GAME_4=20%/SOLVED=20%/max_lvl=1/ARC=0.0014. Chain 3/5, ARC=0.0003. Dual-gradient run-and-tumble (E. coli chemotaxis). Marginal: 2/10 seeds on GAME_4. Better than v22/v23 but 1000x below v21. No chain kill.
  **Step 1090 FAIL (defense v25, systematic+tumble):** 0/3 ARC games. 4th consecutive defense failure after v21 (v22-v25 all killed/failed). **Defense v21 = hard ceiling.** PB30 n=4 defense complexity-kills.
  **Step 1091 FAIL (prosecution v22, seed 28493):** GAME_2=0%, GAME_3=0%, GAME_4=0%. Chain 2/5. ARC=0.0000. Directional attention-trajectory (signed progress instead of magnitude). No improvement over v20. **Attention-trajectory family exhausted: v20=1/3, v21=0/3, v22=0/3.**
  **Step 1092 SIGNAL (prosecution v23, seed 47291):** GAME_4=30%/ARC=0.0099. Forward model action selection (NEW FAMILY). 3/10 seeds.
  **Step 1093 FAIL (defense v26, seed 63841):** 0/3 ARC. Difference-frame reactive. 5th consecutive defense failure post-v21.
  **Step 1094 SIGNAL (prosecution v24, seed 71538):** GAME_3=100%/SOLVED=100%/max_lvl=1/ARC=0.0045. Chain 3/5. 10/10 seeds. Forward model validated with 100-step warmup.
  **Step 1095 DRAW-ROBUSTNESS FAIL (defense v21, seed 42424):** 0/3 ARC games solved. ARC=0.0000. **v21's ARC=0.2973 was a draw artifact** — lucky game selection, not general capability.
  **Step 1096 DRAW-ROBUSTNESS FAIL (prosecution v24, seed 53197):** 0/3 ARC games solved. ARC=0.0000. **v24's ARC=0.0045 was also a draw artifact.** Both sides equally draw-dependent. The 66x efficiency gap was noise. **Debate evidence inconclusive: neither ℓ₁ nor ℓ_π demonstrates general advantage after 14 experiments.**
  **Steps 1023-1025:** 1023: VC33 L1 5/5, separability confirmed (graph=random=800b, all discover same 7 clusters). 1023b: FT09 L1 5/5 (mode map generalizes, 3.2px offset from target). 1024: alpha probing FAILS (FT09 2/5 lucky, VC33 0/5 — prediction error too noisy for WHERE). 1025: mode map + 800b + W_pred = mode map alone (VC33 L1 5/5, L2+ 0/5). Old framework components add zero value for click games.

---

Historical entries (Steps 1-1022): see journal/archive_steps_1_1022.md
