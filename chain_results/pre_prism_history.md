# Pre-PRISM Research History (Steps 944–1005)

Steps 944–1005 ran before the PRISM infrastructure overhaul (2026-03-24).
- No randomized game order (LS20-first gateway)
- No chain kill criterion
- Metrics: LS20 raw score (not L1 rate), seeds explicit

Baseline for comparison: Step 965 (916+h-reset) = LS20=67.0/6/10 @ 10K chain.
PRISM-era baseline: Step 1006 (994) = LS20=100% L1, chain=3/5.

---

## 916-Augmentation Family (Steps 944–947) — DEAD (4 kills)

| Step | Mechanism | LS20 | FT09 | VC33 | Verdict |
|------|-----------|------|------|------|---------|
| 944 | Alpha threshold reset (alpha_conc > 50 → reset) | KILL | 0 | 0 | KILL — concentration load-bearing (Prop 28 FALSIFIED) |
| 945 | Alpha-weighted prediction input | KILL | 0 | 0 | KILL — alpha feedback loop load-bearing |
| 946 | Additive action bias (global augmentation) | KILL | 0 | 0 | KILL — position-independent noise |
| 947 | Persistence-weighted prediction error | KILL | 0 | 0 | KILL — running_mean drift weighting hurts LS20 |

**PROVED:** 916 is a tightly coupled fixed point. Alpha, alpha feedback, and error signal all load-bearing.

---

## Hebbian RNN Family (Steps 948–963) — DEAD (15+ kills)

| Step | Mechanism | LS20 seeds | FT09 | Verdict |
|------|-----------|-----------|------|---------|
| 948 | Hebbian W_a @ h (h_dim=64, sigmoid) | 1/10 (seed 8=96) | 0 | PASS signal — brittle |
| 949 | Softmax exploration (T=1.0) | 1/10 | 0 | Same 1/10 lock |
| 950 | W_a warm-start (delay training) | 0/10 | 0 | KILL — warm-start kills signal |
| 951 | Mean-subtracted delta for W_a | 0/10 | 0 | KILL — W_a cancels |
| 952 | Eligibility traces on W_a | 1/10 | 0 | No improvement |
| 953 | ESN (fixed W_h, spectral_r=0.9) | 1/10 | 0 | Prop 29: architecture irrelevant |
| 955 | ESN sigmoid variant | 1/10 | 0 | KILL — same 1/10 |
| 959 | ReLU-gated W_a (h_gate=relu(h-0.5)) | 1/10 | 0 | No robustness improvement |
| 962 | Decayed W_a + bounded | 1/10 | 0 | KILL |
| 963 | RNN hybrid (trainable W_h sigmoid) | 1/10 | 0 | KILL — worse than fixed W_h |

**Prop 29 (Architecture Irrelevance):** ESN with fixed random W_h = numerically identical to Hebbian RNN.
**Prop 30 (Positive Lock):** sigmoid h ∈ [0,1] → all dot products positive → winner-take-all after 1 update.
**STRUCTURAL GAP:** Action differentiation from uniform-error regime. W_a can't bootstrap reliably.

---

## Chain Diagnostic + h-Reset Fix (Steps 964–965)

| Step | Mechanism | LS20 chain | FT09 | VC33 | Notes |
|------|-----------|-----------|------|------|-------|
| 964 | Chain diagnostic (916, no reset) | 14.7/3/10 | 0 | 0 | Cross-game h contamination |
| 965 | h-reset on set_game() | 67.0/6/10 | 0 | 0 | **Chain floor** — 90% of standalone |

**Step 965 = canonical chain floor.** h contamination was primary degradation.

---

## Sequential Credit / Exploration Mechanism Family (Steps 966–990) — 25 kills

| Step | Mechanism | LS20 | FT09 | Verdict |
|------|-----------|------|------|---------|
| 966 | Action embedding Hebbian | KILL | 0 | 36.5 — noise in embedding |
| 967 | W_pred classification (CIFAR iid) | KILL | 0 | 0.97% chance-level |
| 968 | Action-conditioned h | KILL | 0 | 149.3@25K vs 290.7 baseline |
| 969 | FT09 diagnostic @25K | — | 0/10 | Mechanism-limited CONFIRMED at 25K |
| 970 | Eligibility traces on running_mean | 0 | 0 | KILL — traces destroy 800b signal |
| 971 | Two-stream trace_score | 0 | 0 | KILL — LS20 overwhelms trace buffer |
| 972 | Action-conditioned W_pred | 307.6/0/10 | 0 | LS20-ONLY, not chain (W_pred resets) |
| 973–975 | 972 chain attempts | 38–47 | 0 | Below 965 floor |
| 976 | Ensemble disagreement (K=3) | 11.4 | 0 | KILL — diversity collapses |
| 977 | Action momentum | 47.9 | 0 | KILL |
| 978 | Action suppression | 39.9 | 0 | KILL |
| 979 | Frame-diff augmented encoding (ENC=512) | 41.8 | 0 | KILL |
| 982 | VC33 standalone diagnostic @25K | — | — | 0/10 — same wall as FT09 |
| 983 | avgpool2 hires encoding (1024 dims) | 38.3 | 0 | KILL — needs >10K warmup |
| 984 | Sign confound test (965 exact repro) | 67.0 | 0 | BASELINE CONFIRMED |
| 985 | Attention context K=32 EXT=512 | 49.5 | 0 | KILL |
| 986 | Multi-pass 5×2K steps/game | 26.2 | 0 | KILL — 2K windows too short |
| 987 | Delta-drop cycling | 0 | 0 | KILL — fires during normal nav |
| 988 | Budget diagnostic @50K (FT09/VC33) | 51.0 | 0/10 | MECHANISM-LIMITED at 50K |
| 989 | Dual-horizon prediction | 61.8 | 0 | KILL |
| 990 | Temporal model inconsistency signal | 25.3 | 0 | KILL |

**Debate closed post-989:** Graph ban NOT the wall. Temporal credit IS the wall.
**New rule (Step 987):** 800b action selection FROZEN — any modification auto-reject.

---

## Adaptive Eta Family (Steps 991–1000) — 994 = new chain best

| Step | Mechanism | LS20 chain | FT09 | Verdict |
|------|-----------|-----------|------|---------|
| 991 | Adaptive eta (ETA_H_EMA=0.1, scale=2.0) | 55.4/6/10 | 0 | KILL — h_ema too slow |
| 992 | Adaptive eta (ETA_H_EMA=0.5, scale=0.3) | 65.9/7/10 | 0 | PASS — first non-degrading in 27 exp |
| 993 | Per-action eta boost | 32.5/3/10 | 0 | KILL — per-action differentiation kills |
| **994** | **Fast adapt (h-novelty spike → 500-step 1.5×eta)** | **83.8/8/10** | **0** | **NEW CHAIN BEST +25% vs 965** |
| 995 | Re-triggerable fast adapt | 77.6/7/10 | 0 | Worse — trigger-once is better |
| 996 | FAST_ADAPT_STEPS sweep (200/500/1000) | 29.7/3/10 | 0 | Identical — steps not the variable |
| 997 | h_ema reset ablation | 70.3/7/10 | 0 | h_ema persistence = 80% of gain |
| 998 | Chain duration sweep (200/1000/2000) | 72–75/6–7/10 | 0 | 500 optimal, non-monotonic |
| 999 | Dual-target W_pred | 84.1/7/10 | 0 | LATERAL vs 994 — neutral |
| **1000** | **994 @25K definitive** | **286.2/9/10** | **0/10** | **Prop 31 confirmed at 25K** |

**Mechanism fully characterized (Step 997):**
- Primary (80%): h_ema persists across games → fires at CIFAR→LS20 transition → fast adaptation
- Secondary (20%): CIFAR warm-start (W_pred/alpha carry-over)
- FAST_ADAPT_STEPS value irrelevant — transition detector, not within-game novelty

**Prop 31 (Temporal Credit Wall):** Global running mean SNR→0 for state-dependent actions.
FT09/VC33 mechanism-limited at ANY budget (10K/25K/50K confirmed).

---

## Post-1000: New Families Through PRISM (Steps 1001–1007)

| Step | Family | Mechanism | Verdict |
|------|--------|-----------|---------|
| 1001 | Oscillatory | Stuart-Landau + compression-progress modulator | KILL — LS20=0/10 |
| 1002 | Oscillatory | + delta_obs modulator | KILL — LS20=0/10 |
| 1003 | Oscillatory | Oscillatory encoding (96-dim) + 800b | KILL — LS20=9.5/1/10 |
| 1004 | Multi-horizon | 994 + W_pred_long K=10 gradient ascent | KILL — overflow (long_spread=3557) |
| 1005 | Multi-horizon | + normalization fix | KILL — long_spread=6917 |
| **1006** | **Baseline** | **994 through PRISM** | **LS20=100% L1, chain=3/5. CANONICAL BASELINE** |
| 1007 | Attention-trajectory | softmax(q@K^T)@V buffer | FAIL — LS20 regressed 100%→0% |

---

*Written 2026-03-24. Source: memory files (project_session_948.md, project_session_962.md, project_session_1001_1004.md).*
*Pre-PRISM steps used LS20-first testing — not chain scores. Compare within-era only.*
