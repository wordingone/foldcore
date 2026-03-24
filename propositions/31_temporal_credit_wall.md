# Proposition 31: The Temporal Credit Assignment Wall
Status: CONFIRMED
Steps: 948-990 (43 experiments across 5+ families)

## Statement
Under the graph ban and codebook ban, no mechanism provides temporal credit assignment for ordered multi-step action sequences using prediction error as the sole intrinsic signal. The structural wall is not the ban (Debate v3) — it is the absence of a mechanism that connects "action A at time t contributed to outcome at time t+K" without per-(state,action) data.

## Evidence
1. **800b (running_mean)**: Global per-action average of state-change delta. Works for small action spaces without ordering (LS20, 4 actions). Fails for large action spaces with ordering (FT09/VC33, 68 actions). 0/10 at 10K, 25K, AND 50K (Steps 969, 982, 988).

2. **800b is FROZEN**: 25 modifications to action selection (966-990) all degrade LS20 without producing any FT09/VC33 signal. Any additive bonus to delta_per_action that fires during early training disrupts the 800b warmup. The action mechanism cannot be modified.

3. **Hebbian W_a**: Alternative action learning from prediction error. 1/10 structural bootstrap rate (Proposition 30, positive lock). 15 experiments confirm no variant achieves reliability.

4. **Debate v3**: Graph ban is NOT the wall. Parametric state-conditioned models (Step 972) produce identical FT09 failure. The function (per-state action conditioning) fails regardless of implementation (lookup vs parametric).

5. **Sequential mechanisms**: Eligibility traces (970-971), momentum/suppression (977-978), dual-horizon prediction (989), temporal inconsistency (990) — all killed. Traces corrupt running_mean. Bonuses disrupt warmup. No mechanism provides sequential credit without destroying single-step exploration.

## Formal Statement
Let $\Delta_t(a) = \|f_\theta(s_t) - s_{t+1}\|$ be the one-step prediction error after taking action $a$ at state $s_t$. The running mean $\bar{\Delta}(a) = \text{EMA}(\Delta_t(a))$ averages over all states visited. For an ordered sequence $a_1 \to a_2 \to \ldots \to a_K$, the credit for action $a_1$'s contribution to the outcome at time $t+K$ is diluted by $K$ intermediate averaging steps. When $K \geq 2$ and $n_{\text{actions}} > 10$, the signal-to-noise ratio for sequential credit $\to 0$.

## Implications
1. PRISM-light requires mechanisms outside the prediction-error exploration paradigm
2. Pre-ban graphs bypassed this wall by exhaustive tabular search, not by solving temporal credit
3. The next breakthrough requires temporal credit assignment without external reward — a mechanism class that may not exist in current literature without LLM-based approaches (RICL, 2025)
4. R3 (self-modification) and the temporal credit wall are connected: the substrate can't learn to modify its OWN action sequences because it can't assign credit to sequence elements

## Supersedes / Superseded by
Extends Propositions 29 (architecture irrelevance) and 30 (positive lock). Connected to Debate v3 (graph ban is not the wall). Supersedes the assumption that FT09/VC33 failure is a "ban problem" — it is a "credit problem."
