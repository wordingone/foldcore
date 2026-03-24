# Proposition 30: The Positive Lock
Status: CONFIRMED
Steps: 948-956 (9 experiments across 3 families)

## Statement
For Hebbian action learning with sigmoid activation ($h = \sigma(Wh + b) \in [0,1]^d$), a single Hebbian update creates a winner-take-all lock: the first action taken is reinforced at ALL future states. The structural bootstrap rate $P(\text{escape}) \approx 10\%$ for $n=4$ actions and scales as $P \to 0$ as $n \to \infty$.

## Formal Argument
After one Hebbian update at state $s_0$ taking action $a_0$:
$$W_a[a_0] = \eta \delta \, h(s_0)$$

At any subsequent state $s_1$, the score for $a_0$ is:
$$\text{score}(a_0, s_1) = W_a[a_0] \cdot h(s_1) = \eta \delta \, (h(s_0) \cdot h(s_1))$$

Since $h(s) \in [0,1]^d$ for all $s$ (sigmoid outputs are non-negative), the dot product $h(s_0) \cdot h(s_1) > 0$ for any pair of non-zero states. Therefore $\text{score}(a_0, s_1) > 0 = \text{score}(a_{\neq 0}, s_1)$, and $a_0$ is selected by argmax at every state.

The only escape is epsilon-random action selection (probability $\epsilon$ per step). Bootstrap requires enough epsilon-random actions in the first ~100 steps to create $n$ sufficiently orthogonal $W_a$ rows. Observed rate: ~10% for $n=4$, $\epsilon=0.20$.

## Evidence
- Steps 948-956: 9 experiments across Hebbian RNN, ESN, RNG-free families all show 1/10 (or 0/10)
- Step 955: ESN (sigmoid) = 948 (sigmoid) numerically → architecture irrelevant, lock is the mechanism
- Step 956: RNG-free → lucky seed shifts but rate stays 1/10 → structural property
- Step 954: ESN (tanh) → different rate (1/10 but different seed) due to cancellation masking the lock
- FT09: 0/10 across ALL experiments → consistent with $P \to 0$ as $n \to \infty$ ($n=68$)

## Implications
1. All sigmoid-based Hebbian action learning has this structural brittleness
2. Architecture modifications (recurrence, capacity, spectral radius) cannot fix it (Proposition 29)
3. The fix must change the REPRESENTATION (sparse gating) or the SELECTION (UCB/ensemble)
4. Anti-Hebbian (suppress visited actions) avoids the lock but approximates per-state argmin → graph ban concern

## Supersedes / Superseded by
Explains the mechanism behind Proposition 29 (architecture irrelevance). Connected to Proposition 28 (alpha concentration — both involve degenerate positive accumulation).
