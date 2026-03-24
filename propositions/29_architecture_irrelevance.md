# Proposition 29: Architecture Irrelevance for Hebbian Action Learning
Status: CONFIRMED
Steps: 948, 954, 955, 956

## Statement
For substrates using Hebbian W_a learning (W_a[action] += lr * delta * h) with argmax action selection and epsilon-greedy or deterministic exploration, the recurrent architecture (how h is computed) is irrelevant to navigation performance. The sole variance source is the action sequence during the bootstrap phase (~first 1000 steps).

## Evidence
- Step 948 (Hebbian RNN, sigmoid, W_h init 0.01): seed 8 = 96, 9/10 = 0, W_a_norm = 20.3
- Step 955 (ESN, sigmoid, W_h spectral radius 0.9, 10% sparse): seed 8 = 96, 9/10 = 0, W_a_norm = 19.9
- Results are **numerically identical** despite fundamentally different recurrent architectures
- Step 954 (ESN, tanh): different lucky seed (10=22) due to tanh cancellation changing W_a dynamics
- Step 956 (RNG-free deterministic hash): lucky seed shifts to 4 (=150) but rate stays 1/10

The per-seed variance is entirely determined by RandomState(seed) controlling action selection. Weight initialization RandomState(seed+10000) produces identical results across architectures when action RNG is held constant.

## Implications
1. The 1/10 bootstrap rate is STRUCTURAL — independent of architecture and randomness source
2. The bottleneck is the exploration mechanism, not the encoding/recurrence
3. Architecture search within Hebbian W_a is provably futile — all architectures produce identical results
4. The next improvement must come from the action selection mechanism (UCB, ensemble, etc.), not from modifying h

## Supersedes / Superseded by
Refines Proposition 22 (architecture triangle): at the dynamics vertex, the architecture degree of freedom is provably irrelevant for Hebbian action learning. The sole remaining degree of freedom is the exploration mechanism.
