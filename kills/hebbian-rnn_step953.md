# Kill: Hebbian RNN
Steps: 948-953 (6 experiments) | Trigger: 0/10 seeds on all modifications (953 = 6th experiment, family dead)

## What worked
- Continuous recurrent state h_t = sigmoid(W_h @ h + W_x @ enc) encodes trajectory (seed 8 navigated, 96 L1)
- Hebbian W_a learning CAN drive action selection from recurrent state
- Concurrent W_pred + W_a learning from step 1 is load-bearing (separating kills signal)
- Positive-only accumulation is load-bearing (mean subtraction cancels W_a)

## What failed
- 9/10 seeds fail under ALL configurations (948-953)
- W_a bootstrapping from zero: argmax on near-zero → winner-take-all lock
- Warm-start (950): kills concurrent learning (seed 8: 96→0)
- Mean subtraction (951): cancels W_a (norm 0.338 vs 20.3)
- Epsilon decay (952): pure random early → no directional signal
- Softmax (953): over-distributes, prevents positive feedback
- h_dim=128 (949): capacity irrelevant

## What next family needs
- Strong trajectory encoding from step 1 (not learned from zero)
- Action selection that doesn't rely on lucky W_a bootstrapping
- OR: avoid Hebbian W_a entirely — use fixed action mapping or prediction-error-based selection

## Return condition
If a mechanism is found that reliably bootstraps W_a from zero across seeds, Hebbian RNN could be revisited. Population-based W_a init (evolutionary) or meta-learned Hebbian rules (Najarro & Risi 2020) could work but may violate R1.
