"""
Step 184 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10771.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 184: Iterated k-NN on a SEQUENCE task — Fibonacci mod 10
# Rule: next = (prev + prev_prev) % 10
# Can iterated k-NN predict the sequence N steps ahead?

# State: (a, b) -> next state: (b, (a+b)%10)
n_train = 1000; k = 5

# Generate all possible (a, b) states
states_tr = torch.randint(0, 10, (n_train, 2), device=device).float()
next_states_tr = torch.stack([states_tr[:, 1], (states_tr[:, 0] + states_tr[:, 1]) % 10], dim=1).float()

def predict_step(state, states_db, next_db):
    s = state.unsqueeze(0)
    sims = F.normalize(s, dim=1) @ F.normalize(states_db, dim=1).T
    # Predict next state element-by-element
    pred = torch.zeros(2, device=device)
    for dim in range(2):
        targets = next_db[:, dim]
        # Weighted average of top-k neighbors' next states
        topk_sims, topk_idx = sims[0].topk(min(k, sims.shape[1]))
        topk_targets = targets[topk_idx]
        # Majority vote (round to nearest integer)
        pred[dim] = topk_targets.float().mean().round()
    return pred

# Test: iterate N steps
n_test = 100
print(f'Fibonacci mod 10 — iterated k-NN:')

for n_steps in [1, 3, 5, 10]:
    correct = 0
    for _ in range(n_test):
        a, b = torch.randint(0, 10, (2,)).tolist()
        state = torch.tensor([float(a), float(b)], device=device)
        
        # Ground truth
        true_a, true_b = a, b
        for _ in range(n_steps):
            true_a, true_b = true_b, (true_a + true_b) % 10
        
        # Iterated prediction
        current = state.clone()
        for _ in range(n_steps):
            current = predict_step(current, states_tr, next_states_tr)
        
        if int(current[0].item()) == true_a and int(current[1].item()) == true_b:
            correct += 1
    
    print(f'  {n_steps:2d} steps: {correct}/{n_test} correct ({correct/n_test*100:.0f}%)')
" 2>&1
