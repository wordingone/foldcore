"""
Step 185 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10774.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 185: Fix — use ONE-HOT encoding for Fibonacci states
# One-hot makes each state orthogonal — k-NN becomes exact lookup

n_train = 1000; k = 5

# One-hot encode: state (a, b) -> 20-dim vector (10 for a + 10 for b)
def one_hot_state(a, b):
    v = torch.zeros(20, device=device)
    v[int(a)] = 1
    v[10 + int(b)] = 1
    return v

states_tr = torch.randint(0, 10, (n_train, 2), device=device)
X_tr = torch.stack([one_hot_state(s[0], s[1]) for s in states_tr])
next_states = torch.stack([torch.tensor([s[1], (s[0]+s[1])%10], device=device) for s in states_tr])
y_tr = torch.stack([one_hot_state(n[0], n[1]) for n in next_states])

def predict_step_oh(state_oh, X_db, y_db, k=5):
    sims = state_oh.unsqueeze(0) @ X_db.T
    topk_idx = sims[0].topk(min(k, sims.shape[1])).indices
    # Average top-k neighbors' next states (one-hot space)
    avg_next = y_db[topk_idx].mean(0)
    # Decode: argmax in each half
    a_pred = avg_next[:10].argmax().item()
    b_pred = avg_next[10:].argmax().item()
    return one_hot_state(a_pred, b_pred), a_pred, b_pred

n_test = 100
print(f'Fibonacci mod 10 — one-hot k-NN:')
for n_steps in [1, 3, 5, 10, 20]:
    correct = 0
    for _ in range(n_test):
        a, b = torch.randint(0, 10, (2,)).tolist()
        true_a, true_b = a, b
        for _ in range(n_steps):
            true_a, true_b = true_b, (true_a + true_b) % 10
        
        current = one_hot_state(a, b)
        for _ in range(n_steps):
            current, _, _ = predict_step_oh(current, X_tr, y_tr)
        
        _, pred_a, pred_b = predict_step_oh(current, X_tr, y_tr)  # dummy call to decode
        # Decode current
        pred_a = current[:10].argmax().item()
        pred_b = current[10:].argmax().item()
        
        if pred_a == true_a and pred_b == true_b:
            correct += 1
    
    print(f'  {n_steps:2d} steps: {correct}/{n_test} ({correct}%)')
" 2>&1
