"""
Step 226 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11297.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 226: ITERATED INCREMENT for OOD addition
# Instead of learning a+b directly, learn INCREMENT: (a, counter) -> (a+1, counter-1)
# Then a+b = iterate(a, b times)
# Train increment on 0-7. Test: does it work on 8, 9, 10+?

n_range = 8  # train on 0-7
n_train = 500

# One-hot encode for exact lookup
def one_hot(val, n=15):  # support up to 14
    v = torch.zeros(n, device=device)
    v[min(val, n-1)] = 1
    return v

# Training: (value) -> (value + 1)
X_tr = []; y_tr = []
for _ in range(n_train):
    val = torch.randint(0, n_range, (1,)).item()
    X_tr.append(one_hot(val)); y_tr.append(val + 1)
X_tr = torch.stack(X_tr); y_tr = torch.tensor(y_tr, device=device, dtype=torch.long)

# k-NN increment function
def predict_increment(val_oh, X_db, y_db, k=5):
    sims = val_oh.unsqueeze(0) @ X_db.T
    topk = sims[0].topk(min(k, sims.shape[1]))
    # Majority vote
    preds = y_db[topk.indices]
    return preds.mode().values.item()

# Test: add a + b by iterating increment b times
print('Step 226: Iterated increment for OOD addition')
print(f'Train: increment on 0-{n_range-1}')
print()

for a in [3, 5, 7, 8, 9, 10]:
    for b in [1, 2, 3, 5]:
        true_sum = a + b
        
        # Iterate: start at a, increment b times
        current = a
        success = True
        for _ in range(b):
            if current >= 15: success = False; break  # overflow
            pred_next = predict_increment(one_hot(current), X_tr, y_tr)
            current = pred_next
        
        correct = current == true_sum if success else False
        ood = 'OOD' if a >= n_range else 'IN'
        marker = 'OK' if correct else 'FAIL'
        print(f'  {a}+{b}={true_sum:2d} pred={current:2d} [{ood}] {marker}')
" 2>&1
