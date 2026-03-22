"""
Step 234 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11423.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 234: Pairwise bubble sort — predict one swap at a time
# Input: (seq, position) -> should we swap seq[pos] and seq[pos+1]?
# This is a BINARY classification on a pair — k-NN should handle this.
# Then iterate: scan positions, apply swaps.

vocab = 5; seq_len = 4; n_train = 2000; k = 5

# Training: for each sequence, for each position, should we swap?
X_tr = []; y_tr = []
for _ in range(n_train):
    seq = torch.randint(0, vocab, (seq_len,), device=device).float()
    for pos in range(seq_len - 1):
        # Features: full sequence + position indicator
        feat = torch.cat([seq, torch.tensor([float(pos)], device=device)])
        swap = 1 if seq[pos] > seq[pos+1] else 0
        X_tr.append(feat); y_tr.append(swap)

X_tr = torch.stack(X_tr); y_tr = torch.tensor(y_tr, device=device, dtype=torch.long)

# Test: iterative sort using pairwise swap prediction
n_test = 200
X_te = torch.randint(0, vocab, (n_test, seq_len), device=device).float()
y_te = torch.sort(X_te.long(), dim=1).values.float()

def predict_swap(seq, pos, X_db, y_db, k=5):
    feat = torch.cat([seq, torch.tensor([float(pos)], device=device)])
    sims = F.normalize(feat.unsqueeze(0), dim=1) @ F.normalize(X_db, dim=1).T
    scores = torch.zeros(1, 2, device=device)
    for c in range(2):
        m = y_db == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return scores.argmax(1).item()

# Iterated pairwise sort
correct = 0
for i in range(n_test):
    current = X_te[i].clone()
    for pass_num in range(seq_len):  # max passes = seq_len
        swapped = False
        for pos in range(seq_len - 1):
            if predict_swap(current, pos, X_tr, y_tr) == 1:
                # Swap
                current[pos], current[pos+1] = current[pos+1].item(), current[pos].item()
                swapped = True
        if not swapped: break
    if (current == y_te[i]).all(): correct += 1

print(f'Pairwise iterated bubble sort (vocab={vocab}, len={seq_len}):')
print(f'  Correct: {correct}/{n_test} ({correct/n_test*100:.1f}%)')

# For comparison: single swap accuracy
correct_swap = (torch.tensor([predict_swap(X_tr[j*3][:seq_len], int(X_tr[j*3][seq_len]), X_tr, y_tr) for j in range(min(500,len(X_tr)//3))]) == y_tr[:500:3]).float().mean().item() * 100
print(f'  1-swap prediction accuracy: ~{correct_swap:.0f}%')
" 2>&1
