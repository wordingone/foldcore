"""
Step 140 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10135.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 140: Rule generalization — can k-NN predict UNSEEN combinations?
# Train on sequences starting with symbols 0,1,2. Test with symbol 3.
# The rule s[t] = (s[t-1] + s[t-2]) % 4 is the same, but combinations 
# involving '3' were never seen in training.

vocab = 4
seq_len = 20
context = 2  # minimal context (the rule only needs 2)

# Train: sequences that start with symbols from {0,1,2}
n_train = 500
train_seqs = torch.zeros(n_train, seq_len, dtype=torch.long, device=device)
train_seqs[:, 0] = torch.randint(0, 3, (n_train,))  # only 0,1,2
train_seqs[:, 1] = torch.randint(0, 3, (n_train,))
for t in range(2, seq_len):
    train_seqs[:, t] = (train_seqs[:, t-1] + train_seqs[:, t-2]) % vocab

# Test: sequences that start with symbol 3
n_test = 200
test_seqs = torch.zeros(n_test, seq_len, dtype=torch.long, device=device)
test_seqs[:, 0] = 3  # starts with unseen symbol
test_seqs[:, 1] = torch.randint(0, vocab, (n_test,))
for t in range(2, seq_len):
    test_seqs[:, t] = (test_seqs[:, t-1] + test_seqs[:, t-2]) % vocab

def make_dataset(seqs, context):
    X, Y = [], []
    for seq in seqs:
        for t in range(context, seq_len):
            feat = torch.zeros(context * vocab, device=device)
            for c in range(context):
                feat[c * vocab + seq[t-context+c]] = 1
            X.append(feat)
            Y.append(seq[t].item())
    return torch.stack(X), torch.tensor(Y, device=device)

X_tr, y_tr = make_dataset(train_seqs, context)
X_te, y_te = make_dataset(test_seqs, context)
X_tr = F.normalize(X_tr, dim=1)
X_te = F.normalize(X_te, dim=1)

# k-NN
k = 5
sims = X_te @ X_tr.T
scores = torch.zeros(X_te.shape[0], vocab, device=device)
for c in range(vocab):
    m = y_tr == c; cs = sims[:, m]
    if cs.shape[1] == 0: continue
    scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
preds = scores.argmax(1)
acc_knn = (preds == y_te).float().mean().item() * 100

# Check: how many test contexts involve symbol '3'?
has_3 = (X_te[:, 3] > 0) | (X_te[:, 7] > 0)  # symbol 3 in position 0 or 1
acc_with_3 = (preds[has_3] == y_te[has_3]).float().mean().item() * 100 if has_3.sum() > 0 else 0
acc_without_3 = (preds[~has_3] == y_te[~has_3]).float().mean().item() * 100 if (~has_3).sum() > 0 else 0

print(f'Train: {X_tr.shape[0]} samples (symbols 0,1,2 only)')
print(f'Test: {X_te.shape[0]} samples (starts with symbol 3)')
print(f'')
print(f'k-NN overall:          {acc_knn:.1f}%')
print(f'k-NN on contexts WITH 3:  {acc_with_3:.1f}% ({has_3.sum()} samples)')
print(f'k-NN on contexts WITHOUT 3: {acc_without_3:.1f}% ({(~has_3).sum()} samples)')
print(f'')
print(f'Can k-NN generalize to unseen symbol? {\"YES\" if acc_with_3 > 80 else \"PARTIALLY\" if acc_with_3 > 50 else \"NO\"} ({acc_with_3:.1f}%)')
" 2>&1
