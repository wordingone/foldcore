"""
Step 141 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10141.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 141: PARITY rule — output is parity of last N symbols
# This is a GLOBAL function that doesn't decompose by position
# k-NN should fail because flipping ANY one symbol changes the parity
# but the one-hot representation only changes in one position

vocab = 2  # binary
context = 8  # parity of 8 bits
n_train = 1000
n_test = 256  # all 2^8 = 256 possible 8-bit strings

# Training: random binary sequences, label = parity of context
X_tr = torch.randint(0, 2, (n_train, context), device=device).float()
y_tr = (X_tr.sum(dim=1) % 2).long()
X_te = torch.zeros(256, context, device=device)
for i in range(256):
    for b in range(context):
        X_te[i, b] = (i >> b) & 1
y_te = (X_te.sum(dim=1) % 2).long()

X_tr_n = F.normalize(X_tr, dim=1)
X_te_n = F.normalize(X_te, dim=1)

# k-NN
k = 5
sims = X_te_n @ X_tr_n.T
scores = torch.zeros(256, 2, device=device)
for c in range(2):
    m = y_tr == c; cs = sims[:, m]
    if cs.shape[1] == 0: continue
    scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
preds = scores.argmax(1)
acc_knn = (preds == y_te).float().mean().item() * 100

# Baseline
acc_random = 50.0

# With parity feature (oracle)
X_tr_aug = torch.cat([X_tr, X_tr.sum(1, keepdim=True) % 2], 1)
X_te_aug = torch.cat([X_te, X_te.sum(1, keepdim=True) % 2], 1)
X_tr_aug_n = F.normalize(X_tr_aug, dim=1)
X_te_aug_n = F.normalize(X_te_aug, dim=1)
sims_aug = X_te_aug_n @ X_tr_aug_n.T
scores_aug = torch.zeros(256, 2, device=device)
for c in range(2):
    m = y_tr == c; cs = sims_aug[:, m]
    if cs.shape[1] == 0: continue
    scores_aug[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
acc_oracle = (scores_aug.argmax(1) == y_te).float().mean().item() * 100

# Can coherence discovery find the parity feature?
def class_coherence(V, labels):
    V_n = F.normalize(V, dim=1)
    total = count = 0
    for c in range(labels.max().item() + 1):
        m = labels == c
        if m.sum() < 2: continue
        Vc = V_n[m]
        centroid = F.normalize(Vc.mean(0), dim=0)
        total += (Vc @ centroid).mean().item()
        count += 1
    return total / count if count > 0 else 0

base_coh = class_coherence(X_tr, y_tr)
parity_feat = (X_tr.sum(1) % 2).unsqueeze(1)
aug_coh = class_coherence(torch.cat([X_tr, parity_feat], 1), y_tr)
print(f'Parity task (context={context}):')
print(f'  k-NN:          {acc_knn:.1f}% (random={acc_random:.0f}%)')
print(f'  k-NN + oracle: {acc_oracle:.1f}%')
print(f'  Coherence base: {base_coh:.4f}')
print(f'  Coherence +par: {aug_coh:.4f} (delta={aug_coh-base_coh:+.6f})')
print(f'')
print(f'k-NN handles parity? {\"YES\" if acc_knn > 80 else \"NO\"} ({acc_knn:.1f}%)')
print(f'Coherence detects parity? {\"YES\" if aug_coh - base_coh > 0.01 else \"NO\"}')
" 2>&1
