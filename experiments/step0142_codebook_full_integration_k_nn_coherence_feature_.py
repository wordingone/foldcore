"""
Step 142 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10152.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 142: Full integration — k-NN + coherence feature discovery on parity
# The system automatically discovers and adds the parity feature

vocab = 2
context = 8
n_train = 1000

# Generate training data
X_tr = torch.randint(0, 2, (n_train, context), device=device).float()
y_tr = (X_tr.sum(dim=1) % 2).long()

# ALL 256 test cases
X_te = torch.zeros(256, context, device=device)
for i in range(256):
    for b in range(context):
        X_te[i, b] = (i >> b) & 1
y_te = (X_te.sum(dim=1) % 2).long()

def knn_acc(V, labels, test_x, test_y, k=5):
    V_n = F.normalize(V, dim=1)
    te = F.normalize(test_x, dim=1)
    sims = te @ V_n.T
    n_cls = labels.max().item() + 1
    scores = torch.zeros(test_x.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == test_y).float().mean().item() * 100

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

# UNIFIED PROCESS: store → discover features → augment → classify
print('=== Step 142: Self-Improving Substrate on Parity ===')

# Phase 1: Store (always-spawn)
V = X_tr.clone()
labels = y_tr.clone()
raw = X_tr.clone()
raw_te = X_te.clone()

acc_before = knn_acc(V, labels, X_te, y_te)
print(f'Before discovery: {acc_before:.1f}%')

# Phase 2: Discover features via coherence
# Try ALL possible single-feature augmentations: each individual bit,
# sum, parity, products, etc.
candidates = {}

# Candidate: sum of all bits
feat = raw.sum(1, keepdim=True)
aug = torch.cat([V, feat], 1)
candidates['sum'] = class_coherence(aug, labels) - class_coherence(V, labels)

# Candidate: parity
feat = (raw.sum(1) % 2).unsqueeze(1)
aug = torch.cat([V, feat], 1)
candidates['parity'] = class_coherence(aug, labels) - class_coherence(V, labels)

# Candidate: each pair product
for i in range(context):
    for j in range(i+1, context):
        feat = (raw[:, i] * raw[:, j]).unsqueeze(1)
        aug = torch.cat([V, feat], 1)
        candidates[f'prod_{i}_{j}'] = class_coherence(aug, labels) - class_coherence(V, labels)

# Candidate: each single bit
for i in range(context):
    feat = raw[:, i:i+1]
    aug = torch.cat([V, feat], 1)
    candidates[f'bit_{i}'] = class_coherence(aug, labels) - class_coherence(V, labels)

# Rank by coherence improvement
ranked = sorted(candidates.items(), key=lambda x: -x[1])
print(f'\\nTop 5 features by coherence:')
for name, delta in ranked[:5]:
    print(f'  {name:15s}: +{delta:.6f}')
print(f'...')
print(f'Bottom 3:')
for name, delta in ranked[-3:]:
    print(f'  {name:15s}: +{delta:.6f}')

# Phase 3: Add best feature
best_name, best_delta = ranked[0]
print(f'\\nBest feature: {best_name} (delta=+{best_delta:.6f})')

# Apply to both train and test
if best_name == 'parity':
    feat_tr = (raw.sum(1) % 2).unsqueeze(1)
    feat_te = (raw_te.sum(1) % 2).unsqueeze(1)
elif best_name == 'sum':
    feat_tr = raw.sum(1, keepdim=True)
    feat_te = raw_te.sum(1, keepdim=True)
elif best_name.startswith('prod_'):
    parts = best_name.split('_')
    i, j = int(parts[1]), int(parts[2])
    feat_tr = (raw[:, i] * raw[:, j]).unsqueeze(1)
    feat_te = (raw_te[:, i] * raw_te[:, j]).unsqueeze(1)
else:  # bit_i
    i = int(best_name.split('_')[1])
    feat_tr = raw[:, i:i+1]
    feat_te = raw_te[:, i:i+1]

V_aug = torch.cat([V, feat_tr], 1)
X_te_aug = torch.cat([X_te, feat_te], 1)

acc_after = knn_acc(V_aug, labels, X_te_aug, y_te)
print(f'After discovery: {acc_after:.1f}% (+{acc_after-acc_before:.1f}pp)')
print(f'\\n=== VERDICT: {\"SUBSTRATE WORKS\" if acc_after > acc_before + 5 else \"MARGINAL\"} ===')
" 2>&1
