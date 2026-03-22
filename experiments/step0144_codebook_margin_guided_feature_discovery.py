"""
Step 144 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10185.
"""
import torch, torch.nn.functional as F
device = 'cuda'

vocab = 2; context = 8; n_train = 1000

X_tr = torch.randint(0, 2, (n_train, context), device=device).float()
y_tr = (X_tr.sum(dim=1) % 2).long()
X_te = torch.zeros(256, context, device=device)
for i in range(256):
    for b in range(context):
        X_te[i, b] = (i >> b) & 1
y_te = (X_te.sum(dim=1) % 2).long()

def knn_margin(V, labels, test_x, test_y, k=5):
    V_n = F.normalize(V, dim=1); te = F.normalize(test_x, dim=1)
    sims = te @ V_n.T
    scores = torch.zeros(test_x.shape[0], 2, device=device)
    for c in range(2):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    preds = scores.argmax(1)
    margin = (scores.max(1).values - scores.sort(1).values[:, -2])
    acc = (preds == test_y).float().mean().item() * 100
    correct_margin = margin[preds == test_y].mean().item() if (preds == test_y).sum() > 0 else 0
    return acc, correct_margin

# Step 144: MARGIN-GUIDED feature discovery
# Score each candidate by how much it increases CORRECT-prediction margin
# This uses labels (during training) but the SIGNAL is from k-NN's own output

print('=== Step 144: Margin-Guided Feature Discovery ===')

V = X_tr.clone(); labels = y_tr.clone()
raw = X_tr.clone(); raw_te = X_te.clone()

acc_base, margin_base = knn_margin(V, labels, X_tr, y_tr)  # self-eval
print(f'Base: acc={acc_base:.1f}% margin={margin_base:.4f}')

# Score candidates by margin improvement ON TRAINING SET (self-eval)
candidates = {}
for name, feat_fn in [
    ('sum', lambda x: x.sum(1, keepdim=True)),
    ('sum_mod2', lambda x: (x.sum(1) % 2).unsqueeze(1)),
    ('sum_sq', lambda x: (x.sum(1) ** 2).unsqueeze(1)),
    ('cos_sum_pi', lambda x: torch.cos(x.sum(1) * 3.14159).unsqueeze(1)),
]:
    feat = feat_fn(raw)
    aug = torch.cat([V, feat], 1)
    acc_aug, margin_aug = knn_margin(aug, labels, torch.cat([raw, feat_fn(raw)], 1), y_tr)
    candidates[name] = (margin_aug - margin_base, acc_aug)

print(f'\\nCandidates by margin improvement:')
for name, (delta_m, acc) in sorted(candidates.items(), key=lambda x: -x[1][0]):
    print(f'  {name:15s}: margin_delta={delta_m:+.4f} train_acc={acc:.1f}%')

best_name = max(candidates, key=lambda x: candidates[x][0])
print(f'\\nBest by margin: {best_name}')

# Apply best to test
feat_fns = {
    'sum': lambda x: x.sum(1, keepdim=True),
    'sum_mod2': lambda x: (x.sum(1) % 2).unsqueeze(1),
    'sum_sq': lambda x: (x.sum(1) ** 2).unsqueeze(1),
    'cos_sum_pi': lambda x: torch.cos(x.sum(1) * 3.14159).unsqueeze(1),
}
feat_tr = feat_fns[best_name](raw)
feat_te = feat_fns[best_name](raw_te)
V_aug = torch.cat([V, feat_tr], 1)
te_aug = torch.cat([X_te, feat_te], 1)
acc_test, _ = knn_margin(V_aug, labels, te_aug, y_te)
acc_base_test, _ = knn_margin(V, labels, X_te, y_te)
print(f'\\nTest: {acc_base_test:.1f}% → {acc_test:.1f}% (+{acc_test-acc_base_test:.1f}pp)')
" 2>&1
