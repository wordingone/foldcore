"""
Step 154 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10349.
"""
import sys; sys.path.insert(0, 'substrates/topk-fold')
from self_improving_substrate import SelfImprovingSubstrate
import torch

# Step 154: Label noise robustness
d = 8; n_train = 1000
X = torch.randint(0, 2, (n_train, d)).float()
y_clean = (X.sum(1) % 2).long()  # parity

X_te = torch.zeros(256, d)
for i in range(256):
    for b in range(d):
        X_te[i, b] = (i >> b) & 1
y_te = (X_te.sum(1) % 2).long()

for noise_pct in [0, 5, 10, 20, 30]:
    y = y_clean.clone()
    n_flip = int(n_train * noise_pct / 100)
    if n_flip > 0:
        flip_idx = torch.randperm(n_train)[:n_flip]
        y[flip_idx] = 1 - y[flip_idx]
    
    sub = SelfImprovingSubstrate(d=d, max_features=3)
    sub.train(X, y)
    preds = sub.predict(X_te)
    acc = (preds.cpu() == y_te).float().mean().item() * 100
    
    sub_base = SelfImprovingSubstrate(d=d, max_features=0)
    sub_base.train(X, y)
    acc_base = (sub_base.predict(X_te).cpu() == y_te).float().mean().item() * 100
    
    feats = sub.features
    has_parity = any('cos_sum' in str(f) or 'sum_mod' in str(f) for f in feats)
    print(f'noise={noise_pct:2d}%: base={acc_base:.1f}% improved={acc:.1f}% delta={acc-acc_base:+.1f}pp feats={feats} parity_found={has_parity}')
" 2>&1
