"""
Step 150 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 10278.
"""
import torch, torch.nn.functional as F
device = 'cuda'

d = 10
rules = [
    ('XOR(0,1)', lambda X: (X[:, 0].long() ^ X[:, 1].long())),
    ('AND(2,3)', lambda X: (X[:, 2] * X[:, 3]).long()),
    ('OR(4,5)',  lambda X: ((X[:, 4] + X[:, 5]) > 0).long()),
]

X_te = torch.zeros(1024, d, device=device)
for i in range(1024):
    for b in range(d):
        X_te[i, b] = (i >> b) & 1

# Step 150: Each task is a SEPARATE k-NN with its own codebook
# Features discovered per-task, applied per-task
# This IS task-isolated but tests whether the mechanism works in a multi-task setup

print('=== Step 150: Task-Isolated Rule Discovery ===')

all_discovered = {}
all_accs = {}

for t, (rule_name, rule_fn) in enumerate(rules):
    X_t = torch.randint(0, 2, (500, d), device=device).float()
    y_t = rule_fn(X_t)
    y_te = rule_fn(X_te)
    
    # Base k-NN for this task
    V = X_t.clone()
    V_n = F.normalize(V, dim=1); te_n = F.normalize(X_te, dim=1)
    sims = te_n @ V_n.T
    scores = torch.zeros(1024, 2, device=device)
    for c in range(2):
        m = y_t == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
    acc_base = (scores.argmax(1) == y_te).float().mean().item() * 100
    
    # Margin-guided discovery for this task
    def margin_score(V_aug, labels):
        V_n2 = F.normalize(V_aug, dim=1)
        sims2 = V_n2 @ V_n2.T
        scores2 = torch.zeros(V_aug.shape[0], 2, device=device)
        for c in range(2):
            m2 = labels == c; cs2 = sims2[:, m2]
            if cs2.shape[1] == 0: continue
            scores2[:, c] = cs2.topk(min(5, cs2.shape[1]), dim=1).values.sum(dim=1)
        margin = scores2.sort(1, descending=True).values[:, 0] - scores2.sort(1, descending=True).values[:, 1]
        return margin.mean().item()
    
    m_base = margin_score(F.normalize(V, dim=1), y_t)
    best_pair = None; best_m = m_base
    for i in range(d):
        for j in range(i+1, d):
            feat = (X_t[:, i] * X_t[:, j]).unsqueeze(1)
            aug = torch.cat([V, feat], 1)
            m = margin_score(F.normalize(aug, dim=1), y_t)
            if m > best_m:
                best_m = m; best_pair = (i, j)
    
    if best_pair:
        all_discovered[rule_name] = best_pair
        feat_tr = (X_t[:, best_pair[0]] * X_t[:, best_pair[1]]).unsqueeze(1)
        feat_te_t = (X_te[:, best_pair[0]] * X_te[:, best_pair[1]]).unsqueeze(1)
        V_aug = F.normalize(torch.cat([V, feat_tr], 1), dim=1)
        te_aug = F.normalize(torch.cat([X_te, feat_te_t], 1), dim=1)
        sims = te_aug @ V_aug.T
        scores = torch.zeros(1024, 2, device=device)
        for c in range(2):
            m2 = y_t == c; cs = sims[:, m2]
            if cs.shape[1] == 0: continue
            scores[:, c] = cs.topk(min(5, cs.shape[1]), dim=1).values.sum(dim=1)
        acc_after = (scores.argmax(1) == y_te).float().mean().item() * 100
    else:
        acc_after = acc_base
    
    # Check if discovered pair matches the rule
    rule_pairs = {'XOR(0,1)': (0,1), 'AND(2,3)': (2,3), 'OR(4,5)': (4,5)}
    target = rule_pairs[rule_name]
    correct = best_pair == target if best_pair else False
    
    print(f'{rule_name}: base={acc_base:.1f}% discovered={best_pair} after={acc_after:.1f}% correct_pair={correct}')

print(f'\\nAll discovered: {all_discovered}')
print(f'Correct pairs: {sum(1 for rn, rp in all_discovered.items() if rp == {\"XOR(0,1)\": (0,1), \"AND(2,3)\": (2,3), \"OR(4,5)\": (4,5)}[rn])}/3')
" 2>&1
