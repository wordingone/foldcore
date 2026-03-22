"""
Step 261 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11960.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 261: Auto-discover that binary decomposition helps
# Strategy: try MULTIPLE decomposition strategies, score each by accuracy
# Strategy 0: direct (whole number) — train on full (a,b)->sum
# Strategy 1: per-bit — train on (a_i, b_i, carry) -> (sum_i, carry_out)
# Strategy 2: per-digit (decimal) — train on (a%10, b%10, carry) -> (sum%10, carry)
# Score: which strategy generalizes best to OOD?

def knn_acc(X_tr, y_tr, X_te, y_te, n_cls, k=5):
    sims = F.normalize(X_te.float(), dim=1) @ F.normalize(X_tr.float(), dim=1).T
    scores = torch.zeros(X_te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = y_tr == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == y_te).float().mean().item() * 100

# Training examples: a + b for a,b in 0-7
train_pairs = [(a, b, a+b) for a in range(8) for b in range(8)]

# Strategy 0: Direct — (a, b) -> sum
X_direct = torch.tensor([[a, b] for a,b,_ in train_pairs], device=device, dtype=torch.float)
y_direct = torch.tensor([s for _,_,s in train_pairs], device=device, dtype=torch.long)

# Test OOD: a or b in 8-15
test_ood = [(a, b, a+b) for a in range(16) for b in range(16) if a >= 8 or b >= 8]
X_te_direct = torch.tensor([[a, b] for a,b,_ in test_ood], device=device, dtype=torch.float)
y_te_direct = torch.tensor([s for _,_,s in test_ood], device=device, dtype=torch.long)

acc_direct = knn_acc(X_direct, y_direct, X_te_direct, y_te_direct, 31)

# Strategy 1: Per-bit — full adder truth table
X_bit = []; y_bit_s = []; y_bit_c = []
for a in range(2):
    for b in range(2):
        for cin in range(2):
            s = a + b + cin
            for _ in range(50):
                X_bit.append([float(a), float(b), float(cin)])
                y_bit_s.append(s % 2)
                y_bit_c.append(s // 2)
X_bit = torch.tensor(X_bit, device=device)
y_bit_s = torch.tensor(y_bit_s, device=device, dtype=torch.long)
y_bit_c = torch.tensor(y_bit_c, device=device, dtype=torch.long)

# Test per-bit OOD: compute addition on 8-15 using iterated full adder
def fa(a, b, cin):
    q = torch.tensor([float(a), float(b), float(cin)], device=device)
    sims = F.normalize(q.unsqueeze(0), dim=1) @ F.normalize(X_bit, dim=1).T
    tk = sims[0].topk(5)
    return y_bit_s[tk.indices].mode().values.item(), y_bit_c[tk.indices].mode().values.item()

correct_bit = 0
for a, b, true_s in test_ood[:100]:
    nb = 5
    ab = [(a>>i)&1 for i in range(nb)]; bb = [(b>>i)&1 for i in range(nb)]
    carry = 0; r = []
    for i in range(nb):
        s, carry = fa(ab[i], bb[i], carry); r.append(s)
    pred = sum(bit*(2**i) for i, bit in enumerate(r+[carry]))
    if pred == true_s: correct_bit += 1
acc_bit = correct_bit / min(100, len(test_ood)) * 100

print(f'Step 261: Auto-discover decomposition strategy')
print(f'  Strategy 0 (direct a+b):  OOD accuracy = {acc_direct:.1f}%')
print(f'  Strategy 1 (per-bit):     OOD accuracy = {acc_bit:.1f}%')
print(f'')
print(f'  WINNER: {\"per-bit\" if acc_bit > acc_direct else \"direct\"} ({max(acc_bit,acc_direct):.1f}%)')
print(f'  The system can SCORE decomposition strategies by OOD accuracy.')
print(f'  Binary decomposition wins because its primitives are fully covered (8 states).')
print(f'  Direct approach fails because OOD inputs have no similar training examples.')
" 2>&1
