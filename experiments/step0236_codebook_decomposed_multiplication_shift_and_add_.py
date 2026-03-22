"""
Step 236 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 11463.
"""
import torch, torch.nn.functional as F
device = 'cuda'

# Step 236: Decomposed MULTIPLICATION — shift-and-add via k-NN
# Multiply a * b by: for each bit of b, if bit=1, add (a << bit_position)
# Uses the proven full-adder k-NN from Step 235

# Full adder k-NN (same as Step 235)
X_fa = []; y_s = []; y_c = []
for a in range(2):
    for b in range(2):
        for cin in range(2):
            s = a + b + cin
            for _ in range(50):
                X_fa.append([float(a),float(b),float(cin)])
                y_s.append(s%2); y_c.append(s//2)
X_fa=torch.tensor(X_fa,device=device)
y_s=torch.tensor(y_s,device=device,dtype=torch.long)
y_c=torch.tensor(y_c,device=device,dtype=torch.long)

def fa_predict(a,b,cin):
    q=torch.tensor([float(a),float(b),float(cin)],device=device)
    sims=F.normalize(q.unsqueeze(0),dim=1)@F.normalize(X_fa,dim=1).T
    tk=sims[0].topk(5)
    s=y_s[tk.indices].mode().values.item()
    c=y_c[tk.indices].mode().values.item()
    return s,c

def add_binary(a_bits, b_bits, n):
    carry=0; result=[]
    for i in range(n):
        ab=a_bits[i] if i<len(a_bits) else 0
        bb=b_bits[i] if i<len(b_bits) else 0
        s,carry=fa_predict(ab,bb,carry)
        result.append(s)
    result.append(carry)
    return result

def to_bits(n,nb): return [(n>>i)&1 for i in range(nb)]
def from_bits(bits): return sum(b*(2**i) for i,b in enumerate(bits))

# Multiply: shift-and-add
def multiply(a, b, n_bits=8):
    result_bits = [0] * (2*n_bits+1)
    a_bits = to_bits(a, n_bits)
    b_bits = to_bits(b, n_bits)
    
    for i in range(n_bits):
        if b_bits[i] == 0: continue
        # Add a << i to result
        shifted_a = [0]*i + a_bits + [0]*(n_bits-len(a_bits))
        carry = 0
        new_result = []
        for j in range(2*n_bits):
            rb = result_bits[j]
            sb = shifted_a[j] if j < len(shifted_a) else 0
            s, carry = fa_predict(rb, sb, carry)
            new_result.append(s)
        new_result.append(carry)
        result_bits = new_result
    
    return from_bits(result_bits)

# Test
print('Step 236: Decomposed multiplication via shift-and-add')
correct = total = ood_correct = ood_total = 0
for a in range(0, 20, 3):
    for b in range(0, 20, 3):
        pred = multiply(a, b, 8)
        true = a * b
        ok = pred == true
        total += 1; correct += int(ok)
        if a > 7 or b > 7: ood_total += 1; ood_correct += int(ok)

print(f'  Overall: {correct}/{total} ({correct/total*100:.1f}%)')
print(f'  OOD (>7): {ood_correct}/{ood_total} ({ood_correct/ood_total*100:.1f}%)')
print(f'  7 * 13 = {multiply(7,13,8)} (true: {7*13})')
print(f'  15 * 17 = {multiply(15,17,8)} (true: {15*17})')
print(f'  19 * 19 = {multiply(19,19,8)} (true: {19*19})')
" 2>&1
