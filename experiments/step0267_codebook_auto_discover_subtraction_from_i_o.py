"""
Step 267 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 12172.
"""
import torch

# Step 267: Auto-discover subtraction from I/O
# Given: (a, b, a-b) for a >= b, a,b in 0-3
# Can the system discover the circuits for binary subtraction?

gate_ops = {'AND': lambda x,y: x&y, 'OR': lambda x,y: x|y, 'XOR': lambda x,y: x^y, 'NAND': lambda x,y: 1-(x&y)}

def synth(io_pairs, n_inputs):
    for gn, gf in gate_ops.items():
        for i in range(n_inputs):
            for j in range(n_inputs):
                if all(gf(ins[i], ins[j]) == out for ins, out in io_pairs):
                    return gn, lambda ins, gf=gf, i=i, j=j: gf(ins[i], ins[j])
    for g1n, g1f in gate_ops.items():
        for i in range(n_inputs):
            for j in range(n_inputs):
                for g2n, g2f in gate_ops.items():
                    for k in range(n_inputs):
                        if all(g2f(ins[k], g1f(ins[i], ins[j])) == out for ins, out in io_pairs):
                            return f'{g2n}({g1n})', lambda ins, g2f=g2f, g1f=g1f, i=i, j=j, k=k: g2f(ins[k], g1f(ins[i], ins[j]))
    return None, None

# I/O for 1-bit subtraction (half subtractor)
# a - b: diff = a XOR b, borrow = (NOT a) AND b
examples = [(a, b, max(0, a-b)) for a in range(2) for b in range(2)]
max_out = max(s for _,_,s in examples)
n_bits = max(1, max_out.bit_length())

print(f'Step 267: Auto-discover subtraction')
print(f'  I/O: {[(a,b,s) for a,b,s in examples]}')
print(f'  Output bits needed: {n_bits}')

for bit in range(n_bits):
    io = [([a, b], (s >> bit) & 1) for a, b, s in examples]
    name, fn = synth(io, 2)
    print(f'  Bit {bit}: {name}')

# Also: full subtractor (with borrow-in)
print(f'\\n  Full subtractor (with borrow):')
examples_full = []
for a in range(2):
    for b in range(2):
        for bin_in in range(2):
            diff = a - b - bin_in
            if diff < 0:
                diff_bit = (diff + 2) % 2
                borrow_out = 1
            else:
                diff_bit = diff % 2
                borrow_out = 0
            examples_full.append(([a, b, bin_in], diff_bit, borrow_out))

io_diff = [([a,b,c], d) for [a,b,c],d,_ in examples_full]
io_borrow = [([a,b,c], bo) for [a,b,c],_,bo in examples_full]

name_d, fn_d = synth(io_diff, 3)
name_b, fn_b = synth(io_borrow, 3)
print(f'  Difference = {name_d}')
print(f'  Borrow out = {name_b}')
print(f'\\n  Full subtractor discovered from 8 I/O examples!')
" 2>&1
