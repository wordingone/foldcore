"""
Step 265 — Recovered from session JSONL (inline python -c experiment).
Original execution: foldcore k-NN / torch experiments, March 15 2026.
Source: B--M-avir-leo/0606b161, line 12073.
"""
import torch

# Step 265: Curriculum for multiplication
# Phase 1: 1-bit multiply (AND gate — trivial)
# Phase 2: 2-bit × 1-bit (shift + add — uses Phase 1 + adder from Step 264)

gate_ops = {'AND': lambda x,y: x&y, 'OR': lambda x,y: x|y, 'XOR': lambda x,y: x^y}

def synth(io_pairs, n_inputs):
    for gn, gf in gate_ops.items():
        for i in range(n_inputs):
            for j in range(n_inputs):
                if all(gf(ins[i], ins[j]) == out for ins, out in io_pairs):
                    return f'{gn}(in{i},in{j})', lambda ins, gf=gf, i=i, j=j: gf(ins[i], ins[j])
    for g1n, g1f in gate_ops.items():
        for i in range(n_inputs):
            for j in range(n_inputs):
                for g2n, g2f in gate_ops.items():
                    for k in range(n_inputs):
                        if all(g2f(ins[k], g1f(ins[i], ins[j])) == out for ins, out in io_pairs):
                            return f'{g2n}(in{k},{g1n}(in{i},in{j}))', lambda ins, g2f=g2f, g1f=g1f, i=i, j=j, k=k: g2f(ins[k], g1f(ins[i], ins[j]))
    return None, None

print('Step 265: Curriculum for multiplication')

# Phase 1: 1-bit multiply = AND
io_mul1 = [([a, b], a & b) for a in range(2) for b in range(2)]
circ_mul1, fn_mul1 = synth(io_mul1, 2)
print(f'Phase 1: 1-bit multiply = {circ_mul1}')

# Adder circuits from Phase 264
fn_xor = lambda a, b: a ^ b
fn_and = lambda a, b: a & b
fn_fa_sum = lambda a, b, cin: a ^ b ^ cin
fn_fa_carry = lambda a, b, cin: (a & b) | (cin & (a ^ b))

# Phase 2: 2-bit × 2-bit = shift-and-add
# a = a1*2 + a0, b = b1*2 + b0
# a * b = a0*b0 + (a0*b1 + a1*b0)*2 + a1*b1*4
# This decomposes into: 1-bit multiplies + additions

print('Phase 2: 2-bit x 2-bit via shift-and-add')
correct = 0
for a in range(4):
    for b in range(4):
        a0, a1 = a & 1, (a >> 1) & 1
        b0, b1 = b & 1, (b >> 1) & 1
        
        # Partial products (1-bit multiplies — Phase 1)
        p00 = fn_mul1([a0, b0])
        p01 = fn_mul1([a0, b1])
        p10 = fn_mul1([a1, b0])
        p11 = fn_mul1([a1, b1])
        
        # Sum partial products with adder (Phase 264 curriculum)
        # Bit 0: p00
        r0 = p00
        # Bit 1: p01 + p10
        r1 = fn_xor(p01, p10)
        c1 = fn_and(p01, p10)
        # Bit 2: p11 + c1
        r2 = fn_xor(p11, c1)
        c2 = fn_and(p11, c1)
        # Bit 3: c2
        r3 = c2
        
        pred = r0 + r1*2 + r2*4 + r3*8
        true = a * b
        if pred == true: correct += 1

print(f'  Verification: {correct}/16 ({correct/16*100:.0f}%)')
print(f'  3 x 3 = {fn_mul1([1,1]) + fn_xor(fn_mul1([1,1]),fn_mul1([1,1]))*2 + fn_xor(fn_mul1([1,1]),fn_and(fn_mul1([1,1]),fn_mul1([1,1])))*4}')
print(f'  True: {3*3}')
print()
print(f'  Curriculum: 1-bit multiply (AND) + adder (XOR,AND) = multi-bit multiply')
print(f'  Both primitives DISCOVERED, composition via curriculum')
" 2>&1
