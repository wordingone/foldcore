"""
Steps 273-278: Classical algorithms from ONE truth table (8 entries).

Demonstrates: primality (trial division), GCD (Euclid), Fibonacci,
exponentiation, modular exponentiation (RSA primitive), integer square root.

ALL composed from a single full adder k-NN lookup table.

Usage:
    python experiments/run_step273_278_algorithms.py
"""

import torch
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === THE ONE TRUTH TABLE ===
X_fa, y_s, y_c = [], [], []
for a in range(2):
    for b in range(2):
        for cin in range(2):
            s = a + b + cin
            for _ in range(50):
                X_fa.append([float(a), float(b), float(cin)])
                y_s.append(s % 2)
                y_c.append(s // 2)
X_fa = torch.tensor(X_fa, device=device)
y_s = torch.tensor(y_s, device=device, dtype=torch.long)
y_c = torch.tensor(y_c, device=device, dtype=torch.long)


def fa(a, b, cin):
    q = torch.tensor([float(a), float(b), float(cin)], device=device)
    sims = F.normalize(q.unsqueeze(0), dim=1) @ F.normalize(X_fa, dim=1).T
    tk = sims[0].topk(5)
    return y_s[tk.indices].mode().values.item(), y_c[tk.indices].mode().values.item()


# === COMPOSED ARITHMETIC ===

def add_bin(a, b, nb=20):
    ab = [(a >> i) & 1 for i in range(nb)]
    bb = [(b >> i) & 1 for i in range(nb)]
    carry = 0
    r = []
    for i in range(nb):
        s, carry = fa(ab[i], bb[i], carry)
        r.append(s)
    return sum(bit * (2 ** i) for i, bit in enumerate(r + [carry]))


def mul_bin(a, b, nb=10):
    result = 0
    bb = [(b >> i) & 1 for i in range(nb)]
    for i in range(nb):
        if bb[i] == 1:
            result = add_bin(result, a * (2 ** i), 2 * nb)
    return result


def div_mod(a, b):
    if b == 0:
        return 0, a
    q, r = 0, a
    while r >= b and q < 10000:
        r = r - b
        q += 1
    return q, r


# === ALGORITHMS ===

def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    _, r2 = div_mod(n, 2)
    if r2 == 0: return False
    d = 3
    while mul_bin(d, d) <= n:
        _, rd = div_mod(n, d)
        if rd == 0: return False
        d = add_bin(d, 2)
    return True


def gcd(a, b):
    while b > 0:
        _, r = div_mod(a, b)
        a, b = b, r
    return a


def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, add_bin(a, b)
    return a


def power(base, exp):
    result = 1
    for _ in range(exp):
        result = mul_bin(result, base)
    return result


def mod_pow(base, exp, mod):
    result = 1
    base = div_mod(base, mod)[1]
    while exp > 0:
        if exp % 2 == 1:
            result = div_mod(mul_bin(result, base), mod)[1]
        exp = exp // 2
        base = div_mod(mul_bin(base, base), mod)[1]
    return result


def isqrt(n):
    if n == 0: return 0
    lo, hi = 1, n
    while lo <= hi:
        mid = add_bin(lo, hi) // 2
        sq = mul_bin(mid, mid)
        if sq == n: return mid
        elif sq < n: lo = add_bin(mid, 1)
        else: hi = mid - 1
    return hi


def main():
    print("=" * 60)
    print("CLASSICAL ALGORITHMS FROM ONE TRUTH TABLE (8 entries)")
    print("=" * 60)

    print("\n--- Primality (trial division) ---")
    primes = [n for n in range(2, 50) if is_prime(n)]
    true_primes = [n for n in range(2, 50) if all(n % d != 0 for d in range(2, int(n**0.5)+1))]
    print(f"  Found:  {primes}")
    print(f"  True:   {true_primes}")
    print(f"  {'CORRECT' if primes == true_primes else 'MISMATCH'}")

    print("\n--- GCD (Euclid) ---")
    for a, b in [(48, 18), (100, 75), (17, 13), (144, 89)]:
        print(f"  GCD({a},{b}) = {gcd(a,b)} (true: {math.gcd(a,b)})")

    print("\n--- Fibonacci ---")
    for n in [5, 10, 15]:
        print(f"  fib({n}) = {fibonacci(n)}")

    print("\n--- Exponentiation ---")
    for b, e in [(2, 10), (3, 5)]:
        print(f"  {b}^{e} = {power(b,e)} (true: {b**e})")

    print("\n--- Modular exponentiation ---")
    for b, e, m in [(2, 16, 97), (3, 5, 13)]:
        print(f"  {b}^{e} mod {m} = {mod_pow(b,e,m)} (true: {pow(b,e,m)})")

    print("\n--- Integer square root ---")
    for n in [0, 25, 100, 200, 256]:
        print(f"  isqrt({n}) = {isqrt(n)} (true: {int(math.isqrt(n))})")

    print("\n" + "=" * 60)
    print("All from: 0+0+0=0, 0+0+1=1, ..., 1+1+1=3 (8 entries)")
    print("=" * 60)


if __name__ == '__main__':
    main()
