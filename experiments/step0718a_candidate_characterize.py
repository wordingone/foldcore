"""
Step 718a — Standalone characterization of candidate.c.

DO NOT MODIFY candidate.c. Test what it IS.
DO NOT add 674 or any other infrastructure.

Tests:
1. Constant input (0x00 * N steps): constant/periodic/chaotic?
2. Random input: output distribution, entropy
3. Structured input (repeating 0..255): different from random?
4. Sensitivity: identical except 1 byte flipped at step 1000 — when does output diverge?

candidate.c IO:
- Reads 1 byte/step from stdin (getchar)
- Outputs 1 byte every 256 steps (f() -> putchar)
- n=1<<20 default -> 4096 output bytes for 1M steps
- Compile: /mingw64/bin/gcc -O2 -o candidate.exe candidate.c (under MSYS2 bash)
"""
import subprocess
import numpy as np
import sys
import os
import time

EXE = "B:/M/the-search/substrates/candidate/candidate.exe"
N_STEPS = 1 << 20  # 1M steps -> 4096 output bytes
N_OUT = N_STEPS // 256


def run_candidate(input_bytes, n_steps=N_STEPS):
    """Feed input_bytes to candidate.exe, return output bytes."""
    # Pad or truncate input to n_steps
    if len(input_bytes) < n_steps:
        input_bytes = input_bytes + bytes(n_steps - len(input_bytes))
    else:
        input_bytes = input_bytes[:n_steps]
    r = subprocess.run(
        [EXE, str(n_steps)],
        input=input_bytes, capture_output=True, timeout=60
    )
    return r.stdout


def entropy(data):
    """Shannon entropy in bits."""
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def autocorr_lag1(data):
    """Lag-1 autocorrelation of byte values."""
    a = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    if len(a) < 2:
        return 0.0
    return float(np.corrcoef(a[:-1], a[1:])[0, 1])


def run_tests():
    print(f"Step 718a: candidate.c characterization")
    print(f"EXE: {EXE}")
    print(f"N_STEPS: {N_STEPS:,}  N_OUT: {N_OUT}")
    print()

    # --- Test 1: Constant zero input ---
    print("=== Test 1: Constant input (0x00) ===")
    t0 = time.time()
    data_zero = run_candidate(bytes(N_STEPS))
    print(f"  Output bytes: {len(data_zero)}")
    if data_zero:
        counts = np.bincount(np.frombuffer(data_zero, dtype=np.uint8), minlength=256)
        unique = int((counts > 0).sum())
        h = entropy(data_zero)
        ac = autocorr_lag1(data_zero)
        print(f"  Unique values: {unique}/256")
        print(f"  Entropy: {h:.4f} bits (max 8.0)")
        print(f"  Lag-1 autocorr: {ac:.4f}")
        print(f"  First 32 bytes: {list(data_zero[:32])}")
        # Check periodicity
        arr = np.frombuffer(data_zero, dtype=np.uint8)
        periodic = False
        for period in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            if len(arr) >= 2 * period and np.all(arr[:len(arr) - period] == arr[period:]):
                print(f"  PERIODIC: period={period}")
                periodic = True
                break
        if not periodic:
            print(f"  Not periodic (checked periods 1-256)")
    print(f"  Time: {time.time()-t0:.2f}s")
    print()

    # --- Test 2: Random input ---
    print("=== Test 2: Random input (fixed seed) ===")
    rng = np.random.RandomState(42)
    rand_bytes = bytes(rng.randint(0, 256, N_STEPS, dtype=np.uint8).tolist())
    t0 = time.time()
    data_rand = run_candidate(rand_bytes)
    print(f"  Output bytes: {len(data_rand)}")
    if data_rand:
        counts = np.bincount(np.frombuffer(data_rand, dtype=np.uint8), minlength=256)
        unique = int((counts > 0).sum())
        h = entropy(data_rand)
        ac = autocorr_lag1(data_rand)
        mode_val = int(counts.argmax())
        mode_freq = int(counts.max())
        print(f"  Unique values: {unique}/256")
        print(f"  Entropy: {h:.4f} bits (max 8.0)")
        print(f"  Lag-1 autocorr: {ac:.4f}")
        print(f"  Mode: {mode_val} (freq={mode_freq}/{len(data_rand)})")
        print(f"  First 32 bytes: {list(data_rand[:32])}")
        # Histogram spread (min/max of top 10 counts)
        top10 = np.sort(counts)[-10:]
        print(f"  Top 10 counts: {top10.tolist()}")
    print(f"  Time: {time.time()-t0:.2f}s")
    print()

    # --- Test 3: Structured input (repeating 0..255) ---
    print("=== Test 3: Structured input (repeating 0..255) ===")
    struct_bytes = bytes([i % 256 for i in range(N_STEPS)])
    t0 = time.time()
    data_struct = run_candidate(struct_bytes)
    print(f"  Output bytes: {len(data_struct)}")
    if data_struct:
        counts = np.bincount(np.frombuffer(data_struct, dtype=np.uint8), minlength=256)
        unique = int((counts > 0).sum())
        h = entropy(data_struct)
        ac = autocorr_lag1(data_struct)
        print(f"  Unique values: {unique}/256")
        print(f"  Entropy: {h:.4f} bits (max 8.0)")
        print(f"  Lag-1 autocorr: {ac:.4f}")
        print(f"  First 32 bytes: {list(data_struct[:32])}")
        # Compare to random
        if data_rand:
            same = sum(a == b for a, b in zip(data_struct, data_rand))
            print(f"  Match with random: {same}/{min(len(data_struct),len(data_rand))} bytes")
    print(f"  Time: {time.time()-t0:.2f}s")
    print()

    # --- Test 4: Sensitivity (bit flip at step 1000) ---
    print("=== Test 4: Sensitivity (1-bit flip at step 1000) ===")
    base_bytes = bytes(rng.randint(0, 256, N_STEPS, dtype=np.uint8).tolist())
    flip_bytes = bytearray(base_bytes)
    flip_bytes[1000] ^= 0x01
    flip_bytes = bytes(flip_bytes)

    t0 = time.time()
    data_base = run_candidate(base_bytes)
    data_flip = run_candidate(flip_bytes)
    print(f"  Base output bytes: {len(data_base)}")
    print(f"  Flip output bytes: {len(data_flip)}")
    if data_base and data_flip:
        # Find first divergence in output
        first_diff = None
        for i, (a, b) in enumerate(zip(data_base, data_flip)):
            if a != b:
                first_diff = i
                break
        n = min(len(data_base), len(data_flip))
        diffs = sum(a != b for a, b in zip(data_base, data_flip))
        # Flip at step 1000 -> output at step 1000//256 = output byte 3
        flip_output_idx = 1000 // 256
        print(f"  Flip at step 1000 -> expected output impact at output byte ~{flip_output_idx}")
        print(f"  First divergence at output byte: {first_diff}")
        print(f"  Total differing bytes: {diffs}/{n}")
        print(f"  Divergence fraction: {diffs/n:.4f}")
        # Show divergence pattern (first 64 bytes)
        diff_mask = [int(a != b) for a, b in zip(data_base[:64], data_flip[:64])]
        print(f"  Diff mask (first 64): {diff_mask}")
    print(f"  Time: {time.time()-t0:.2f}s")
    print()

    print("=== Summary ===")
    if data_zero and data_rand and data_struct:
        h0 = entropy(data_zero)
        hr = entropy(data_rand)
        hs = entropy(data_struct)
        print(f"  Entropy: zero={h0:.3f} random={hr:.3f} structured={hs:.3f}")
        print(f"  Unique vals: zero={int((np.bincount(np.frombuffer(data_zero,dtype=np.uint8),minlength=256)>0).sum())} "
              f"random={int((np.bincount(np.frombuffer(data_rand,dtype=np.uint8),minlength=256)>0).sum())} "
              f"structured={int((np.bincount(np.frombuffer(data_struct,dtype=np.uint8),minlength=256)>0).sum())}")


if __name__ == "__main__":
    run_tests()
