#!/usr/bin/env python3
"""
Step 96 -- Temporal order discrimination.
Spec. k=8, spectral Phi, Formula C.

Task: sequences of length 5-10 from {A,B,C,D}.
Class 0: A appears before B (first A before first B).
Class 1: B appears before A.
Both classes contain both A and B.

Key test: does spectral substrate beat 50% where order-blind methods can't?
"""
import random, math, time

SEED = 42
K = 8
LANDSCAPE_STEPS = 200
COMPOSE_STEPS = 200
CONVERGE_TOL = 0.01
BASIN_COS = 0.95
KNOWN_COS = 0.90
N_EFS = 4  # A=EF_0, B=EF_1, C=EF_2, D=EF_3
N_TRAIN = 500
N_TEST = 200
SEQ_LEN_MIN = 5
SEQ_LEN_MAX = 10
SYMBOLS = ['A', 'B', 'C', 'D']


def target_norm(k): return math.sqrt(k)


def mmul(A, B, k):
    return [[sum(A[i][l]*B[l][j] for l in range(k)) for j in range(k)] for i in range(k)]


def mmt(M, k):
    return [[sum(M[i][l]*M[j][l] for l in range(k)) for j in range(k)] for i in range(k)]


def madd(A, B, k):
    return [[A[i][j]+B[i][j] for j in range(k)] for i in range(k)]


def msub(A, B, k):
    return [[A[i][j]-B[i][j] for j in range(k)] for i in range(k)]


def mscale(M, s, k):
    return [[M[i][j]*s for j in range(k)] for i in range(k)]


def frob(M, k):
    return math.sqrt(sum(M[i][j]**2 for i in range(k) for j in range(k)))


def cosine(A, B, k):
    dot = sum(A[i][j]*B[i][j] for i in range(k) for j in range(k))
    na, nb = frob(A, k), frob(B, k)
    return dot/(na*nb) if na > 1e-10 and nb > 1e-10 else 0.0


def copy_mat(M, k):
    return [[M[i][j] for j in range(k)] for i in range(k)]


def melem_mean(mats, k):
    """Element-wise mean of a list of matrices."""
    n = len(mats)
    result = [[0.0]*k for _ in range(k)]
    for M in mats:
        for i in range(k):
            for j in range(k):
                result[i][j] += M[i][j] / n
    return result


def phi(M, k):
    C = mmt(M, k)
    n = frob(C, k)
    if n < 1e-10: return copy_mat(M, k)
    return mscale(C, target_norm(k)/n, k)


def converge(M, k, max_steps, tol=CONVERGE_TOL):
    for _ in range(max_steps):
        p = phi(M, k)
        d = frob(msub(p, M, k), k)
        M = p
        if d < tol: return M, True
    return M, False


def psi_C(A, B, k):
    AB = mmul(A, B, k)
    n = frob(AB, k)
    if n < 1e-10: return madd(A, B, k)
    scaled = mscale(AB, target_norm(k)/n, k)
    return msub(madd(A, B, k), scaled, k)


def compose(A, B, k):
    return converge(psi_C(A, B, k), k, COMPOSE_STEPS)


def chain(symbol_seq, efs, k):
    """Left-to-right composition of symbol sequence."""
    M, conv = compose(efs[symbol_seq[0]], efs[symbol_seq[1]], k)
    if not conv: return None, False
    for i in range(2, len(symbol_seq)):
        M, conv = compose(M, efs[symbol_seq[i]], k)
        if not conv: return None, False
    return M, True


def same_basin(A, B, k):
    return abs(cosine(A, B, k)) > BASIN_COS


def find_efs(n, k, seed):
    rng = random.Random(seed)
    found = []
    for _ in range(2000):
        M0 = [[rng.uniform(-1, 1) for _ in range(k)] for _ in range(k)]
        M_f, conv = converge(M0, k, LANDSCAPE_STEPS)
        if conv and all(not same_basin(M_f, ef, k) for ef in found):
            found.append(M_f)
            if len(found) == n: break
    return found


# ─── Sequence generation ─────────────────────────────────────────────────────

def make_sequence(cls, rng):
    """Generate sequence of length 5-10.
    cls 0: A appears before B.
    cls 1: B appears before A.
    Both contain at least one A and one B.
    Starts with C or D to eliminate first-element class signal.
    Middle positions are random {A,B,C,D}.
    """
    while True:
        length = rng.randint(SEQ_LEN_MIN, SEQ_LEN_MAX)
        # Force first element to be C(2) or D(3)
        seq = [rng.randint(2, 3)] + [rng.randint(0, 3) for _ in range(length - 1)]
        has_a = 0 in seq
        has_b = 1 in seq
        if not has_a or not has_b: continue
        first_a = next(i for i, s in enumerate(seq) if s == 0)
        first_b = next(i for i, s in enumerate(seq) if s == 1)
        if cls == 0 and first_a < first_b: return seq
        if cls == 1 and first_b < first_a: return seq


def make_dataset(n, seed):
    """Make balanced dataset of n sequences (n//2 per class)."""
    rng = random.Random(seed)
    data = []
    per_class = n // 2
    for cls in range(2):
        for _ in range(per_class):
            seq = make_sequence(cls, rng)
            data.append((seq, cls))
    rng.shuffle(data)
    return data


# ─── Classifiers ─────────────────────────────────────────────────────────────

def classify_compositional(train_data, test_data, efs, k):
    """Classify by cosine to element-wise mean of class chain results."""
    # Build class prototypes
    class_results = {0: [], 1: []}
    n_fail = 0
    for seq, cls in train_data:
        M_f, conv = chain(seq, efs, k)
        if conv:
            class_results[cls].append(M_f)
        else:
            n_fail += 1

    protos = {}
    for cls in [0, 1]:
        if class_results[cls]:
            protos[cls] = melem_mean(class_results[cls], k)

    # Cosine similarity between prototypes
    if 0 in protos and 1 in protos:
        cos_p = cosine(protos[0], protos[1], k)
        print(f"  Proto cosine similarity: {cos_p:.4f}")

    # Classify test
    correct = 0
    for seq, cls in test_data:
        M_f, conv = chain(seq, efs, k)
        if not conv:
            pred = 0
        else:
            cos0 = cosine(M_f, protos.get(0, M_f), k)
            cos1 = cosine(M_f, protos.get(1, M_f), k)
            pred = 0 if cos0 >= cos1 else 1
        if pred == cls: correct += 1

    acc = correct / len(test_data)
    print(f"  Compositional: {correct}/{len(test_data)} ({acc*100:.1f}%), "
          f"train_fail={n_fail}/{len(train_data)}")
    return acc


def classify_bag_of_symbols(train_data, test_data):
    """Order-blind: count symbol occurrences, classify by count vector."""
    # Build class mean counts
    class_counts = {0: [0.0]*4, 1: [0.0]*4}
    class_n = {0: 0, 1: 0}
    for seq, cls in train_data:
        for s in seq:
            class_counts[cls][s] += 1
        class_n[cls] += 1
    protos = {cls: [class_counts[cls][s]/class_n[cls] for s in range(4)]
              for cls in [0, 1]}

    def dot(a, b): return sum(x*y for x, y in zip(a, b))
    def norm(a): return math.sqrt(sum(x*x for x in a) + 1e-15)

    correct = 0
    for seq, cls in test_data:
        counts = [seq.count(s) for s in range(4)]
        n = norm(counts)
        v = [x/n for x in counts]
        cos0 = dot(v, [x/norm(protos[0]) for x in protos[0]])
        cos1 = dot(v, [x/norm(protos[1]) for x in protos[1]])
        pred = 0 if cos0 >= cos1 else 1
        if pred == cls: correct += 1

    acc = correct / len(test_data)
    print(f"  Bag-of-symbols (order-blind): {correct}/{len(test_data)} ({acc*100:.1f}%)")
    return acc


def classify_last_element(train_data, test_data):
    """Classify by last symbol."""
    # Build class distribution of last elements
    class_last = {0: [0]*4, 1: [0]*4}
    for seq, cls in train_data:
        class_last[cls][seq[-1]] += 1
    correct = 0
    for seq, cls in test_data:
        last = seq[-1]
        p0 = class_last[0][last] + 1  # Laplace smoothing
        p1 = class_last[1][last] + 1
        pred = 0 if p0 >= p1 else 1
        if pred == cls: correct += 1
    acc = correct / len(test_data)
    print(f"  Last-element: {correct}/{len(test_data)} ({acc*100:.1f}%)")
    return acc


def classify_vector_mean(train_data, test_data, efs, k):
    """Order-blind: mean of symbol embeddings (flattened)."""
    def seq_to_vec(seq):
        mean = [[0.0]*k for _ in range(k)]
        for s in seq:
            ef = efs[s]
            for i in range(k):
                for j in range(k):
                    mean[i][j] += ef[i][j] / len(seq)
        return mean

    class_vecs = {0: [], 1: []}
    for seq, cls in train_data:
        class_vecs[cls].append(seq_to_vec(seq))
    protos = {cls: melem_mean(class_vecs[cls], k) for cls in [0, 1]}

    correct = 0
    for seq, cls in test_data:
        v = seq_to_vec(seq)
        cos0 = cosine(v, protos[0], k)
        cos1 = cosine(v, protos[1], k)
        pred = 0 if cos0 >= cos1 else 1
        if pred == cls: correct += 1
    acc = correct / len(test_data)
    print(f"  Vector mean (order-blind): {correct}/{len(test_data)} ({acc*100:.1f}%)")
    return acc


def classify_first_element(train_data, test_data):
    """Classify by first symbol. Should be informative since class depends on first A/B."""
    class_first = {0: [0]*4, 1: [0]*4}
    for seq, cls in train_data:
        class_first[cls][seq[0]] += 1
    correct = 0
    for seq, cls in test_data:
        first = seq[0]
        p0 = class_first[0][first] + 1
        p1 = class_first[1][first] + 1
        pred = 0 if p0 >= p1 else 1
        if pred == cls: correct += 1
    acc = correct / len(test_data)
    print(f"  First-element: {correct}/{len(test_data)} ({acc*100:.1f}%)")
    return acc


def main():
    t0 = time.time()
    print(f"Step 96 -- Temporal order discrimination, k={K}", flush=True)
    print(f"Task: Class 0 = A before B, Class 1 = B before A", flush=True)
    print(f"Sequences length {SEQ_LEN_MIN}-{SEQ_LEN_MAX}, alphabet {{A,B,C,D}}", flush=True)
    print()

    # Find 4 eigenforms
    print(f"Finding {N_EFS} eigenforms...", flush=True)
    efs = find_efs(N_EFS, K, SEED + K)
    print(f"  Found {len(efs)}: A=EF_0, B=EF_1, C=EF_2, D=EF_3")

    # Verify A o B != B o A
    AB, c1 = compose(efs[0], efs[1], K)
    BA, c2 = compose(efs[1], efs[0], K)
    if c1 and c2:
        cos_ab_ba = cosine(AB, BA, K)
        print(f"  Verify: A o B vs B o A cosine = {cos_ab_ba:.4f} "
              f"({'DIFFER' if abs(cos_ab_ba) < BASIN_COS else 'SAME'})")
    print()

    # Generate data
    print(f"Generating sequences (train={N_TRAIN}, test={N_TEST})...", flush=True)
    train_data = make_dataset(N_TRAIN, SEED)
    test_data = make_dataset(N_TEST, SEED + 1)
    print(f"  Train class balance: "
          f"{sum(1 for _,c in train_data if c==0)}/{sum(1 for _,c in train_data if c==1)}")
    print(f"  Avg seq length train: {sum(len(s) for s,_ in train_data)/len(train_data):.1f}")
    print()

    # Run classifiers
    print("=== Results ===", flush=True)
    accs = {}

    t1 = time.time()
    accs['compositional'] = classify_compositional(train_data, test_data, efs, K)
    print(f"  [time: {time.time()-t1:.1f}s]")
    print()

    accs['bag'] = classify_bag_of_symbols(train_data, test_data)
    accs['last'] = classify_last_element(train_data, test_data)
    accs['first'] = classify_first_element(train_data, test_data)
    accs['vec_mean'] = classify_vector_mean(train_data, test_data, efs, K)

    elapsed = time.time() - t0
    print(f"\n=== Summary ===")
    print(f"  Compositional (Formula C):   {accs['compositional']*100:.1f}%")
    print(f"  Vector mean (order-blind):   {accs['vec_mean']*100:.1f}%")
    print(f"  Bag-of-symbols (order-blind):{accs['bag']*100:.1f}%")
    print(f"  Last-element:                {accs['last']*100:.1f}%")
    print(f"  First-element (sanity):      {accs['first']*100:.1f}%")
    print(f"  Random baseline:             50.0%")
    delta = (accs['compositional'] - 0.5) * 100
    print(f"  Compositional vs random: {delta:+.1f}pp")
    print(f"Elapsed: {elapsed:.1f}s")

    if accs['compositional'] > 0.55:
        print(f"\n  VERDICT: Substrate encodes temporal order "
              f"({accs['compositional']*100:.1f}% > 55% threshold)")
    elif accs['compositional'] > 0.50:
        print(f"\n  VERDICT: Weak order signal ({accs['compositional']*100:.1f}%, marginal)")
    else:
        print(f"\n  VERDICT: No order encoding ({accs['compositional']*100:.1f}%, at or below random)")


if __name__ == '__main__':
    main()
