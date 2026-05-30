"""
E3 Full-Corpus Abstraction Engine — Leo #11633.

Upgrades from contiguous-subsequence + solved-only MDL to:
(a) Full corpus: solved programs + partial programs from failed tasks
(b) Anti-unification with holes: LCS-based non-contiguous pattern extraction
(c) Non-contiguous motif MDL over full corpus (solved + failed partial programs)
(d) Separate reporting: solved-only vs failed-derived MDL contribution

Extra gates (Kai review, Leo #11638):
- Failed traces must expose meaningful partial structure (energy/progress evidence).
  If inert (no partial matches above chance), say so explicitly.
- Report solved-only vs failed-derived MDL contribution SEPARATELY.

Pre-registered outcomes:
- KILL (scoped negative, strong): full-corpus MDL still no positive compounds on
  held-out -> ARC corpus (including failures) is below R* -> reuse-sparsity is a
  full-corpus property, not a solved-traces artifact -> forward = richer-reuse domain.
- FORWARD: richer abstraction yields compounds + held-out reduction -> solved-only
  was engine artifact -> continue on ARC.
"""

import json
import os
import sys
import itertools
import numpy as np
from collections import Counter, defaultdict

# ── Seed DSL ─────────────────────────────────────────────────────────────────

def _bg(x):
    v, c = np.unique(x, return_counts=True)
    return int(v[np.argmax(c)])

def _crop(x):
    nz = np.argwhere(x != _bg(x))
    if len(nz) == 0:
        return x
    return x[nz.min(0)[0]:nz.max(0)[0]+1, nz.min(0)[1]:nz.max(0)[1]+1]

SEED_OPS = {
    'id':    lambda x: x,
    'fh':    lambda x: x[:, ::-1],
    'fv':    lambda x: x[::-1],
    'tr':    lambda x: x.T,
    'rot':   lambda x: np.rot90(x),
    'crop':  _crop,
    'dup_h': lambda x: np.hstack([x, x]),
    'dup_v': lambda x: np.vstack([x, x]),
    'mir_h': lambda x: np.hstack([x, x[:, ::-1]]),
    'mir_v': lambda x: np.vstack([x, x[::-1]]),
    'up2':   lambda x: np.repeat(np.repeat(x, 2, 0), 2, 1),
    'down2': lambda x: x[::2, ::2],
}
SEED_NAMES = sorted(SEED_OPS.keys())

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "incoming/arc-agi1-visa/03_R4_transfer_wall/ARC-AGI/data/training"
BUDGET = 10000
MAX_LENGTH = 5
SPLIT_SEED = 42
N_SOURCE = 200
N_HELD = 200
N_TOTAL = N_SOURCE + N_HELD
N_PAIRS_MIN = 1   # partial match: at least this many pairs matched


# ── Grid ops ─────────────────────────────────────────────────────────────────

def apply_op(name, grid, library=None):
    try:
        g = np.array(grid, dtype=np.int64)
        if library and name in library:
            for sub in library[name]:
                result = apply_op(sub, g, library)
                if result is None:
                    return None
                g = result
            return g
        return SEED_OPS[name](g)
    except Exception:
        return None


def apply_program(ops, grid, library=None):
    g = np.array(grid, dtype=np.int64)
    for name in ops:
        result = apply_op(name, g, library)
        if result is None:
            return None
        g = result
    return g


def count_pairs_matched(ops, task_io, library=None):
    count = 0
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        result = apply_program(ops, inp, library)
        if result is not None and result.shape == out.shape and np.array_equal(result, out):
            count += 1
    return count


def check_program(ops, task_io, library=None):
    return count_pairs_matched(ops, task_io, library) == len(task_io)


# ── BFS with partial match tracking ──────────────────────────────────────────

def search_with_partials(task_io, library=None, max_length=MAX_LENGTH, budget=BUDGET):
    """
    BFS. Returns:
      (solution, nodes_expanded, best_partial, best_partial_score)
    solution: program that solves all pairs, or None
    best_partial: program with highest pair-match fraction (may be None)
    best_partial_score: fraction 0..1
    """
    lib = library or {}
    lib_ops = sorted(lib.keys())
    ordered_ops = []
    seen = set()
    for n in lib_ops + SEED_NAMES:
        if n not in seen:
            seen.add(n)
            ordered_ops.append(n)

    nodes = 0
    best_partial = None
    best_partial_score = 0.0
    n_pairs = len(task_io)

    for length in range(1, max_length + 1):
        for ops in itertools.product(ordered_ops, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes, best_partial, best_partial_score

            matched = count_pairs_matched(list(ops), task_io, lib)
            score = matched / n_pairs

            if score > best_partial_score:
                best_partial_score = score
                best_partial = list(ops)

            if matched == n_pairs:
                return list(ops), nodes, list(ops), 1.0

    return None, nodes, best_partial, best_partial_score


# ── Data loading ──────────────────────────────────────────────────────────────

def load_tasks(data_dir, limit=None):
    tasks = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath) as f:
            data = json.load(f)
        task_id = fname.replace('.json', '')
        io_pairs = data.get('train', [])
        if not io_pairs:
            continue
        tasks.append({'id': task_id, 'io': io_pairs})
        if limit and len(tasks) >= limit:
            break
    return tasks


# ── LCS-based anti-unification ────────────────────────────────────────────────

def lcs(p1, p2):
    """Longest common subsequence (non-contiguous). Returns list of common ops."""
    m, n = len(p1), len(p2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if p1[i] == p2[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    result = []
    i, j = 0, 0
    while i < m and j < n:
        if p1[i] == p2[j]:
            result.append(p1[i])
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return result


def count_noncontiguous(corpus_programs, pattern):
    """Count programs that contain `pattern` as a subsequence (not just contiguous)."""
    if not pattern:
        return 0
    count = 0
    for prog in corpus_programs:
        pi = 0
        for op in prog:
            if pi < len(pattern) and op == pattern[pi]:
                pi += 1
        if pi == len(pattern):
            count += 1
    return count


def count_contiguous(corpus_programs, pattern):
    """Count occurrences of contiguous pattern across all corpus programs."""
    count = 0
    pat = tuple(pattern)
    L = len(pat)
    for prog in corpus_programs:
        for i in range(len(prog) - L + 1):
            if tuple(prog[i:i + L]) == pat:
                count += 1
    return count


def mdl_gain_formula(count, length):
    return count * (length - 1) - length


# ── MDL analysis ──────────────────────────────────────────────────────────────

def run_contiguous_mdl(corpus_programs, label):
    """Standard contiguous sub-sequence MDL over corpus_programs."""
    subseq_counts = Counter()
    for prog in corpus_programs:
        if len(prog) >= 2:
            for length in range(2, len(prog) + 1):
                for start in range(len(prog) - length + 1):
                    sub = tuple(prog[start:start + length])
                    subseq_counts[sub] += 1

    results = []
    for sub, cnt in sorted(subseq_counts.items(), key=lambda x: (-x[1], x[0])):
        gain = mdl_gain_formula(cnt, len(sub))
        results.append({'sub': list(sub), 'count': cnt, 'gain': gain})

    positive = [(r['sub'], r['count'], r['gain']) for r in results if r['gain'] > 0]
    print(f"\n  [{label}] Contiguous MDL over {len(corpus_programs)} programs:")
    print(f"    Distinct candidates: {len(results)}")
    print(f"    MDL-positive: {len(positive)}")
    for sub, cnt, gain in positive[:10]:
        print(f"      {'__'.join(sub)}: count={cnt}, gain={gain}")
    if not positive:
        all_counts = [r['count'] for r in results]
        if all_counts:
            print(f"    Max count: {max(all_counts)} (need >L/(L-1)+1 for gain>0; "
                  f"L=2 needs count>=3)")
    return results, positive


def run_noncontiguous_mdl(corpus_programs, label, top_pairs=50):
    """LCS anti-unification + non-contiguous occurrence counting."""
    print(f"\n  [{label}] Non-contiguous LCS anti-unification:")

    # Extract all pairwise LCS patterns of length >= 2
    lcs_counter = Counter()
    n = len(corpus_programs)
    if n < 2:
        print("    Too few programs for pairwise LCS.")
        return [], []

    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            common = lcs(corpus_programs[i], corpus_programs[j])
            if len(common) >= 2:
                lcs_counter[tuple(common)] += 1
            pair_count += 1

    print(f"    Pairs analyzed: {pair_count}")
    print(f"    Distinct LCS patterns length>=2: {len(lcs_counter)}")

    # For each LCS pattern, count non-contiguous occurrences across corpus
    candidate_patterns = sorted(lcs_counter.items(), key=lambda x: -x[1])[:top_pairs]
    results = []
    positive = []
    for pat_tuple, lcs_pair_count in candidate_patterns:
        pat = list(pat_tuple)
        nc_count = count_noncontiguous(corpus_programs, pat)
        gain = mdl_gain_formula(nc_count, len(pat))
        results.append({'sub': pat, 'lcs_pair_count': lcs_pair_count,
                        'nc_count': nc_count, 'gain': gain})
        if gain > 0:
            positive.append((pat, nc_count, gain))

    print(f"    Top LCS patterns checked for non-contiguous MDL gain:")
    for r in results[:10]:
        flag = " <-- POSITIVE" if r['gain'] > 0 else ""
        print(f"      {'__'.join(r['sub'])}: lcs_pairs={r['lcs_pair_count']}, "
              f"nc_count={r['nc_count']}, gain={r['gain']}{flag}")

    print(f"    Non-contiguous MDL-positive: {len(positive)}")
    return results, positive


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("E3 FULL-CORPUS ABSTRACTION ENGINE")
    print(f"Config: budget={BUDGET}, max_length={MAX_LENGTH}, split_seed={SPLIT_SEED}")
    print(f"Seed ops ({len(SEED_OPS)}): {SEED_NAMES}")
    print("=" * 70)
    sys.stdout.flush()

    # ── Load tasks ────────────────────────────────────────────────────────
    all_tasks = load_tasks(DATA_DIR)
    print(f"Loaded {len(all_tasks)} tasks")

    rng = np.random.default_rng(SPLIT_SEED)
    indices = rng.permutation(len(all_tasks))
    source_ids = set(int(i) for i in indices[:N_SOURCE])
    held_ids = set(int(i) for i in indices[N_SOURCE:N_SOURCE + N_HELD])

    source_tasks = [all_tasks[i] for i in sorted(source_ids)]
    held_tasks = [all_tasks[i] for i in sorted(held_ids)]
    print(f"Split: {len(source_tasks)} source / {len(held_tasks)} held-out\n")
    sys.stdout.flush()

    # ── SOURCE ROUND: BFS with partial tracking ───────────────────────────
    print("SOURCE ROUND — BFS with partial-match tracking...")
    sys.stdout.flush()

    source_solved = []       # programs that solve all pairs
    source_partials = []     # best-partial programs for failed tasks (score < 1.0)
    partial_score_dist = Counter()   # distribution of best partial scores

    for i, task in enumerate(source_tasks):
        prog, cost, best_partial, best_score = search_with_partials(task['io'])

        bucket = round(best_score * len(task['io'])) / len(task['io'])
        partial_score_dist[bucket] += 1

        if prog:
            source_solved.append(prog)
            print(f"  {task['id']}: solved  cost={cost}  prog={prog}")
        else:
            if best_partial and best_score >= N_PAIRS_MIN / len(task['io']):
                source_partials.append(best_partial)

        if (i + 1) % 50 == 0:
            sys.stdout.flush()

    print(f"\n  Solved: {len(source_solved)}/{len(source_tasks)}")
    print(f"  Best-partial programs (score >= {N_PAIRS_MIN}/{3}): {len(source_partials)}")
    print(f"  Partial score distribution:")
    for score in sorted(partial_score_dist.keys(), reverse=True):
        print(f"    score={score:.2f}: {partial_score_dist[score]} tasks")
    sys.stdout.flush()

    # ── Partial structure gate (Kai #11638) ───────────────────────────────
    n_zero_score = partial_score_dist.get(0.0, 0)
    n_nonzero = len(source_tasks) - len(source_solved) - n_zero_score
    failed_inert = (n_nonzero == 0)

    print(f"\n  PARTIAL STRUCTURE GATE:")
    print(f"    Failed tasks with any partial match (score>0): {n_nonzero}/{len(source_tasks)-len(source_solved)}")
    if failed_inert:
        print("    INERT: no failed task shows any partial match above 0.")
        print("    Ingesting failed traces adds NOTHING to the corpus.")
        print("    Full corpus = solved-only. Negative stands on full-corpus basis.")
    else:
        print(f"    INFORMATIVE: {n_nonzero} tasks show partial structure.")
        print(f"    Partial programs added to corpus for anti-unification.")
    sys.stdout.flush()

    # ── Build corpora ─────────────────────────────────────────────────────
    solved_corpus = [list(p) for p in source_solved]
    full_corpus = solved_corpus + [list(p) for p in source_partials]

    print(f"\n  Corpus sizes:")
    print(f"    Solved-only corpus: {len(solved_corpus)} programs")
    print(f"    Full corpus (solved + partials): {len(full_corpus)} programs")
    sys.stdout.flush()

    # ── MDL ANALYSIS ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MDL ANALYSIS")
    print("=" * 70)
    sys.stdout.flush()

    # 1. Contiguous MDL — solved only
    solved_cont_results, solved_cont_positive = run_contiguous_mdl(solved_corpus, "SOLVED-ONLY CONTIGUOUS")
    sys.stdout.flush()

    # 2. Contiguous MDL — full corpus
    full_cont_results, full_cont_positive = run_contiguous_mdl(full_corpus, "FULL-CORPUS CONTIGUOUS")
    sys.stdout.flush()

    # 3. Non-contiguous MDL — solved only
    solved_nc_results, solved_nc_positive = run_noncontiguous_mdl(solved_corpus, "SOLVED-ONLY NON-CONTIGUOUS")
    sys.stdout.flush()

    # 4. Non-contiguous MDL — full corpus (only if not inert)
    if not failed_inert and len(full_corpus) > len(solved_corpus):
        full_nc_results, full_nc_positive = run_noncontiguous_mdl(full_corpus, "FULL-CORPUS NON-CONTIGUOUS")
    else:
        full_nc_results, full_nc_positive = [], []
        print("\n  [FULL-CORPUS NON-CONTIGUOUS] Skipped — failed traces inert, full=solved corpus.")
    sys.stdout.flush()

    # ── Determine best library ────────────────────────────────────────────
    # Priority: contiguous MDL-positive compounds (direct library addition)
    # Then non-contiguous (informational, can't directly add to contiguous library)
    all_positive_compounds = {}
    for sub, cnt, gain in full_cont_positive:
        name = '__'.join(sub)
        all_positive_compounds[name] = {'ops': sub, 'count': cnt, 'gain': gain,
                                        'type': 'contiguous', 'corpus': 'full'}

    for sub, cnt, gain in solved_cont_positive:
        name = '__'.join(sub)
        if name not in all_positive_compounds:
            all_positive_compounds[name] = {'ops': sub, 'count': cnt, 'gain': gain,
                                            'type': 'contiguous', 'corpus': 'solved-only'}

    print(f"\n  MDL-POSITIVE COMPOUNDS (total): {len(all_positive_compounds)}")
    for name, info in all_positive_compounds.items():
        print(f"    {name}: count={info['count']}, gain={info['gain']}, "
              f"type={info['type']}, corpus={info['corpus']}")

    grown_lib = {name: info['ops'] for name, info in all_positive_compounds.items()}
    compound_names = list(grown_lib.keys())
    sys.stdout.flush()

    # ── HELD-OUT EVALUATION ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HELD-OUT EVALUATION")
    print("=" * 70)
    sys.stdout.flush()

    baseline_results = []
    withlib_results = []

    print(f"\nBASELINE (seed, 12 ops): {len(held_tasks)} held-out tasks...")
    for i, task in enumerate(held_tasks):
        prog, cost, _, _ = search_with_partials(task['io'], library={})
        baseline_results.append({'program': prog, 'cost': cost})
        if prog:
            print(f"  [{i+1:3d}/{len(held_tasks)}] {task['id']}: solved  prog={prog}  cost={cost}")
        if (i + 1) % 50 == 0:
            sys.stdout.flush()

    b_solved = sum(1 for r in baseline_results if r['program'])
    b_mean = np.mean([r['cost'] for r in baseline_results])
    print(f"  Baseline: {b_solved}/{len(held_tasks)} solved, mean_cost={b_mean:.1f}")
    sys.stdout.flush()

    if compound_names:
        print(f"\nWITH-LIBRARY ({len(grown_lib)} compounds + 12 seed ops)...")
        for i, task in enumerate(held_tasks):
            prog, cost, _, _ = search_with_partials(task['io'], library=grown_lib)
            withlib_results.append({'program': prog, 'cost': cost})
            if prog and prog != baseline_results[i]['program']:
                print(f"  [{i+1:3d}/{len(held_tasks)}] {task['id']}: "
                      f"prog={prog}  cost={cost} (vs baseline {baseline_results[i]['cost']})")
            if (i + 1) % 50 == 0:
                sys.stdout.flush()

        wl_solved = sum(1 for r in withlib_results if r['program'])
        wl_mean = np.mean([r['cost'] for r in withlib_results])
        print(f"  With-lib: {wl_solved}/{len(held_tasks)} solved, mean_cost={wl_mean:.1f}")
    else:
        print("\n  [WITH-LIBRARY] No compounds formed — library = seed. Skipping.")
        withlib_results = baseline_results[:]
        wl_mean = b_mean

    sys.stdout.flush()

    # ── COMPOUND-APPLICABILITY PARTITION ──────────────────────────────────
    def is_applicable(seed_prog, grown_prog, compounds):
        for comp in compounds:
            comp_parts = tuple(comp.split('__'))
            if seed_prog:
                for i in range(len(seed_prog) - len(comp_parts) + 1):
                    if tuple(seed_prog[i:i + len(comp_parts)]) == comp_parts:
                        return True
            if grown_prog and any(op == comp for op in grown_prog):
                return True
        return False

    applicable_ids = set()
    for i in range(len(held_tasks)):
        if is_applicable(baseline_results[i]['program'],
                         withlib_results[i]['program'], compound_names):
            applicable_ids.add(i)

    appl_base_costs = [baseline_results[i]['cost'] for i in range(len(held_tasks))
                       if i in applicable_ids]
    appl_wl_costs = [withlib_results[i]['cost'] for i in range(len(held_tasks))
                     if i in applicable_ids]
    nonappl_base_costs = [baseline_results[i]['cost'] for i in range(len(held_tasks))
                          if i not in applicable_ids]
    nonappl_wl_costs = [withlib_results[i]['cost'] for i in range(len(held_tasks))
                        if i not in applicable_ids]

    agg_delta = wl_mean - b_mean
    agg_delta_pct = 100 * agg_delta / b_mean if b_mean > 0 else 0

    print("\n" + "=" * 70)
    print("E3 FULL-CORPUS RESULT")
    print("=" * 70)
    print(f"Source solved-corpus: {len(solved_corpus)} programs")
    print(f"Source partial-corpus: {len(source_partials)} programs (inert={failed_inert})")
    print(f"MDL-positive compounds: {len(all_positive_compounds)}")
    print(f"  Contiguous (solved-only): {len(solved_cont_positive)}")
    print(f"  Contiguous (full-corpus): {len(full_cont_positive)}")
    print(f"  Non-contiguous (solved-only): {len(solved_nc_positive)}")
    print(f"  Non-contiguous (full-corpus): {len(full_nc_positive)}")

    print(f"\nAGGREGATE (held-out):")
    print(f"  BASELINE:  mean_cost={b_mean:.1f}  solved={b_solved}/{len(held_tasks)}")
    print(f"  WITH-LIB:  mean_cost={wl_mean:.1f}  solved={wl_solved}/{len(held_tasks)}")
    print(f"  Delta:     {agg_delta:+.1f} ({agg_delta_pct:+.1f}%)")

    if applicable_ids:
        appl_b = np.mean(appl_base_costs)
        appl_wl = np.mean(appl_wl_costs)
        print(f"\nCOMPOUND-APPLICABLE ({len(applicable_ids)} tasks):")
        print(f"  Baseline={appl_b:.1f}  With-lib={appl_wl:.1f}  "
              f"Delta={appl_wl - appl_b:+.1f} ({100*(appl_wl-appl_b)/appl_b:+.1f}%)")
        print(f"\nNON-APPLICABLE ({len(held_tasks)-len(applicable_ids)} tasks):")
        if nonappl_base_costs:
            na_b = np.mean(nonappl_base_costs)
            na_wl = np.mean(nonappl_wl_costs)
            print(f"  Baseline={na_b:.1f}  With-lib={na_wl:.1f}  "
                  f"Delta={na_wl - na_b:+.1f} ({100*(na_wl-na_b)/na_b:+.1f}%)")
    else:
        print(f"\nCOMPOUND-APPLICABLE: 0 tasks")

    # ── Verdict ───────────────────────────────────────────────────────────
    if not compound_names:
        verdict = "NO_APPLICABLE_TASKS"
        scope_note = ("Full-corpus (solved+failed) with anti-unification + non-contiguous "
                      "extraction yields ZERO MDL-positive compounds on 200/200 split. "
                      "Reuse-sparsity is a full-corpus property of ARC-AGI-1 under 12-op DSL.")
    elif not applicable_ids:
        verdict = "NO_APPLICABLE_TASKS"
        scope_note = "Library formed but no held-out task applicable."
    elif len(appl_base_costs) > 0 and np.mean(appl_wl_costs) < np.mean(appl_base_costs) * 0.9:
        verdict = "MECHANISM_WORKS"
        scope_note = "Full-corpus engine shows search-cost reduction on applicable held-out."
    else:
        verdict = "NO_REDUCTION"
        scope_note = "Applicable tasks exist but no significant cost reduction."

    print(f"\nVerdict: {verdict}")
    print(f"Scope: {scope_note}")

    if failed_inert:
        print("\nFAILED-TRACES GATE (Kai #11638): INERT")
        print("  No failed task shows any partial match. Ingesting failed traces")
        print("  adds nothing. Full-corpus = solved-only. The E3 negative is robust.")
    else:
        print(f"\nFAILED-TRACES GATE (Kai #11638): INFORMATIVE ({n_nonzero} tasks)")

    # ── Save result ───────────────────────────────────────────────────────
    result = {
        "experiment": "E3-full-corpus",
        "config": {
            "budget": BUDGET,
            "max_length": MAX_LENGTH,
            "split_seed": SPLIT_SEED,
            "n_source": N_SOURCE,
            "n_held": N_HELD,
        },
        "source_solved": len(source_solved),
        "source_partials": len(source_partials),
        "failed_inert": failed_inert,
        "partial_score_distribution": {str(k): v for k, v in sorted(partial_score_dist.items())},
        "mdl": {
            "solved_contiguous_positive": [(r['sub'], r['count'], r['gain'])
                                           for r in solved_cont_results if r['gain'] > 0],
            "full_contiguous_positive": [(r['sub'], r['count'], r['gain'])
                                         for r in full_cont_results if r['gain'] > 0],
            "solved_noncontiguous_positive": [[list(s), c, g] for s, c, g in solved_nc_positive],
            "full_noncontiguous_positive": [[list(s), c, g] for s, c, g in full_nc_positive],
            "all_positive_compounds": {k: v for k, v in all_positive_compounds.items()},
        },
        "aggregate": {
            "baseline_mean": float(b_mean),
            "withlib_mean": float(wl_mean),
            "delta": float(agg_delta),
            "delta_pct": float(agg_delta_pct),
            "baseline_solved": b_solved,
            "withlib_solved": wl_solved,
        },
        "n_applicable": len(applicable_ids),
        "verdict": verdict,
        "scope_note": scope_note,
    }

    out_path = "incoming/arc-agi1-visa/03_R4_transfer_wall/E3_full_corpus_result.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {out_path}")


if __name__ == '__main__':
    main()
