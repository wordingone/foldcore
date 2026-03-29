"""
Masked PRISM infrastructure — permanent.

Structural enforcement: game IDs never appear in any readable output.
Only labels (MBPP, Game A, Game B, ...) appear in logs, filenames, and results.
The sealed mapping (.sealed_game_mapping.json) is written once and not read
by Eli or Leo during the session.

Protocol (Jun directive, 2026-03-29):
- Substrate initialization is DETERMINISTIC — no seeds, no draws.
- 3 games × N conditions = N×3 runs per experiment. No multi-draw loops.
- Environment breaks symmetry through observations, not random init.

Level masking (Jun directive, 2026-03-29):
- Level numbers (L1, L2, L3) are NOT reported in any output.
- Metric: steps_to_first_progress (opaque — first time game state advanced, no level label).
- Diagnostics: progress_count (how many advances) instead of max_level / level_first_step.
- Speedup: steps_to_first_progress(try1) / steps_to_first_progress(try2).
- Apply to step 1333 and all future experiment scripts.

RHAE optimal_steps (Leo directive, 2026-03-29):
- MBPP: optimal_steps = len(correct_solution) via mbpp_game.compute_solver_steps.
- ARC: no solver endpoint. Use ARC_OPTIMAL_STEPS_PROXY = 10 (proxy until solver available).
  10 is conservative (most ARC games need more than 10 clicks), makes RHAE non-zero
  when progress reached, allowing condition comparison. RHAE values on ARC are relative,
  not absolute, until a real solver is available.

Usage in experiment scripts:
    from prism_masked import select_games, seal_mapping, label_filename, det_weights
    from prism_masked import compute_progress_speedup, compute_rhae_try2, write_experiment_results
    from prism_masked import ARC_OPTIMAL_STEPS_PROXY, get_arc_optimal_steps

    GAMES, GAME_LABELS = select_games(seed=STEP)
    # GAMES is internal-only — never print, never log, never pass to Leo.
    # GAME_LABELS maps game_id -> label string (MBPP, Game A, Game B, ...).
    # All output uses GAME_LABELS[game] or just the label string directly.

    # Deterministic weight init (no np.random.seed — same shape = same weights):
    W1 = det_weights(128, 256)  # orthogonal init from fixed QR

    # Progress-based speedup (level-masked):
    speedup = compute_progress_speedup(first_progress_try1, first_progress_try2)
    # first_progress = step at which any game state advancement occurred (opaque)
    # None if no advancement in the episode.
"""

import random
import os
import json
import numpy as np

# ARC optimal_steps proxy (Leo directive, 2026-03-29).
# No solver endpoint available. Proxy = 10 makes RHAE non-zero when progress reached.
# Replace with game-specific values when a solver/reference is available.
ARC_OPTIMAL_STEPS_PROXY = 10


def get_arc_optimal_steps(game_name=None):
    """Return optimal_steps for ARC games (proxy until solver available)."""
    return ARC_OPTIMAL_STEPS_PROXY

ARC_POOL = [
    'ft09', 'ls20', 'vc33', 'tr87', 'sp80', 'sb26',
    'tu93', 'cn04', 'cd82', 'lp85',
]


def select_games(seed, n_arc=2, include_mbpp=True):
    """Select games for an experiment step.

    Game selection uses a seed (experiment-level selection, not substrate init).
    This is kept: which 2 ARC games to use is experiment selection, not substrate randomness.

    Args:
        seed: Experiment step number (used as RNG seed for game selection only).
        n_arc: Number of random ARC games to include (default 2, per Jun directive).
        include_mbpp: If True, MBPP is always included and labeled 'MBPP' (not masked).

    Returns:
        games: list of game IDs — INTERNAL USE ONLY. Never print or log.
        labels: dict mapping game_id -> label string for all output.

    Example:
        GAMES, GAME_LABELS = select_games(seed=1313)
        # GAMES = ['mbpp', 'ft09', 'vc33']  ← internal, do NOT expose
        # GAME_LABELS = {'mbpp': 'MBPP', 'ft09': 'Game A', 'vc33': 'Game B'}
    """
    rng = random.Random(seed)
    arc_games = sorted(rng.sample(ARC_POOL, n_arc))

    if include_mbpp:
        games = ['mbpp'] + arc_games
        labels = {'mbpp': 'MBPP'}
        for i, g in enumerate(arc_games):
            labels[g] = f'Game {chr(65 + i)}'  # Game A, Game B, ...
    else:
        games = arc_games
        labels = {}
        for i, g in enumerate(arc_games):
            labels[g] = f'Game {chr(65 + i)}'

    return games, labels


def det_weights(m, n):
    """Deterministic weight initialization — orthogonal from fixed QR decomposition.

    No randomness. Same (m, n) = same initial weights every time.
    Environment breaks symmetry through observations, not random init.

    Args:
        m: Output dimension (rows).
        n: Input dimension (cols).

    Returns:
        numpy array of shape (m, n) with orthonormal rows (if m <= n) or cols (if m > n).
    """
    # Fixed deterministic input: shaped array normalized to prevent QR overflow
    k = max(m, n)
    A = np.arange(1, k * k + 1, dtype=np.float64).reshape(k, k)
    A = A / np.linalg.norm(A)
    Q, _ = np.linalg.qr(A)
    return Q[:m, :n]


def seal_mapping(results_dir, games, labels):
    """Write game mapping to sealed file in results dir.

    The sealed file is NOT to be read by Eli or Leo during the session.
    It exists for post-session audit only.
    Filename starts with '.' to signal: do not open.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, '.sealed_game_mapping.json')
    with open(path, 'w') as f:
        json.dump({'games': games, 'labels': labels}, f, indent=2)


def label_filename(label, step):
    """Return the JSONL output filename for a game label.

    Uses label, NOT game name. Enforces structural masking.

    Args:
        label: Label string, e.g. 'MBPP', 'Game A', 'Game B'.
        step: Experiment step number.

    Returns:
        Filename string, e.g. 'mbpp_1313.jsonl', 'game_a_1313.jsonl'.
    """
    safe = label.lower().replace(' ', '_')
    return f'{safe}_{step}.jsonl'


def masked_game_list(labels):
    """Return sorted label list for display in logs (no real game IDs)."""
    return sorted(labels.values(), key=lambda s: (s != 'MBPP', s))


def masked_run_log(label, elapsed_seconds):
    """Return the per-run log line for masked experiments (no-draws protocol).

    STRUCTURAL ENFORCEMENT: Only timing is shown — no stats that could reveal
    game type (action_KL, wdrift, cr, RHAE). Stats exist only in chain summary.

    Example output: "  Game A:  (5.5s)"
    """
    return f"  {label}:  ({elapsed_seconds:.1f}s)"


# ---------------------------------------------------------------------------
# Single-metric output (Jun directive, 2026-03-29) — apply to step 1320+
# ---------------------------------------------------------------------------

def compute_progress_speedup(p1, p2):
    """Compute second-exposure speedup from steps_to_first_progress values.

    Level-masked: no L1/L2/L3 labels. 'Progress' = any game state advancement.

    Args:
        p1: steps_to_first_progress in try1 (int or None if no progress).
        p2: steps_to_first_progress in try2 (int or None if no progress).

    Returns:
        float: >1 means faster on try2. 0.0 means try1 succeeded, try2 didn't.
        float('inf'): try1 failed, try2 succeeded.
        None: neither try made progress.
    """
    if p1 is not None and p2 is not None and p2 > 0:
        return round(p1 / p2, 4)
    if p1 is None and p2 is not None:
        return float('inf')
    if p1 is not None and p2 is None:
        return 0.0
    return None


def format_speedup(speedup):
    """Format speedup value for stdout display.

    Args:
        speedup: float (>1 = learned faster on try2), None (L1 not reached), or inf.

    Returns:
        Formatted string for stdout.
    """
    if speedup is None:
        return "N/A — L1 not reached"
    if speedup == float('inf'):
        return "inf (failed try1, succeeded try2)"
    return f"{speedup:.4f}"


def speedup_for_chain(speedup):
    """Convert speedup to chain-mean-ready value (Leo directive, 2026-03-29).

    Convention: speedup = 0 when progress never reached or ratio is undefined.
    Only finite values represent computable transfer. No N/A exclusions.

    None → 0.0   (neither try reached progress)
    inf  → 0.0   (try1 failed; try2 success is luck, not transfer from try1)
    0.0  → 0.0   (try1 success, try2 failed — negative transfer)
    float → float (both tries reached progress — valid measurement)
    """
    if speedup is None or speedup == float('inf'):
        return 0.0
    return float(speedup)


# ---------------------------------------------------------------------------
# RHAE(try2) — primary metric (Jun directive via Leo, 2026-03-29)
# ---------------------------------------------------------------------------

def compute_rhae_try2(try2_progress_by_label, optimal_steps_by_label):
    """Compute RHAE(try2) = mean(efficiency²) across all games.

    ARC Prize scoring IS the metric. efficiency = optimal_steps / actual_steps,
    capped at 1. efficiency = 0 when no progress reached.

    Args:
        try2_progress_by_label: dict mapping label -> steps_to_first_progress
            (int if progress reached, None if not).
        optimal_steps_by_label: dict mapping label -> optimal_steps (int).
            Pass None or missing key for unknown games → treated as 0 efficiency.

    Returns:
        float: RHAE(try2), mean of efficiency² across all games (0.0 to 1.0).
    """
    if not try2_progress_by_label:
        return 0.0
    sq = []
    for label, steps in try2_progress_by_label.items():
        if steps is None:
            sq.append(0.0)
            continue
        opt = (optimal_steps_by_label or {}).get(label)
        if opt is None or opt <= 0:
            sq.append(0.0)
            continue
        eff = min(1.0, opt / steps)
        sq.append(eff ** 2)
    return round(sum(sq) / len(sq), 6) if sq else 0.0


def write_experiment_results(results_dir, step, rhae_by_condition,
                              all_results, conditions, game_labels=None,
                              speedup_by_condition=None):
    """Write summary.json (RHAE(try2) primary) and diagnostics.json.

    summary.json: ONE number per condition — rhae_try2.
    diagnostics.json: all raw results. NOT printed. NOT referenced in kill
    decisions. Game IDs are MASKED to labels when game_labels is provided.

    Args:
        results_dir: Output directory.
        step: Experiment step number.
        rhae_by_condition: dict mapping condition -> rhae_try2 float.
        all_results: list of all per-game per-condition result dicts.
        conditions: list of condition names.
        game_labels: Optional dict mapping game_id -> label (PRISM masking).
        speedup_by_condition: Optional diagnostic dict (speedup per condition).
    """
    os.makedirs(results_dir, exist_ok=True)

    # summary.json — ONE number primary
    summary = {
        'step': step,
        'rhae_try2': {c: rhae_by_condition.get(c) for c in conditions},
        'diagnostics': {},
    }
    if speedup_by_condition is not None:
        summary['diagnostics']['speedup'] = {
            c: speedup_by_condition.get(c) for c in conditions
        }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # diagnostics.json — full results, game IDs masked to labels
    if game_labels:
        masked_results = []
        for r in all_results:
            mr = dict(r)
            if 'game' in mr and mr['game'] in game_labels:
                mr['game'] = game_labels[mr['game']]
            masked_results.append(mr)
    else:
        masked_results = all_results

    with open(os.path.join(results_dir, 'diagnostics.json'), 'w') as f:
        json.dump({'step': step, 'results': masked_results}, f, indent=2, default=str)
