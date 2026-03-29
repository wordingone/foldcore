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

Usage in experiment scripts:
    from prism_masked import select_games, seal_mapping, label_filename, det_weights

    GAMES, GAME_LABELS = select_games(seed=STEP)
    # GAMES is internal-only — never print, never log, never pass to Leo.
    # GAME_LABELS maps game_id -> label string (MBPP, Game A, Game B, ...).
    # All output uses GAME_LABELS[game] or just the label string directly.

    # Deterministic weight init (no np.random.seed — same shape = same weights):
    W1 = det_weights(128, 256)  # orthogonal init from fixed QR
"""

import random
import os
import json
import numpy as np

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


def write_experiment_results(results_dir, step, speedup_by_condition,
                              all_results, conditions):
    """Write summary.json (speedup only) and diagnostics.json (everything else).

    summary.json: ONE metric per condition — second_exposure_speedup.
    diagnostics.json: all raw results for post-hoc analysis. NOT printed. NOT
    referenced in kill decisions.

    Args:
        results_dir: Output directory.
        step: Experiment step number.
        speedup_by_condition: dict mapping condition -> speedup value (float or None).
        all_results: list of all per-game per-condition result dicts.
        conditions: list of condition names.
    """
    os.makedirs(results_dir, exist_ok=True)

    # summary.json — ONE metric only
    summary = {
        'step': step,
        'speedup': {c: speedup_by_condition.get(c) for c in conditions},
    }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # diagnostics.json — full results, NOT for kill decisions
    with open(os.path.join(results_dir, 'diagnostics.json'), 'w') as f:
        json.dump({'step': step, 'results': all_results}, f, indent=2, default=str)
