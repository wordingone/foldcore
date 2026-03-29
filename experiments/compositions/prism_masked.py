"""
Masked PRISM infrastructure — permanent.

Structural enforcement: game IDs never appear in any readable output.
Only labels (MBPP, Game A, Game B, ...) appear in logs, filenames, and results.
The sealed mapping (.sealed_game_mapping.json) is written once and not read
by Eli or Leo during the session.

Usage in experiment scripts:
    from prism_masked import select_games, seal_mapping, label_filename, GAME_LABELS_DISPLAY

    GAMES, GAME_LABELS = select_games(seed=STEP)
    # GAMES is internal-only — never print, never log, never pass to Leo.
    # GAME_LABELS maps game_id -> label string (MBPP, Game A, Game B, ...).
    # All output uses GAME_LABELS[game] or just the label string directly.
"""

import random
import os
import json

ARC_POOL = [
    'ft09', 'ls20', 'vc33', 'tr87', 'sp80', 'sb26',
    'tu93', 'cn04', 'cd82', 'lp85',
]


def select_games(seed, n_arc=2, include_mbpp=True):
    """Select games for an experiment step.

    Args:
        seed: Experiment step number (used as RNG seed).
        n_arc: Number of random ARC games to include (default 2, per Jun directive).
        include_mbpp: If True, MBPP is always included and labeled 'MBPP' (not masked).

    Returns:
        games: list of game IDs — INTERNAL USE ONLY. Never print or log.
        labels: dict mapping game_id -> label string for all output.

    Example:
        GAMES, GAME_LABELS = select_games(seed=1310)
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
        Filename string, e.g. 'mbpp_1310.jsonl', 'game_a_1310.jsonl'.
    """
    safe = label.lower().replace(' ', '_')
    return f'{safe}_{step}.jsonl'


def masked_game_list(labels):
    """Return sorted label list for display in logs (no real game IDs)."""
    return sorted(labels.values(), key=lambda s: (s != 'MBPP', s))
