"""
PRISM — Unified evaluation framework for the-search.

Single entry point for all experiment modes. Deterministic given mode + seed.

Modes:
    masked      — MBPP + N random masked ARC games (default: N=2). Current standard.
    full_10     — All 10 solved ARC games. No masking.
    full_25     — All 25 known ARC games. No masking.
    full_pool   — All available ARC games (150+). No masking.
    single      — One named game. For diagnostics.
    custom      — Caller provides explicit game list.

Benchmarks (combinable with any mode):
    arc         — ARC-AGI-3 interactive games (default: on)
    mbpp        — Text/code generation, 128 ASCII actions (default: on in masked mode)
    cifar       — CIFAR-100 classification (legacy, Steps 506-1006)
    pmnist      — Permuted MNIST (legacy, Steps 1-416)
    atari       — Atari 100K, 26 games (legacy, Steps 722-766)
    split_cifar — Split-CIFAR-100 transfer test (legacy, Steps 506-1006)

Protocol:
    Each experiment = (substrate, mode, benchmarks, n_draws, max_steps, max_seconds, seed).
    Deterministic: same config = same game selection, same weight init, same results.
    PRISM enforces: masking, RHAE computation, result writing, sealed mappings.

Usage:
    from prism import PRISM

    # Current standard (masked, MBPP + 2 ARC)
    p = PRISM(seed=1392, mode='masked')
    p.run(MySubstrate, conditions=['EXP', 'CTRL'], n_draws=30)

    # Full 10-game evaluation
    p = PRISM(seed=1392, mode='full_10', benchmarks=['arc'])
    p.run(MySubstrate, conditions=['EXP'], n_draws=5)

    # Reproduce old chain benchmark (Steps 778-1006)
    p = PRISM(seed=994, mode='custom', games=['cifar', 'ls20', 'ft09', 'vc33', 'cifar'],
              benchmarks=['arc', 'cifar'], max_steps=10000)
    p.run(OldSubstrate, conditions=['REF'])

    # Single game diagnostic
    p = PRISM(seed=1379, mode='single', game='ft09', benchmarks=['arc'])
    p.run(SSMSubstrate, conditions=['FULL', 'MASKED'], n_draws=3)
"""

import random
import os
import json
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Game pools
# ---------------------------------------------------------------------------

ARC_SOLVED_10 = [
    'ft09', 'ls20', 'vc33', 'tr87', 'sp80',
    'sb26', 'tu93', 'cn04', 'cd82', 'lp85',
]

ARC_KNOWN_25 = ARC_SOLVED_10 + [
    're86', 'r11l', 's5i5', 'm0r0', 'su15',
    'ar25', 'dc22', 'sc25', 'g50t', 'wa30',
    'bp35', 'lf52', 'ka59', 'sk48', 'tn36',
]

# Optimal action counts from prescriptions (per-game, all levels).
# Used for RHAE computation. None = unknown (RHAE uses proxy).
OPTIMAL_STEPS = {
    'ft09': 75, 'ls20': 311, 'vc33': 176, 'lp85': 79,
    'tr87': 123, 'sb26': 124, 'sp80': 107, 'cd82': 140,
    'cn04': 107, 'tu93': 185,
    # Partial solves — L1 only
    're86': 210, 'r11l': 65, 's5i5': 39, 'm0r0': 15,
    'su15': 25, 'ar25': 26, 'dc22': 20, 'sc25': 17,
    'g50t': 17, 'wa30': 77, 'bp35': 29, 'lf52': 8,
}

ARC_OPTIMAL_STEPS_PROXY = 10  # For games without known optimal

# ---------------------------------------------------------------------------
# Deterministic utilities
# ---------------------------------------------------------------------------

def det_weights(m, n):
    """Deterministic weight initialization — orthogonal from fixed QR.
    No randomness. Same (m, n) = same weights every time.
    """
    k = max(m, n)
    A = np.arange(1, k * k + 1, dtype=np.float64).reshape(k, k)
    A = A / np.linalg.norm(A)
    Q, _ = np.linalg.qr(A)
    return Q[:m, :n]


# ---------------------------------------------------------------------------
# RHAE computation
# ---------------------------------------------------------------------------

def compute_rhae(progress_by_game, optimal_by_game=None):
    """RHAE = mean(efficiency²) across all games.

    Args:
        progress_by_game: dict {game_label: steps_to_first_progress or None}
        optimal_by_game: dict {game_label: optimal_steps or None}

    Returns:
        float: RHAE value (0.0 to 1.0)
    """
    if not progress_by_game:
        return 0.0
    optimal_by_game = optimal_by_game or {}
    sq = []
    for label, steps in progress_by_game.items():
        if steps is None:
            sq.append(0.0)
            continue
        opt = optimal_by_game.get(label)
        if opt is None or opt <= 0:
            opt = ARC_OPTIMAL_STEPS_PROXY
        eff = min(1.0, opt / steps)
        sq.append(eff ** 2)
    return round(sum(sq) / len(sq), 6) if sq else 0.0


def compute_speedup(p1, p2):
    """Second-exposure speedup from steps_to_first_progress values.
    >1 = faster on try2. 0 = try1 ok, try2 failed. None = both failed.
    """
    if p1 is not None and p2 is not None and p2 > 0:
        return round(p1 / p2, 4)
    if p1 is None and p2 is not None:
        return float('inf')
    if p1 is not None and p2 is None:
        return 0.0
    return None


# ---------------------------------------------------------------------------
# PRISM class
# ---------------------------------------------------------------------------

class PRISM:
    """Unified evaluation framework.

    Deterministic: same (seed, mode, benchmarks) = same game selection every time.
    """

    MODES = {'masked', 'full_10', 'full_25', 'full_pool', 'single', 'custom'}
    BENCHMARKS = {'arc', 'mbpp', 'cifar', 'pmnist', 'atari', 'split_cifar'}

    def __init__(self, seed, mode='masked', benchmarks=None,
                 n_arc=2, game=None, games=None,
                 max_steps=2000, max_seconds=300,
                 mask_game_ids=True, mask_levels=True,
                 results_dir=None):
        """
        Args:
            seed: Experiment step number. Determines game selection for masked mode.
            mode: One of MODES.
            benchmarks: Set of benchmark names to include. Default: {'arc', 'mbpp'} for masked,
                        {'arc'} for full modes.
            n_arc: Number of random ARC games for masked mode (default 2).
            game: Game ID for single mode.
            games: Explicit game list for custom mode.
            max_steps: Max steps per episode (default 2000).
            max_seconds: Max seconds per episode (default 300).
            mask_game_ids: If True, output uses labels (Game A, Game B) not real IDs.
            mask_levels: If True, output uses progress events, not level numbers.
            results_dir: Override results output directory.
        """
        assert mode in self.MODES, f"Unknown mode: {mode}. Choose from {self.MODES}"

        self.seed = seed
        self.mode = mode
        self.n_arc = n_arc
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.mask_game_ids = mask_game_ids
        self.mask_levels = mask_levels

        # Default benchmarks
        if benchmarks is None:
            benchmarks = {'arc', 'mbpp'} if mode == 'masked' else {'arc'}
        self.benchmarks = set(benchmarks) & self.BENCHMARKS

        # Select games
        self.games, self.labels = self._select_games(game, games)

        # Results directory
        if results_dir is None:
            results_dir = os.path.join('experiments', 'results', f'results_{seed}')
        self.results_dir = results_dir

    def _select_games(self, game, games):
        """Select games based on mode. Returns (game_list, label_dict)."""
        if self.mode == 'single':
            assert game is not None, "single mode requires game= argument"
            game_list = [game]
        elif self.mode == 'custom':
            assert games is not None, "custom mode requires games= argument"
            game_list = list(games)
        elif self.mode == 'masked':
            rng = random.Random(self.seed)
            game_list = sorted(rng.sample(ARC_SOLVED_10, self.n_arc))
        elif self.mode == 'full_10':
            game_list = list(ARC_SOLVED_10)
        elif self.mode == 'full_25':
            game_list = list(ARC_KNOWN_25)
        elif self.mode == 'full_pool':
            # Full pool requires runtime discovery from arc_agi API
            game_list = list(ARC_KNOWN_25)  # fallback to known 25

        # Add benchmarks
        if 'mbpp' in self.benchmarks:
            game_list = ['mbpp'] + [g for g in game_list if g != 'mbpp']
        if 'cifar' in self.benchmarks:
            game_list = ['cifar'] + [g for g in game_list if g != 'cifar']
        if 'pmnist' in self.benchmarks:
            game_list = ['pmnist'] + [g for g in game_list if g != 'pmnist']

        # Build labels
        labels = {}
        arc_idx = 0
        for g in game_list:
            if g in ('mbpp', 'cifar', 'pmnist'):
                labels[g] = g.upper()
            elif self.mask_game_ids:
                labels[g] = f'Game {chr(65 + arc_idx)}'
                arc_idx += 1
            else:
                labels[g] = g

        return game_list, labels

    def get_optimal_steps(self, game):
        """Return optimal steps for a game (for RHAE computation)."""
        return OPTIMAL_STEPS.get(game, ARC_OPTIMAL_STEPS_PROXY)

    def seal_mapping(self, draw=None):
        """Write sealed game mapping for audit trail."""
        d = self.results_dir
        if draw is not None:
            d = os.path.join(d, f'draw{draw}')
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, '.sealed_game_mapping.json')
        with open(path, 'w') as f:
            json.dump({
                'seed': self.seed,
                'mode': self.mode,
                'benchmarks': sorted(self.benchmarks),
                'games': self.games,
                'labels': self.labels,
                'max_steps': self.max_steps,
                'max_seconds': self.max_seconds,
            }, f, indent=2)

    def label(self, game):
        """Get display label for a game (masked or real ID)."""
        return self.labels.get(game, game)

    def label_filename(self, game):
        """Get output filename for a game's results."""
        safe = self.label(game).lower().replace(' ', '_')
        return f'{safe}_{self.seed}.jsonl'

    def write_results(self, step, rhae_by_condition, all_results, conditions,
                      speedup_by_condition=None):
        """Write summary.json and diagnostics.json."""
        os.makedirs(self.results_dir, exist_ok=True)

        summary = {
            'step': step,
            'mode': self.mode,
            'benchmarks': sorted(self.benchmarks),
            'n_games': len(self.games),
            'rhae_try2': {c: rhae_by_condition.get(c) for c in conditions},
        }
        if speedup_by_condition:
            summary['diagnostics'] = {'speedup': speedup_by_condition}

        with open(os.path.join(self.results_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        # Mask game IDs in diagnostics
        masked = []
        for r in all_results:
            row = dict(r)
            if self.mask_game_ids and 'game' in row and row['game'] in self.labels:
                row['game'] = self.labels[row['game']]
            masked.append(row)

        with open(os.path.join(self.results_dir, 'diagnostics.json'), 'w') as f:
            json.dump({'step': step, 'results': masked}, f, indent=2, default=str)

    def describe(self):
        """Return human-readable description of this PRISM config."""
        game_desc = ', '.join(self.label(g) for g in self.games)
        return (f"PRISM(mode={self.mode}, seed={self.seed}, "
                f"games=[{game_desc}], "
                f"benchmarks={sorted(self.benchmarks)}, "
                f"max_steps={self.max_steps})")
