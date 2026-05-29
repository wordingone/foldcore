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
import sys
import time
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

    # -----------------------------------------------------------------------
    # Game / environment factory
    # -----------------------------------------------------------------------

    def _make_game(self, game):
        """Create game environment for the given game ID."""
        # Ensure environments directory is on path
        env_dir = str(Path(__file__).parent / 'environments')
        if env_dir not in sys.path:
            sys.path.insert(0, env_dir)
        steps_dir = str(Path(__file__).parent / 'steps')
        if steps_dir not in sys.path:
            sys.path.insert(0, steps_dir)

        gn = game.lower().strip()
        if gn == 'mbpp' or gn.startswith('mbpp_'):
            import mbpp_game
            return mbpp_game.make(gn)
        try:
            import arcagi3
            return arcagi3.make(game.upper())
        except ImportError:
            import util_arcagi3
            return util_arcagi3.make(game.upper())

    # -----------------------------------------------------------------------
    # Episode runner
    # -----------------------------------------------------------------------

    def _run_episode(self, substrate, env, n_actions, seed, max_steps, max_seconds):
        """Run one episode. Returns (steps_to_first_progress, elapsed_seconds)."""
        try:
            obs = env.reset(seed=seed)
        except TypeError:
            obs = env.reset()

        steps = 0
        level = 0
        steps_to_first_progress = None
        t_start = time.time()
        fresh = True

        while steps < max_steps:
            if time.time() - t_start > max_seconds:
                break
            if obs is None:
                try:
                    obs = env.reset(seed=seed)
                except TypeError:
                    obs = env.reset()
                level = 0
                fresh = True
                continue
            obs_arr = np.asarray(obs, dtype=np.float32)
            action = int(substrate.process(obs_arr)) % n_actions
            obs_next, reward, done, info = env.step(action)
            steps += 1
            if fresh:
                fresh = False
                obs = obs_next
                continue
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                if steps_to_first_progress is None:
                    steps_to_first_progress = steps
                level = cl
            if done:
                try:
                    obs = env.reset(seed=seed)
                except TypeError:
                    obs = env.reset()
                level = 0
                fresh = True
            else:
                obs = obs_next

        return steps_to_first_progress, round(time.time() - t_start, 2)

    # -----------------------------------------------------------------------
    # Full evaluation runner
    # -----------------------------------------------------------------------

    def run(self, substrate_cls, conditions, n_draws=1, try2=True):
        """Execute full PRISM evaluation.

        For each draw × condition × game:
          1. Instantiate substrate (deterministic)
          2. Try1: run max_steps actions
          3. Try2 (if enabled): load try1 weights, reset state, rerun same episode
          4. Compute RHAE(try2), collect results

        Args:
            substrate_cls: Class with interface process(obs)->int, get_weights(),
                           load_weights(w), reset(). Constructor: __init__(n_actions).
                           Optional condition kwarg: __init__(n_actions, condition=cond).
            conditions: List of condition name strings. Each condition runs the same
                        substrate_cls, instantiated with condition=cond_name.
            n_draws: Number of draws (different env seeds).
            try2: If True, run second try with try1 weights loaded.

        Returns:
            dict: {condition: rhae_try2_float}
        """
        if isinstance(conditions, str):
            conditions = [conditions]

        os.makedirs(self.results_dir, exist_ok=True)
        self.seal_mapping()

        raw_results = []

        for draw_idx in range(n_draws):
            draw_seed = self.seed * 1000 + draw_idx

            for game_idx, game in enumerate(self.games):
                env = self._make_game(game)
                try:
                    n_actions = int(env.n_actions)
                except AttributeError:
                    n_actions = 128

                label = self.label(game)
                env_seed = draw_seed * 100 + game_idx
                opt = self.get_optimal_steps(game)

                for cond in conditions:
                    try:
                        substrate = substrate_cls(n_actions, condition=cond)
                    except TypeError:
                        substrate = substrate_cls(n_actions)

                    # Try1
                    np.random.seed(env_seed)
                    p1, t1 = self._run_episode(
                        substrate, env, n_actions,
                        env_seed, self.max_steps, self.max_seconds)

                    # Try2
                    p2, t2 = None, 0.0
                    if try2:
                        weights = substrate.get_weights()
                        substrate.reset()
                        substrate.load_weights(weights)
                        np.random.seed(int(env_seed * 1000 + 1) % (2**32))
                        p2, t2 = self._run_episode(
                            substrate, env, n_actions,
                            env_seed, self.max_steps, self.max_seconds)

                    rhae_t2 = (min(1.0, opt / p2) ** 2) if (p2 is not None and opt > 0) else 0.0

                    raw_results.append({
                        'draw': draw_idx,
                        'game': game,
                        'label': label,
                        'condition': cond,
                        'steps_to_first_progress_t1': p1,
                        'steps_to_first_progress_t2': p2,
                        'rhae_t2': rhae_t2,
                        'elapsed_t1': t1,
                        'elapsed_t2': t2,
                    })

        # Aggregate RHAE per condition
        rhae_by_condition = {}
        for cond in conditions:
            vals = [r['rhae_t2'] for r in raw_results if r['condition'] == cond]
            rhae_by_condition[cond] = round(sum(vals) / len(vals), 6) if vals else 0.0

        # Build output rows (mask game IDs if needed)
        all_out = []
        for r in raw_results:
            all_out.append({
                'draw': r['draw'],
                'game': r['label'] if self.mask_game_ids else r['game'],
                'condition': r['condition'],
                'steps_to_first_progress_t1': r['steps_to_first_progress_t1'],
                'steps_to_first_progress_t2': r['steps_to_first_progress_t2'],
                'rhae_t2': r['rhae_t2'],
                'elapsed_t1': r['elapsed_t1'],
                'elapsed_t2': r['elapsed_t2'],
            })

        self.write_results(self.seed, rhae_by_condition, all_out, conditions)
        return rhae_by_condition


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import importlib.util
    import inspect

    parser = argparse.ArgumentParser(description='PRISM unified evaluation framework')
    parser.add_argument('--step', type=int, required=True,
                        help='Experiment step number (seed for game selection)')
    parser.add_argument('--substrate', type=str, required=True,
                        help='Path to substrate .py file (must define class Substrate or '
                             'a class with process/get_weights/load_weights/reset)')
    parser.add_argument('--mode', type=str, default='masked',
                        choices=['masked', 'full_10', 'full_25', 'full_pool', 'single', 'custom'],
                        help='Evaluation mode (default: masked)')
    parser.add_argument('--n_draws', type=int, default=None,
                        help='Number of draws (default: 30 for masked, 5 for others)')
    parser.add_argument('--n_arc', type=int, default=2,
                        help='Number of ARC games in masked mode (default: 2)')
    parser.add_argument('--game', type=str, default=None,
                        help='Game ID for single mode')
    parser.add_argument('--games', type=str, default=None,
                        help='Comma-separated game list for custom mode')
    parser.add_argument('--benchmarks', type=str, default=None,
                        help='Comma-separated benchmarks (default: arc,mbpp)')
    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Max steps per episode (default: 2000)')
    parser.add_argument('--max_seconds', type=int, default=300,
                        help='Max seconds per episode (default: 300)')
    parser.add_argument('--conditions', type=str, default='EXP',
                        help='Comma-separated condition names (default: EXP)')
    parser.add_argument('--no_try2', action='store_true',
                        help='Skip try2')
    parser.add_argument('--no_mask', action='store_true',
                        help="Don't mask game IDs in output")
    args = parser.parse_args()

    # Defaults
    if args.n_draws is None:
        args.n_draws = 30 if args.mode == 'masked' else 5

    # Load substrate from file
    substrate_path = os.path.abspath(args.substrate)
    spec = importlib.util.spec_from_file_location('_substrate_mod', substrate_path)
    mod = importlib.util.module_from_spec(spec)
    # Add substrate directory to path (for its own imports)
    sys.path.insert(0, str(Path(substrate_path).parent))
    spec.loader.exec_module(mod)

    substrate_cls = getattr(mod, 'Substrate', None)
    if substrate_cls is None:
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if (obj.__module__ == '_substrate_mod'
                    and hasattr(obj, 'process')
                    and hasattr(obj, 'get_weights')):
                substrate_cls = obj
                break
    if substrate_cls is None:
        raise RuntimeError(
            f'No Substrate class found in {args.substrate}. '
            'Define a class named Substrate with process(), get_weights(), '
            'load_weights(), and reset() methods.')

    conditions = [c.strip() for c in args.conditions.split(',')]
    benchmarks = [b.strip() for b in args.benchmarks.split(',')] if args.benchmarks else None
    games_list = [g.strip() for g in args.games.split(',')] if args.games else None

    p = PRISM(
        seed=args.step,
        mode=args.mode,
        benchmarks=benchmarks,
        n_arc=args.n_arc,
        game=args.game,
        games=games_list,
        max_steps=args.max_steps,
        max_seconds=args.max_seconds,
        mask_game_ids=not args.no_mask,
    )

    print(p.describe())
    results = p.run(substrate_cls, conditions, n_draws=args.n_draws, try2=not args.no_try2)
    print(f'RHAE: {results}')
    print(f'Results: {p.results_dir}')

    # ---------------------------------------------------------------------------
    # Environment factory
    # ---------------------------------------------------------------------------

    def _make_env(self, game):
        """Create environment for a game."""
        gn = game.lower().strip()
        if gn == 'mbpp' or gn.startswith('mbpp_'):
            env_dir = os.path.join(os.path.dirname(__file__), 'environments')
            if env_dir not in sys.path:
                sys.path.insert(0, env_dir)
            import mbpp_game
            return mbpp_game.make(gn)
        try:
            import arcagi3
            return arcagi3.make(gn.upper())
        except ImportError:
            import util_arcagi3
            return util_arcagi3.make(gn.upper())

    # ---------------------------------------------------------------------------
    # Episode runner
    # ---------------------------------------------------------------------------

    def _run_episode(self, env, substrate, seed, max_steps):
        """Run one try on env with substrate. Returns (steps_to_first_progress, elapsed)."""
        obs = env.reset(seed=seed)
        steps = 0
        level = 0
        steps_to_first_progress = None
        t_start = time.time()
        fresh_episode = True

        while steps < max_steps:
            if time.time() - t_start > self.max_seconds:
                break
            if obs is None:
                obs = env.reset(seed=seed)
                if hasattr(substrate, 'on_level_transition'):
                    substrate.on_level_transition(0)
                level = 0
                fresh_episode = True
                continue
            obs_arr = np.asarray(obs, dtype=np.float32)
            n_actions = getattr(env, 'n_actions', 7)
            action = substrate.process(obs_arr) % int(n_actions)
            obs_next, reward, done, info = env.step(action)
            steps += 1
            if obs_next is not None and hasattr(substrate, 'update_after_step'):
                substrate.update_after_step(obs_next, action, reward)
            if fresh_episode:
                fresh_episode = False
                obs = obs_next
                continue
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                if steps_to_first_progress is None:
                    steps_to_first_progress = steps
                level = cl
                if hasattr(substrate, 'on_level_transition'):
                    substrate.on_level_transition(cl)
            if done:
                obs = env.reset(seed=seed)
                if hasattr(substrate, 'on_level_transition'):
                    substrate.on_level_transition(0)
                level = 0
                fresh_episode = True
            else:
                obs = obs_next

        return steps_to_first_progress, round(time.time() - t_start, 2)

    # ---------------------------------------------------------------------------
    # Main runner
    # ---------------------------------------------------------------------------

    def _draw_games(self, draw_seed):
        """Select games for a single draw (used when mode == 'masked')."""
        if self.mode == 'masked':
            rng = random.Random(draw_seed)
            arc_games = sorted(rng.sample(ARC_SOLVED_10, self.n_arc))
        elif self.mode == 'full_10':
            arc_games = list(ARC_SOLVED_10)
        elif self.mode == 'full_25':
            arc_games = list(ARC_KNOWN_25)
        else:
            # single / custom / full_pool — use pre-selected games (no per-draw variation)
            return self.games, self.labels

        game_list = arc_games
        if 'mbpp' in self.benchmarks:
            game_list = ['mbpp'] + game_list

        labels = {}
        arc_idx = 0
        for g in game_list:
            if g == 'mbpp':
                labels[g] = 'MBPP'
            elif self.mask_game_ids:
                labels[g] = f'Game {chr(65 + arc_idx)}'
                arc_idx += 1
            else:
                labels[g] = g

        return game_list, labels

    def run(self, substrates, conditions=None, n_draws=1, try2=True):
        """Execute full PRISM evaluation.

        Args:
            substrates: Single class, or dict {condition_name: class}.
                        Class interface: __init__(n_actions), process(obs)->int,
                        get_weights()->dict, load_weights(dict), reset().
            conditions: List of condition names. Required if substrates is a class.
            n_draws:    Number of independent draws.
            try2:       Whether to run try2 after try1.

        Returns:
            dict: {condition: chain_mean_rhae} summary.
        """
        if isinstance(substrates, dict):
            substrate_map = substrates
            conditions = conditions or list(substrates.keys())
        else:
            assert conditions is not None, "conditions required when substrates is a class"
            substrate_map = {c: substrates for c in conditions}

        os.makedirs(self.results_dir, exist_ok=True)
        all_results = []
        rhae_by_condition = {c: [] for c in conditions}

        for draw_idx in range(n_draws):
            draw_seed = self.seed * 100 + draw_idx
            games, game_labels = self._draw_games(draw_seed)

            draw_dir = os.path.join(self.results_dir, f'draw{draw_idx}')
            os.makedirs(draw_dir, exist_ok=True)
            # Write sealed mapping for this draw
            mapping_path = os.path.join(draw_dir, '.sealed_game_mapping.json')
            with open(mapping_path, 'w') as f:
                json.dump({
                    'seed': self.seed, 'draw_seed': draw_seed, 'draw_idx': draw_idx,
                    'mode': self.mode, 'benchmarks': sorted(self.benchmarks),
                    'games': games, 'labels': game_labels,
                    'max_steps': self.max_steps, 'max_seconds': self.max_seconds,
                }, f, indent=2)

            for cond in conditions:
                substrate_cls = substrate_map[cond]
                cond_dir = os.path.join(self.results_dir, cond, f'draw{draw_idx}')
                os.makedirs(cond_dir, exist_ok=True)

                for game in games:
                    env = self._make_env(game)
                    n_actions = int(getattr(env, 'n_actions', 7))

                    substrate = substrate_cls(n_actions=n_actions)

                    # Try1
                    p1, t1 = self._run_episode(env, substrate, seed=0, max_steps=self.max_steps)

                    if try2:
                        if hasattr(substrate, 'get_weights'):
                            weights = substrate.get_weights()
                            substrate.reset()
                            substrate.load_weights(weights)
                        else:
                            substrate.reset()
                        np.random.seed(int(draw_seed * 1000 + 1) % (2**32))  # PRNG fix: isolated try2 RNG; % 2^32 prevents uint32 overflow
                        p2, t2 = self._run_episode(env, substrate, seed=4,
                                                    max_steps=self.max_steps)
                    else:
                        p2, t2 = None, 0.0

                    opt = self.get_optimal_steps(game)
                    eff_sq = 0.0
                    if p2 is not None and opt > 0:
                        eff_sq = round(min(1.0, opt / p2) ** 2, 6)
                    speedup = compute_speedup(p1, p2)

                    rhae_by_condition[cond].append(eff_sq)
                    label = game_labels.get(game, game)
                    result = {
                        'draw': draw_idx, 'condition': cond, 'game': label,
                        'steps_to_progress_t1': p1, 'steps_to_progress_t2': p2,
                        'rhae_t2': eff_sq, 'speedup': speedup,
                        'runtime_t1': t1, 'runtime_t2': t2,
                    }
                    all_results.append(result)

                    # Per-game JSONL
                    safe = label.lower().replace(' ', '_')
                    out_fn = os.path.join(cond_dir, f'{safe}_{self.seed}.jsonl')
                    with open(out_fn, 'a') as f:
                        f.write(json.dumps(result) + '\n')

        # Aggregate and write summary
        rhae_summary = {
            c: round(float(np.mean(v)), 6) if v else 0.0
            for c, v in rhae_by_condition.items()
        }
        nz_by_condition = {c: sum(1 for x in v if x > 0) for c, v in rhae_by_condition.items()}
        self.write_results(self.seed, rhae_summary, all_results, conditions,
                           speedup_by_condition=nz_by_condition)
        return rhae_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import importlib.util

    parser = argparse.ArgumentParser(description='PRISM evaluation runner')
    parser.add_argument('--step',       type=int,   required=True,  help='Step/experiment number')
    parser.add_argument('--substrate',              required=True,  help='Path to substrate .py file')
    parser.add_argument('--mode',       default='masked',           help='Evaluation mode')
    parser.add_argument('--n_draws',    type=int,   default=None,   help='Number of draws')
    parser.add_argument('--n_arc',      type=int,   default=2,      help='ARC games per draw (masked)')
    parser.add_argument('--game',       default=None,               help='Game ID for single mode')
    parser.add_argument('--games',      default=None,               help='Comma-separated games (custom)')
    parser.add_argument('--benchmarks', default='arc,mbpp',         help='Benchmarks to include')
    parser.add_argument('--max_steps',  type=int,   default=2000,   help='Max steps per episode')
    parser.add_argument('--max_seconds',type=int,   default=300,    help='Max seconds per episode')
    parser.add_argument('--conditions', default=None,               help='Comma-separated conditions')
    parser.add_argument('--no_try2',    action='store_true',        help='Skip try2')
    parser.add_argument('--no_mask',    action='store_true',        help='Show real game IDs')
    args = parser.parse_args()

    # Load substrate file via importlib
    sub_path = os.path.abspath(args.substrate)
    spec = importlib.util.spec_from_file_location('_substrate_module', sub_path)
    module = importlib.util.module_from_spec(spec)
    # Add substrate's directory to path so its imports resolve
    sub_dir = os.path.dirname(sub_path)
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    spec.loader.exec_module(module)

    # Resolve substrate classes
    if hasattr(module, 'SUBSTRATES') and isinstance(module.SUBSTRATES, dict):
        substrate_map = module.SUBSTRATES
        conditions = args.conditions.split(',') if args.conditions else list(substrate_map.keys())
        substrate_map = {c: substrate_map[c] for c in conditions if c in substrate_map}
    elif hasattr(module, 'SUBSTRATE'):
        conditions = args.conditions.split(',') if args.conditions else ['default']
        substrate_map = {c: module.SUBSTRATE for c in conditions}
    else:
        raise SystemExit(
            f"ERROR: {args.substrate} must export SUBSTRATES (dict) or SUBSTRATE (class)"
        )

    n_draws = args.n_draws or (30 if args.mode == 'masked' else 5)
    benchmarks = set(args.benchmarks.split(',')) if args.benchmarks else None
    games_list = args.games.split(',') if args.games else None

    p = PRISM(
        seed=args.step,
        mode=args.mode,
        benchmarks=benchmarks,
        n_arc=args.n_arc,
        game=args.game,
        games=games_list,
        max_steps=args.max_steps,
        max_seconds=args.max_seconds,
        mask_game_ids=not args.no_mask,
    )

    print(p.describe())
    result = p.run(substrate_map, conditions=conditions, n_draws=n_draws, try2=not args.no_try2)
    print(f"\nResults (chain_mean RHAE):")
    for cond, rhae in result.items():
        print(f"  {cond}: {rhae:.6e}")
