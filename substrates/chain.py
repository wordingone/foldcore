"""
chain.py — PRISM: Progressive Recursive Intelligence Sequence Metric.

The benchmark for self-modifying substrates. One system, one config, no reward,
sequential diverse tasks. The parts of the chain are one problem seen from
different angles — classification, navigation, sequential puzzles, transfer.

Sequences a substrate through multiple environments in order.
Handles LS20/FT09/VC33 (ARC games), CIFAR-100, and any gymnasium env.

Budget: n_steps is PRIMARY. elapsed is an efficiency metric, not a cap.
Safety timeout = 10× expected time (prevents runaway, not a budget).
"""
import time
import numpy as np
import sys
import os

# Known action counts — used as fallback when env.action_space unavailable
# LS20: 4 directions. FT09/VC33: 68 (4 dirs + 64 grid clicks).
GAME_N_ACTIONS = {
    "LS20": 4,
    "FT09": 68,
    "VC33": 68,
}

DEFAULT_STEPS = 10_000
SAFETY_MULTIPLIER = 10   # safety timeout = 10× expected step time


# ─── Substrate compatibility shim ───
# Old substrates (pre-2026-03-24) use set_game(n_actions) + _n_actions (private).
# ChainRunner calls reset(seed) and substrate.n_actions (public).
# These helpers bridge the gap without modifying existing substrates.

def _substrate_reset(sub, seed: int):
    """Call sub.reset(seed) if available. Old substrates use set_game() as reset."""
    if hasattr(sub, 'reset'):
        sub.reset(seed)
    # else: set_game() was already called by the wrapper — functions as reset.


def _substrate_n_actions(sub, fallback: int) -> int:
    """Return sub.n_actions, or sub._n_actions (private), or fallback."""
    return int(getattr(sub, 'n_actions', getattr(sub, '_n_actions', fallback)))


def _make_arc_env(game_name: str):
    """Return factory for ARC game environment."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

    def factory():
        try:
            import arcagi3
            return arcagi3.make(game_name)
        except ImportError:
            import util_arcagi3
            return util_arcagi3.make(game_name)

    return factory


class ArcGameWrapper:
    """Wraps an ARC game env into a uniform interface for the chain runner.

    Budget: n_steps (primary). safety_timeout is a watchdog, not a budget.
    """

    def __init__(self, game_name: str, n_steps: int = DEFAULT_STEPS,
                 safety_timeout: float = 300.0,
                 # Legacy compat: per_seed_time maps to safety_timeout
                 per_seed_time: float = None):
        self.game_name = game_name
        self.n_steps = n_steps
        self.safety_timeout = per_seed_time if per_seed_time is not None else safety_timeout
        self._factory = _make_arc_env(game_name)
        self._env = None

    def run_seed(self, substrate, seed: int) -> dict:
        """Run one seed. Returns structured result dict."""
        if self._env is None:
            self._env = self._factory()

        # Detect n_actions from env or fallback to known table
        try:
            n_actions = self._env.action_space.n
        except AttributeError:
            n_actions = GAME_N_ACTIONS.get(self.game_name, 4)
        if hasattr(substrate, 'set_game'):
            substrate.set_game(n_actions)

        _substrate_reset(substrate, seed)
        obs = self._env.reset(seed=seed)
        level = 0
        l1_step = l2_step = None
        steps = 0
        t_start = time.time()
        fresh_episode = True  # skip first step for fresh_episode bug

        while steps < self.n_steps:
            if (time.time() - t_start) > self.safety_timeout:
                break  # safety timeout only — not primary budget
            if obs is None:
                obs = self._env.reset(seed=seed)
                substrate.on_level_transition()
                fresh_episode = True
                continue

            action = substrate.process(np.array(obs, dtype=np.float32))
            obs, reward, done, info = self._env.step(action % _substrate_n_actions(substrate, n_actions))
            steps += 1

            if fresh_episode:
                fresh_episode = False
                continue

            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                if cl == 1 and l1_step is None:
                    l1_step = steps
                if cl == 2 and l2_step is None:
                    l2_step = steps
                level = cl
                substrate.on_level_transition()

            if done:
                obs = self._env.reset(seed=seed)
                substrate.on_level_transition()
                fresh_episode = True

        elapsed = time.time() - t_start
        return {
            "game": self.game_name,
            "seed": seed,
            "steps": steps,
            "elapsed": round(elapsed, 2),
            "l1": l1_step,
            "l2": l2_step,
            "level_reached": level,
        }


class GymWrapper:
    """Wraps ANY gymnasium-compatible environment for the chain.

    Handles Atari, CartPole, ProcGen, or any gym env.
    No modality-specific code — raw obs passed directly to substrate.

    Budget: n_steps (primary). safety_timeout is a watchdog.
    """

    def __init__(self, env_id: str, n_steps: int = DEFAULT_STEPS,
                 safety_timeout: float = 300.0,
                 obs_processor=None,
                 frameskip: int = 1,
                 # Legacy: per_seed_time maps to safety_timeout
                 per_seed_time: float = None):
        self.env_id = env_id
        self.n_steps = n_steps
        self.safety_timeout = per_seed_time if per_seed_time is not None else safety_timeout
        self.obs_processor = obs_processor or (lambda x: np.array(x, dtype=np.float32) / 255.0
                                                if np.array(x).max() > 1.0
                                                else np.array(x, dtype=np.float32))
        self.frameskip = frameskip

    def run_seed(self, substrate, seed: int) -> dict:
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)

        env = gym.make(self.env_id, render_mode=None,
                       frameskip=self.frameskip if self.frameskip > 1 else None)
        n_actions = env.action_space.n if hasattr(env.action_space, 'n') else _substrate_n_actions(substrate, 4)

        _substrate_reset(substrate, seed)
        obs, info = env.reset(seed=seed)
        steps = 0
        total_reward = 0.0
        unique_states = set()
        t_start = time.time()

        while steps < self.n_steps:
            if (time.time() - t_start) > self.safety_timeout:
                break  # safety watchdog
            obs_proc = self.obs_processor(obs)
            action = substrate.process(obs_proc) % n_actions
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward   # external measurement only — NOT passed to substrate (R1 mode)
            steps += 1

            if steps % 100 == 0:
                h = hash(np.array(obs, dtype=np.float32).tobytes()[:64])
                unique_states.add(h)

            if terminated or truncated:
                obs, info = env.reset(seed=seed)
                substrate.on_level_transition()

        env.close()
        elapsed = time.time() - t_start
        return {
            "game": self.env_id,
            "seed": seed,
            "steps": steps,
            "elapsed": round(elapsed, 2),
            "total_reward": total_reward,
            "unique_states": len(unique_states),
            "l1": None,   # game-specific — caller sets threshold
            "l2": None,
        }


class CIFARWrapper:
    """CIFAR-100 classification task as a chain environment.

    Budget: n_steps (primary). safety_timeout is a watchdog.
    """

    def __init__(self, n_steps: int = DEFAULT_STEPS,
                 safety_timeout: float = 300.0,
                 split: str = "test",
                 # Legacy compat
                 per_seed_time: float = None):
        self.n_steps = n_steps
        self.safety_timeout = per_seed_time if per_seed_time is not None else safety_timeout
        self.split = split
        self._data = None

    def _load(self):
        if self._data is not None:
            return True
        try:
            import torchvision
            import torchvision.transforms as transforms
            ds = torchvision.datasets.CIFAR100(
                root=os.path.join(os.path.dirname(__file__), '..', 'data'),
                train=(self.split == "train"),
                download=True,
                transform=transforms.ToTensor()
            )
            images = np.array([np.array(ds[i][0]).transpose(1, 2, 0) * 255
                                for i in range(len(ds))], dtype=np.uint8)
            labels = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
            self._data = (images, labels)
            return True
        except Exception:
            return False

    def run_seed(self, substrate, seed: int) -> dict:
        if not self._load():
            return {
                "game": "CIFAR-100",
                "seed": seed,
                "steps": 0,
                "elapsed": 0.0,
                "l1": None,
                "l2": None,
                "accuracy": None,
                "error": "CIFAR-100 not available",
            }

        images, labels = self._data
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(images))[:self.n_steps]

        _substrate_reset(substrate, seed)
        correct = 0
        window = []
        l1_step = None
        steps = 0
        t_start = time.time()

        for i, img_idx in enumerate(idx):
            if (time.time() - t_start) > self.safety_timeout:
                break  # safety watchdog only
            obs = images[img_idx].astype(np.float32) / 255.0
            label = int(labels[img_idx])
            action = substrate.process(obs)
            hit = int(action == label)
            correct += hit
            window.append(hit)
            if len(window) > 100:
                window.pop(0)
            steps += 1
            if l1_step is None and len(window) >= 100:
                if sum(window) / len(window) > 0.10:
                    l1_step = steps

        elapsed = time.time() - t_start
        accuracy = correct / max(steps, 1)
        return {
            "game": "CIFAR-100",
            "seed": seed,
            "steps": steps,
            "elapsed": round(elapsed, 2),
            "l1": l1_step,
            "l2": None,
            "accuracy": round(accuracy, 4),
        }


class SplitCIFAR100Wrapper:
    """Split-CIFAR-100: the standard continual learning benchmark.

    20 sequential tasks, 5 classes per task (classes 0-4, 5-9, ..., 95-99).
    Standard CL metrics: avg accuracy + backward transfer.

    Budget: n_images_per_task × N_TASKS = total steps (primary).
    safety_timeout is a watchdog for runaway substrates.

    R1 mode: no reward or label signal to substrate.
    """

    N_TASKS = 20
    CLASSES_PER_TASK = 5

    def __init__(self, n_images_per_task: int = 500,
                 safety_timeout: float = 300.0,
                 split: str = "test",
                 # Legacy compat
                 per_seed_time: float = None):
        self.n_images_per_task = n_images_per_task
        self.safety_timeout = per_seed_time if per_seed_time is not None else safety_timeout
        self.split = split
        self._data = None

    def _load(self):
        if self._data is not None:
            return True
        try:
            import torchvision
            import torchvision.transforms as transforms
            ds = torchvision.datasets.CIFAR100(
                root=os.path.join(os.path.dirname(__file__), '..', 'data'),
                train=(self.split == "train"),
                download=True,
                transform=transforms.ToTensor()
            )
            images = np.array([np.array(ds[i][0]).transpose(1, 2, 0) * 255
                                for i in range(len(ds))], dtype=np.uint8)
            labels = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
            tasks = []
            for t in range(self.N_TASKS):
                class_start = t * self.CLASSES_PER_TASK
                class_end = class_start + self.CLASSES_PER_TASK
                mask = (labels >= class_start) & (labels < class_end)
                task_images = images[mask]
                task_labels = labels[mask] - class_start
                tasks.append((task_images, task_labels))
            self._data = tasks
            return True
        except Exception:
            return False

    def run_seed(self, substrate, seed: int) -> dict:
        """Run full 20-task sequence. Returns per-task accuracy + backward transfer."""
        if not self._load():
            return {
                "game": "Split-CIFAR-100",
                "seed": seed,
                "error": "CIFAR-100 not available",
                "task_accuracies": [],
                "avg_accuracy": None,
                "backward_transfer": None,
            }

        rng = np.random.RandomState(seed)
        task_accuracies = []
        task0_acc_after = None
        t_start = time.time()

        _substrate_reset(substrate, seed)

        for task_id in range(self.N_TASKS):
            if (time.time() - t_start) > self.safety_timeout:
                break  # safety watchdog only

            task_images, task_labels = self._data[task_id]
            idx = rng.choice(len(task_images),
                             min(self.n_images_per_task, len(task_images)),
                             replace=False)
            correct = 0
            for i in idx:
                obs = task_images[i].astype(np.float32) / 255.0
                action = substrate.process(obs) % self.CLASSES_PER_TASK
                correct += int(action == task_labels[i])

            acc = correct / len(idx)
            task_accuracies.append(round(acc, 4))
            substrate.on_level_transition()

        # Backward transfer: re-evaluate task 0 after all tasks
        if len(task_accuracies) >= 2:
            task0_images, task0_labels = self._data[0]
            idx0 = rng.choice(len(task0_images),
                              min(self.n_images_per_task, len(task0_images)),
                              replace=False)
            correct0 = 0
            for i in idx0:
                obs = task0_images[i].astype(np.float32) / 255.0
                action = substrate.process(obs) % self.CLASSES_PER_TASK
                correct0 += int(action == task0_labels[i])
            task0_acc_after = correct0 / len(idx0)

        avg_acc = float(np.mean(task_accuracies)) if task_accuracies else None
        bwt = (task0_acc_after - task_accuracies[0]) if task0_acc_after is not None else None

        elapsed = time.time() - t_start
        return {
            "game": "Split-CIFAR-100",
            "seed": seed,
            "steps": len(task_accuracies),
            "elapsed": round(elapsed, 2),
            "l1": 1 if avg_acc and avg_acc > 0.05 else None,
            "l2": None,
            "task_accuracies": task_accuracies,
            "avg_accuracy": round(avg_acc, 4) if avg_acc else None,
            "backward_transfer": round(bwt, 4) if bwt is not None else None,
            "tasks_completed": len(task_accuracies),
        }


class AtariWrapper(GymWrapper):
    """Atari wrapper — thin subclass of GymWrapper.

    R1 mode: substrate never sees reward. total_reward is external measurement.
    Default: Montezuma's Revenge (hardest exploration game).
    """

    def __init__(self, game: str = "ALE/MontezumaRevenge-v5",
                 n_steps: int = 100_000,
                 safety_timeout: float = 300.0,
                 frame_skip: int = 4,
                 per_seed_time: float = None):
        super().__init__(
            env_id=game,
            n_steps=n_steps,
            safety_timeout=per_seed_time if per_seed_time is not None else safety_timeout,
            frameskip=frame_skip,
        )
        self.game = game

    def run_seed(self, substrate, seed: int) -> dict:
        result = super().run_seed(substrate, seed)
        # L1: total_reward > 0 (got any reward in R1 mode without reward signal)
        result["l1"] = 1 if result.get("total_reward", 0) > 0 else None
        return result


class ChainRunner:
    """Runs a substrate through a sequence of (name, wrapper) tasks.

    Budget is set per-wrapper (n_steps). ChainRunner enforces minimum n_seeds.
    """

    MIN_SEEDS = 10  # Fix 6: minimum seeds for any chain result

    def __init__(self, chain: list, n_seeds: int = 10,
                 verbose: bool = True,
                 randomize_order: bool = True,
                 # Legacy compat
                 per_seed_time: float = None):
        """
        chain: list of (name, wrapper_instance)
        n_seeds: seeds per game (minimum 10 per Fix 6)
        randomize_order: MUST be True (Jun directive 2026-03-24).
            Game order randomized per seed. Neither Leo nor Eli knows the order.
            CIFAR included in shuffle. Prevents per-game optimization.
        """
        if not randomize_order:
            print("[ChainRunner] WARNING: randomize_order=False is DEPRECATED. "
                  "Jun directive 2026-03-24: game order MUST be randomized. "
                  "Forcing randomize_order=True.")
            randomize_order = True
        self.chain = chain
        self.n_seeds = max(n_seeds, self.MIN_SEEDS)
        self.verbose = verbose
        self.randomize_order = randomize_order
        if n_seeds < self.MIN_SEEDS:
            print(f"[ChainRunner] n_seeds={n_seeds} < minimum {self.MIN_SEEDS}. "
                  f"Using n_seeds={self.MIN_SEEDS}.")

    def run(self, substrate_cls: type, substrate_kwargs: dict = None) -> dict:
        """Run full chain. ONE substrate per seed, persists across all games.

        Outer loop = seeds. Inner loop = games. The substrate accumulates
        state across the entire chain trajectory. This IS the chain test:
        does experience on game A help or hurt game B?
        """
        if substrate_kwargs is None:
            substrate_kwargs = {}

        # Collect per-game results across all seeds
        results = {name: [] for name, _ in self.chain}

        for seed in range(self.n_seeds):
            sub = substrate_cls(**substrate_kwargs)  # ONE substrate per seed
            if self.verbose:
                print(f"\n=== Seed {seed} ===")

            # Randomize phase order per seed (Jun 2026-03-24: ALWAYS)
            seed_chain = list(self.chain)
            if self.randomize_order:
                rng = np.random.RandomState(seed + 10000)
                rng.shuffle(seed_chain)
                # NOTE: game order is HIDDEN during execution.
                # Neither Leo nor Eli should know which game is running.
                # Order revealed only in final summary after all seeds complete.

            for phase_idx, (name, wrapper) in enumerate(seed_chain):
                r = wrapper.run_seed(sub, seed)  # SAME substrate persists
                results[name].append(r)
                if self.verbose:
                    l1 = r.get('l1')
                    acc = r.get('accuracy') or r.get('avg_accuracy')
                    reward = r.get('total_reward')
                    if acc is not None:
                        metric = f"acc={acc:.3f}"
                    elif reward is not None:
                        metric = f"reward={reward:.0f}"
                    else:
                        metric = f"l1={l1}"
                    # Hide game name — print phase number only
                    print(f"  Phase {phase_idx+1}/{len(seed_chain)}: {metric} steps={r['steps']} t={r['elapsed']}s")

        # Aggregate per game
        aggregated = {}
        for name, task_results in results.items():
            aggregated[name] = {
                "seeds": task_results,
                "l1_rate": sum(1 for r in task_results if r.get('l1')) / self.n_seeds,
                "l2_rate": sum(1 for r in task_results if r.get('l2')) / self.n_seeds,
                "avg_steps": np.mean([r['steps'] for r in task_results]),
                "mean_elapsed": np.mean([r['elapsed'] for r in task_results]),
            }
            if self.verbose:
                print(f"\n{name}: L1={aggregated[name]['l1_rate']:.0%} "
                      f"avg_t={aggregated[name]['mean_elapsed']:.1f}s")

        # Compute aggregate chain score (Jun 2026-03-24)
        # Single number: mean L1 rate across all games. 1.0 = perfect chain.
        game_l1_rates = [v['l1_rate'] for v in aggregated.values()
                         if isinstance(v, dict) and 'l1_rate' in v]
        chain_score = np.mean(game_l1_rates) if game_l1_rates else 0.0
        games_with_signal = sum(1 for r in game_l1_rates if r > 0)

        if self.verbose:
            print(f"\n{'='*50}")
            print(f"CHAIN SCORE: {chain_score:.3f} ({games_with_signal}/{len(game_l1_rates)} games with signal)")
            print(f"{'='*50}")
            # Per-game breakdown revealed AFTER aggregate
            for name in aggregated:
                if isinstance(aggregated[name], dict):
                    print(f"  {name}: L1={aggregated[name]['l1_rate']:.0%}")

        aggregated['_chain_score'] = chain_score
        aggregated['_games_with_signal'] = games_with_signal
        aggregated['_total_games'] = len(game_l1_rates)

        return aggregated

    def save_results(self, aggregated: dict, substrate_name: str,
                     step: int = 0, config: dict = None,
                     output_dir: str = None, mode: str = None,
                     chain_kill: dict = None) -> str:
        """Save chain results as structured JSON to chain_results/runs/.

        Returns path to saved file.
        """
        import json
        from datetime import datetime

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'chain_results', 'runs')
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{date_str}_{step}_{substrate_name}.json"

        # Build chain score
        phases_passed = sum(1 for v in aggregated.values()
                           if isinstance(v, dict) and v.get('l1_rate', 0) > 0)
        phases_total = len(aggregated)

        output = {
            "substrate": substrate_name,
            "timestamp": timestamp,
            "step": step,
            "prism_mode": mode,
            "chain": [name for name, _ in self.chain],
            "budget_per_phase": getattr(self.chain[0][1], 'n_steps',
                                        getattr(self.chain[0][1], 'n_images_per_task', 0)) if self.chain else 0,
            "n_seeds": self.n_seeds,
            "config": config or {},
            "results": {},
            "chain_score": {
                "phases_passed": phases_passed,
                "phases_total": phases_total,
                "chain_complete": phases_passed == phases_total,
            },
            "chain_kill": chain_kill if chain_kill is not None else {"verdict": "NO_BASELINE"},
        }

        for name, data in aggregated.items():
            if isinstance(data, dict):
                # Strip per-seed raw data for compactness, keep summary
                output["results"][name] = {
                    k: v for k, v in data.items() if k != "seeds"
                }
                # Add per-seed L1 list
                if "seeds" in data:
                    output["results"][name]["per_seed_l1"] = [
                        s.get("l1") or s.get("level_reached", 0)
                        for s in data["seeds"]
                    ]

        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        # Update latest.json
        latest_path = os.path.join(os.path.dirname(output_dir), 'latest.json')
        with open(latest_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        if self.verbose:
            print(f"\nResults saved to {filepath}")
            print(f"Chain score: {phases_passed}/{phases_total}"
                  f"{' — CHAIN COMPLETE' if phases_passed == phases_total else ''}")

        return filepath


def make_prism(n_steps: int = DEFAULT_STEPS,
               safety_timeout: float = 300.0) -> list:
    """PRISM: Progressive Recursive Intelligence Sequence Metric.

    Standard chain: Split-CIFAR-100 → LS20 → FT09 → VC33 → Split-CIFAR-100.
    One problem, many angles. The substrate that solves one solves all.

    n_steps: primary step budget per env (not time).
    safety_timeout: watchdog (not primary budget).
    """
    return [
        ("Split-CIFAR-100-before", SplitCIFAR100Wrapper(500, safety_timeout)),
        ("LS20", ArcGameWrapper("LS20", n_steps, safety_timeout)),
        ("FT09", ArcGameWrapper("FT09", n_steps, safety_timeout)),
        ("VC33", ArcGameWrapper("VC33", n_steps, safety_timeout)),
        ("Split-CIFAR-100-after", SplitCIFAR100Wrapper(500, safety_timeout)),
    ]


def make_default_chain(n_steps: int = DEFAULT_STEPS,
                       safety_timeout: float = 300.0) -> list:
    """Legacy chain: CIFAR-100 → LS20 → FT09 → VC33 → CIFAR-100."""
    return [
        ("CIFAR-100-before", CIFARWrapper(n_steps, safety_timeout)),
        ("LS20", ArcGameWrapper("LS20", n_steps, safety_timeout)),
        ("FT09", ArcGameWrapper("FT09", n_steps, safety_timeout)),
        ("VC33", ArcGameWrapper("VC33", n_steps, safety_timeout)),
        ("CIFAR-100-after", CIFARWrapper(n_steps, safety_timeout)),
    ]


def make_game_only_chain(n_steps: int = DEFAULT_STEPS,
                         safety_timeout: float = 300.0) -> list:
    """Chain without CIFAR-100."""
    return [
        ("LS20", ArcGameWrapper("LS20", n_steps, safety_timeout)),
        ("FT09", ArcGameWrapper("FT09", n_steps, safety_timeout)),
        ("VC33", ArcGameWrapper("VC33", n_steps, safety_timeout)),
    ]


def make_prism_from_config(config_path: str = None, mode: str = "light",
                            n_steps: int = DEFAULT_STEPS,
                            safety_timeout: float = 300.0) -> list:
    """Build PRISM chain from config file. No hardcoded games.

    Adding a game = editing prism_config.json, not this code.
    Post March 25: add 150 ARC-AGI-3 games to the 'full' config.
    """
    import json
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'prism_config.json')
    with open(config_path) as f:
        config = json.load(f)

    phases = config.get(mode, config.get("light"))
    chain = []
    for phase in phases:
        t = phase["type"]
        name = phase["name"]
        if t == "arc_game":
            chain.append((name, ArcGameWrapper(phase["game_name"], n_steps, safety_timeout)))
        elif t == "split_cifar":
            chain.append((name, SplitCIFAR100Wrapper(phase.get("n_images_per_task", 500),
                                                      safety_timeout)))
        elif t == "cifar":
            chain.append((name, CIFARWrapper(n_steps, safety_timeout)))
        elif t == "gym":
            chain.append((name, GymWrapper(phase["env_id"], n_steps, safety_timeout)))
        elif t == "atari":
            chain.append((name, AtariWrapper(phase.get("game", "ALE/MontezumaRevenge-v5"),
                                             n_steps, safety_timeout)))
        # Future: terminal_bench, browsecomp, hle — add when available
    return chain


def make_prism_mode(mode: str = "C", config_path: str = None,
                    n_steps: int = DEFAULT_STEPS,
                    safety_timeout: float = 300.0) -> tuple:
    """PRISM modes (Jun, 2026-03-23):

      A: full config, fixed order
      B: full config, randomized order per seed
      C: light config, randomized order per seed (default until March 25)

    Returns: (chain_list, randomize_order)
    Usage: chain, randomize = make_prism_mode("C")
           runner = ChainRunner(chain, randomize_order=randomize)
    """
    mode = mode.upper()
    if mode == "A":
        return make_prism_from_config(config_path, "full", n_steps, safety_timeout), False
    elif mode == "B":
        return make_prism_from_config(config_path, "full", n_steps, safety_timeout), True
    elif mode == "C":
        return make_prism_from_config(config_path, "light", n_steps, safety_timeout), True
    else:
        raise ValueError(f"Unknown PRISM mode: {mode}. Use A, B, or C.")


def compute_chain_kill(aggregated: dict, baseline_path: str = None,
                       baseline: dict = None) -> dict:
    """Compare aggregated results against baseline. Returns chain kill verdict.

    Chain kill rule (Jun 2026-03-24):
      PASS  — all games neutral-or-improved vs baseline
      KILL  — any game improved while another degraded (per-game tuning)
      FAIL  — all games degraded (bad mechanism, not per-game tuning)
      NO_BASELINE — no baseline available yet

    Args:
        aggregated: dict from ChainRunner.run()
        baseline_path: path to baseline JSON file (chain_results/baseline_994.json)
        baseline: pre-loaded baseline dict (alternative to baseline_path)

    Returns dict with: verdict, per_game_delta, chain_score, details
    """
    import json

    # Load baseline
    if baseline is None and baseline_path is not None:
        try:
            with open(baseline_path) as f:
                raw = json.load(f)
            baseline = raw.get('results', raw)
        except (FileNotFoundError, KeyError):
            baseline = None

    if baseline is None:
        return {"verdict": "NO_BASELINE", "per_game_delta": {}, "details": "baseline not found"}

    # Compute per-game L1 rate deltas
    per_game_delta = {}
    for name, data in aggregated.items():
        if not isinstance(data, dict) or 'l1_rate' not in data:
            continue
        base_data = baseline.get(name, {})
        base_l1 = base_data.get('l1_rate', None)
        if base_l1 is None:
            per_game_delta[name] = {"delta": None, "current": data['l1_rate'], "baseline": None}
            continue
        delta = data['l1_rate'] - base_l1
        per_game_delta[name] = {
            "delta": round(delta, 4),
            "current": round(data['l1_rate'], 4),
            "baseline": round(base_l1, 4),
        }

    # Verdict
    valid_deltas = [v['delta'] for v in per_game_delta.values() if v['delta'] is not None]
    if not valid_deltas:
        verdict = "NO_BASELINE"
    else:
        improved = sum(1 for d in valid_deltas if d > 0.05)   # >5% improvement
        degraded = sum(1 for d in valid_deltas if d < -0.05)  # >5% degradation
        if improved > 0 and degraded > 0:
            verdict = "KILL"   # per-game tuning: some games up, some down
        elif improved == 0 and degraded > 0:
            verdict = "FAIL"   # no improvement, at least one degraded
        else:
            verdict = "PASS"   # all improved or neutral

    # Chain score summary
    phases_passed = sum(1 for v in aggregated.values()
                       if isinstance(v, dict) and v.get('l1_rate', 0) > 0)
    phases_total = len([v for v in aggregated.values() if isinstance(v, dict) and 'l1_rate' in v])

    return {
        "verdict": verdict,
        "per_game_delta": per_game_delta,
        "chain_score": {"phases_passed": phases_passed, "phases_total": phases_total},
        "details": f"improved={improved if valid_deltas else '?'} degraded={degraded if valid_deltas else '?'} of {len(valid_deltas)} games",
    }
