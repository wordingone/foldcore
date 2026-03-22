"""
chain.py — Chain benchmark runner.

Sequences a substrate through multiple environments in order.
Handles LS20/FT09/VC33 (ARC games) and CIFAR-100 (classification).
5-minute cap per seed. Reports structured dict.
"""
import time
import numpy as np
import sys
import os

# LS20/FT09/VC33 action counts (confirmed from game metadata)
GAME_N_ACTIONS = {
    "LS20": 68,   # 4 dir + 64 grid clicks
    "FT09": 68,   # same action space structure
    "VC33": 68,   # same
}

PER_SEED_TIME = 300   # 5 minutes
DEFAULT_STEPS = 10_000


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
    """Wraps an ARC game env into a uniform interface for the chain runner."""

    def __init__(self, game_name: str, n_steps: int = DEFAULT_STEPS,
                 per_seed_time: float = PER_SEED_TIME):
        self.game_name = game_name
        self.n_steps = n_steps
        self.per_seed_time = per_seed_time
        self._factory = _make_arc_env(game_name)
        self._env = None

    def run_seed(self, substrate, seed: int) -> dict:
        """Run one seed. Returns structured result dict."""
        if self._env is None:
            self._env = self._factory()

        substrate.reset(seed)
        obs = self._env.reset(seed=seed)
        level = 0
        l1_step = l2_step = None
        steps = 0
        t_start = time.time()
        fresh_episode = True  # skip first step for fresh_episode bug

        while steps < self.n_steps and (time.time() - t_start) < self.per_seed_time:
            if obs is None:
                obs = self._env.reset(seed=seed)
                substrate.on_level_transition()
                fresh_episode = True
                continue

            action = substrate.process(np.array(obs, dtype=np.float32))
            obs, reward, done, info = self._env.step(action)
            steps += 1

            if fresh_episode:
                fresh_episode = False
                continue  # skip first step for fresh_episode spurious level

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


class CIFARWrapper:
    """CIFAR-100 classification task as a chain environment.

    Each 'episode' is one image. Action = class prediction (0-99).
    L1 metric = rolling accuracy > 10% (better than random).
    """

    def __init__(self, n_steps: int = DEFAULT_STEPS,
                 per_seed_time: float = PER_SEED_TIME,
                 split: str = "test"):
        self.n_steps = n_steps
        self.per_seed_time = per_seed_time
        self.split = split
        self._data = None  # (images, labels) loaded lazily

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
        except Exception as e:
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

        substrate.reset(seed)
        correct = 0
        window = []
        l1_step = None
        steps = 0
        t_start = time.time()

        for i, img_idx in enumerate(idx):
            if (time.time() - t_start) >= self.per_seed_time:
                break
            obs = images[img_idx].astype(np.float32) / 255.0
            label = int(labels[img_idx])
            action = substrate.process(obs)
            hit = int(action == label)
            correct += hit
            window.append(hit)
            if len(window) > 100:
                window.pop(0)
            steps += 1
            # L1: rolling accuracy > 10% (2x random)
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


class ChainRunner:
    """Runs a substrate through a sequence of (name, wrapper, n_steps) tasks."""

    def __init__(self, chain: list, n_seeds: int = 5,
                 per_seed_time: float = PER_SEED_TIME, verbose: bool = True):
        """
        chain: list of (name, wrapper_instance)
        n_seeds: seeds per game
        per_seed_time: max seconds per seed
        """
        self.chain = chain
        self.n_seeds = n_seeds
        self.per_seed_time = per_seed_time
        self.verbose = verbose

    def run(self, substrate_cls: type, substrate_kwargs: dict = None) -> dict:
        """Run full chain. Returns structured results dict."""
        if substrate_kwargs is None:
            substrate_kwargs = {}

        results = {}
        for (name, wrapper) in self.chain:
            if self.verbose:
                print(f"\n--- {name} ---")
            task_results = []
            for seed in range(self.n_seeds):
                sub = substrate_cls(**substrate_kwargs)
                r = wrapper.run_seed(sub, seed)
                task_results.append(r)
                if self.verbose:
                    l1 = r.get('l1')
                    acc = r.get('accuracy')
                    metric = f"acc={acc:.3f}" if acc is not None else f"l1={l1}"
                    print(f"  s{seed}: {metric} steps={r['steps']} t={r['elapsed']}s")
            results[name] = {
                "seeds": task_results,
                "l1_rate": sum(1 for r in task_results if r.get('l1')) / self.n_seeds,
                "l2_rate": sum(1 for r in task_results if r.get('l2')) / self.n_seeds,
                "avg_steps": np.mean([r['steps'] for r in task_results]),
            }
            if self.verbose:
                print(f"  L1={results[name]['l1_rate']:.0%} L2={results[name]['l2_rate']:.0%}")

        return results


def make_default_chain(n_steps: int = DEFAULT_STEPS,
                       per_seed_time: float = PER_SEED_TIME) -> list:
    """Standard chain: CIFAR-100 → LS20 → FT09 → VC33 → CIFAR-100.
    CIFAR-100 entries are optional (skipped if not available).
    """
    return [
        ("CIFAR-100-before", CIFARWrapper(n_steps, per_seed_time)),
        ("LS20", ArcGameWrapper("LS20", n_steps, per_seed_time)),
        ("FT09", ArcGameWrapper("FT09", n_steps, per_seed_time)),
        ("VC33", ArcGameWrapper("VC33", n_steps, per_seed_time)),
        ("CIFAR-100-after", CIFARWrapper(n_steps, per_seed_time)),
    ]


def make_game_only_chain(n_steps: int = DEFAULT_STEPS,
                         per_seed_time: float = PER_SEED_TIME) -> list:
    """Chain without CIFAR-100, for substrates tuned to game environments."""
    return [
        ("LS20", ArcGameWrapper("LS20", n_steps, per_seed_time)),
        ("FT09", ArcGameWrapper("FT09", n_steps, per_seed_time)),
        ("VC33", ArcGameWrapper("VC33", n_steps, per_seed_time)),
    ]
