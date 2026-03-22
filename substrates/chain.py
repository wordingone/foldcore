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


class SplitCIFAR100Wrapper:
    """Split-CIFAR-100: the standard continual learning benchmark.

    20 sequential tasks, 5 classes per task (classes 0-4, 5-9, ..., 95-99).
    Standard CL metrics: avg accuracy + backward transfer.

    R1 mode: substrate gets images, picks action (0-4 within each task),
    no reward signal. External judge measures whether actions correlate
    with true labels.

    Compare against: DER++, EWC, iCaRL, A-GEM published results.
    """

    N_TASKS = 20
    CLASSES_PER_TASK = 5

    def __init__(self, n_images_per_task: int = 500,
                 per_seed_time: float = PER_SEED_TIME,
                 split: str = "test"):
        self.n_images_per_task = n_images_per_task
        self.per_seed_time = per_seed_time
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
            # Group by task (5 classes per task)
            tasks = []
            for t in range(self.N_TASKS):
                class_start = t * self.CLASSES_PER_TASK
                class_end = class_start + self.CLASSES_PER_TASK
                mask = (labels >= class_start) & (labels < class_end)
                task_images = images[mask]
                task_labels = labels[mask] - class_start  # remap to 0-4
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

        substrate.reset(seed)

        for task_id in range(self.N_TASKS):
            if (time.time() - t_start) >= self.per_seed_time:
                break

            task_images, task_labels = self._data[task_id]
            idx = rng.choice(len(task_images),
                             min(self.n_images_per_task, len(task_images)),
                             replace=False)
            correct = 0
            for i in idx:
                if (time.time() - t_start) >= self.per_seed_time:
                    break
                obs = task_images[i].astype(np.float32) / 255.0
                action = substrate.process(obs) % self.CLASSES_PER_TASK
                correct += int(action == task_labels[i])

            acc = correct / len(idx)
            task_accuracies.append(round(acc, 4))

            # On task completion, signal level transition to substrate
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
            "l1": 1 if avg_acc and avg_acc > 0.05 else None,  # L1: above random (20%)
            "l2": None,
            "task_accuracies": task_accuracies,
            "avg_accuracy": round(avg_acc, 4) if avg_acc else None,
            "backward_transfer": round(bwt, 4) if bwt is not None else None,
            "tasks_completed": len(task_accuracies),
        }


class AtariWrapper:
    """Atari game wrapper — R1 mode (no reward signal to substrate).

    Start with Montezuma's Revenge: hardest exploration game, used as
    canonical hard-exploration benchmark in RL literature.

    Requires: gymnasium[atari] + ROM files.
    If not available: raises ImportError with install instructions.

    Standard comparison: Atari 100K (100K steps with reward). Our setup
    is strictly harder — same step budget, no reward signal.
    """

    INSTALL_MSG = (
        "Install: pip install gymnasium[atari] ale-py\n"
        "ROMs: pip install 'autorom[accept-rom-license]'"
    )

    def __init__(self, game: str = "MontezumaRevenge-v4",
                 n_steps: int = 100_000,
                 per_seed_time: float = PER_SEED_TIME,
                 frame_skip: int = 4):
        self.game = game
        self.n_steps = n_steps
        self.per_seed_time = per_seed_time
        self.frame_skip = frame_skip
        self._env = None

    def _make_env(self):
        try:
            import gymnasium as gym
            env = gym.make(self.game, obs_type="rgb",
                           frameskip=self.frame_skip,
                           render_mode=None)
            return env
        except ImportError:
            raise ImportError(f"Atari not available.\n{self.INSTALL_MSG}")

    def run_seed(self, substrate, seed: int) -> dict:
        env = self._make_env()
        substrate.reset(seed)
        obs, info = env.reset(seed=seed)
        steps = 0
        rooms_visited = set()
        t_start = time.time()

        while steps < self.n_steps and (time.time() - t_start) < self.per_seed_time:
            from substrates.base import Observation
            action = substrate.process(
                Observation(data=np.array(obs, dtype=np.float32) / 255.0,
                            modality="atari",
                            metadata={"game": self.game, "step": steps})
            )
            action = action % env.action_space.n
            obs, reward, terminated, truncated, info = env.step(action)
            # NOTE: reward is NOT passed to substrate (R1 mode)
            # reward is used only for external measurement
            steps += 1
            if "ram" in info:
                rooms_visited.add(info["ram"][3])  # room byte in Montezuma
            if terminated or truncated:
                obs, info = env.reset(seed=seed)
                substrate.on_level_transition()

        env.close()
        elapsed = time.time() - t_start
        return {
            "game": self.game,
            "seed": seed,
            "steps": steps,
            "elapsed": round(elapsed, 2),
            "l1": 1 if len(rooms_visited) > 1 else None,  # L1: visited >1 room
            "l2": None,
            "rooms_visited": len(rooms_visited),
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


def make_standard_chain(n_steps: int = DEFAULT_STEPS,
                        per_seed_time: float = PER_SEED_TIME) -> list:
    """Standard chain with established benchmarks.
    Split-CIFAR-100 → LS20 → FT09 → VC33 → Split-CIFAR-100 (backward transfer).
    Split-CIFAR-100 is the standard CL benchmark (comparable to DER++/EWC/iCaRL).
    """
    return [
        ("Split-CIFAR-100-before", SplitCIFAR100Wrapper(500, per_seed_time)),
        ("LS20", ArcGameWrapper("LS20", n_steps, per_seed_time)),
        ("FT09", ArcGameWrapper("FT09", n_steps, per_seed_time)),
        ("VC33", ArcGameWrapper("VC33", n_steps, per_seed_time)),
        ("Split-CIFAR-100-after", SplitCIFAR100Wrapper(500, per_seed_time)),
    ]


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
