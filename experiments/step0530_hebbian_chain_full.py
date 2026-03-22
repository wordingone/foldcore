"""
Step 530 — Hebbian full chain: CIFAR->LS20->FT09->VC33->CIFAR.

W is (256, 69): max_actions = max(4, 69, 3) = 69 (FT09 max).
Each phase uses its own n_actions slice of W.
CIFAR: random action in [0,3]. LS20: argmin W[:,:4]. FT09: argmin W[:,:69]. VC33: argmin W[:,:3].

Predictions: LS20 WIN (Step 527 WIN@45K < 50K budget).
FT09 3/3 (round-robin over 69 acts). VC33 1/3 (magic pixel timing). Deterministic.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)

MAX_ACTIONS = 6   # max(LS20=4, FT09=6, VC33=1, CIFAR=4)
LR = 1e-3
N_CIFAR = 10_000
TIME_CAP = 300

PHASES = [
    ('LS20', 50_000),
    ('FT09', 50_000),
    ('VC33', 30_000),
]


def encode_game(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def encode_cifar(img):
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32) +
            0.587 * img[:, :, 1].astype(np.float32) +
            0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


class HebbianAgent:
    def __init__(self, enc_dim=256, max_actions=MAX_ACTIONS, lr=LR):
        self.W = np.zeros((enc_dim, max_actions), dtype=np.float32)
        self.lr = lr

    def act(self, x, n_actions):
        scores = self.W[:, :n_actions].T @ x
        return int(np.argmin(scores))

    def update(self, x, action):
        self.W[:, action] += self.lr * x

    def weight_stats(self, n_actions):
        norms = np.linalg.norm(self.W[:, :n_actions], axis=0)
        return float(norms.min()), float(norms.max()), float(norms.mean())


def run_cifar_phase(agent, X, label, n_actions, rng):
    t0 = time.time()
    for i in range(len(X)):
        x = encode_cifar(X[i])
        a = int(rng.randint(n_actions))
        agent.update(x, a)
    w_min, w_max, w_mean = agent.weight_stats(n_actions)
    print(f"  {label}: W_norms=[{w_min:.2f},{w_max:.2f},{w_mean:.2f}]  {time.time()-t0:.1f}s",
          flush=True)


def run_game_phase(agent, arc, game_name, max_steps):
    from arcengine import GameState
    games = arc.get_environments()
    game_id = next(g.game_id for g in games if game_name.lower() in g.game_id.lower())
    env = arc.make(game_id)
    action_space = env.action_space
    n_actions = len(action_space)  # detect from env
    obs = env.reset()
    go = ts = 0
    level_step = win_step = None
    t0 = time.time()
    while ts < max_steps:
        if time.time() - t0 > TIME_CAP:
            break
        if obs is None or not obs.frame:
            obs = env.reset(); go += 1; continue
        if obs.state == GameState.GAME_OVER:
            obs = env.reset(); go += 1; continue
        x = encode_game(obs.frame)
        a = agent.act(x, n_actions)
        prev_lvls = obs.levels_completed
        obs = env.step(action_space[a])
        agent.update(x, a)
        ts += 1
        if obs and obs.levels_completed > prev_lvls and level_step is None:
            level_step = ts
        if obs and obs.state == GameState.WIN:
            win_step = ts
            break
    w_min, w_max, w_mean = agent.weight_stats(n_actions)
    tag = f"WIN@{win_step}" if win_step else (f"L1@{level_step}" if level_step else "FAIL")
    print(f"  {game_name}(n={n_actions}): {tag}  go={go}  steps={ts}  "
          f"W_norms=[{w_min:.2f},{w_max:.2f},{w_mean:.2f}]  {time.time()-t0:.0f}s",
          flush=True)
    return win_step is not None


def t1():
    agent = HebbianAgent()
    assert agent.W.shape == (256, MAX_ACTIONS), f"W shape: {agent.W.shape} expected (256,{MAX_ACTIONS})"
    # argmin of zeros returns 0 for any n_actions
    x = np.ones(256, dtype=np.float32)
    assert agent.act(x, 4) == 0
    assert agent.act(x, 69) == 0
    # after updating action 0, argmin should return 1 (not 0)
    agent.update(x, 0)
    assert agent.act(x, 4) in [1, 2, 3], "argmin should avoid action 0 after update"
    # weight stats
    wmin, wmax, wmean = agent.weight_stats(4)
    assert wmax > 0
    print("T1 PASS")


def main():
    t1()

    import torchvision, arc_agi
    ds = torchvision.datasets.CIFAR100('./data/cifar100',
                                        train=False, download=True)
    X = np.array(ds.data[:N_CIFAR])
    arc = arc_agi.Arcade()

    agent = HebbianAgent()

    print("\n--- Phase 1: CIFAR (random actions in [0,3]) ---", flush=True)
    run_cifar_phase(agent, X, "CIFAR_P1", n_actions=4, rng=np.random.RandomState(0))

    results = {}
    for game_name, max_steps in PHASES:
        print(f"\n--- Phase: {game_name} ({max_steps//1000}K steps) ---", flush=True)
        results[game_name] = run_game_phase(agent, arc, game_name, max_steps)

    print("\n--- Phase 5: CIFAR return ---", flush=True)
    run_cifar_phase(agent, X, "CIFAR_P2", n_actions=4, rng=np.random.RandomState(0))
    w_global_min, w_global_max, _ = agent.weight_stats(MAX_ACTIONS)
    print(f"  Full W norms (69 cols): [{w_global_min:.2f},{w_global_max:.2f}]", flush=True)

    print(f"\n{'='*55}", flush=True)
    print("STEP 530 SUMMARY", flush=True)
    for game, win in results.items():
        print(f"  {game}: {'WIN' if win else 'FAIL'}", flush=True)
    wins = sum(results.values())
    print(f"\n{wins}/3 games full WIN (WIN = all levels)", flush=True)
    print("NOTE: LS20 L1 = level 1 only (not full WIN); FT09 actual n_actions=6 not 69; VC33 actual n_actions=1 not 3.", flush=True)


if __name__ == "__main__":
    main()
