#!/usr/bin/env python3
"""
Step 527 -- Hebbian learning substrate on chain: CIFAR->LS20->CIFAR.

Reuses Step 524 Hebbian substrate (Variant A: unbounded W, argmin(W.T@x)).
Chain: CIFAR 1-pass (10K images) -> LS20 50K steps (3 seeds) -> CIFAR 1-pass.

CIFAR phase design: Option C (random action assignment per image).
W[:,a] += lr*x for randomly assigned action a.
~2500 images per action column (uniform random over 4 actions).

Predictions:
- CIFAR phase: W grows, each column accumulates ~2500 diverse CIFAR vectors
- LS20: 1/3 WIN. W saturated by CIFAR -> argmin nearly random -> slow nav.
  Some seeds may still navigate (W is not truly uniform after CIFAR).
- CIFAR return: same as P1 (LS20 observations negligible vs 10K CIFAR)

Kill: 0/3 -> Hebbian negative transfer confirmed.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)

N_CIFAR = 10_000
N_ACTIONS_LS20 = 4
N_ACTIONS_CIFAR = 4    # Option C: random assignment over LS20 action space
MAX_LS20 = 50_000
N_SEEDS = 3
LR = 1e-3


def encode_cifar(img):
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32) +
            0.587 * img[:, :, 1].astype(np.float32) +
            0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


def encode_arc(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class HebbianAgent:
    def __init__(self, enc_dim=256, n_actions=N_ACTIONS_LS20, lr=LR):
        self.W = np.zeros((enc_dim, n_actions), dtype=np.float32)
        self.lr = lr
        self.n_actions = n_actions

    def act(self, x):
        scores = self.W.T @ x    # (n_actions,)
        return int(np.argmin(scores))

    def update(self, x, action):
        self.W[:, action] += self.lr * x

    def weight_stats(self):
        norms = np.linalg.norm(self.W, axis=0)
        return float(norms.min()), float(norms.max()), float(norms.mean())


def run_cifar(agent, X, y, label, rng):
    """CIFAR phase: random action assignment (Option C)."""
    correct = 0
    t0 = time.time()
    for i in range(len(X)):
        x = encode_cifar(X[i])
        # Random action assignment — don't use argmin (no game semantics for CIFAR)
        a = int(rng.randint(agent.n_actions))
        agent.update(x, a)
        # Track "accuracy" via argmin as diagnostic (not used for update)
        a_pred = agent.act(x)
        if a_pred == int(y[i] % agent.n_actions): correct += 1
    w_min, w_max, w_mean = agent.weight_stats()
    n_imgs_per_col = N_CIFAR // agent.n_actions
    print(f"  {label}: W_norms=[{w_min:.3f},{w_max:.3f},{w_mean:.3f}]  "
          f"~{n_imgs_per_col} imgs/col  {time.time()-t0:.1f}s", flush=True)
    return agent


def run_ls20(agent, arc, game_id, seed=0):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < MAX_LS20:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = agent.act(x)
        obs_before = obs.levels_completed
        obs = env.step(action_space[a])
        agent.update(x, a)
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    w_min, w_max, w_mean = agent.weight_stats()
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"  seed={seed}: {status}  go={go}  steps={ts}  "
          f"W_norms=[{w_min:.3f},{w_max:.3f},{w_mean:.3f}]  {time.time()-t0:.0f}s",
          flush=True)
    return lvls > 0


def main():
    t_total = time.time()
    print("Step 527: Hebbian chain CIFAR->LS20->CIFAR", flush=True)
    print(f"lr={LR}  cifar_n={N_CIFAR}  cifar_action=Option_C(random,n={N_ACTIONS_CIFAR})  "
          f"max_ls20={MAX_LS20//1000}K  seeds={N_SEEDS}", flush=True)
    print(f"Prediction: 1/3 LS20 WIN (W saturated by CIFAR, slow navigation)", flush=True)

    import torchvision, arc_agi
    ds = torchvision.datasets.CIFAR100('./data/cifar100',
                                        train=False, download=True)
    X = np.array(ds.data[:N_CIFAR])
    y = np.array(ds.targets[:N_CIFAR])
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    # W diagnostic before CIFAR: all zeros
    rng_cifar = np.random.RandomState(0)

    # Phase 1: CIFAR (build W contamination)
    print("\n--- Phase 1: CIFAR (1-pass, Option C: random action per image) ---", flush=True)
    agent_p1 = HebbianAgent()
    run_cifar(agent_p1, X, y, "P1", rng_cifar)
    W_after_cifar = agent_p1.W.copy()
    print(f"  W snapshot taken. Action column norms: "
          f"{np.linalg.norm(W_after_cifar, axis=0).tolist()}", flush=True)

    # Phase 2: LS20 (3 seeds, each starting from CIFAR-contaminated W)
    print(f"\n--- Phase 2: LS20 ({MAX_LS20//1000}K steps, {N_SEEDS} seeds) ---", flush=True)
    wins = 0
    for s in range(N_SEEDS):
        agent = HebbianAgent()
        agent.W = W_after_cifar.copy()    # restore CIFAR state for each seed
        result = run_ls20(agent, arc, ls20.game_id, seed=s)
        if result: wins += 1
    print(f"  LS20: {wins}/{N_SEEDS} WIN", flush=True)

    # Phase 3: CIFAR return (measure forgetting)
    print("\n--- Phase 3: CIFAR return (1-pass) ---", flush=True)
    agent_p3 = HebbianAgent()
    agent_p3.W = W_after_cifar.copy()    # start from CIFAR snapshot (no LS20 contamination)
    rng_p3 = np.random.RandomState(0)
    run_cifar(agent_p3, X, y, "P3", rng_p3)

    # W comparison: P3 norms vs P1 norms
    w_p1 = np.linalg.norm(W_after_cifar, axis=0)
    w_p3 = np.linalg.norm(agent_p3.W, axis=0)
    print(f"  P1 norms: {w_p1.tolist()}", flush=True)
    print(f"  P3 norms: {w_p3.tolist()}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("STEP 527 SUMMARY", flush=True)
    print(f"  CIFAR P1:  W contaminated ({N_CIFAR} images, random actions)", flush=True)
    print(f"  LS20:      {wins}/{N_SEEDS} WIN", flush=True)
    print(f"  CIFAR P3:  W grows further (additive)", flush=True)

    print(f"\nVERDICT:", flush=True)
    if wins == 0:
        print(f"  KILL: Hebbian negative transfer confirmed. W saturation breaks LS20.", flush=True)
        print(f"  CIFAR contamination (W[:,a] contains avg CIFAR vectors) prevents navigation.", flush=True)
    elif wins == 1:
        print(f"  PARTIAL: 1/3. W saturation slows but doesn't kill navigation.", flush=True)
        print(f"  Hebbian is more robust to cross-domain contamination than predicted.", flush=True)
    else:
        print(f"  UNEXPECTED: {wins}/3 WIN. W saturation doesn't prevent navigation.", flush=True)
        print(f"  Hebbian continuous aggregation survives CIFAR contamination.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
