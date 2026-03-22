#!/usr/bin/env python3
"""
Step 524 -- Hebbian learning substrate on LS20. New architecture family.

No centroids, no projections. Weight matrix W: (256, 4).
Observation: avgpool16 + centered_enc -> 256D vector x.
Action selection: a = argmin(W.T @ x) -- least-activated action.
Update: W[:, a] += lr * x  (Hebbian -- strengthen connection).

Variants:
  A: Basic Hebbian (W unbounded)
  B: Normalized Hebbian (W normalized after each update)

Prediction: 0/5 at 50K. W saturates, W.T @ x stops discriminating.
Kill: 0/5 at 50K for both variants.
5-min cap.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)

N_ACTIONS = 4
MAX_STEPS = 50_000
N_SEEDS = 5
LR = 1e-3

# Tier 1 constants
T1_MAX_STEPS = 200


def encode_arc(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class HebbianAgent:
    """Hebbian learning: W(256,4), action=argmin(W.T @ x), update W[:,a] += lr*x."""

    def __init__(self, enc_dim=256, n_actions=N_ACTIONS, lr=LR, normalize=False):
        self.W = np.zeros((enc_dim, n_actions), dtype=np.float32)
        self.lr = lr
        self.normalize = normalize
        self.n_actions = n_actions
        self.step_count = 0

    def act(self, x):
        scores = self.W.T @ x          # (n_actions,)
        action = int(np.argmin(scores))
        return action

    def update(self, x, action):
        self.W[:, action] += self.lr * x
        if self.normalize:
            norms = np.linalg.norm(self.W, axis=0, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self.W /= norms
        self.step_count += 1

    def weight_stats(self):
        norms = np.linalg.norm(self.W, axis=0)
        return float(norms.min()), float(norms.max()), float(norms.mean())


def run_ls20(agent, arc, game_id, seed=0, max_steps=MAX_STEPS):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < max_steps:
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
          f"W_norms=[{w_min:.3f},{w_max:.3f},{w_mean:.3f}]  "
          f"{time.time()-t0:.0f}s", flush=True)
    return lvls > 0


def tier1_sanity():
    """T1: Verify shapes and basic update, < 1 second."""
    print("T1: Sanity check", flush=True)
    rng = np.random.RandomState(0)
    x = rng.randn(256).astype(np.float32)
    x -= x.mean()

    # Basic agent
    ag = HebbianAgent(normalize=False)
    assert ag.W.shape == (256, 4)
    scores_before = ag.W.T @ x
    a = ag.act(x)
    assert a == int(np.argmin(scores_before))
    ag.update(x, a)
    scores_after = ag.W.T @ x
    # After update: score for action a increased (Hebbian strengthens)
    assert scores_after[a] > scores_before[a], "Hebbian update must increase score for chosen action"

    # Normalized agent
    ag_n = HebbianAgent(normalize=True)
    ag_n.update(x, 0)
    norms = np.linalg.norm(ag_n.W, axis=0)
    assert abs(norms[0] - 1.0) < 1e-5 or norms[0] == 0.0, f"Norm should be 1.0 after normalize, got {norms[0]}"

    # W=zeros -> argmin=0 always (all tied, expected). Verify this.
    ag_zero = HebbianAgent()
    actions_zero = [ag_zero.act((rng.randn(256) - 0).astype(np.float32)) for _ in range(10)]
    assert all(a == 0 for a in actions_zero), "W=zeros should always pick action 0"
    print(f"  T1 NOTE: W=zeros -> argmin=0 always (expected — all scores tied at 0)", flush=True)

    # Short run to check update changes actions
    ag2 = HebbianAgent(normalize=False)
    t0 = time.time()
    for step in range(T1_MAX_STEPS):
        xx = rng.randn(256).astype(np.float32)
        xx -= xx.mean()
        a = ag2.act(xx)
        ag2.update(xx, a)
    actions_after = [ag2.act((rng.randn(256) - rng.randn(256).mean()).astype(np.float32))
                     for _ in range(20)]
    unique_after = len(set(actions_after))
    w_min, w_max, w_mean = ag2.weight_stats()
    print(f"  T1: {T1_MAX_STEPS} steps, W_norms=[{w_min:.3f},{w_max:.3f},{w_mean:.3f}]  "
          f"unique_actions={unique_after}  {time.time()-t0:.3f}s", flush=True)
    print(f"  T1 PASS", flush=True)


def main():
    t_total = time.time()
    print("Step 524: Hebbian learning substrate on LS20", flush=True)
    print(f"W=(256,4)  lr={LR}  n_seeds={N_SEEDS}  max_steps={MAX_STEPS//1000}K", flush=True)
    print(f"Prediction: 0/5 FAIL (W saturates, no navigation)", flush=True)

    tier1_sanity()

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    seeds = list(range(N_SEEDS))

    # Variant A: Basic Hebbian (no normalization)
    print(f"\n--- Variant A: Basic Hebbian (no normalize, lr={LR}) ---", flush=True)
    wins_a = 0
    for s in seeds:
        agent = HebbianAgent(normalize=False, lr=LR)
        result = run_ls20(agent, arc, ls20.game_id, seed=s)
        if result: wins_a += 1
    print(f"  Variant A: {wins_a}/{N_SEEDS} WIN", flush=True)

    # Variant B: Normalized Hebbian
    print(f"\n--- Variant B: Normalized Hebbian (||W||=1 per column, lr={LR}) ---", flush=True)
    wins_b = 0
    for s in seeds:
        agent = HebbianAgent(normalize=True, lr=LR)
        result = run_ls20(agent, arc, ls20.game_id, seed=s)
        if result: wins_b += 1
    print(f"  Variant B: {wins_b}/{N_SEEDS} WIN", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("STEP 524 SUMMARY", flush=True)
    print(f"  Variant A (basic):      {wins_a}/{N_SEEDS} WIN", flush=True)
    print(f"  Variant B (normalized): {wins_b}/{N_SEEDS} WIN", flush=True)

    print(f"\nVERDICT:", flush=True)
    if wins_a == 0 and wins_b == 0:
        print(f"  KILL confirmed: Hebbian learning cannot navigate LS20.", flush=True)
        print(f"  W.T @ x argmin does not discriminate game states.", flush=True)
        print(f"  Hebbian association (obs->action) is insufficient for navigation.", flush=True)
    elif wins_a > 0 or wins_b > 0:
        print(f"  UNEXPECTED WIN: Hebbian substrate navigates LS20.", flush=True)
        print(f"  A={wins_a}/{N_SEEDS}  B={wins_b}/{N_SEEDS}", flush=True)
        print(f"  Direct obs->action association can learn navigation.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
