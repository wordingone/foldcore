"""
Step 591 -- Death penalty on Hebbian substrate (family transfer test).

Step 524: Hebbian gets 5/5 L1 on LS20. W(256,4), act=argmin(W.T @ x), update W[:,a]+=lr*x.
Death penalty found useful on LSH (581d). Does it transfer to a different family?

Two conditions, 5 seeds, 10K steps:
  A) Hebbian alone  -- baseline (should replicate 524's 5/5)
  B) Hebbian+Death  -- on_death: W[:,a_death] += PENALTY_SCALE*LR*x_death
                       (100 fake Hebbian updates for the fatal action)

Key question: is the death-penalty speed improvement universal across families?
"""
import numpy as np
import time
import sys

N_A = 4
LR = 1e-3
DIM = 256
PENALTY_SCALE = 100   # 100 fake Hebbian updates on death (analog of LSH PENALTY=100)
MAX_STEPS = 10_000
TIME_CAP = 60
N_SEEDS = 5


def enc_vec(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class Hebbian:
    """W(256,4), act=argmin(W.T @ x), update W[:,a]+=lr*x."""

    def __init__(self, seed=0):
        np.random.seed(seed)
        self.W = np.zeros((DIM, N_A), dtype=np.float64)
        self._cx = None
        self._px = None
        self._pa = None
        self.total_deaths = 0

    def observe(self, frame):
        self._cx = enc_vec(frame)

    def act(self):
        scores = self.W.T @ self._cx
        action = int(np.argmin(scores))
        self._px = self._cx.copy()
        self._pa = action
        return action

    def on_step(self, action):
        if self._px is not None:
            self.W[:, action] += LR * self._px

    def on_death(self):
        pass

    def on_reset(self):
        self._px = None
        self._pa = None


class HebbianDeath(Hebbian):
    """Hebbian + soft death penalty.
    On death: W[:,a_death] += PENALTY_SCALE * LR * x_death
    Makes the fatal (state, action) pair look like it was visited PENALTY_SCALE extra times.
    """

    def on_death(self):
        if self._px is not None and self._pa is not None:
            self.W[:, self._pa] += PENALTY_SCALE * LR * self._px
            self.total_deaths += 1


def run_seed(mk, seed, SubClass):
    env = mk()
    sub = SubClass(seed=seed * 100 + 7)
    obs = env.reset(seed=seed)
    sub.on_reset()

    l1 = go = step = 0
    prev_cl = 0
    fresh = True
    t0 = time.time()

    while step < MAX_STEPS and time.time() - t0 < TIME_CAP:
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue
        sub.observe(obs)
        action = sub.act()
        obs, _, done, info = env.step(action)
        sub.on_step(action)
        step += 1
        if done:
            sub.on_death()
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh:
            prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1:
            l1 += 1
            if l1 <= 2:
                print(f"    s{seed} L1@{step}", flush=True)
        prev_cl = cl

    print(f"  s{seed}: L1={l1} go={go} step={step} deaths={sub.total_deaths} "
          f"{time.time()-t0:.0f}s", flush=True)
    return dict(seed=seed, l1=l1, go=go, steps=step, deaths=sub.total_deaths)


def run_condition(mk, label, SubClass):
    print(f"\n--- {label} ---", flush=True)
    results = []
    for seed in range(N_SEEDS):
        try:
            results.append(run_seed(mk, seed, SubClass))
        except Exception as e:
            print(f"  s{seed} FAIL: {e}", flush=True)
    wins = sum(1 for r in results if r['l1'] > 0)
    l1 = sum(r['l1'] for r in results)
    print(f"  {label}: {wins}/{N_SEEDS} L1={l1}", flush=True)
    return wins, l1


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 591: Hebbian + death penalty (family transfer test)", flush=True)
    print(f"  {N_SEEDS} seeds x {MAX_STEPS} steps | LR={LR} | PENALTY_SCALE={PENALTY_SCALE}", flush=True)

    t_total = time.time()
    h_wins, h_l1 = run_condition(mk, "Hebbian alone", Hebbian)
    hd_wins, hd_l1 = run_condition(mk, "Hebbian+Death", HebbianDeath)

    print(f"\n{'='*60}", flush=True)
    print(f"Step 591: Family transfer -- death penalty on Hebbian", flush=True)
    print(f"  Hebbian alone:  {h_wins}/{N_SEEDS} L1={h_l1}", flush=True)
    print(f"  Hebbian+Death:  {hd_wins}/{N_SEEDS} L1={hd_l1}", flush=True)

    if hd_wins > h_wins:
        print(f"\n  TRANSFER SIGNAL: Death penalty helps Hebbian ({hd_wins} > {h_wins}).", flush=True)
        print(f"  Speed improvement is NOT LSH-specific. Universal mechanism.", flush=True)
    elif hd_wins == h_wins:
        print(f"\n  NEUTRAL: Death penalty no transfer ({hd_wins} == {h_wins}).", flush=True)
        print(f"  Benefit may be LSH-specific (discrete cells enable precise avoidance).", flush=True)
    else:
        print(f"\n  DEGRADED: Death penalty hurts Hebbian ({hd_wins} < {h_wins}).", flush=True)
        print(f"  Continuous W update is incompatible with this penalty approach.", flush=True)

    print(f"\n  Total elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
