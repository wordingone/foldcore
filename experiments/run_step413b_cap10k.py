#!/usr/bin/env python3
"""
Step 413b -- Same rotation as 413, cb_cap=10K per codebook.

4 codebooks at {8, 16, 32, 64}. V stored at EACH codebook's own resolution.
Rotate driver per life. Lock on level. Codebooks persist across lives.

Test: does cb_cap=10K (vs 1K in 413) restore exploration?
Memory: 8x8=2.5MB, 16x16=10MB, 32x32=40MB, 64x64=160MB → ~212MB total.

LS20. 50K steps.
Script: scripts/run_step413b_cap10k.py
"""

import time, math, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CB_CAP = 10000; THRESH_INT = 100

RESOLUTIONS = [8, 16, 32, 64]


class Codebook:
    """Single-resolution codebook. V stored at native resolution (N²D)."""
    def __init__(self, res, nc, k=3, dev=DEVICE):
        self.res = res; self.nc = nc; self.k = k; self.dev = dev
        self.dim = res * res
        self.V = torch.zeros(0, self.dim, device=dev)
        self.labels = torch.zeros(0, dtype=torch.long, device=dev)
        self.thresh = 0.7; self.spawn_count = 0

    def _pool(self, x_raw):
        """Pool 4096D raw input to this codebook's resolution."""
        N = self.res
        if N == 64: return x_raw
        img = x_raw.reshape(64, 64)
        k = 64 // N
        return img.reshape(N, k, N, k).mean(dim=(1, 3)).flatten()

    def _ut(self):
        n = self.V.shape[0]
        if n < 2: return
        Vn = F.normalize(self.V, dim=1)
        ss = min(500, n); idx = torch.randperm(n, device=self.dev)[:ss]
        s = Vn[idx] @ Vn.T
        t = s.topk(min(2, n), dim=1).values
        self.thresh = float((t[:, 1] if t.shape[1] >= 2 else t[:, 0]).median())

    def seed(self, x_raw, label):
        x_raw = x_raw.to(self.dev).float()
        x_p = self._pool(x_raw)
        self.V = torch.cat([self.V, x_p.unsqueeze(0)])
        self.labels = torch.cat([self.labels, torch.tensor([label], device=self.dev)])
        self.spawn_count += 1
        if self.spawn_count % THRESH_INT == 0: self._ut()

    def process_votes(self, x_raw):
        """Return class scores (for action selection by driver)."""
        x_raw = x_raw.to(self.dev).float()
        if self.V.shape[0] == 0:
            return torch.zeros(self.nc, device=self.dev)
        x_p = F.normalize(self._pool(x_raw), dim=0)
        Vn = F.normalize(self.V, dim=1)
        si = Vn @ x_p
        ac = int(self.labels.max().item()) + 1
        sc = torch.zeros(max(ac, self.nc), device=self.dev)
        for c in range(ac):
            m = (self.labels == c)
            if m.sum() == 0: continue
            cs = si[m]; sc[c] = cs.topk(min(self.k, len(cs))).values.sum()
        return sc[:self.nc]

    def process_learn(self, x_raw, action):
        """Attract/spawn using action label. Stores at native resolution."""
        x_raw = x_raw.to(self.dev).float()
        x_p = self._pool(x_raw)

        if self.V.shape[0] == 0:
            self.V = x_p.unsqueeze(0)
            self.labels = torch.tensor([action], device=self.dev)
            self.spawn_count += 1
            return

        x_n = F.normalize(x_p, dim=0)
        Vn = F.normalize(self.V, dim=1)
        si = Vn @ x_n
        p = action; tm = (self.labels == p)

        if (tm.sum() == 0 or si[tm].max() < self.thresh) and self.V.shape[0] < CB_CAP:
            self.V = torch.cat([self.V, x_p.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([p], device=self.dev)])
            self.spawn_count += 1
            if self.spawn_count % THRESH_INT == 0: self._ut()
        else:
            w_i = int(si.argmax().item())
            a = 1.0 - float(si[w_i].item())
            self.V[w_i] = self.V[w_i] + a * (x_p - self.V[w_i])


class ParallelResolution:
    def __init__(self, nc):
        self.nc = nc
        self.codebooks = {N: Codebook(N, nc) for N in RESOLUTIONS}
        self.driving_idx = 0
        self.locked = False
        self.locked_res = None
        self.life_log = []

    @property
    def driving_res(self):
        if self.locked: return self.locked_res
        return RESOLUTIONS[self.driving_idx]

    def step(self, x_raw):
        """All codebooks learn. Driving codebook picks action."""
        driver = self.codebooks[self.driving_res]
        scores = driver.process_votes(x_raw)
        action = scores[:self.nc].argmin().item()

        for N, cb in self.codebooks.items():
            cb.process_learn(x_raw, action)

        return action

    def seed(self, x_raw, label):
        for cb in self.codebooks.values():
            cb.seed(x_raw, label)

    def on_game_over(self, found_level):
        dr = self.driving_res
        self.life_log.append((dr, found_level))

        if not self.locked and found_level:
            self.locked = True
            self.locked_res = dr
            print(f"    LOCKED resolution: {dr}x{dr} (found level while driving)", flush=True)

        if not self.locked:
            self.driving_idx = (self.driving_idx + 1) % len(RESOLUTIONS)


def main():
    t0 = time.time()
    print(f"Step 413b -- Parallel resolution, cb_cap=10K. Rotate driver per life.", flush=True)
    print(f"Device: {DEVICE}  resolutions={RESOLUTIONS}  cb_cap={CB_CAP}", flush=True)
    print(flush=True)

    import arc_agi; from arcengine import GameState
    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    env = arc.make(ls20.game_id); obs = env.reset()
    na = len(env.action_space)
    pr = ParallelResolution(nc=na)

    ts = 0; go = 0; lvls = 0; seeded = False
    action_counts = {}
    life_start_lvls = 0

    while ts < 50000 and go < 500:
        if obs is None or obs.state == GameState.GAME_OVER:
            if ts > 0:
                found = (lvls > life_start_lvls)
                pr.on_game_over(found)
            go += 1; obs = env.reset(); seeded = False
            if obs is None: break
            life_start_lvls = lvls
            continue
        if obs.state == GameState.WIN:
            print(f"    WIN at step {ts}!", flush=True); break

        raw = torch.from_numpy(np.array(obs.frame[0], dtype=np.float32).flatten() / 15.0)

        # Seed phase
        if not seeded and pr.codebooks[RESOLUTIONS[0]].V.shape[0] < na:
            i = pr.codebooks[RESOLUTIONS[0]].V.shape[0]
            pr.seed(raw, i)
            obs = env.step(env.action_space[i]); ts += 1
            action_counts[env.action_space[i].name] = action_counts.get(env.action_space[i].name, 0) + 1
            if pr.codebooks[RESOLUTIONS[0]].V.shape[0] >= na:
                seeded = True
                for cb in pr.codebooks.values():
                    cb._ut()
            continue
        if not seeded: seeded = True

        c = pr.step(raw)
        action = env.action_space[c % na]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1
        ol = obs.levels_completed
        obs = env.step(action); ts += 1
        if obs is None: break

        if obs.levels_completed > ol:
            lvls = obs.levels_completed
            dr = pr.driving_res
            print(f"    LEVEL {lvls} at step {ts} driver={dr}x{dr}"
                  f"  cb_sizes={{{','.join(f'{N}:{cb.V.shape[0]}' for N,cb in pr.codebooks.items())}}}"
                  f"  go={go}", flush=True)

        if ts % 5000 == 0:
            dom = max(action_counts.values()) / sum(action_counts.values()) * 100 if action_counts else 0
            dr = pr.driving_res
            mem_mb = sum(cb.V.shape[0] * cb.dim * 4 / 1e6 for cb in pr.codebooks.values())
            print(f"    [step {ts:5d}] driver={dr}x{dr}"
                  f"  {'LOCKED' if pr.locked else 'ROTATING'}"
                  f"  cb_sizes={{{','.join(f'{N}:{cb.V.shape[0]}' for N,cb in pr.codebooks.items())}}}"
                  f"  mem={mem_mb:.0f}MB  dom={dom:.0f}%  levels={lvls}  go={go}", flush=True)

    elapsed = time.time() - t0
    print(flush=True); print("=" * 60, flush=True)
    print("STEP 413b SUMMARY", flush=True); print("=" * 60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    print(f"Locked: {pr.locked}  resolution={pr.locked_res}", flush=True)
    print(flush=True)

    # Per-resolution stats (skip full life log — too verbose)
    for N in RESOLUTIONS:
        driving_lives = [(i, fl) for i, (dr, fl) in enumerate(pr.life_log) if dr == N]
        found_count = sum(1 for _, fl in driving_lives if fl)
        total = len(driving_lives)
        cb = pr.codebooks[N]
        print(f"  {N:2d}x{N:2d}: drove {total} lives, found level in {found_count}"
              f"  cb_size={cb.V.shape[0]}  thresh={cb.thresh:.3f}", flush=True)

    print(f"\nactions: {action_counts}", flush=True)
    if lvls > 0 and pr.locked:
        print(f"\nPASS: Level found! Locked resolution: {pr.locked_res}x{pr.locked_res}", flush=True)
    elif lvls > 0:
        print(f"\nPASS: Level found (resolution not locked yet)", flush=True)
    else:
        print(f"\nKILL: No levels found in 50K steps.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__': main()
