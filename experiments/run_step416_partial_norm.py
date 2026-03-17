#!/usr/bin/env python3
"""
Step 416 -- Partial normalization at 64x64 raw. Stage 7 test.

x_normed = x / ||x||^p. p=1→cosine (saturates at 4096D). p=0→raw. p=0.5→half-cosine.

Run 3 values: p=0.25, p=0.5, p=0.75. LS20. 50K steps each. 64x64 raw input.
centered_enc: partial_normalize then subtract codebook mean.

Script: scripts/run_step416_partial_norm.py
"""

import sys, time, logging, numpy as np, torch
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_STEPS = 50000; THRESH_SAMPLE = 500

POWERS = [0.25, 0.5, 0.75]


def partial_normalize(x, power):
    norm = torch.norm(x)
    if norm > 0:
        return x / (norm ** power)
    return x


class CompressedFold:
    def __init__(self, d, k=3, power=0.5, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.0; self.k = k; self.d = d; self.device = device
        self.power = power

    def _pnorm(self, x):
        return partial_normalize(x.to(self.device).float(), self.power)

    def _force_add(self, x, label):
        x_n = self._pnorm(x)
        self.V = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels, torch.tensor([label], device=self.device)])
        self._update_thresh()

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        if n <= THRESH_SAMPLE:
            G = self.V @ self.V.T
            G.fill_diagonal_(-float('inf'))
            self.thresh = float(G.max(dim=1).values.median())
        else:
            idx = torch.randperm(n, device=self.device)[:THRESH_SAMPLE]
            S = self.V[idx] @ self.V[idx].T
            S.fill_diagonal_(-float('inf'))
            self.thresh = float(S.max(dim=1).values.median())

    def process(self, x, nc, label=None):
        x_n = self._pnorm(x)
        if self.V.shape[0] == 0:
            spawn_label = label if label is not None else 0
            self.V = x_n.unsqueeze(0)
            self.labels = torch.tensor([spawn_label], device=self.device)
            return spawn_label

        sims = self.V @ x_n

        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(max(n_cls, nc), device=self.device)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0: continue
            cs = sims[mask]; scores[c] = cs.topk(min(self.k, len(cs))).values.sum()

        prediction = scores[:nc].argmin().item()
        spawn_label = label if label is not None else prediction
        target_mask = (self.labels == prediction)

        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([spawn_label], device=self.device)])
        else:
            target_sims = sims.clone()
            target_sims[~target_mask] = -float('inf')
            winner = target_sims.argmax().item()
            sim_val = float(sims[winner].item())
            alpha = 1.0 / (1.0 + abs(sim_val))
            self.V[winner] = self._pnorm(self.V[winner] + alpha * (x_n - self.V[winner]))

        self._update_thresh()
        return prediction


def centered_enc_raw(frame, fold):
    """64x64 raw → partial_normalize → subtract codebook mean."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    t = torch.from_numpy(arr.flatten()).to(fold.device)
    t_n = partial_normalize(t, fold.power)
    if fold.V.shape[0] > 2:
        mean_V = fold.V.mean(dim=0)
        t_n = t_n - mean_V
    return t_n


def run_power(arc, game_id, power, max_steps=MAX_STEPS, verbose=True):
    from arcengine import GameState

    fold = CompressedFold(d=4096, k=3, power=power)
    env = arc.make(game_id); obs = env.reset()
    na = len(env.action_space)

    ts = 0; go = 0; lvls = 0; seeded = False
    action_counts = {}
    sim_snapshots = []

    while ts < max_steps and go < 500:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); seeded = False
            if obs is None: break
            continue
        if obs.state == GameState.WIN:
            if verbose: print(f"      WIN at step {ts}!", flush=True)
            break

        x = centered_enc_raw(obs.frame, fold)

        if not seeded and fold.V.shape[0] < na:
            i = fold.V.shape[0]
            fold._force_add(x, i)
            obs = env.step(env.action_space[i]); ts += 1
            action_counts[env.action_space[i].name] = action_counts.get(env.action_space[i].name, 0) + 1
            if fold.V.shape[0] >= na: seeded = True
            continue
        if not seeded: seeded = True

        # Snapshot sim stats before process
        if ts % 5000 == 0 and fold.V.shape[0] > 0:
            x_n = fold._pnorm(x)
            sims = fold.V @ x_n
            sim_snapshots.append({
                'step': ts, 'mean': float(sims.mean()), 'std': float(sims.std()),
                'min': float(sims.min()), 'max': float(sims.max()),
            })

        c = fold.process(x, na)
        action = env.action_space[c % na]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1
        ol = obs.levels_completed
        obs = env.step(action); ts += 1
        if obs is None: break

        if obs.levels_completed > ol:
            lvls = obs.levels_completed
            if verbose:
                print(f"      LEVEL {lvls} at step {ts}  p={power}"
                      f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}  go={go}", flush=True)

        if ts % 5000 == 0 and verbose:
            dom = max(action_counts.values()) / sum(action_counts.values()) * 100 if action_counts else 0
            ss = sim_snapshots[-1] if sim_snapshots else {}
            print(f"      [step {ts:5d}] cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}"
                  f"  dom={dom:.0f}%  levels={lvls}  go={go}"
                  f"  sim=[{ss.get('min',0):.2f},{ss.get('mean',0):.2f},{ss.get('max',0):.2f}]", flush=True)

    dom = max(action_counts.values()) / sum(action_counts.values()) * 100 if action_counts else 0
    return {
        'power': power, 'steps': ts, 'game_overs': go, 'levels': lvls,
        'cb_size': fold.V.shape[0], 'thresh': fold.thresh,
        'dom': dom, 'actions': action_counts, 'sims': sim_snapshots,
    }


def main():
    t0 = time.time()
    print(f"Step 416 -- Partial normalization at 64x64 raw. LS20.", flush=True)
    print(f"Device: {DEVICE}  powers={POWERS}  max_steps={MAX_STEPS}", flush=True)
    print(flush=True)

    import arc_agi
    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    results = []
    for p in POWERS:
        print(f"  === p={p} ===", flush=True)
        pt0 = time.time()
        stats = run_power(arc, ls20.game_id, p)
        elapsed = time.time() - pt0
        results.append(stats)
        print(f"      DONE: levels={stats['levels']}  cb={stats['cb_size']}"
              f"  thresh={stats['thresh']:.4f}  dom={stats['dom']:.0f}%"
              f"  go={stats['game_overs']}  elapsed={elapsed:.1f}s", flush=True)
        print(flush=True)

    total_elapsed = time.time() - t0
    print("=" * 60, flush=True)
    print("STEP 416 SUMMARY", flush=True)
    print("=" * 60, flush=True)

    for s in results:
        mark = " PASS" if s['levels'] > 0 else " KILL"
        print(f"  p={s['power']:.2f}: levels={s['levels']}  steps={s['steps']}"
              f"  cb={s['cb_size']}  thresh={s['thresh']:.4f}"
              f"  dom={s['dom']:.0f}%  go={s['game_overs']}{mark}", flush=True)

    print(f"\nElapsed: {total_elapsed:.2f}s", flush=True)


if __name__ == '__main__': main()
