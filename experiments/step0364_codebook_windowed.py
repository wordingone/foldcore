#!/usr/bin/env python3
"""
Step 364 -- Windowed encoding: x = [state_t, state_{t-1}, state_{t-2}].

768 dims (3 x 256). Codebook stores trajectory windows.
Kill: unique_windows >> unique_states → encoding adds information.
2000 steps on LS20.
Script: scripts/run_step364_windowed.py
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC  = 256 * 3  # window=3
WINDOW = 3


class CompressedFold:
    def __init__(self, d, k=3, device=DEVICE):
        self.V      = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7
        self.k, self.d, self.device = k, d, device

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels, torch.tensor([label], device=self.device)])
        self._update_thresh()

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        ss = min(500, n)
        idx = torch.randperm(n, device=self.device)[:ss]
        sims = self.V[idx] @ self.V.T
        topk = sims.topk(min(2, n), dim=1).values
        self.thresh = float((topk[:, 1] if topk.shape[1] >= 2 else topk[:, 0]).median())

    def process_novelty(self, x, n_cls, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            sl = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([sl], device=self.device)
            return sl
        sims = self.V @ x
        ac = int(self.labels.max().item()) + 1
        scores = torch.zeros(max(ac, n_cls), device=self.device)
        for c in range(ac):
            m = (self.labels == c)
            if m.sum() == 0: continue
            cs = sims[m]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()
        pred = scores[:n_cls].argmin().item()
        sl = label if label is not None else pred
        tm = (self.labels == pred)
        if tm.sum() == 0 or sims[tm].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([sl], device=self.device)])
            self._update_thresh()
        else:
            ts = sims.clone(); ts[~tm] = -float('inf')
            w = ts.argmax().item()
            a = 1.0 - float(sims[w].item())
            self.V[w] = F.normalize(self.V[w] + a * (x - self.V[w]), dim=0)
        return pred


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()  # (256,)


def centered_single(pooled, fold):
    """Center a single 256-dim frame using codebook mean (projected to 256)."""
    t = F.normalize(torch.from_numpy(pooled.astype(np.float32)), dim=0)
    if fold.V.shape[0] > 2:
        # Mean of codebook projected to first 256 dims (current frame portion)
        mean_V = fold.V[:, :256].mean(dim=0).cpu()
        t = t - mean_V
    return t


def make_window(history, fold):
    """Concatenate last WINDOW centered frames into one vector."""
    parts = []
    for pooled in history[-WINDOW:]:
        parts.append(centered_single(pooled, fold))
    # Pad with zeros if not enough history
    while len(parts) < WINDOW:
        parts.insert(0, torch.zeros(256))
    return torch.cat(parts)  # (768,)


def main():
    t0 = time.time()
    print(f"Step 364 -- Windowed encoding (window={WINDOW}, {D_ENC} dims)", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("Kill: unique_windows >> unique_states", flush=True)
    print(flush=True)

    import arc_agi
    from arcengine import GameState

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    print(f"Running: {ls20.title} ({ls20.game_id})", flush=True)
    print(f"max_steps=2000  k=3  window={WINDOW}", flush=True)
    print(flush=True)

    fold = CompressedFold(d=D_ENC, k=3)
    env  = arc.make(ls20.game_id)
    obs  = env.reset()
    n_acts = len(env.action_space)

    total_steps     = 0
    game_over_count = 0
    total_levels    = 0
    unique_states   = set()
    unique_windows  = set()
    seeded          = False
    history         = []  # list of pooled frames (256-dim each)

    max_steps = 2000

    while total_steps < max_steps and game_over_count < 50:
        if obs is None or obs.state == GameState.GAME_OVER:
            game_over_count += 1
            obs = env.reset()
            history = []  # reset history on death
            if obs is None: break
            continue

        if obs.state == GameState.WIN:
            print(f"    WIN at step {total_steps}!", flush=True)
            break

        curr_pooled = avgpool16(obs.frame)
        history.append(curr_pooled)
        unique_states.add(hash(curr_pooled.tobytes()))

        win_enc = make_window(history, fold)
        unique_windows.add(hash(win_enc.numpy().tobytes()))

        if not seeded and fold.V.shape[0] < n_acts:
            i = fold.V.shape[0]
            fold._force_add(win_enc, label=i)
            obs = env.step(env.action_space[i])
            total_steps += 1
            if fold.V.shape[0] >= n_acts:
                seeded = True
                print(f"    [seed done, step {total_steps}]"
                      f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
            continue

        if not seeded: seeded = True

        cls = fold.process_novelty(win_enc, n_cls=n_acts)
        ol = obs.levels_completed
        obs = env.step(env.action_space[cls % n_acts])
        total_steps += 1
        if obs is None: break

        if obs.levels_completed > ol:
            total_levels = obs.levels_completed
            print(f"    LEVEL {total_levels} at step {total_steps}"
                  f"  cb={fold.V.shape[0]}", flush=True)

        if total_steps % 500 == 0:
            print(f"    [step {total_steps:5d}] cb={fold.V.shape[0]}"
                  f"  unique_states={len(unique_states)}"
                  f"  unique_windows={len(unique_windows)}"
                  f"  go={game_over_count}", flush=True)

    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 364 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"levels={total_levels}  steps={total_steps}  go={game_over_count}", flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
    print(f"unique_states={len(unique_states)}", flush=True)
    print(f"unique_windows={len(unique_windows)}", flush=True)
    print(f"ratio windows/states = {len(unique_windows)/max(len(unique_states),1):.2f}x", flush=True)
    print(flush=True)
    if len(unique_windows) > len(unique_states) * 1.5:
        print("PASS: windowed encoding adds information (windows >> states).", flush=True)
    else:
        print("KILL: windowed encoding doesn't add enough information.", flush=True)
    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
