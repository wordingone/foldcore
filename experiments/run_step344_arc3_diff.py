#!/usr/bin/env python3
"""
Step 344 -- ARC-AGI-3: diff encoding.

Encoding: diff = avgpool8(frame[t]) - avgpool8(frame[t-1]), 64-dim, normalize.
  Step 0:     use avgpool8(frame[0]) directly (no prev)
  Zero diff:  skip codebook update, repeat previous action
  The diff encodes WHAT HAPPENED, not WHERE THINGS ARE.
  Substrate learns transition dynamics.

Bootstrap: force-add first len(action_space) entries directly to fold.V/fold.labels.
  Don't go through process() — just append. Then state-derived threshold computes.

Reward: stamp (prev_diff, prev_action_cls) when levels_completed increases.

Exploration: 30% random for first 200 steps, decay to 5%.

Run on LS20 first (most diff diversity). 1000 steps max.
Script: scripts/run_step344_arc3_diff.py
"""

import time
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC  = 64   # 8x8 avg pool


# ══════════════════════════════════════════════════════════════════════════════
# CompressedFold — Step 342 (all 7 stages)
# ══════════════════════════════════════════════════════════════════════════════

class CompressedFold:
    def __init__(self, d, k=3, device=DEVICE):
        self.V      = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7
        self.k      = k
        self.d      = d
        self.device = device

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        self.thresh = float(G.max(dim=1).values.median())

    def _force_add(self, x, label):
        """Directly append entry — bypass threshold check (used for bootstrap)."""
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V      = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels,
                                  torch.tensor([label], device=self.device)])
        self._update_thresh()

    def process(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            spawn_label = label if label is not None else 0
            self.V      = x.unsqueeze(0)
            self.labels = torch.tensor([spawn_label], device=self.device)
            return spawn_label
        sims = self.V @ x
        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(n_cls, device=self.device)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0: continue
            cs = sims[mask]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()
        prediction = scores.argmax().item()
        attract_target = prediction
        spawn_label    = label if label is not None else prediction
        target_mask = (self.labels == attract_target)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V      = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([spawn_label], device=self.device)])
        else:
            target_sims = sims.clone()
            target_sims[~target_mask] = -float('inf')
            winner = target_sims.argmax().item()
            alpha = 1.0 - float(sims[winner].item())
            self.V[winner] = F.normalize(
                self.V[winner] + alpha * (x - self.V[winner]), dim=0)
        self._update_thresh()
        return prediction


# ══════════════════════════════════════════════════════════════════════════════
# Encoding
# ══════════════════════════════════════════════════════════════════════════════

def avgpool8(frame):
    """avgpool2d(8x8) -> 64-dim float32 numpy array, normalized 0-1."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(8, 8, 8, 8).mean(axis=(2, 3)).flatten()   # (64,)


def diff_encode(pooled_curr, pooled_prev):
    """Diff of avg-pooled frames as torch tensor."""
    diff = pooled_curr - pooled_prev
    return torch.from_numpy(diff)  # (64,), may be zero


def is_zero(t, eps=1e-6):
    return float(t.norm()) < eps


# ══════════════════════════════════════════════════════════════════════════════
# Game runner
# ══════════════════════════════════════════════════════════════════════════════

def run_game_diff(arc, game_id, max_steps=1000, max_resets=20,
                  eps_start=0.30, eps_end=0.05, eps_steps=200, k=3, verbose=True):
    """
    Run game with diff encoding.
    Returns metrics dict.
    """
    from arcengine import GameState

    fold = CompressedFold(d=D_ENC, k=k)
    env  = arc.make(game_id)
    obs  = env.reset()

    total_steps   = 0
    total_resets  = 0
    total_levels  = 0
    steps_per_lvl = []
    cb_snapshots  = []   # (step, cb_size, thresh)
    action_counts = {}   # action.name -> count

    # Epsilon schedule: linear decay from eps_start to eps_end over eps_steps, then flat
    def get_epsilon(step):
        if step >= eps_steps:
            return eps_end
        return eps_start + (eps_end - eps_start) * (step / eps_steps)

    bootstrapped    = False
    prev_pooled     = None
    prev_diff_enc   = None
    prev_cls_used   = None
    prev_action     = None
    lvl_step_start  = 0
    win             = False

    while total_steps < max_steps and total_resets < max_resets:
        if obs is None or obs.state == GameState.GAME_OVER:
            total_resets += 1
            if total_resets >= max_resets:
                break
            obs = env.reset()
            if obs is None: break
            prev_pooled = prev_diff_enc = prev_cls_used = prev_action = None
            lvl_step_start = total_steps
            continue

        if obs.state == GameState.WIN:
            win = True
            break

        action_space = env.action_space
        n_acts       = len(action_space)

        # Bootstrap: force-add one entry per action class
        if not bootstrapped:
            curr_pooled = avgpool8(obs.frame)
            if fold.V.shape[0] < n_acts:
                i = fold.V.shape[0]
                enc = torch.from_numpy(curr_pooled)
                fold._force_add(enc, label=i)
                action = action_space[i % n_acts]
            else:
                bootstrapped = True

            if bootstrapped or fold.V.shape[0] >= n_acts:
                bootstrapped = True
                if verbose and fold.V.shape[0] >= n_acts:
                    print(f"    [bootstrap done] cb={fold.V.shape[0]}"
                          f"  thresh={fold.thresh:.4f}", flush=True)

            prev_pooled = curr_pooled
            if not bootstrapped:
                # Execute bootstrap action
                data = {}
                if action.is_complex():
                    arr = np.array(obs.frame[0])
                    cy, cx = divmod(int(np.argmax(arr)), 64)
                    data = {"x": cx, "y": cy}
                obs = env.step(action, data=data)
                total_steps += 1
                action_counts[action.name] = action_counts.get(action.name, 0) + 1
                if obs and obs.state == GameState.WIN:
                    win = True
                    break
                continue

        # Compute diff encoding
        curr_pooled = avgpool8(obs.frame)
        if prev_pooled is None:
            # First step after reset: use abs encoding
            enc = torch.from_numpy(curr_pooled)
        else:
            enc = diff_encode(curr_pooled, prev_pooled)

        # Zero diff: skip codebook update, repeat previous action
        if is_zero(enc):
            if prev_action is not None:
                action = prev_action
            else:
                action = random.choice(action_space)
            # Don't call process(), just execute
            data = {}
            if action.is_complex():
                arr = np.array(obs.frame[0])
                cy, cx = divmod(int(np.argmax(arr)), 64)
                data = {"x": cx, "y": cy}
            prev_obs_levels = obs.levels_completed
            obs = env.step(action, data=data)
            total_steps += 1
            action_counts[action.name] = action_counts.get(action.name, 0) + 1
            if obs is not None and obs.levels_completed > prev_obs_levels:
                total_levels = obs.levels_completed
                steps_this = total_steps - lvl_step_start
                steps_per_lvl.append(steps_this)
                lvl_step_start = total_steps
                if verbose:
                    print(f"    [step {total_steps}] Level {obs.levels_completed}"
                          f"  steps_this_lvl={steps_this}"
                          f"  cb={fold.V.shape[0]}", flush=True)
            if obs is not None and obs.state == GameState.WIN:
                win = True
                break
            # prev_pooled stays (no new useful diff)
            continue

        # Normal step: process() with diff encoding
        epsilon = get_epsilon(total_steps)

        if random.random() < epsilon:
            # Random exploration: pick action, stamp it as label
            action = random.choice(action_space)
            cls_used = action_space.index(action) if action in action_space else 0
            fold.process(enc, label=cls_used)
        else:
            cls_used = fold.process(enc, label=None)
            action   = action_space[cls_used % n_acts]

        action_counts[action.name] = action_counts.get(action.name, 0) + 1

        # Execute
        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}

        enc_this_step  = enc
        cls_this_step  = cls_used
        prev_obs_levels = obs.levels_completed

        obs = env.step(action, data=data)
        total_steps += 1

        if obs is None: break

        # Codebook snapshot
        if total_steps % 100 == 0:
            cb_snapshots.append((total_steps, fold.V.shape[0], fold.thresh))

        # Reward stamp on level completion
        if obs.levels_completed > prev_obs_levels:
            total_levels = obs.levels_completed
            steps_this = total_steps - lvl_step_start
            steps_per_lvl.append(steps_this)
            lvl_step_start = total_steps
            if verbose:
                print(f"    [step {total_steps}] Level {obs.levels_completed}"
                      f"  steps={steps_this}"
                      f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
            # Reward: stamp (this diff, this action class) as positive
            fold.process(enc_this_step, label=cls_this_step)

        if obs.state == GameState.WIN:
            win = True
            if verbose:
                print(f"    [step {total_steps}] WIN! levels={obs.levels_completed}"
                      f"  cb={fold.V.shape[0]}", flush=True)
            break

        prev_pooled   = curr_pooled
        prev_diff_enc = enc_this_step
        prev_cls_used = cls_this_step
        prev_action   = action

    # Class distribution
    cls_dist = {}
    if fold.V.shape[0] > 0:
        for lbl in fold.labels.cpu().numpy():
            k_ = int(lbl)
            cls_dist[k_] = cls_dist.get(k_, 0) + 1

    return {
        'win': win,
        'levels': total_levels,
        'steps': total_steps,
        'resets': total_resets,
        'steps_per_level': steps_per_lvl,
        'cb_final': fold.V.shape[0],
        'thresh_final': fold.thresh,
        'cb_snapshots': cb_snapshots,
        'cls_dist': dict(sorted(cls_dist.items())),
        'action_counts': action_counts,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("Step 344 -- ARC-AGI-3: diff encoding", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("Encoding: diff(avgpool8(frame[t]) - avgpool8(frame[t-1])) -> 64-dim", flush=True)
    print("Bootstrap: force-add one entry per action. Epsilon: 30%->5% over 200 steps.", flush=True)
    print(flush=True)

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()

    # Quick diff cos-sim sanity check on LS20
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())
    env_chk = arc.make(ls20.game_id)
    obs_chk  = env_chk.reset()
    pooleds, diffs = [], []
    prev_p = None
    for i in range(20):
        p = avgpool8(obs_chk.frame)
        pooleds.append(p)
        if prev_p is not None:
            d = p - prev_p
            if not (abs(d).max() < 1e-6):
                diffs.append(d)
        prev_p = p
        a = env_chk.action_space[i % len(env_chk.action_space)]
        obs_chk = env_chk.step(a)
        if obs_chk is None: break

    if diffs:
        vecs = [F.normalize(torch.from_numpy(d.astype(np.float32)).unsqueeze(0), dim=1).squeeze(0)
                for d in diffs]
        sims = [float(F.cosine_similarity(vecs[i].unsqueeze(0), vecs[i+1].unsqueeze(0)))
                for i in range(len(vecs)-1)]
        print(f"Diff encoding sanity (LS20, {len(diffs)} non-zero diffs):", flush=True)
        print(f"  cos_sim: min={min(sims):.4f} mean={np.mean(sims):.4f} max={max(sims):.4f}", flush=True)
    print(flush=True)

    results = {}

    # Run LS20 first (most diff diversity), then others
    ordered = [g for g in games if 'ls20' in g.game_id.lower()] + \
              [g for g in games if 'ls20' not in g.game_id.lower()]

    for g in ordered:
        game_id = g.game_id
        title   = g.title
        print("=" * 60, flush=True)
        print(f"Game: {title} ({game_id})", flush=True)
        t1 = time.time()

        r = run_game_diff(arc, game_id,
                          max_steps=1000, max_resets=20,
                          eps_start=0.30, eps_end=0.05, eps_steps=200,
                          k=3, verbose=True)

        elapsed_g = time.time() - t1
        print(f"  win={r['win']}  levels={r['levels']}  steps={r['steps']}"
              f"  resets={r['resets']}", flush=True)
        print(f"  cb={r['cb_final']}  thresh={r['thresh_final']:.4f}", flush=True)
        print(f"  steps_per_level: {r['steps_per_level']}", flush=True)
        print(f"  cls_dist: {r['cls_dist']}", flush=True)
        print(f"  action_counts: {r['action_counts']}", flush=True)
        if r['cb_snapshots']:
            snaps = '  '.join(f"s{s}:cb{c}:t{th:.3f}" for s, c, th in r['cb_snapshots'][:6])
            print(f"  cb_growth: {snaps}", flush=True)
        print(f"  elapsed: {elapsed_g:.1f}s", flush=True)

        results[title] = r
        print(flush=True)

    elapsed = time.time() - t0
    print("=" * 60, flush=True)
    print("STEP 344 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print("Encoding: diff(avgpool8) -> 64-dim", flush=True)
    print(flush=True)

    total_wins = total_levels = 0
    for title, r in results.items():
        win_str = "WIN" if r['win'] else "---"
        spl = f"~{int(np.mean(r['steps_per_level']))}/lvl" if r['steps_per_level'] else "no lvls"
        print(f"  {title}: [{win_str}]  levels={r['levels']}  steps={r['steps']}"
              f"  cb={r['cb_final']}  {spl}", flush=True)
        if r['win']: total_wins += 1
        total_levels += r['levels']

    print(flush=True)
    print(f"Games won: {total_wins}/{len(results)}", flush=True)
    print(f"Total levels: {total_levels}", flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
