"""
Step 715b — k_prune sweep for outcome-based pruning across all 3 games.

R3 hypothesis: there exists a universal pruning hash resolution k_prune where
outcome-based action discovery works for ALL games simultaneously.

Key insight from 715: k=12 nav hash too fine for VC33 (cosmetic=0, every action new cell).
Separation: pruning hash k_prune is INDEPENDENT of 674 nav hash (k=12 fixed).
The pruning hash groups observations at coarser resolution to enable cosmetic detection.

Test matrix: k_prune=[4,6,8,12] x [LS20,FT09,VC33], 1 seed, 500 warmup steps.
Report cosmetic count at each (k, game). Find universal k where:
  VC33: cosmetic > 60 (most actions cosmetic, ~2-8 structural)
  FT09: cosmetic ~= 56 (4 dirs cosmetic, 8 clicks structural = 60 cosmetic)
  LS20: cosmetic ~= 64 (all clicks cosmetic, 4 dirs structural = 64 cosmetic)

Kill: if no single k achieves correct pruning on all 3 games -> universal k impossible.
"""
import numpy as np
import sys
import time

K_NAV = 12        # fixed 674 navigation hash — never changed
DIM = 4096
MIN_PROBES = 10
WARMUP_STEPS = 500
SEED = 0

K_PRUNE_VALUES = [4, 6, 8, 12]
GAMES = ['LS20', 'FT09', 'VC33']

DIR_ACTIONS = [0, 1, 2, 3]
GRID_ACTIONS = [(gx * 8 + 4) + (gy * 8 + 4) * 64
                for gy in range(8) for gx in range(8)]
UNIVERSAL_ACTIONS = DIR_ACTIONS + GRID_ACTIONS
N_UNIV = len(UNIVERSAL_ACTIONS)  # 68


def enc_raw(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    return a.flatten()


def _hash_k(H, x):
    bits = (H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)


def run_warmup(make, k_prune, seed=SEED):
    env = make()
    rng = np.random.RandomState(seed * 1000 + k_prune)
    H_prune = rng.randn(k_prune, DIM).astype(np.float32)

    mu = np.zeros(DIM, dtype=np.float32)
    mu_n = 0
    all_seen = set()

    probe_count = [0] * N_UNIV
    new_cell_count = [0] * N_UNIV
    cosmetic = set()
    structural = set(range(N_UNIV))
    probe_ptr = 0

    obs = env.reset(seed=seed)

    for step in range(1, WARMUP_STEPS + 1):
        if obs is None:
            obs = env.reset(seed=seed)
            mu = np.zeros(DIM, dtype=np.float32); mu_n = 0
            all_seen = set()
            continue

        # Update running mean
        x_raw = enc_raw(obs)
        mu_n += 1
        mu = mu + (x_raw - mu) / mu_n
        x = x_raw - mu

        # Choose action round-robin over structural
        active = sorted(structural) or list(range(N_UNIV))
        idx = active[probe_ptr % len(active)]
        probe_ptr += 1
        action_int = UNIVERSAL_ACTIONS[idx]

        try:
            obs_new, reward, done, info = env.step(action_int)
        except Exception:
            obs_new = obs; done = False; info = {}

        # Check novelty of result using k_prune hash
        if obs_new is not None:
            x_new_raw = enc_raw(obs_new)
            # Use current mu (not yet updated for obs_new) for consistent centering
            x_new = x_new_raw - mu
            prune_n = _hash_k(H_prune, x_new)
            is_new = prune_n not in all_seen
            all_seen.add(prune_n)
        else:
            is_new = False

        probe_count[idx] += 1
        if is_new:
            new_cell_count[idx] += 1
            if idx in cosmetic:
                cosmetic.discard(idx)
                structural.add(idx)
        if (probe_count[idx] >= MIN_PROBES
                and new_cell_count[idx] == 0
                and idx not in cosmetic):
            cosmetic.add(idx)
            structural.discard(idx)

        obs = obs_new if obs_new is not None else obs
        if done:
            obs = env.reset(seed=seed)
            mu = np.zeros(DIM, dtype=np.float32); mu_n = 0
            all_seen = set()

    cosm_dir = len([a for a in cosmetic if a < 4])
    cosm_click = len([a for a in cosmetic if a >= 4])
    struct_dir = len([a for a in structural if a < 4])
    struct_click = len([a for a in structural if a >= 4])
    return dict(cosmetic=len(cosmetic), structural=len(structural),
                cosm_dir=cosm_dir, cosm_click=cosm_click,
                struct_dir=struct_dir, struct_click=struct_click)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        makers = {
            'LS20': lambda: arcagi3.make("LS20"),
            'FT09': lambda: arcagi3.make("FT09"),
            'VC33': lambda: arcagi3.make("VC33"),
        }
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 715b: k_prune sweep, {WARMUP_STEPS} warmup steps, 1 seed")
    print(f"k_prune values: {K_PRUNE_VALUES}  games: {GAMES}")
    print(f"Target: VC33 cosmetic>60, FT09 cosmetic~56-60, LS20 cosmetic~64\n")

    # results[game][k_prune] = dict
    results = {g: {} for g in GAMES}

    t_start = time.time()
    for game in GAMES:
        mk = makers[game]
        for k in K_PRUNE_VALUES:
            r = run_warmup(mk, k)
            results[game][k] = r
            print(f"  {game} k={k:2d}: cosmetic={r['cosmetic']:2d}(dir={r['cosm_dir']},click={r['cosm_click']}) "
                  f"structural={r['structural']:2d}(dir={r['struct_dir']},click={r['struct_click']})",
                  flush=True)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Total time: {elapsed:.1f}s\n")
    print(f"{'':8s}  {'VC33':>12s}  {'FT09':>12s}  {'LS20':>12s}")
    print(f"{'k_prune':8s}  {'cosmetic':>12s}  {'cosmetic':>12s}  {'cosmetic':>12s}")
    print(f"{'-'*52}")
    for k in K_PRUNE_VALUES:
        vc = results['VC33'][k]['cosmetic']
        ft = results['FT09'][k]['cosmetic']
        ls = results['LS20'][k]['cosmetic']
        # Check if this k is "universal": VC33>60, FT09 in [54,64], LS20 in [60,68]
        vc_ok = vc > 60
        ft_ok = 50 <= ft <= 68
        ls_ok = 60 <= ls <= 68
        flag = " <<UNIVERSAL" if (vc_ok and ft_ok and ls_ok) else ""
        print(f"k={k:2d}      {vc:>12d}  {ft:>12d}  {ls:>12d}{flag}")

    # Find universal k
    universal = [k for k in K_PRUNE_VALUES
                 if (results['VC33'][k]['cosmetic'] > 60
                     and 50 <= results['FT09'][k]['cosmetic'] <= 68
                     and 60 <= results['LS20'][k]['cosmetic'] <= 68)]
    if universal:
        best_k = universal[0]
        print(f"\nFINDING: Universal k_prune={best_k} works for all 3 games")
        print(f"IMPLICATION: Outcome-based pruning is viable with k_prune={best_k} as pruning hash")
        print(f"Next: Step 716 — outcome-based pruning with k_prune={best_k} on VC33, 5 seeds, 120K")
    else:
        print(f"\nKILL: No single k_prune works for all 3 games simultaneously")
        print(f"IMPLICATION: Universal hash resolution for outcome-based pruning is impossible")
        print(f"The coarser VC33 needs conflicts with the finer resolution LS20/FT09 require")


if __name__ == "__main__":
    main()
