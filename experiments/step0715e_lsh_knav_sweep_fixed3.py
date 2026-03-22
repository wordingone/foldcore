"""
Step 715e — k_prune sweep, third fix: MIN_PROBES=5 (was 10).

Bug in 715d: MIN_PROBES=10 but 500 steps / 68 actions = ~7 probes each.
No action ever reaches threshold -> cosmetic=0 everywhere.
Fix: MIN_PROBES=5 (matches DEAD_THRESHOLD=5 from observation-based pruning).
With 500 warmup steps, each action gets ~7 probes > 5 -> threshold reachable.

All other fixes retained:
  - Raw uncentered prune hash (frame identity stable)
  - Current obs added to all_seen before action (no-change = already-seen)
"""
import numpy as np
import sys
import time

DIM = 4096
MIN_PROBES = 5    # FIXED: was 10, but 500 steps / 68 actions < 10 per action
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


def _hash_k(H, x_raw):
    bits = (H @ x_raw > 0).astype(np.uint8)
    return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)


def run_warmup(make, k_prune, seed=SEED):
    env = make()
    rng = np.random.RandomState(seed * 1000 + k_prune)
    H_prune = rng.randn(k_prune, DIM).astype(np.float32)

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
            all_seen = set()
            continue

        # Add current obs to all_seen: no-change action -> result already seen -> not novel
        x_curr_raw = enc_raw(obs)
        curr_hash = _hash_k(H_prune, x_curr_raw)
        all_seen.add(curr_hash)

        active = sorted(structural) or list(range(N_UNIV))
        idx = active[probe_ptr % len(active)]
        probe_ptr += 1
        action_int = UNIVERSAL_ACTIONS[idx]

        try:
            obs_new, reward, done, info = env.step(action_int)
        except Exception:
            obs_new = obs; done = False; info = {}

        if obs_new is not None:
            x_new_raw = enc_raw(obs_new)
            prune_n = _hash_k(H_prune, x_new_raw)
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
            all_seen = set()

    cosm_dir = len([a for a in cosmetic if a < 4])
    cosm_click = len([a for a in cosmetic if a >= 4])
    struct_dir = len([a for a in structural if a < 4])
    struct_click = len([a for a in structural if a >= 4])
    return dict(cosmetic=len(cosmetic), structural=len(structural),
                cosm_dir=cosm_dir, cosm_click=cosm_click,
                struct_dir=struct_dir, struct_click=struct_click,
                probes_per_action=WARMUP_STEPS // N_UNIV)


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

    probes = WARMUP_STEPS // N_UNIV
    print(f"Step 715e: k_prune sweep (MIN_PROBES={MIN_PROBES}, ~{probes} probes/action), {WARMUP_STEPS} warmup")
    print(f"Fixes: raw hash + current-obs-in-all_seen + MIN_PROBES={MIN_PROBES} (was 10)")
    print(f"k_prune: {K_PRUNE_VALUES}  games: {GAMES}\n")

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
        vc_ok = vc > 60
        ft_ok = 50 <= ft <= 68
        ls_ok = 60 <= ls <= 68
        flag = " <<UNIVERSAL" if (vc_ok and ft_ok and ls_ok) else ""
        print(f"k={k:2d}      {vc:>12d}  {ft:>12d}  {ls:>12d}{flag}")

    universal = [k for k in K_PRUNE_VALUES
                 if (results['VC33'][k]['cosmetic'] > 60
                     and 50 <= results['FT09'][k]['cosmetic'] <= 68
                     and 60 <= results['LS20'][k]['cosmetic'] <= 68)]

    print(f"\nR3 finding:")
    if universal:
        print(f"SIGNAL: Universal k_prune={universal[0]} works for all 3 games")
        print(f"Next: Step 716 — outcome-based pruning (k_prune={universal[0]}) on VC33, 5 seeds, 120K")
    else:
        for k in K_PRUNE_VALUES:
            vc = results['VC33'][k]['cosmetic']
            ft = results['FT09'][k]['cosmetic']
            ls = results['LS20'][k]['cosmetic']
            print(f"  k={k:2d}: VC33={'OK' if vc > 60 else f'FAIL({vc})'} "
                  f"FT09={'OK' if 50<=ft<=68 else f'FAIL({ft})'} "
                  f"LS20={'OK' if 60<=ls<=68 else f'FAIL({ls})'}")
        print(f"KILL: No universal k_prune")


if __name__ == "__main__":
    main()
