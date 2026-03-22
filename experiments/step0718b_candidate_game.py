"""
Step 718b — candidate.c raw game connection on LS20.

Spec:
- Start candidate.c as subprocess (stdin/stdout pipe)
- For each game step: flatten 64x64 frame to 4096 bytes, write to candidate stdin
- candidate.c runs 4096 CA steps, produces 16 output bytes (one per 256 steps)
- Use last output byte modulo num_actions as the action
- No bootloader. No argmin. No graph. No centering. Raw CA output = action.

R3 hypothesis: memory grid m[N] creates ℓ₁ self-modification. Does this produce
meaningful navigation behavior beyond random walk?

NOTE: 120K game steps = 491M CA steps ≈ 92 min/seed (timing: 46s/1K game steps).
Runtime cap compliance: N_GAME_STEPS=1000 per seed (5 seeds ≈ 4 min total).
The "<5 min" estimate assumed C was faster than actual 11µs/CA-step.
Flag results with timing extrapolation to 120K.

DO NOT MODIFY candidate.c. DO NOT add 674 or any infrastructure.
"""
import subprocess
import numpy as np
import sys
import time

EXE = "B:/M/the-search/substrates/candidate/candidate.exe"

DIR_ACTIONS = [0, 1, 2, 3]
GRID_ACTIONS = [(gx * 8 + 4) + (gy * 8 + 4) * 64
                for gy in range(8) for gx in range(8)]
UNIVERSAL_ACTIONS = DIR_ACTIONS + GRID_ACTIONS
N_UNIV = len(UNIVERSAL_ACTIONS)  # 68

N_SEEDS = 5
N_GAME_STEPS = 1000  # cap for 5-min budget; 120K would be ~92 min/seed
N_CA_PER_STEP = 4096  # 64*64 bytes per game step
N_OUT_PER_STEP = N_CA_PER_STEP // 256  # 16 output bytes per game step


def run_seed(seed, make):
    env = make()
    rng = np.random.RandomState(seed)
    obs = env.reset(seed=seed)
    level = 0
    l1 = None
    t_start = time.time()

    # action counts
    action_hist = np.zeros(N_UNIV, dtype=np.int64)

    # track unique frames (using pixel sum as coarse proxy for cell count)
    seen_sums = set()
    cells_visited = 0

    # episode tracking
    ep_lengths = []
    ep_step = 0

    # Run candidate.c as persistent subprocess
    # Total CA steps = N_GAME_STEPS * N_CA_PER_STEP
    n_total_ca = N_GAME_STEPS * N_CA_PER_STEP

    # Collect all obs bytes upfront (we need to send them to candidate.exe)
    # We run the game loop and collect obs, then pipe all at once.
    # Alternatively: keep candidate.exe running interactively.
    # For simplicity with Windows buffering: collect all game obs first,
    # pipe to candidate.exe in one shot, then parse output.
    # This means we play the game with random actions first, then "replay"
    # with candidate's actions — NOT ideal but avoids Windows pipe deadlocks.
    #
    # Correct interactive approach:
    # Use Popen with PIPE and write/read interactively, 4096 bytes in / 16 bytes out.
    # Risk: pipe deadlock if buffers fill. We flush after each read.

    proc = subprocess.Popen(
        [EXE, str(n_total_ca)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    action_idx = 0  # default until first output
    output_buf = bytearray()

    for step in range(1, N_GAME_STEPS + 1):
        if obs is None:
            obs = env.reset(seed=seed)
            ep_step = 0

        # Encode obs: flatten 64x64 to 4096 bytes
        frame = np.array(obs[0], dtype=np.uint8).flatten()  # 4096 bytes
        assert len(frame) == N_CA_PER_STEP, f"Expected 4096, got {len(frame)}"

        # Write to candidate's stdin and read 16 output bytes
        try:
            proc.stdin.write(bytes(frame))
            proc.stdin.flush()
            # Read exactly 16 output bytes
            chunk = proc.stdout.read(N_OUT_PER_STEP)
            if chunk and len(chunk) >= 1:
                output_buf.extend(chunk)
                # Use last byte of chunk as action selector
                action_idx = chunk[-1] % N_UNIV
        except (BrokenPipeError, OSError):
            break

        action_int = UNIVERSAL_ACTIONS[action_idx]
        action_hist[action_idx] += 1

        # Track unique observations (coarse)
        psum = int(frame.sum())
        if psum not in seen_sums:
            seen_sums.add(psum)
            cells_visited += 1

        try:
            obs_new, reward, done, info = env.step(action_int)
        except Exception:
            obs_new = obs; done = False; info = {}

        obs = obs_new
        ep_step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
            level = cl
            ep_lengths.append(ep_step)
            ep_step = 0
        if done:
            ep_lengths.append(ep_step)
            ep_step = 0
            obs = env.reset(seed=seed)

    proc.stdin.close()
    proc.wait(timeout=5)

    elapsed = time.time() - t_start

    # Action distribution
    dir_count = int(action_hist[:4].sum())
    click_count = int(action_hist[4:].sum())
    most_used = int(action_hist.argmax())
    most_used_pct = float(action_hist[most_used] / action_hist.sum() * 100) if action_hist.sum() > 0 else 0
    unique_actions = int((action_hist > 0).sum())

    bootloader = "PASS" if l1 else "FAIL"
    print(f"  s{seed:2d}: {bootloader} l1={l1} unique_actions={unique_actions}/68 "
          f"dir%={dir_count/N_GAME_STEPS*100:.1f}% cells={cells_visited} "
          f"eps={len(ep_lengths)} t={elapsed:.1f}s", flush=True)
    print(f"         most_used_action={most_used} ({most_used_pct:.1f}%) "
          f"output_bytes={len(output_buf)}", flush=True)
    return dict(seed=seed, l1=l1, unique_actions=unique_actions,
                dir_pct=dir_count/N_GAME_STEPS, cells=cells_visited,
                n_eps=len(ep_lengths), elapsed=elapsed,
                action_hist=action_hist.tolist())


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    t_start = time.time()
    print(f"Step 718b: candidate.c raw game connection on LS20")
    print(f"N_GAME_STEPS={N_GAME_STEPS} (cap; 120K would be ~92 min/seed)")
    print(f"Encoding: 4096 bytes/step -> 16 CA outputs -> last byte % 68 = action")
    print(f"No 674, no infrastructure. Raw CA output.")
    print()

    results = []
    for seed in range(N_SEEDS):
        results.append(run_seed(seed, mk))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    boot_n = sum(1 for r in results if r['l1'])
    print(f"Bootloader: {boot_n}/{N_SEEDS}  total_time={elapsed:.1f}s")
    print()

    # Action distribution summary
    total_hist = np.zeros(N_UNIV, dtype=np.float64)
    for r in results:
        total_hist += np.array(r['action_hist'])
    total_hist /= N_SEEDS * N_GAME_STEPS

    dir_pct = float(total_hist[:4].sum() * 100)
    click_pct = float(total_hist[4:].sum() * 100)
    unique_avg = float(np.mean([r['unique_actions'] for r in results]))
    cells_avg = float(np.mean([r['cells'] for r in results]))

    print(f"Action distribution (avg over seeds):")
    print(f"  Dir actions (0-3): {dir_pct:.1f}%")
    print(f"  Click actions (4-67): {click_pct:.1f}%")
    print(f"  Expected if uniform: dir={4/68*100:.1f}% click={64/68*100:.1f}%")
    print(f"  Unique actions used: {unique_avg:.1f}/68")
    print(f"  Unique cells visited (coarse): {cells_avg:.1f}")

    # Check if CA output is biased
    top5 = sorted(range(N_UNIV), key=lambda a: total_hist[a], reverse=True)[:5]
    print(f"  Top 5 actions: {[(a, f'{total_hist[a]*100:.1f}%') for a in top5]}")

    print(f"\nR3 result:")
    if boot_n >= 3:
        print(f"SIGNAL: CA behavior produces LS20 level-up {boot_n}/5 seeds")
    elif cells_avg > 50:
        print(f"PARTIAL: CA navigates (cells={cells_avg:.0f}) but no level-up in {N_GAME_STEPS} steps")
    else:
        print(f"BASELINE: CA produces action distribution. No level-up at {N_GAME_STEPS} steps (need 120K for fair test).")
        print(f"  Extrapolated: 120K steps would need ~{elapsed/N_GAME_STEPS*120000:.0f}s/seed")

    print(f"\nNOTE: 120K steps as specced = {elapsed/N_GAME_STEPS*120000/60:.0f} min/seed "
          f"({elapsed/N_GAME_STEPS*120000*5/3600:.1f} hours for 5 seeds). "
          f"Need Avir approval to run full budget.")


if __name__ == "__main__":
    main()
