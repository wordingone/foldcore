"""
Step 610: VC33 full chain benchmark.

Tests the complete 7-level solution chain for VC33.
Runs 5 times to confirm determinism.

SOLUTIONS[0-6] — all verified deterministic solutions.

Level 7 (SOLUTIONS[6]) mechanics:
  Level 7: 48x48 grid, vertical mode (TiD=[0,-2]), camera offset +8.
  Win: ChX.y=18(keB), PPS.y=10(khL), VAJ.y=42(khL)
  Solved via analytical BFS (3.47s, 49 clicks, no NOPs needed).
"""
import sys
sys.path.insert(0, 'B:/M/the-search/experiments/environment_files/vc33/9851e02b')
sys.path.insert(0, 'B:/M/the-search/experiments')
import logging
logging.getLogger().setLevel(logging.WARNING)
import time

# ---- Level solutions (display coordinates) ----

GA=(0,27); GB=(0,33); GC=(24,33); GD=(24,27)
S1=(6,30); S2=(30,30); NOP_L5=(60,60)

# Level 6 (LEVELS[5]) — horizontal, BFS solution with (60,60) NOP
SOLN_5 = [GA,GD,GD,GD,S1,NOP_L5,GB,GB,GC,GC,GC,GC,GC,GC,S2,NOP_L5,GD,GD,GD,GD,GD,GD]

# Level 7 (LEVELS[6]) — vertical (TiD=[0,-2]), camera offset +8, analytical BFS
# Grid actions + 8 offset:  HMp->RmM=(24,8)  RmM->HMp=(20,8)  wmR->RmM=(24,32)
#   RmM->wmR=(20,32)  AEF->RmM=(38,32)  RmM->AEF=(42,32)  HfU->RmM=(38,8)  RmM->HfU=(42,8)
#   SW_WMRM=(22,38)  SW_RMHF=(40,16)  SW_RMAEF=(40,38)
HM=(24,8); MH=(20,8); WM=(24,32); MW=(20,32)
AM=(38,32); MA=(42,32); FM=(38,8); MF=(42,8)
SW1=(22,38); SW2=(40,16); SW3=(40,38)

SOLN_6 = (
    [HM]*10 +       # 10x HMp->RmM: RmM.h 8->28, HMp.h 20->0
    [WM, SW1] +     # wmR->RmM (RmM.h 28->30, wmR.h 8->6), SW1 fires (wmR.utq=wmR.y+6=30=RmM.utq): ChX->RmM
    [MH]*10 +       # 10x RmM->HMp: RmM.h 30->10 (ChX moves up), HMp.h 0->20
    [MF, SW2] +     # RmM->HfU (RmM.h 10->8, HfU.h 6->8), SW2 fires (RmM.utq=HfU.utq=8): VAJ->RmM
    [HM]*10 +       # 10x HMp->RmM: RmM.h 8->28, HMp.h 20->0
    [AM, AM] +      # 2x AEF->RmM: RmM.h 28->32... wait need RmM.h=30 for SW1,SW3
    # Actually: after HM*10: RmM.h=28. Need RmM.h=30 for next SW.
    # AM (AEF->RmM): RmM.h 28->30, AEF.h 10->8... need AEF.h=6 for SW3
    # One more: AM again? AEF.h 8->6, RmM.h 30->32. But SW1 needs RmM.h=30 not 32.
    # BFS found: [AM,AM, MF, SW1, SW3, MW*6, MF*4]
    # Let me use the exact BFS sequence:
    [MF] +          # RmM->HfU
    [SW1] +         # SW1 fires (wmR.utq=? RmM.utq=?)
    [SW3] +         # SW3 fires
    [MW]*6 +        # 6x RmM->wmR
    [MF]*4          # 4x RmM->HfU
)

# Exact BFS solution (verified: level advances at step 48/0-indexed = click 49)
SOLN_6_EXACT = [
    (24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),  # 10x HMp->RmM
    (24,32),  # wmR->RmM
    (22,38),  # SW1 (wmR<->RmM): ChX moves to RmM
    (20,8),(20,8),(20,8),(20,8),(20,8),(20,8),(20,8),(20,8),(20,8),(20,8),  # 10x RmM->HMp
    (42,8),   # RmM->HfU
    (40,16),  # SW2 (RmM<->HfU): VAJ moves to RmM
    (24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),  # 10x HMp->RmM
    (38,32),(38,32),  # 2x AEF->RmM
    (42,8),   # RmM->HfU
    (22,38),  # SW1 fires again (wmR<->RmM): moves between wmR and RmM
    (40,38),  # SW3 (RmM<->AEF): PPS moves to RmM
    (20,32),(20,32),(20,32),(20,32),(20,32),(20,32),  # 6x RmM->wmR: ChX.y decreases
    (42,8),(42,8),(42,8),(42,8),  # 4x RmM->HfU: final adjustments
]

SOLUTIONS = {
    0: [(62,34),(62,34),(62,34)],
    1: [(0,24),(0,24),(0,44),(0,44),(0,44),(0,44),(0,44)],
    2: [(12,56),(24,56),(12,56),(24,56),(12,56),(34,56),(24,56),(12,56),(34,56),(24,56),(12,56),(34,56),(24,56),(12,56),
        (46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56)],
    3: [(15,61),(15,61),(12,43),(32,32),(15,61),(15,61),(15,61),
        (39,61),(39,61),(51,61),(39,61),(27,34),(32,32),
        (51,61),(39,61),(51,61),(39,61),(51,61),(39,61),(51,61),(39,61),(51,61),(39,61)],
    4: [(61,17),(61,17),(61,17),(61,17),(61,35),(61,35),(61,35),(61,35),(61,35),(61,52),(61,52),(25,49),(32,32),
        (61,29),(61,29),(61,29),(61,52),(61,52),(40,32),(32,32),
        (61,17),(61,17),(61,17),(61,17),(28,14),(32,32),
        (61,11),(61,11),(61,11),(61,11),(40,32),(32,32),
        (61,11),(61,35),(61,35),(61,35),(61,46),(61,46),(25,49),(32,32),
        (61,29),(61,11),(61,52),(61,52),(61,52),(61,52),(61,52),(61,52),(61,52)],
    5: SOLN_5,
    6: SOLN_6_EXACT,
}


def run_full_chain(env, action6, seed=None):
    """Run the full 7-level chain. Returns (success, levels_completed, total_clicks)."""
    from arcengine import GameState

    obs = env.reset()
    game = getattr(env, 'game', None) or getattr(env, '_game', None)
    total_clicks = 0

    for lvl in range(7):
        soln = SOLUTIONS[lvl]
        level_clicks = 0
        for cx, cy in soln:
            obs = env.step(action6, data={'x': cx, 'y': cy})
            total_clicks += 1
            level_clicks += 1

            if obs.state == GameState.WIN:
                return True, obs.levels_completed, total_clicks, 'WIN'
            if obs.state == GameState.GAME_OVER:
                return False, obs.levels_completed, total_clicks, f'GAME_OVER at L{lvl} click {level_clicks}'
            if obs.levels_completed > lvl:
                break  # Level cleared, move to next

        else:
            # Ran out of solution clicks without advancing
            if obs.levels_completed <= lvl:
                return False, obs.levels_completed, total_clicks, f'STUCK at L{lvl} after {level_clicks} clicks'

    # If we get here without WIN, check final state
    if obs.state == GameState.WIN:
        return True, obs.levels_completed, total_clicks, 'WIN'
    return False, obs.levels_completed, total_clicks, f'NO_WIN (levels={obs.levels_completed})'


def main():
    import arc_agi

    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    if vc33 is None:
        print("VC33 not found"); return

    env = arc.make(vc33.game_id)
    action6 = env.action_space[0]

    print("=== Step 610: VC33 Full Chain Benchmark ===\n")
    print(f"Solution lengths: {[len(SOLUTIONS[i]) for i in range(7)]}")
    print(f"Total clicks: {sum(len(SOLUTIONS[i]) for i in range(7))}\n")

    n_runs = 5
    results = []
    t_start = time.time()

    for run in range(n_runs):
        t0 = time.time()
        success, levels, clicks, status = run_full_chain(env, action6, seed=run)
        elapsed = time.time() - t0
        results.append((success, levels, clicks, status))
        print(f"Run {run}: success={success}, levels={levels}, clicks={clicks}, "
              f"status={status} ({elapsed:.2f}s)")

    total_elapsed = time.time() - t_start
    wins = sum(1 for r in results if r[0])
    print(f"\n{'='*50}")
    print(f"Results: {wins}/{n_runs} wins")
    print(f"Total time: {total_elapsed:.2f}s")

    if wins == n_runs:
        print("\nRESULT: DETERMINISTIC WIN — VC33 full chain SOLVED")
        print(f"\nSOLUTIONS summary:")
        for i in range(7):
            print(f"  SOLUTIONS[{i}]: {len(SOLUTIONS[i])} clicks")
        print(f"\nSOLUTIONS[6] = {SOLUTIONS[6]}")
    else:
        print(f"\nRESULT: PARTIAL ({wins}/{n_runs} wins)")


if __name__ == "__main__":
    main()
