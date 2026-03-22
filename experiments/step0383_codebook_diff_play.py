#!/usr/bin/env python3
"""
Step 383 -- Raw 64x64 diffs, fixed thresh=0.5, 10K steps. Does it PLAY?

Diff encoding (what changed). Fixed thresh (no state-derived).
Kill: level 1 from raw pixels.
Script: scripts/run_step383_diff_play.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; FIXED_THRESH = 0.5

class CF:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=FIXED_THRESH; self.k=k; self.d=d; self.dev=dev
    def _fa(self,x,l):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
    def pn(self,x,nc):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0
        si=self.V@x; ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:nc].argmin().item(); tm=(self.labels==p)
        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
        else:
            ts=si.clone(); ts[~tm]=-float('inf'); w=ts.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return p


def main():
    t0=time.time()
    print(f"Step 383 -- Raw 64x64 diffs, thresh={FIXED_THRESH}, 10K steps. LS20.",flush=True)
    print(f"Device: {DEVICE}",flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=CF(d=D,k=3); env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False
    action_counts={}; prev_raw=None

    while ts<10000 and go<200:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset(); prev_raw=None
            if obs is None: break; continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!",flush=True); break

        raw=torch.from_numpy(np.array(obs.frame[0],dtype=np.float32).flatten()/15.0)

        if prev_raw is not None:
            diff=raw-prev_raw
            if diff.abs().max()<1e-6:
                # No frame change — still step but don't process
                prev_raw=raw.clone()
                obs=env.step(env.action_space[ts%na]); ts+=1
                continue
            enc=F.normalize(diff,dim=0)
        else:
            enc=F.normalize(raw,dim=0)

        if not sd and fold.V.shape[0]<na:
            i=fold.V.shape[0]; fold._fa(enc,i)
            action=env.action_space[i]
            action_counts[action.name]=action_counts.get(action.name,0)+1
            prev_raw=raw.clone()
            obs=env.step(action); ts+=1
            if fold.V.shape[0]>=na: sd=True
            continue
        if not sd: sd=True

        c=fold.pn(enc,nc=na)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        prev_raw=raw.clone()
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]} go={go}",flush=True)

        if ts%2000==0:
            print(f"    [step {ts:6d}] cb={fold.V.shape[0]} go={go}"
                  f"  levels={lvls} acts={action_counts}",flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60,flush=True)
    print("STEP 383 SUMMARY",flush=True); print("="*60,flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}",flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh}",flush=True)
    print(f"action_counts: {action_counts}",flush=True)
    if lvls>0:
        print("\nPASS: Level completed from raw 64x64 diffs!",flush=True)
    else:
        print(f"\nKILL: 0 levels in {ts} steps.",flush=True)
    print(f"\nElapsed: {elapsed:.2f}s",flush=True)

if __name__=='__main__': main()
