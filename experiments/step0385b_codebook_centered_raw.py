#!/usr/bin/env python3
"""
Step 385b -- Raw 64x64 + proper centering. 50K steps.

Store RAW in V. Center both V and x by codebook mean before similarity.
Recompute mean every 100 spawns. The mean = background. Signal = deviation.

50K steps on LS20. Kill: sim mean < 0.95 at 5K steps.
Script: scripts/run_step385b_centered_raw.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INT = 100

class CenteredFold:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev
        self.spawn_count=0; self.mean=torch.zeros(d,device=dev)

    def _update_mean(self):
        if self.V.shape[0]>0:
            self.mean=self.V.mean(dim=0)

    def _centered_sims(self, x, V=None):
        """Compute sims in centered+normalized space."""
        if V is None: V=self.V
        cx=F.normalize(x-self.mean,dim=0)
        cV=F.normalize(V-self.mean.unsqueeze(0),dim=1)
        return cV@cx, cx, cV

    def _fa(self,x,l):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        self.V=torch.cat([self.V,x.unsqueeze(0)])
        self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INT==0:
            self._update_mean(); self._ut()

    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        self._update_mean()
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        cV=F.normalize(self.V-self.mean.unsqueeze(0),dim=1)
        s=cV[idx]@cV.T; t=s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())

    def pn(self,x,nc):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev)
            return 0,0.0

        si,cx,cV=self._centered_sims(x)
        ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:nc].argmin().item(); tm=(self.labels==p)
        nsim=float(si.max().item())

        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)])  # store RAW
            self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0:
                self._update_mean(); self._ut()
        else:
            # Attract in raw space
            w_i=si.argmax().item()
            raw_sim=float(si[w_i].item())
            a=1.0-raw_sim
            self.V[w_i]=F.normalize(self.V[w_i]+a*(x-self.V[w_i]),dim=0)
        return p,nsim


def main():
    t0=time.time()
    print(f"Step 385b -- Raw 64x64 + proper centering. 50K steps. LS20.",flush=True)
    print(f"Device: {DEVICE}  CB cap={CB_CAP}",flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=CenteredFold(d=D,k=3); env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False
    action_counts={}; sims_log=[]

    while ts<50000 and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break; continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!",flush=True); break

        raw=torch.from_numpy(np.array(obs.frame[0],dtype=np.float32).flatten()/15.0)

        if not sd and fold.V.shape[0]<na:
            i=fold.V.shape[0]; fold._fa(raw,i)
            obs=env.step(env.action_space[i]); ts+=1
            action_counts[env.action_space[i].name]=action_counts.get(env.action_space[i].name,0)+1
            if fold.V.shape[0]>=na: sd=True; fold._update_mean(); fold._ut()
            continue
        if not sd: sd=True

        c,nsim=fold.pn(raw,nc=na)
        sims_log.append(nsim)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]} go={go}",flush=True)

        if ts%5000==0:
            avg_sim=np.mean(sims_log[-500:]) if sims_log else 0
            sim_std=np.std(sims_log[-500:]) if len(sims_log)>10 else 0
            print(f"    [step {ts:6d}] cb={fold.V.shape[0]} thresh={fold.thresh:.4f}"
                  f"  avg_sim={avg_sim:.4f} sim_std={sim_std:.4f}"
                  f"  levels={lvls} go={go}",flush=True)
            print(f"      acts: {action_counts}",flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60,flush=True)
    print("STEP 385b SUMMARY",flush=True); print("="*60,flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}",flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.4f}",flush=True)
    print(f"action_counts: {action_counts}",flush=True)
    if sims_log:
        sl=np.array(sims_log)
        print(f"Centered sim: min={sl.min():.4f} max={sl.max():.4f} mean={sl.mean():.4f} std={sl.std():.4f}",flush=True)
        print(f"Compare raw (377): mean=0.984 std=0.009",flush=True)
    if lvls>0:
        print("\nPASS: Level from raw 64x64 with self-derived centering!",flush=True)
    elif sims_log and np.mean(sims_log[-500:])<0.95:
        print("\nMARGINAL: centering improved sim but no level.",flush=True)
    else:
        print("\nKILL.",flush=True)
    print(f"\nElapsed: {elapsed:.2f}s",flush=True)

if __name__=='__main__': main()
