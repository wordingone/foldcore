#!/usr/bin/env python3
"""
Step 390 -- Raw centered dot product, NO rescaling. LS20 64x64. 50K steps.

Phase 1 (200 spawns): cosine similarity.
Phase 2: centered raw dot product. No normalization. No rescaling.
alpha = 1/(1+sim). Thresh in raw-dot space.

Script: scripts/run_step388_raw_dot.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 500; THRESH_INT = 100; MEAN_BOOT = 200

class RawDotFold:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.99; self.k=k; self.d=d; self.dev=dev
        self.spawn_count=0; self.mean=None; self.phase2=False

    def _update_mean(self):
        if self.V.shape[0]>=MEAN_BOOT:
            self.mean=self.V.mean(dim=0)
            if not self.phase2: self.phase2=True

    def _fa(self,x,l):
        self.V=torch.cat([self.V,x.to(self.dev).float().unsqueeze(0)])
        self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count==MEAN_BOOT:
            self._update_mean(); self._ut()
        elif self.spawn_count%THRESH_INT==0:
            if self.phase2: self._update_mean()
            self._ut()

    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        if self.phase2 and self.mean is not None:
            Vc=self.V-self.mean.unsqueeze(0)
            s=Vc[idx]@Vc.T
        else:
            Vn=F.normalize(self.V,dim=1)
            s=Vn[idx]@Vn.T
        # Nearest neighbor (exclude self)
        for i,j in enumerate(idx): s[i,j]=-1e9
        nn=s.max(dim=1).values
        self.thresh=float(nn.median())

    def pn(self,x,nc):
        x=x.to(self.dev).float()
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0,0.0

        if self.phase2 and self.mean is not None:
            xc=x-self.mean
            Vc=self.V-self.mean.unsqueeze(0)
            si=Vc@xc
        else:
            xn=F.normalize(x,dim=0); Vn=F.normalize(self.V,dim=1)
            si=Vn@xn

        ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:nc].argmin().item(); tm=(self.labels==p)
        nsim=float(si.max().item())

        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)])
            self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count==MEAN_BOOT:
                self._update_mean(); self._ut()
            elif self.spawn_count%THRESH_INT==0:
                if self.phase2: self._update_mean()
                self._ut()
        else:
            w_i=si.argmax().item()
            alpha=1.0/(1.0+max(float(si[w_i].item()),0.0))
            self.V[w_i]=self.V[w_i]+alpha*(x-self.V[w_i])
        return p,nsim


def main():
    t0=time.time()
    print(f"Step 390 -- Raw centered dot product, no rescaling. 64x64. 50K.",flush=True)
    print(f"Device: {DEVICE}  Cap={CB_CAP}  Bootstrap={MEAN_BOOT}",flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=RawDotFold(d=D,k=3); env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False
    action_counts={}; sim_log=[]

    while ts<200000 and go<2000:
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
            if fold.V.shape[0]>=na: sd=True; fold._ut()
            continue
        if not sd: sd=True

        c,nsim=fold.pn(raw,nc=na)
        sim_log.append(nsim)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]} go={go}",flush=True)

        if ts%5000==0:
            sl=np.array(sim_log[-500:])
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            print(f"    [step {ts:6d}] cb={fold.V.shape[0]} thresh={fold.thresh:.2f}"
                  f"  phase={'DOT' if fold.phase2 else 'COS'}"
                  f"  sim: mean={sl.mean():.2f} std={sl.std():.2f} min={sl.min():.2f} max={sl.max():.2f}"
                  f"  dom={dom:.0f}%  levels={lvls} go={go}",flush=True)
            print(f"      acts: {action_counts}",flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60,flush=True)
    print("STEP 390 SUMMARY",flush=True); print("="*60,flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}",flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.2f}  phase2={fold.phase2}",flush=True)
    print(f"action_counts: {action_counts}",flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
    if sim_log:
        sl=np.array(sim_log)
        print(f"Sim: min={sl.min():.2f} max={sl.max():.2f} mean={sl.mean():.2f} std={sl.std():.2f}",flush=True)
    if lvls>0:
        print("\nPASS: Level from raw 64x64 with centered dot product!",flush=True)
    elif dom<60:
        print(f"\nMARGINAL: balanced actions ({dom:.0f}% dominant) but no level.",flush=True)
    else:
        print(f"\nKILL: {dom:.0f}% action dominance.",flush=True)
    print(f"\nElapsed: {elapsed:.2f}s",flush=True)

if __name__=='__main__': main()
