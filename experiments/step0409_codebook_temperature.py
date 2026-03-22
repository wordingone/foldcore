#!/usr/bin/env python3
"""
Step 409 -- Centered unnorm + self-tuning temperature.

Gaussian noise on sims for action selection (exploration).
Clean sims for winner+attract (learning). Temperature self-tunes
from spawn rate (target 5%).

LS20. 64x64. 50K steps.
Script: scripts/run_step409_temperature.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
from collections import deque
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INT = 100; MEAN_BOOT = 200
TARGET_SPAWN_RATE = 0.05; T_WINDOW = 1000

class TempFold:
    def __init__(self, d, nc, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.99; self.k=k; self.d=d; self.nc=nc; self.dev=dev
        self.spawn_count=0; self.mean=None; self.phase2=False
        self.temperature=0.0
        self.spawn_history=deque(maxlen=T_WINDOW)

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
        for i,j in enumerate(idx): s[i,j]=-1e9
        nn=s.max(dim=1).values
        self.thresh=float(nn.median())

    def _tune_temperature(self):
        if len(self.spawn_history)<100: return
        actual_rate=sum(self.spawn_history)/len(self.spawn_history)
        if actual_rate<TARGET_SPAWN_RATE:
            self.temperature*=1.01
            if self.temperature<0.01: self.temperature=0.01
        else:
            self.temperature*=0.99

    def process(self, x):
        x=x.to(self.dev).float()
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev)
            return 0

        # Clean sims (for learning)
        if self.phase2 and self.mean is not None:
            xc=x-self.mean; Vc=self.V-self.mean.unsqueeze(0)
            si_clean=Vc@xc
        else:
            xn=F.normalize(x,dim=0); Vn=F.normalize(self.V,dim=1)
            si_clean=Vn@xn

        # Noisy sims (for action selection)
        if self.temperature>0:
            noise=self.temperature*torch.randn_like(si_clean)
            si_noisy=si_clean+noise
        else:
            si_noisy=si_clean

        # Action selection: argmin on noisy class votes
        ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,self.nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si_noisy[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        action=sc[:self.nc].argmin().item()

        # Spawn/attract using CLEAN sims
        p=action; tm=(self.labels==p)
        spawned=False
        if (tm.sum()==0 or si_clean[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)])
            self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1; spawned=True
            if self.spawn_count==MEAN_BOOT:
                self._update_mean(); self._ut()
            elif self.spawn_count%THRESH_INT==0:
                if self.phase2: self._update_mean()
                self._ut()
        else:
            w_i=int(si_clean.argmax().item())
            alpha=1.0/(1.0+max(float(si_clean[w_i].item()),0.0))
            self.V[w_i]=self.V[w_i]+alpha*(x-self.V[w_i])

        self.spawn_history.append(1 if spawned else 0)
        self._tune_temperature()

        return action


def main():
    t0=time.time()
    print(f"Step 409 -- Self-tuning temperature. 64x64. 50K. LS20.", flush=True)
    print(f"Device: {DEVICE}  target_spawn={TARGET_SPAWN_RATE}  cb_cap={CB_CAP}", flush=True)
    print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    fold=TempFold(d=D, nc=na, k=3)
    ts=0; go=0; lvls=0; sd=False
    action_counts={}

    while ts<50000 and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break
            continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!", flush=True); break

        raw=torch.from_numpy(np.array(obs.frame[0],dtype=np.float32).flatten()/15.0)

        if not sd and fold.V.shape[0]<na:
            i=fold.V.shape[0]; fold._fa(raw,i)
            obs=env.step(env.action_space[i]); ts+=1
            action_counts[env.action_space[i].name]=action_counts.get(env.action_space[i].name,0)+1
            if fold.V.shape[0]>=na: sd=True; fold._ut()
            continue
        if not sd: sd=True

        c=fold.process(raw)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]}"
                  f"  T={fold.temperature:.2f} go={go}", flush=True)

        if ts%5000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            sr=sum(fold.spawn_history)/max(len(fold.spawn_history),1)
            print(f"    [step {ts:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.2f}"
                  f"  T={fold.temperature:.2f}  spawn_rate={sr:.3f}"
                  f"  phase={'DOT' if fold.phase2 else 'COS'}"
                  f"  dom={dom:.0f}%  levels={lvls} go={go}", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 409 SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    print(f"cb={fold.V.shape[0]}  thresh={fold.thresh:.2f}  T={fold.temperature:.2f}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
    if lvls>0:
        print(f"\nPASS: Level with self-tuned temperature at 64x64!", flush=True)
    elif dom<60:
        print(f"\nMARGINAL: balanced ({dom:.0f}%) but no level.", flush=True)
    else:
        print(f"\nKILL: {dom:.0f}% dominance.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
