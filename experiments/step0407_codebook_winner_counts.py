#!/usr/bin/env python3
"""
Step 407 -- Winner-identity encoding. Discrete state. Count-based exploration.

Winner index from centered unnormalized dot = discrete position.
Action selection: least-taken action from this winner's state.
Epsilon-greedy on a discrete state space derived from the codebook.

LS20. 64x64. 50K steps.
Script: scripts/run_step407_winner_counts.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INT = 100; MEAN_BOOT = 200

class WinnerCountFold:
    def __init__(self, d, nc, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.99; self.k=k; self.d=d; self.nc=nc; self.dev=dev
        self.spawn_count=0; self.mean=None; self.phase2=False
        # Count-based action selection
        self.win_action_counts={}  # {winner_idx: [count_per_action]}
        self.unique_winners=set()

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

    def process(self, x):
        """Returns action index using count-based selection on winner identity."""
        x=x.to(self.dev).float()
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev)
            return 0

        # Compute sims (centered unnormalized or cosine)
        if self.phase2 and self.mean is not None:
            xc=x-self.mean; Vc=self.V-self.mean.unsqueeze(0)
            si=Vc@xc
        else:
            xn=F.normalize(x,dim=0); Vn=F.normalize(self.V,dim=1)
            si=Vn@xn

        # Find winner (global argmax)
        w_i=int(si.argmax().item())
        self.unique_winners.add(w_i)

        # Count-based action selection: least-taken action from this winner
        if w_i not in self.win_action_counts:
            self.win_action_counts[w_i]=[0]*self.nc
        counts=self.win_action_counts[w_i]
        action=min(range(self.nc), key=lambda a: counts[a])

        # Record
        self.win_action_counts[w_i][action]+=1

        # Standard spawn/attract (using class vote for labeling)
        ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,self.nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        p=sc[:self.nc].argmin().item(); tm=(self.labels==p)

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
            alpha=1.0/(1.0+max(float(si[w_i].item()),0.0))
            self.V[w_i]=self.V[w_i]+alpha*(x-self.V[w_i])

        return action


def main():
    t0=time.time()
    print(f"Step 407 -- Winner-identity count-based exploration. 64x64. 50K. LS20.", flush=True)
    print(f"Device: {DEVICE}  mean_boot={MEAN_BOOT}  cb_cap={CB_CAP}", flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    fold=WinnerCountFold(d=D, nc=na, k=3)
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
                  f"  unique_winners={len(fold.unique_winners)} go={go}", flush=True)

        if ts%5000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            # Winner stability: how many unique winners in last 500 steps
            n_uw=len(fold.unique_winners)
            # Count balance across winners
            bal_scores=[]
            for w,counts in fold.win_action_counts.items():
                if sum(counts)>3:
                    mx=max(counts); mn=min(counts)
                    bal_scores.append(mx/(sum(counts)/na))
            avg_bal=np.mean(bal_scores) if bal_scores else 0
            print(f"    [step {ts:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.2f}"
                  f"  phase={'DOT' if fold.phase2 else 'COS'}"
                  f"  unique_winners={n_uw}  avg_balance={avg_bal:.2f}"
                  f"  dom={dom:.0f}%  levels={lvls} go={go}", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 407 SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    print(f"cb={fold.V.shape[0]}  thresh={fold.thresh:.2f}  phase2={fold.phase2}", flush=True)
    print(f"unique_winners={len(fold.unique_winners)}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0

    # Winner count analysis
    total_visits=sum(sum(c) for c in fold.win_action_counts.values())
    active_winners=sum(1 for c in fold.win_action_counts.values() if sum(c)>5)
    print(f"active_winners(>5 visits)={active_winners}  total_visits={total_visits}", flush=True)

    # Per-action balance across active winners
    action_totals=[0]*na
    for counts in fold.win_action_counts.values():
        for a in range(na):
            action_totals[a]+=counts[a]
    print(f"action_totals: {action_totals}", flush=True)

    if lvls>0:
        print(f"\nPASS: Level with winner-count exploration at 64x64!", flush=True)
    elif dom<60:
        print(f"\nMARGINAL: balanced ({dom:.0f}%) but no level.", flush=True)
    else:
        print(f"\nKILL: {dom:.0f}% dominance.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
