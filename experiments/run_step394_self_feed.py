#!/usr/bin/env python3
"""
Step 394 -- Self-feeding consolidation at 64x64. LS20. 200K steps.

Between lives: feed 10 random codebook entries through process().
Same centered unnormalized dot as Step 388.
Codebook should consolidate (shrink, merge timer noise, keep position).

Script: scripts/run_step394_self_feed.py
"""

import time, random, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INT = 100; MEAN_BOOT = 200; SELF_FEED = 10

class RawDotFold:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.99; self.k=k; self.d=d; self.dev=dev
        self.spawn_count=0; self.mean=None; self.phase2=False
        self.self_feed_self_wins=0; self.self_feed_other_wins=0

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

    def pn(self,x,nc,is_self_feed=False,src_idx=-1):
        x=x.to(self.dev).float()
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0

        if self.phase2 and self.mean is not None:
            xc=x-self.mean; Vc=self.V-self.mean.unsqueeze(0)
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
            if is_self_feed:
                if w_i==src_idx: self.self_feed_self_wins+=1
                else: self.self_feed_other_wins+=1
            alpha=1.0/(1.0+max(float(si[w_i].item()),0.0))
            self.V[w_i]=self.V[w_i]+alpha*(x-self.V[w_i])
        return p

    def do_self_feed(self, nc, steps=SELF_FEED):
        n=self.V.shape[0]
        if n<10: return
        for _ in range(steps):
            idx=random.randint(0,n-1)
            self.pn(self.V[idx].clone(), nc=nc, is_self_feed=True, src_idx=idx)


def main():
    t0=time.time()
    print(f"Step 394 -- Self-feeding consolidation. 64x64. 200K. LS20.",flush=True)
    print(f"Device: {DEVICE}  self_feed={SELF_FEED}/life",flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=RawDotFold(d=D,k=3); env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    ts=0; go=0; lvls=0; sd=False
    action_counts={}; cb_sizes=[]

    while ts<200000 and go<2000:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1
            # Self-feed between lives
            if fold.V.shape[0]>=10:
                fold.do_self_feed(nc=na)
            cb_sizes.append(fold.V.shape[0])
            obs=env.reset()
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

        c=fold.pn(raw,nc=na)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]} go={go}",flush=True)

        if ts%20000==0:
            sf_total=fold.self_feed_self_wins+fold.self_feed_other_wins
            sf_other_pct=fold.self_feed_other_wins/max(sf_total,1)*100
            dom=max(action_counts.values())/sum(action_counts.values())*100
            print(f"    [step {ts:6d}] cb={fold.V.shape[0]} thresh={fold.thresh:.2f}"
                  f"  phase={'DOT' if fold.phase2 else 'COS'}"
                  f"  sf_other={sf_other_pct:.0f}%  dom={dom:.0f}%"
                  f"  levels={lvls} go={go}",flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60,flush=True)
    print("STEP 394 SUMMARY",flush=True); print("="*60,flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}",flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.2f}  phase2={fold.phase2}",flush=True)
    print(f"action_counts: {action_counts}",flush=True)
    sf_total=fold.self_feed_self_wins+fold.self_feed_other_wins
    print(f"Self-feed: total={sf_total}  self_wins={fold.self_feed_self_wins}"
          f"  other_wins={fold.self_feed_other_wins}"
          f"  other%={fold.self_feed_other_wins/max(sf_total,1)*100:.0f}%",flush=True)
    if cb_sizes:
        print(f"cb_size evolution: start={cb_sizes[0] if cb_sizes else 0}"
              f"  mid={cb_sizes[len(cb_sizes)//2] if cb_sizes else 0}"
              f"  end={cb_sizes[-1] if cb_sizes else 0}",flush=True)
    if lvls>0:
        print("\nPASS: Level with self-feeding consolidation!",flush=True)
    else:
        print("\nKILL.",flush=True)
    print(f"\nElapsed: {elapsed:.2f}s",flush=True)

if __name__=='__main__': main()
