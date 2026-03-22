#!/usr/bin/env python3
"""
Step 400 -- Change-rate weighted similarity at 64x64.

Track per-dimension change frequency. Weight similarity by rate*(1-rate):
timer (rate~1) -> weight~0. Background (rate=0) -> weight=0.
Sprite (rate~0.04) -> peak weight. Weighted cosine in effective ~300 dims.

LS20. 50K steps.
Script: scripts/run_step400_changeweight.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 10000; THRESH_INT = 100; WARMUP = 100; EPS = 0.01

class WeightedFold:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev; self.spawn_count=0
        self.x_prev=None; self.change_count=torch.zeros(d,device=dev)
        self.frame_count=0; self.weight=None

    def _update_weight(self, x):
        x=x.to(self.dev).float()
        if self.x_prev is not None:
            changed=(torch.abs(x-self.x_prev)>EPS).float()
            self.change_count+=changed
        self.frame_count+=1
        self.x_prev=x.clone()
        if self.frame_count>=WARMUP:
            rate=self.change_count/self.frame_count
            w=rate*(1.0-rate)
            s=w.sum()
            self.weight=w/(s+1e-8) if s>0 else None

    def _weighted_normalize(self, x):
        """Apply weight and normalize."""
        if self.weight is not None:
            xw=x*torch.sqrt(self.weight)
        else:
            xw=x
        return F.normalize(xw, dim=-1 if xw.dim()==1 else -1)

    def _fa(self,x,l):
        self._update_weight(x)
        self.V=torch.cat([self.V,x.to(self.dev).float().unsqueeze(0)])
        self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INT==0: self._ut()

    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        if self.weight is not None:
            sw=torch.sqrt(self.weight).unsqueeze(0)
            Vw=F.normalize(self.V*sw, dim=1)
            s=Vw[idx]@Vw.T
        else:
            Vn=F.normalize(self.V,dim=1)
            s=Vn[idx]@Vn.T
        t=s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())

    def pn(self,x,nc):
        x=x.to(self.dev).float()
        self._update_weight(x)

        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0

        # Weighted cosine similarity
        if self.weight is not None:
            sw=torch.sqrt(self.weight)
            xw=F.normalize(x*sw, dim=0)
            Vw=F.normalize(self.V*sw.unsqueeze(0), dim=1)
            si=Vw@xw
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
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            ts_=si.clone(); ts_[~tm]=-float('inf'); w_i=ts_.argmax().item()
            a=1.0-float(si[w_i].item())
            self.V[w_i]=self.V[w_i]+a*(x-self.V[w_i])
        return p


def main():
    t0=time.time()
    print(f"Step 400 -- Change-rate weighted cosine. 64x64. 50K. LS20.", flush=True)
    print(f"Device: {DEVICE}  warmup={WARMUP}  eps={EPS}", flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    fold=WeightedFold(d=D,k=3); env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

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

        c=fold.pn(raw,nc=na)
        action=env.action_space[c%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            print(f"    LEVEL {lvls} at step {ts} cb={fold.V.shape[0]} go={go}", flush=True)

        if ts%5000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            w=fold.weight
            if w is not None:
                w_np=w.cpu().numpy()
                n_active=(w_np>0.001).sum()
                top_idx=np.argsort(w_np)[-10:][::-1]
                rate_np=(fold.change_count/fold.frame_count).cpu().numpy()
                print(f"    [step {ts:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.3f}"
                      f"  dom={dom:.0f}%  levels={lvls} go={go}", flush=True)
                print(f"      weight: active(>0.001)={n_active}  max={w_np.max():.4f}"
                      f"  sum_top10={w_np[top_idx].sum():.3f}", flush=True)
                print(f"      top dims: {list(top_idx)}", flush=True)
                print(f"      top rates: [{', '.join(f'{rate_np[i]:.3f}' for i in top_idx)}]", flush=True)
            else:
                print(f"    [step {ts:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.3f}"
                      f"  dom={dom:.0f}%  levels={lvls} go={go} (no weight yet)", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 400 SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    print(f"cb={fold.V.shape[0]}  thresh={fold.thresh:.3f}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
    w=fold.weight
    if w is not None:
        w_np=w.cpu().numpy()
        rate_np=(fold.change_count/fold.frame_count).cpu().numpy()
        print(f"weight: active(>0.001)={(w_np>0.001).sum()}"
              f"  active(>0.01)={(w_np>0.01).sum()}", flush=True)
        print(f"rate distribution: min={rate_np.min():.4f} max={rate_np.max():.4f}"
              f"  mean={rate_np.mean():.4f} std={rate_np.std():.4f}", flush=True)
        # Rate histogram
        for lo,hi,label in [(0,0.01,'bg'),(0.01,0.1,'sprite'),(0.1,0.5,'active'),(0.5,1.01,'timer')]:
            n=((rate_np>=lo)&(rate_np<hi)).sum()
            print(f"  rate [{lo:.2f},{hi:.2f}): {n} dims ({n/len(rate_np)*100:.1f}%)", flush=True)
    if lvls>0:
        print(f"\nPASS: Level with change-rate weighted cosine at 64x64!", flush=True)
    elif dom<60:
        print(f"\nMARGINAL: balanced ({dom:.0f}%) but no level.", flush=True)
    else:
        print(f"\nKILL: {dom:.0f}% action dominance.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
