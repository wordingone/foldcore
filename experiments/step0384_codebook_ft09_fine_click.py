#!/usr/bin/env python3
"""
Step 384 -- FT09 level 2 with 256 click regions (16x16 grid). 50K steps.

261 classes (256 click + 5 simple). Codebook reset on level transition.
16x16 centered cosine. 10K cap, sparse thresh.
Script: scripts/run_step384_ft09_fine_click.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 256; CB_CAP = 10000; THRESH_INTERVAL = 100

CLICK_GRID = [(gx*4+2, gy*4+2) for gy in range(16) for gx in range(16)]
NC = len(CLICK_GRID)  # 256

class CF:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev; self.spawn_count=0
    def reset_cb(self):
        self.V=torch.zeros(0,self.d,device=self.dev)
        self.labels=torch.zeros(0,dtype=torch.long,device=self.dev)
        self.thresh=0.7; self.spawn_count=0
    def _fa(self,x,l):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INTERVAL==0: self._ut()
    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        s=self.V[idx]@self.V.T; t=s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())
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
            self.spawn_count+=1
            if self.spawn_count%THRESH_INTERVAL==0: self._ut()
        else:
            ts=si.clone(); ts[~tm]=-float('inf'); w=ts.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return p

def ap16(f):
    a=np.array(f[0],dtype=np.float32)/15.0; return a.reshape(16,4,16,4).mean(axis=(1,3)).flatten()

def ce(p,fold):
    t=F.normalize(torch.from_numpy(p.astype(np.float32)),dim=0)
    if fold.V.shape[0]>2: t=t-fold.V.mean(dim=0).cpu()
    return t

def main():
    t0=time.time()
    n_cls=NC+5
    print(f"Step 384 -- FT09 fine click (16x16={NC} regions + 5 simple = {n_cls} classes). 50K steps.",flush=True)
    print(f"Device: {DEVICE}",flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ft09=next(g for g in games if 'ft09' in g.game_id.lower())

    fold=CF(d=D,k=3); env=arc.make(ft09.game_id); obs=env.reset()
    as_=env.action_space; a6=next(a for a in as_ if a.is_complex())
    sa=[a for a in as_ if not a.is_complex()]

    ts=0; go=0; lvls=0; sd=False; ls=0; evts=[]

    while ts<50000 and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset(); ls=ts; sd=False
            if obs is None: break; continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}! levels={obs.levels_completed}",flush=True); break

        enc=ce(ap16(obs.frame),fold)
        if not sd and fold.V.shape[0]<n_cls:
            i=fold.V.shape[0]; fold._fa(enc,i)
            if i<NC: cx,cy=CLICK_GRID[i]; act,d_=a6,{"x":cx,"y":cy}
            else: act,d_=sa[(i-NC)%len(sa)],{}
            obs=env.step(act,data=d_); ts+=1
            if fold.V.shape[0]>=n_cls: sd=True; fold._ut()
            continue
        if not sd: sd=True

        c=fold.pn(enc,nc=n_cls); ol=obs.levels_completed
        if c<NC: cx,cy=CLICK_GRID[c]; act,d_=a6,{"x":cx,"y":cy}
        else: act,d_=sa[(c-NC)%len(sa)],{}
        obs=env.step(act,data=d_); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed; life_s=ts-ls
            evts.append({'level':lvls,'step':ts,'life':life_s,'go':go})
            ls=ts; fold.reset_cb(); sd=False
            print(f"    LEVEL {lvls} at step {ts} (life={life_s}) go={go} -> RESET",flush=True)

        if ts%10000==0:
            print(f"    [step {ts:6d}] cb={fold.V.shape[0]} lvls={lvls} go={go}",flush=True)

        if obs.state==GameState.WIN:
            print(f"    WIN!",flush=True); break

    elapsed=time.time()-t0
    print(flush=True); print("="*60,flush=True)
    print("STEP 384 SUMMARY",flush=True); print("="*60,flush=True)
    print(f"levels={lvls}  steps={ts}  go={go}",flush=True)
    print(f"cb_final={fold.V.shape[0]}",flush=True)
    for e in evts:
        print(f"  Level {e['level']}: step={e['step']} life={e['life']} go={e['go']}",flush=True)
    if lvls>=2:
        print("\nPASS: FT09 level 2+ reached with fine click grid!",flush=True)
    print(f"\nElapsed: {elapsed:.2f}s",flush=True)

if __name__=='__main__': main()
