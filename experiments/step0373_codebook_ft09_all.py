#!/usr/bin/env python3
"""
Step 373 -- FT09 ALL LEVELS. 100K steps. Click-space (69 classes).
FT09 has 6 win_levels. Diagnostic: what changes after level 1?
Script: scripts/run_step373_ft09_all.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 256; NC = 64

CLICK_GRID = []
for gy in range(8):
    for gx in range(8):
        CLICK_GRID.append((gx*8+4,gy*8+4))

class CF:
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev
    def _fa(self,x,l):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self._ut()
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
        if tm.sum()==0 or si[tm].max()<self.thresh:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self._ut()
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
    print("Step 373 -- FT09 ALL LEVELS. 100K steps. Click-space.",flush=True)
    print(f"Device: {DEVICE}",flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ft09=next(g for g in games if 'ft09' in g.game_id.lower())

    n_cls=NC+5  # 64 clicks + 5 simple
    fold=CF(d=D,k=3); env=arc.make(ft09.game_id); obs=env.reset()
    action_space=env.action_space
    action6=next(a for a in action_space if a.is_complex())
    simple_actions=[a for a in action_space if not a.is_complex()]

    ts=0; go=0; lvls=0; sd=False; ls=0; lvl_events=[]; win=False
    unique_states=set(); prev_frame=None
    mx=100000

    while ts<mx and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset(); ls=ts; prev_frame=None
            if obs is None: break; continue
        if obs.state==GameState.WIN:
            win=True; print(f"    WIN at step {ts}! levels={obs.levels_completed}",flush=True); break

        pooled=ap16(obs.frame)
        unique_states.add(hash(pooled.tobytes()))
        enc=ce(pooled,fold)

        # Diagnostic: frame diff at level transition
        if prev_frame is not None and obs.levels_completed>0 and len(lvl_events)>0:
            if lvl_events[-1]['step']==ts-1:  # just completed a level
                diff=np.abs(pooled-prev_frame)
                n_changed=(diff>0.01).sum()
                print(f"    Frame diff at level {obs.levels_completed} transition:"
                      f" max={diff.max():.4f} cells_changed={n_changed}/256",flush=True)
        prev_frame=pooled.copy()

        if not sd and fold.V.shape[0]<n_cls:
            i=fold.V.shape[0]; fold._fa(enc,i)
            if i<NC:
                cx,cy=CLICK_GRID[i]; action,data=action6,{"x":cx,"y":cy}
            else:
                action,data=simple_actions[(i-NC)%len(simple_actions)],{}
            obs=env.step(action,data=data); ts+=1
            if fold.V.shape[0]>=n_cls: sd=True
            continue
        if not sd: sd=True

        c=fold.pn(enc,nc=n_cls); ol=obs.levels_completed
        if c<NC:
            cx,cy=CLICK_GRID[c]; action,data=action6,{"x":cx,"y":cy}
        else:
            action,data=simple_actions[(c-NC)%len(simple_actions)],{}
        obs=env.step(action,data=data); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed; life_s=ts-ls
            lvl_events.append({'level':lvls,'step':ts,'life_steps':life_s,'cb':fold.V.shape[0],'go':go})
            ls=ts
            print(f"    LEVEL {lvls} at step {ts} (life={life_s}) cb={fold.V.shape[0]} go={go}",flush=True)

        if ts%10000==0:
            print(f"    [step {ts:6d}] cb={fold.V.shape[0]} thresh={fold.thresh:.4f}"
                  f" unique={len(unique_states)} lvls={lvls} go={go}",flush=True)

        if obs.state==GameState.WIN:
            win=True; print(f"    WIN at step {ts}!",flush=True); break

    el=time.time()-t0
    print(flush=True); print("="*60,flush=True); print("STEP 373 SUMMARY",flush=True); print("="*60,flush=True)
    print(f"win={win} levels={lvls} steps={ts} go={go}",flush=True)
    print(f"cb_final={fold.V.shape[0]} thresh={fold.thresh:.4f}",flush=True)
    print(f"unique_states={len(unique_states)}",flush=True)
    print(f"\nLevel events:",flush=True)
    for e in lvl_events:
        print(f"  Level {e['level']}: step={e['step']} life_steps={e['life_steps']} cb={e['cb']} go={e['go']}",flush=True)
    print(f"\nElapsed: {el:.2f}s",flush=True)

if __name__=='__main__': main()
