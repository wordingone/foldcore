#!/usr/bin/env python3
"""
Step 411 -- process() evolving its own metric. Random mutation + natural selection.

Weight vector w modulates similarity. Between lives, mutate w. Keep if
life was longer. The substrate discovers its own Goldilocks encoding.

LS20. 64x64. 500K steps.
Script: scripts/run_step411_evolve_metric.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = 4096; CB_CAP = 5000; THRESH_INT = 100; MUTATE_FRAC = 0.1; MUTATE_SCALE = 0.1

class EvolvingFold:
    def __init__(self, d, nc, k=3, dev=DEVICE):
        self.d=d; self.nc=nc; self.k=k; self.dev=dev
        self.w=torch.ones(d,device=dev)
        self.w_best=self.w.clone()
        self.best_score=0.0; self.life_actions=[]
        self.generation=0
        self._reset_cb()

    def _reset_cb(self):
        self.V=torch.zeros(0,self.d,device=self.dev)
        self.labels=torch.zeros(0,dtype=torch.long,device=self.dev)
        self.thresh=0.7; self.spawn_count=0

    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        Vw=F.normalize(self.V*self.w.unsqueeze(0), dim=1)
        s=Vw[idx]@Vw.T
        t=s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())

    def _fa(self,x,l):
        self.V=torch.cat([self.V,x.to(self.dev).float().unsqueeze(0)])
        self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INT==0: self._ut()

    def process(self, x):
        x=x.to(self.dev).float()

        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev); return 0

        xw=F.normalize(x*self.w, dim=0)
        Vw=F.normalize(self.V*self.w.unsqueeze(0), dim=1)
        si=Vw@xw

        ac=int(self.labels.max().item())+1
        sc=torch.zeros(max(ac,self.nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc[c]=cs.topk(min(self.k,len(cs))).values.sum()
        action=sc[:self.nc].argmin().item()
        p=action; tm=(self.labels==p)

        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)])
            self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            w_i=int(si.argmax().item())
            a=1.0-float(si[w_i].item())
            self.V[w_i]=self.V[w_i]+a*(x-self.V[w_i])

        self.life_actions.append(action)
        return action

    def _action_entropy(self):
        """Compute action entropy for this life."""
        import math
        if not self.life_actions: return 0.0
        n=len(self.life_actions)
        counts={}
        for a in self.life_actions:
            counts[a]=counts.get(a,0)+1
        ent=0.0
        for c in counts.values():
            if c>0:
                p=c/n; ent-=p*math.log2(p)
        return ent

    def on_game_over(self):
        fitness=self._action_entropy()
        if fitness>self.best_score:
            self.best_score=fitness
            self.w_best=self.w.clone()
        else:
            self.w=self.w_best.clone()

        # Mutate
        mask=(torch.rand(self.d,device=self.dev)<MUTATE_FRAC).float()
        perturbation=torch.randn(self.d,device=self.dev)*MUTATE_SCALE
        self.w=self.w+mask*perturbation
        self.w=self.w.clamp(min=0)

        self.life_actions=[]
        self.generation+=1
        self._reset_cb()


def main():
    t0=time.time()
    print(f"Step 411 -- Evolving metric. 64x64. 500K. LS20.", flush=True)
    print(f"Device: {DEVICE}  mutate={MUTATE_FRAC}  scale={MUTATE_SCALE}  cb_cap={CB_CAP}", flush=True)
    print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)
    fold=EvolvingFold(d=D, nc=na, k=3)

    ts=0; go=0; lvls=0; sd=False
    action_counts={}
    score_history=[]

    while ts<500000 and go<5000:
        if obs is None or obs.state==GameState.GAME_OVER:
            score_history.append(fold._action_entropy())
            fold.on_game_over(); sd=False
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
            print(f"    LEVEL {lvls} at step {ts} gen={fold.generation}"
                  f"  best_entropy={fold.best_score:.3f} go={go}", flush=True)

        if ts%50000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            w_np=fold.w_best.cpu().numpy()
            top_idx=np.argsort(w_np)[-10:][::-1]
            top_rows=[i//64 for i in top_idx]
            recent=score_history[-100:] if score_history else [0]
            print(f"    [step {ts:6d}] gen={fold.generation}  best_entropy={fold.best_score:.3f}"
                  f"  recent_avg={np.mean(recent):.2f}  dom={dom:.0f}%  levels={lvls} go={go}", flush=True)
            print(f"      w: min={w_np.min():.3f} max={w_np.max():.3f}"
                  f"  mean={w_np.mean():.3f} std={w_np.std():.3f}", flush=True)
            print(f"      top rows: {top_rows}", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 411 SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}  generations={fold.generation}", flush=True)
    print(f"best_entropy={fold.best_score:.3f}", flush=True)
    if score_history:
        sh=np.array(score_history)
        print(f"life_score: min={sh.min()} max={sh.max()} mean={sh.mean():.1f} std={sh.std():.1f}", flush=True)
        # Evolution: compare first 100 vs last 100
        if len(sh)>200:
            early=sh[:100]; late=sh[-100:]
            print(f"  early_100: mean={early.mean():.1f}  late_100: mean={late.mean():.1f}", flush=True)
    w_np=fold.w_best.cpu().numpy()
    print(f"w_best: min={w_np.min():.3f} max={w_np.max():.3f} mean={w_np.mean():.3f} std={w_np.std():.3f}", flush=True)
    top_idx=np.argsort(w_np)[-20:][::-1]
    top_rows=sorted(set([i//64 for i in top_idx]))
    print(f"top 20 weight rows: {top_rows}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
    if lvls>0:
        print(f"\nPASS: Level with evolved metric at 64x64!", flush=True)
    elif fold.best_score>1.8:
        print(f"\nMARGINAL: good entropy ({fold.best_score:.3f}) but no level.", flush=True)
    else:
        print(f"\nKILL: best_entropy={fold.best_score:.3f} (max=2.0 for 4 actions).", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
