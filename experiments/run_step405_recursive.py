#!/usr/bin/env python3
"""
Step 405 -- Recursive self-composition with cached depth.

Phase 1 (1K steps): recursive process() discovers which depth has thresh<0.999.
Phase 2 (49K steps): lock structure, run flat composition at discovered depth.

LS20. 64x64. 50K steps.
Script: scripts/run_step405_recursive.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CB_CAP = 1000; THRESH_INT = 100; MAX_DEPTH = 5; THRESH_SPLIT = 0.999
WARMUP = 1000

class LevelCB:
    def __init__(self, d, nc, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.nc=nc; self.dev=dev; self.spawn_count=0
    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        s=self.V[idx]@self.V.T; t=s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())
    def process_votes(self, x):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        sc=torch.zeros(self.nc,device=self.dev)
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev)
            return sc
        si=self.V@x; ac=int(self.labels.max().item())+1
        sc_full=torch.zeros(max(ac,self.nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc_full[c]=cs.topk(min(self.k,len(cs))).values.sum()
        sc=sc_full[:self.nc]
        p=sc.argmin().item(); tm=(self.labels==p)
        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<CB_CAP:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            ts_=si.clone(); ts_[~tm]=-float('inf'); w=ts_.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return sc
    def update_last_label(self, label):
        if self.labels.shape[0]>0: self.labels[-1]=label


class RecursiveSubstrate:
    def __init__(self, na, dev=DEVICE):
        self.na=na; self.dev=dev
        self.codebooks={}
        self.depth_stats={}

    def _get_cb(self, key, dim):
        if key not in self.codebooks:
            self.codebooks[key]=LevelCB(d=dim, nc=self.na, dev=self.dev)
        return self.codebooks[key]

    def process_recursive(self, x, depth=0):
        """Phase 1: recursive with per-step splitting."""
        x=x.to(self.dev).float()
        d=x.shape[0]
        cb=self._get_cb(f"raw_{depth}_{d}", d)

        can_discriminate=cb.V.shape[0]>10 and cb.thresh<THRESH_SPLIT
        should_split=not can_discriminate and depth<MAX_DEPTH and d>16

        if depth not in self.depth_stats:
            self.depth_stats[depth]={'splits':0,'direct':0,'dim':d,'thresh':1.0}
        self.depth_stats[depth]['thresh']=cb.thresh if cb.V.shape[0]>10 else 1.0
        if should_split:
            self.depth_stats[depth]['splits']+=1
            mid=d//2
            vl=self.process_recursive(x[:mid], depth+1)
            vr=self.process_recursive(x[mid:], depth+1)
            meta_x=F.normalize(torch.cat([vl, vr]), dim=0)
            meta_cb=self._get_cb(f"meta_{depth}", 2*self.na)
            return meta_cb.process_votes(meta_x)
        else:
            self.depth_stats[depth]['direct']+=1
            return cb.process_votes(x)

    def update_labels(self, action):
        for cb in self.codebooks.values():
            cb.update_last_label(action)


class FlatComposition:
    """Locked flat composition at a specific chunk size."""
    def __init__(self, chunk_dim, n_chunks, na, dev=DEVICE):
        self.chunk_dim=chunk_dim; self.n_chunks=n_chunks; self.na=na; self.dev=dev
        self.chunk_cbs=[LevelCB(d=chunk_dim, nc=na, dev=dev) for _ in range(n_chunks)]
        self.meta_cb=LevelCB(d=n_chunks*na, nc=na, dev=dev)

    def process(self, x_raw):
        x=x_raw.to(self.dev).float()
        chunks=x.reshape(self.n_chunks, self.chunk_dim)
        all_votes=[]
        for i in range(self.n_chunks):
            v=self.chunk_cbs[i].process_votes(chunks[i])
            all_votes.append(v)
        meta_x=F.normalize(torch.cat(all_votes), dim=0)
        votes=self.meta_cb.process_votes(meta_x)
        return votes

    def update_labels(self, action):
        for cb in self.chunk_cbs: cb.update_last_label(action)
        self.meta_cb.update_last_label(action)


def main():
    t0=time.time()
    print(f"Step 405 -- Recursive self-composition (cached). LS20. 50K.", flush=True)
    print(f"Device: {DEVICE}  warmup={WARMUP}  max_depth={MAX_DEPTH}", flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    # Phase 1: Recursive discovery
    print("=== Phase 1: Recursive discovery (1K steps) ===", flush=True)
    sub=RecursiveSubstrate(na=na)
    ts=0; go=0; lvls=0; action_counts={}

    while ts<WARMUP and go<50:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break; continue
        if obs.state==GameState.WIN: break

        raw=torch.from_numpy(np.array(obs.frame[0],dtype=np.float32).flatten()/15.0)
        votes=sub.process_recursive(raw, depth=0)
        action_idx=votes[:na].argmin().item()
        sub.update_labels(action_idx)
        action=env.action_space[action_idx%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        obs=env.step(action); ts+=1

    # Log depth analysis
    print(f"  Warmup done: {ts} steps, {go} game_overs", flush=True)
    best_depth=MAX_DEPTH
    for depth in sorted(sub.depth_stats.keys()):
        ds=sub.depth_stats[depth]
        total=ds['splits']+ds['direct']
        print(f"  depth {depth}: dim={ds['dim']}  thresh={ds['thresh']:.4f}"
              f"  splits={ds['splits']}/{total}", flush=True)
        if ds['thresh']<THRESH_SPLIT and best_depth==MAX_DEPTH:
            best_depth=depth

    if best_depth<MAX_DEPTH:
        chunk_dim=4096//(2**best_depth)
        n_chunks=2**best_depth
        print(f"\n  LOCKED: depth={best_depth} -> {n_chunks} chunks of {chunk_dim}D", flush=True)
    else:
        # All depths saturated — use max depth
        chunk_dim=4096//(2**MAX_DEPTH) if 4096//(2**MAX_DEPTH)>=16 else 16
        n_chunks=4096//chunk_dim
        print(f"\n  ALL DEPTHS SATURATED. Using max: {n_chunks} chunks of {chunk_dim}D", flush=True)

    # Phase 2: Flat composition at locked depth
    print(f"\n=== Phase 2: Flat composition ({n_chunks}x{chunk_dim}D, 49K steps) ===", flush=True)
    flat=FlatComposition(chunk_dim=chunk_dim, n_chunks=n_chunks, na=na)
    action_counts={}

    while ts<50000 and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break; continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!", flush=True); break

        raw=torch.from_numpy(np.array(obs.frame[0],dtype=np.float32).flatten()/15.0)
        votes=flat.process(raw)
        action_idx=votes[:na].argmin().item()
        flat.update_labels(action_idx)
        action=env.action_space[action_idx%na]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            chunk_sizes=[cb.V.shape[0] for cb in flat.chunk_cbs]
            print(f"    LEVEL {lvls} at step {ts} meta_cb={flat.meta_cb.V.shape[0]}"
                  f"  chunks={sum(chunk_sizes)} go={go}", flush=True)

        if ts%5000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            chunk_sizes=[cb.V.shape[0] for cb in flat.chunk_cbs]
            print(f"    [step {ts:5d}] meta_cb={flat.meta_cb.V.shape[0]}"
                  f"  meta_thresh={flat.meta_cb.thresh:.3f}"
                  f"  chunks: min={min(chunk_sizes)} max={max(chunk_sizes)} total={sum(chunk_sizes)}"
                  f"  dom={dom:.0f}%  levels={lvls} go={go}", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 405 SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    print(f"Locked: {n_chunks} chunks of {chunk_dim}D", flush=True)
    chunk_sizes=[cb.V.shape[0] for cb in flat.chunk_cbs]
    print(f"meta: cb={flat.meta_cb.V.shape[0]}  thresh={flat.meta_cb.thresh:.3f}", flush=True)
    print(f"chunks: min={min(chunk_sizes)} max={max(chunk_sizes)} total={sum(chunk_sizes)}", flush=True)
    for i,cb in enumerate(flat.chunk_cbs):
        if i<20 or cb.V.shape[0]>100:  # only show first 20 or notable
            print(f"  chunk {i:3d}: cb={cb.V.shape[0]:4d}  thresh={cb.thresh:.3f}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
    if lvls>0:
        print(f"\nPASS: Level with recursive self-composition!", flush=True)
    elif dom<60:
        print(f"\nMARGINAL: balanced ({dom:.0f}%) but no level.", flush=True)
    else:
        print(f"\nKILL: {dom:.0f}% dominance.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
