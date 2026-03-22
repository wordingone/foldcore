"""Step 583: Combined — permanent soft death penalty (581d) + surprise boost (582 op3) + per-edge predictions.
581d: death_edges permanent set, PENALTY=100.
582 op3: prediction error -> BONUS=50 count reduction.
Full model: argmin reads (count + death_penalty - surprise_bonus)."""
import time, numpy as np, sys
K=12; DIM=256; N_A=4; MAX_STEPS=50_000; TIME_CAP=280; PENALTY=100; BONUS=50

def encode(frame, H):
    arr=np.array(frame[0],dtype=np.float32); x=arr.reshape(16,4,16,4).mean(axis=(1,3)).flatten()/15.0; x-=x.mean()
    bits=(H@x>0).astype(np.int64); return int(np.dot(bits,1<<np.arange(K)))

class CombinedSub:
    def __init__(self,lsh_seed=0):
        self.H=np.random.RandomState(lsh_seed).randn(K,DIM).astype(np.float32)
        self.G={}; self.death_edges=set(); self.surprise_edges=set(); self.edge_pred={}
        self._prev_node=self._prev_action=None; self.cells=set(); self.total_deaths=0; self.surprises=0
    def observe(self,frame):
        node=encode(frame,self.H); self.cells.add(node); self._curr_node=node
        if self._prev_node is not None:
            key=(self._prev_node,self._prev_action)
            d=self.G.setdefault(key,{}); d[node]=d.get(node,0)+1
            pred=self.edge_pred.get(key)
            if pred is None: self.edge_pred[key]=node
            elif pred==node: self.surprise_edges.discard(key)  # confirmed: remove surprise
            else: self.edge_pred[key]=node; self.surprise_edges.add(key); self.surprises+=1
    def on_death(self):
        if self._prev_node is not None:
            key=(self._prev_node,self._prev_action); self.death_edges.add(key); self.total_deaths+=1
    def act(self):
        node=self._curr_node; eff=[]
        for a in range(N_A):
            key=(node,a); v=sum(self.G.get(key,{}).values())
            if key in self.death_edges: v+=PENALTY
            if key in self.surprise_edges: v=max(0,v-BONUS)
            eff.append(v)
        action=int(np.argmin(eff)); self._prev_node=node; self._prev_action=action; return action
    def on_reset(self): self._prev_node=self._prev_action=None

class ArgminSub:
    def __init__(self,lsh_seed=0):
        self.H=np.random.RandomState(lsh_seed).randn(K,DIM).astype(np.float32)
        self.G={}; self._prev_node=self._prev_action=None; self.cells=set()
    def observe(self,frame):
        node=encode(frame,self.H); self.cells.add(node); self._curr_node=node
    def act(self):
        node=self._curr_node; counts=[sum(self.G.get((node,a),{}).values()) for a in range(N_A)]
        action=int(np.argmin(counts))
        if self._prev_node is not None: d=self.G.setdefault((self._prev_node,self._prev_action),{}); d[self._curr_node]=d.get(self._curr_node,0)+1
        self._prev_node=node; self._prev_action=action; return action
    def on_reset(self): self._prev_node=self._prev_action=None
    def on_death(self): pass

def run_seed(mk,seed,SubClass,time_cap=TIME_CAP):
    env=mk(); sub=SubClass(lsh_seed=seed*100+7); obs=env.reset(seed=seed); sub.on_reset()
    prev_cl=0; fresh=True; l1=l2=go=step=0; t0=time.time()
    while step<MAX_STEPS and time.time()-t0<time_cap:
        if obs is None: obs=env.reset(seed=seed); sub.on_reset(); prev_cl=0; fresh=True; go+=1; continue
        sub.observe(obs); action=sub.act(); obs,_,done,info=env.step(action); step+=1
        if done:
            sub.on_death(); obs=env.reset(seed=seed); sub.on_reset(); prev_cl=0; fresh=True; go+=1; continue
        cl=info.get('level',0) if isinstance(info,dict) else 0
        if fresh: prev_cl=cl; fresh=False
        elif cl>=1 and prev_cl<1: l1+=1; print(f"    s{seed} L1@{step}",flush=True)
        elif cl>=2 and prev_cl<2: l2+=1
        prev_cl=cl
    print(f"  s{seed}: L1={l1} L2={l2} go={go} step={step} cells={len(sub.cells)} {time.time()-t0:.0f}s",flush=True)
    if hasattr(sub,'surprises'): print(f"    death_edges={len(sub.death_edges)} surprise_edges={len(sub.surprise_edges)} surprises={sub.surprises}",flush=True)
    return dict(seed=seed,l1=l1,l2=l2,go=go,steps=step,cells=len(sub.cells))

def main():
    try: sys.path.insert(0,'.'); import arcagi3; mk=lambda: arcagi3.make("LS20")
    except Exception as e: print(f"arcagi3: {e}"); return
    print("Step 583: Combined (permanent death penalty + surprise boost)",flush=True)
    r1=[]; t=time.time()
    for seed in range(5):
        if time.time()-t>1380: break
        print(f"\nseed {seed}:",flush=True); r1.append(run_seed(mk,seed,CombinedSub))
    r2=[]
    for seed in range(5):
        if time.time()-t>1380: break
        print(f"\nseed {seed}:",flush=True); r2.append(run_seed(mk,seed,ArgminSub))
    l1a=sum(r['l1'] for r in r1); l1b=sum(r['l1'] for r in r2)
    print(f"\n{'='*60}\nStep 583: Combined substrate")
    print(f"  Combined: {sum(1 for r in r1 if r['l1']>0)}/5 seeds L1, total={l1a}")
    for r in r1: print(f"    s{r['seed']}: L1={r['l1']} cells={r['cells']}")
    print(f"  Argmin:   {sum(1 for r in r2 if r['l1']>0)}/5 seeds L1, total={l1b}")
    for r in r2: print(f"    s{r['seed']}: L1={r['l1']} cells={r['cells']}")
    if l1a>l1b: print(f"\nSIGNAL: combined ({l1a}) > argmin ({l1b})")
    elif l1a==l1b: print(f"\nNEUTRAL: combined ({l1a}) == argmin ({l1b})")
    else: print(f"\nFAIL: combined ({l1a}) < argmin ({l1b})")

if __name__=="__main__": main()
