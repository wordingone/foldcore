#!/usr/bin/env python3
"""
Step 404 -- Chunk composition. 16 codebooks x 256D + meta codebook.

Split 64x64 (4096D) into 16 chunks of 256D. Each chunk has its own
codebook producing class votes. Concatenate 16x4=64D meta-encoding.
Meta codebook selects action from composed encoding.

LS20. 50K steps.
Script: scripts/run_step404_chunk_compose.py
"""

import time, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CHUNKS = 16; CHUNK_DIM = 256; CB_CAP_CHUNK = 1000; CB_CAP_META = 500
THRESH_INT = 100

class ChunkCB:
    """Standard normalized cosine codebook for one chunk."""
    def __init__(self, d, nc, k=3, cap=1000, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.nc=nc; self.dev=dev
        self.spawn_count=0; self.cap=cap
    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        s=self.V[idx]@self.V.T; t=s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())
    def process_votes(self, x, nc):
        """Process and return class vote vector. Updates codebook."""
        x=F.normalize(x.to(self.dev).float(),dim=0)
        sc=torch.zeros(nc,device=self.dev)
        if self.V.shape[0]==0:
            self.V=x.unsqueeze(0); self.labels=torch.tensor([0],device=self.dev)
            return sc
        si=self.V@x; ac=int(self.labels.max().item())+1
        sc_full=torch.zeros(max(ac,nc),device=self.dev)
        for c in range(ac):
            m=(self.labels==c)
            if m.sum()==0: continue
            cs=si[m]; sc_full[c]=cs.topk(min(self.k,len(cs))).values.sum()
        sc=sc_full[:nc]
        p=sc.argmin().item(); tm=(self.labels==p)
        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<self.cap:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            ts_=si.clone(); ts_[~tm]=-float('inf'); w=ts_.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return sc
    def update_last_label(self, label):
        if self.labels.shape[0]>0:
            self.labels[-1]=label

class MetaCB:
    """Meta codebook on composed vote vectors."""
    def __init__(self, d, nc, k=3, cap=500, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.nc=nc; self.dev=dev
        self.spawn_count=0; self.cap=cap
    def _fa(self,x,l):
        x=F.normalize(x.to(self.dev).float(),dim=0)
        self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([l],device=self.dev)])
        self.spawn_count+=1
        if self.spawn_count%THRESH_INT==0: self._ut()
    def _ut(self):
        n=self.V.shape[0]
        if n<2: return
        ss=min(500,n); idx=torch.randperm(n,device=self.dev)[:ss]
        s=self.V[idx]@self.V.T; t=s.topk(min(2,n),dim=1).values
        self.thresh=float((t[:,1] if t.shape[1]>=2 else t[:,0]).median())
    def process(self, x, nc):
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
        if (tm.sum()==0 or si[tm].max()<self.thresh) and self.V.shape[0]<self.cap:
            self.V=torch.cat([self.V,x.unsqueeze(0)]); self.labels=torch.cat([self.labels,torch.tensor([p],device=self.dev)])
            self.spawn_count+=1
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            ts_=si.clone(); ts_[~tm]=-float('inf'); w=ts_.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return p


def main():
    t0=time.time()
    print(f"Step 404 -- Chunk composition. {N_CHUNKS}x{CHUNK_DIM}D + meta. LS20. 50K.", flush=True)
    print(f"Device: {DEVICE}  chunk_cap={CB_CAP_CHUNK}  meta_cap={CB_CAP_META}", flush=True)
    print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    chunks=[ChunkCB(d=CHUNK_DIM, nc=na, k=3, cap=CB_CAP_CHUNK) for _ in range(N_CHUNKS)]
    meta=MetaCB(d=N_CHUNKS*na, nc=na, k=3, cap=CB_CAP_META)

    ts=0; go=0; lvls=0; meta_seeded=False
    action_counts={}

    while ts<50000 and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break
            continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!", flush=True); break

        raw=torch.from_numpy(np.array(obs.frame[0],dtype=np.float32).flatten()/15.0)
        raw_chunks=raw.reshape(N_CHUNKS, CHUNK_DIM)

        # Get votes from all chunk codebooks
        all_votes=[]
        for i in range(N_CHUNKS):
            votes=chunks[i].process_votes(raw_chunks[i], nc=na)
            all_votes.append(votes)
        meta_input=torch.cat(all_votes)  # N_CHUNKS * na dims

        # Seed meta
        if not meta_seeded and meta.V.shape[0]<na:
            i=meta.V.shape[0]; meta._fa(meta_input, i)
            action=env.action_space[i]
            # Update chunk labels with chosen action
            for cb in chunks: cb.update_last_label(i)
            action_counts[action.name]=action_counts.get(action.name,0)+1
            obs=env.step(action); ts+=1
            if meta.V.shape[0]>=na: meta_seeded=True; meta._ut()
            continue

        # Meta selects action
        c=meta.process(meta_input, nc=na)
        action=env.action_space[c%na]
        # Update chunk labels with chosen action
        for cb in chunks: cb.update_last_label(c%na)
        action_counts[action.name]=action_counts.get(action.name,0)+1
        ol=obs.levels_completed
        obs=env.step(action); ts+=1
        if obs is None: break

        if obs.levels_completed>ol:
            lvls=obs.levels_completed
            chunk_cbs=[cb.V.shape[0] for cb in chunks]
            print(f"    LEVEL {lvls} at step {ts} meta_cb={meta.V.shape[0]}"
                  f"  chunk_cbs={sum(chunk_cbs)} go={go}", flush=True)

        if ts%5000==0:
            dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
            chunk_cbs=[cb.V.shape[0] for cb in chunks]
            # Find which chunks have the most diverse votes
            chunk_var=[]
            for v in all_votes:
                vn=v.cpu().numpy()
                chunk_var.append(float(vn.max()-vn.min()))
            print(f"    [step {ts:5d}] meta_cb={meta.V.shape[0]} meta_thresh={meta.thresh:.3f}"
                  f"  dom={dom:.0f}%  levels={lvls} go={go}", flush=True)
            print(f"      chunk_cbs: min={min(chunk_cbs)} max={max(chunk_cbs)} total={sum(chunk_cbs)}", flush=True)
            top_var=sorted(enumerate(chunk_var), key=lambda x:-x[1])[:5]
            print(f"      top vote_var: {[(i,f'{v:.2f}') for i,v in top_var]}", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 404 SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    chunk_cbs=[cb.V.shape[0] for cb in chunks]
    print(f"meta: cb={meta.V.shape[0]}  thresh={meta.thresh:.3f}", flush=True)
    print(f"chunks: min={min(chunk_cbs)} max={max(chunk_cbs)} total={sum(chunk_cbs)}", flush=True)
    for i,cb in enumerate(chunks):
        print(f"  chunk {i:2d}: cb={cb.V.shape[0]:4d}  thresh={cb.thresh:.3f}"
              f"  rows={i*4}-{i*4+3}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
    if lvls>0:
        print(f"\nPASS: Level with chunk composition at 64x64!", flush=True)
    elif dom<60:
        print(f"\nMARGINAL: balanced ({dom:.0f}%) but no level.", flush=True)
    else:
        print(f"\nKILL: {dom:.0f}% dominance.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
