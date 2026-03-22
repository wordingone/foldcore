#!/usr/bin/env python3
"""
Step 403 -- 2D-aware substrate. Temporal detection + kernel discovery.

1. Track change rates -> find active pixels
2. Bounding box -> signal region (2D patch)
3. Try avgpool kernels {2,4,8} on patch, pick most discriminative
4. Run game at selected kernel

LS20. 64x64. 50K steps.
Script: scripts/run_step403_2d_kernel.py
"""

import time, random, logging, numpy as np, torch, torch.nn.functional as F
logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CB_CAP = 10000; THRESH_INT = 100; EPS = 0.01
WARMUP = 100; KERNEL_TRIAL = 200

class CF:
    """Standard normalized cosine codebook."""
    def __init__(self, d, k=3, dev=DEVICE):
        self.V=torch.zeros(0,d,device=dev); self.labels=torch.zeros(0,dtype=torch.long,device=dev)
        self.thresh=0.7; self.k=k; self.d=d; self.dev=dev; self.spawn_count=0
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
            if self.spawn_count%THRESH_INT==0: self._ut()
        else:
            ts_=si.clone(); ts_[~tm]=-float('inf'); w=ts_.argmax().item()
            a=1.0-float(si[w].item()); self.V[w]=F.normalize(self.V[w]+a*(x-self.V[w]),dim=0)
        return p


def encode_patch(frame_2d, r0, r1, c0, c1, kernel):
    """Extract signal region and avgpool with kernel."""
    patch = frame_2d[r0:r1, c0:c1]
    ph, pw = patch.shape
    # Pad to make divisible by kernel
    ph_pad = ((ph + kernel - 1) // kernel) * kernel
    pw_pad = ((pw + kernel - 1) // kernel) * kernel
    padded = np.zeros((ph_pad, pw_pad), dtype=np.float32)
    padded[:ph, :pw] = patch
    pooled = padded.reshape(ph_pad//kernel, kernel, pw_pad//kernel, kernel).mean(axis=(1,3))
    return pooled.flatten()


def main():
    t0=time.time()
    print("Step 403 -- 2D-aware substrate. Temporal detection + kernel discovery. LS20.", flush=True)
    print(f"Device: {DEVICE}  warmup={WARMUP}  kernel_trial={KERNEL_TRIAL}", flush=True); print(flush=True)

    import arc_agi; from arcengine import GameState
    arc=arc_agi.Arcade(); games=arc.get_environments()
    ls20=next(g for g in games if 'ls20' in g.game_id.lower())

    env=arc.make(ls20.game_id); obs=env.reset()
    na=len(env.action_space)

    # Stage 1: Temporal detection
    print("=== Stage 1: Temporal detection ===", flush=True)
    x_prev=None; change_count=np.zeros(4096); frame_count=0; ts=0; go=0
    action_counts={}

    while ts<WARMUP and go<20:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break; continue
        if obs.state==GameState.WIN: break
        frame=np.array(obs.frame[0],dtype=np.float32).flatten()/15.0
        if x_prev is not None:
            change_count+=(np.abs(frame-x_prev)>EPS).astype(np.float32)
        frame_count+=1; x_prev=frame.copy()
        action=env.action_space[random.randint(0,na-1)]
        action_counts[action.name]=action_counts.get(action.name,0)+1
        obs=env.step(action); ts+=1

    rate=change_count/max(frame_count,1)
    active_mask=rate>0.001
    r=rate[active_mask]
    threshold=r.mean()+2*r.std()
    signal_mask=rate>threshold
    active_idx=np.where(signal_mask)[0]
    rows=active_idx//64; cols=active_idx%64
    r0,r1=int(rows.min()),int(rows.max()+1)
    c0,c1=int(cols.min()),int(cols.max()+1)
    print(f"  Active pixels: {len(active_idx)}", flush=True)
    print(f"  Bounding box: rows [{r0},{r1}), cols [{c0},{c1})"
          f"  = {r1-r0}x{c1-c0} = {(r1-r0)*(c1-c0)} pixels", flush=True)

    # Stage 2: Kernel discovery
    print(f"\n=== Stage 2: Kernel discovery ({KERNEL_TRIAL} steps each) ===", flush=True)
    kernel_results=[]
    for kernel in [2, 4, 8]:
        d=len(encode_patch(np.zeros((64,64),dtype=np.float32), r0,r1,c0,c1, kernel))
        fold=CF(d=d, k=3); sd=False; trial_ts=0; trial_go=0
        obs_t=env.reset()
        while trial_ts<KERNEL_TRIAL and trial_go<10:
            if obs_t is None or obs_t.state==GameState.GAME_OVER:
                trial_go+=1; obs_t=env.reset()
                if obs_t is None: break; continue
            if obs_t.state==GameState.WIN: break
            frame_2d=np.array(obs_t.frame[0],dtype=np.float32)/15.0
            enc=torch.from_numpy(encode_patch(frame_2d, r0,r1,c0,c1, kernel))
            if not sd and fold.V.shape[0]<na:
                i=fold.V.shape[0]; fold._fa(enc,i)
                obs_t=env.step(env.action_space[i]); trial_ts+=1
                if fold.V.shape[0]>=na: sd=True; fold._ut()
                continue
            if not sd: sd=True
            fold.pn(enc, nc=na)
            obs_t=env.step(env.action_space[random.randint(0,na-1)]); trial_ts+=1

        sim_std=0.0
        if fold.V.shape[0]>=5:
            ss=min(200,fold.V.shape[0])
            idx=torch.randperm(fold.V.shape[0],device=fold.dev)[:ss]
            Vn=F.normalize(fold.V,dim=1)
            s=Vn[idx]@Vn.T
            sim_std=float(s.std().item())

        kernel_results.append({'kernel':kernel, 'dims':d, 'cb':fold.V.shape[0],
                               'sim_std':sim_std, 'thresh':fold.thresh})
        print(f"  {kernel}x{kernel}: dims={d}  cb={fold.V.shape[0]}"
              f"  sim_std={sim_std:.4f}  thresh={fold.thresh:.3f}", flush=True)

    best=max(kernel_results, key=lambda r: r['sim_std'])
    kernel=best['kernel']
    d=best['dims']
    print(f"\n  SELECTED: {kernel}x{kernel} kernel (sim_std={best['sim_std']:.4f}, dims={d})", flush=True)

    # Stage 3: Play game at selected kernel
    print(f"\n=== Stage 3: Gameplay ({kernel}x{kernel} kernel, {d}D, 50K steps) ===", flush=True)
    fold=CF(d=d, k=3); sd=False; lvls=0
    obs=env.reset(); action_counts={}

    while ts<50000 and go<500:
        if obs is None or obs.state==GameState.GAME_OVER:
            go+=1; obs=env.reset()
            if obs is None: break; continue
        if obs.state==GameState.WIN:
            print(f"    WIN at step {ts}!", flush=True); break
        frame_2d=np.array(obs.frame[0],dtype=np.float32)/15.0
        enc=torch.from_numpy(encode_patch(frame_2d, r0,r1,c0,c1, kernel))

        if not sd and fold.V.shape[0]<na:
            i=fold.V.shape[0]; fold._fa(enc,i)
            obs=env.step(env.action_space[i]); ts+=1
            action_counts[env.action_space[i].name]=action_counts.get(env.action_space[i].name,0)+1
            if fold.V.shape[0]>=na: sd=True; fold._ut()
            continue
        if not sd: sd=True

        c=fold.pn(enc, nc=na)
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
            print(f"    [step {ts:5d}] cb={fold.V.shape[0]} thresh={fold.thresh:.3f}"
                  f"  dom={dom:.0f}%  levels={lvls} go={go}", flush=True)

    elapsed=time.time()-t0
    print(flush=True); print("="*60, flush=True)
    print("STEP 403 SUMMARY", flush=True); print("="*60, flush=True)
    print(f"steps={ts}  go={go}  levels={lvls}", flush=True)
    print(f"Signal region: rows [{r0},{r1}), cols [{c0},{c1})", flush=True)
    print(f"Selected kernel: {kernel}x{kernel}  dims={d}", flush=True)
    print(f"Kernel comparison:", flush=True)
    for kr in kernel_results:
        m=" <-- SELECTED" if kr['kernel']==kernel else ""
        print(f"  {kr['kernel']}x{kr['kernel']}: sim_std={kr['sim_std']:.4f}  dims={kr['dims']}{m}", flush=True)
    print(f"cb={fold.V.shape[0]}  thresh={fold.thresh:.3f}", flush=True)
    print(f"actions: {action_counts}", flush=True)
    dom=max(action_counts.values())/sum(action_counts.values())*100 if action_counts else 0
    if lvls>0:
        print(f"\nPASS: Level with self-discovered kernel at 64x64!", flush=True)
    elif dom<60:
        print(f"\nMARGINAL: balanced ({dom:.0f}%) but no level.", flush=True)
    else:
        print(f"\nKILL: {dom:.0f}% dominance.", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__=='__main__': main()
