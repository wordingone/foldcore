import math,random
D=12;NC=6;W=72

def vcosine(a,b):
    dot=na2=nb2=0.0
    for ai,bi in zip(a,b): dot+=ai*bi;na2+=ai*ai;nb2+=bi*bi
    na=math.sqrt(na2+1e-15);nb=math.sqrt(nb2+1e-15)
    if na<1e-10 or nb<1e-10: return 0.0
    return max(-1.0,min(1.0,dot/(na*nb)))

def vnorm(v): return math.sqrt(sum(vi*vi for vi in v)+1e-15)

class Organism:
    def __init__(self,seed=42,alive=False,init_lo=0.3,init_hi=1.8,clip_lo=0.3,clip_hi=1.8):
        self.beta=0.5;self.gamma=0.9;self.eps=0.15;self.tau=0.3;self.delta=0.35
        self.noise=0.005;self.clip=4.0;self.eta=0.0003
        self.sbm=0.3;self.am=0.5;self.dm=0.1
        self.thr=0.01;self.aclo=clip_lo;self.achi=clip_hi
        self.seed=seed;self.alive=alive
        random.seed(seed)
        self.alpha=[[1.1+0.7*(random.random()*2-1) for _ in range(D)] for _ in range(NC)]
        for i in range(NC):
            for k in range(D):
                self.alpha[i][k]=max(init_lo,min(init_hi,self.alpha[i][k]))
        flat=[self.alpha[i][k] for i in range(NC) for k in range(D)]
        self.bm=sum(flat)/len(flat)
        self.bs=math.sqrt(sum((a-self.bm)**2 for a in flat)/len(flat))

    def step(self,xs,signal=None):
        b,g=self.beta,self.gamma
        pb=[]
        for i in range(NC):
            row=[]
            for k in range(D):
                kp=(k+1)%D;km=(k-1)%D
                row.append(math.tanh(self.alpha[i][k]*xs[i][k]+b*xs[i][kp]*xs[i][km]))
            pb.append(row)
        if signal:
            ps=[]
            for i in range(NC):
                row=[]
                for k in range(D):
                    kp=(k+1)%D;km=(k-1)%D
                    row.append(math.tanh(self.alpha[i][k]*xs[i][k]+b*(xs[i][kp]+g*signal[kp])*(xs[i][km]+g*signal[km])))
                ps.append(row)
        else:
            ps=pb
        if self.alive and signal:
            resp=[[abs(ps[i][k]-pb[i][k]) for k in range(D)] for i in range(NC)]
            ar=[resp[i][k] for i in range(NC) for k in range(D)]
            om=sum(ar)/len(ar)
            osd=math.sqrt(sum((r-om)**2 for r in ar)/len(ar))+1e-10
            for i in range(NC):
                for k in range(D):
                    rz=(resp[i][k]-om)/osd
                    cm=sum(self.alpha[j][k] for j in range(NC))/NC
                    dev=self.alpha[i][k]-cm
                    if abs(dev)<self.thr: push=self.eta*self.sbm*random.gauss(0,1.0)
                    elif rz>0: push=self.eta*math.tanh(rz)*(1.0 if dev>0 else -1.0)*self.am
                    else: push=self.eta*self.dm*random.gauss(0,1.0)
                    nv=self.alpha[i][k]+push
                    self.alpha[i][k]=max(self.aclo,min(self.achi,nv))
        wts=[]
        for i in range(NC):
            raw=[-1e10 if i==j else sum(xs[i][k]*xs[j][k] for k in range(D))/(D*self.tau) for j in range(NC)]
            mx=max(raw)
            ex=[math.exp(min(v-mx,50)) for v in raw]
            s=sum(ex)+1e-15
            wts.append([e/s for e in ex])
        new=[]
        for i in range(NC):
            p=list(ps[i])
            bd=[pb[i][k]-xs[i][k] for k in range(D)]
            fpd=vnorm(bd)/max(vnorm(xs[i]),1.0)
            pl=math.exp(-(fpd*fpd)/0.0225)
            if pl>0.01 and self.eps>0:
                pull=[0.0]*D
                for j in range(NC):
                    if i==j or wts[i][j]<1e-8: continue
                    for k in range(D): pull[k]+=wts[i][j]*(pb[j][k]-pb[i][k])
                p=[p[k]+pl*self.eps*pull[k] for k in range(D)]
            nx=[]
            for k in range(D):
                v=(1-self.delta)*xs[i][k]+self.delta*p[k]+random.gauss(0,self.noise)
                nx.append(max(-self.clip,min(self.clip,v)))
            new.append(nx)
        return new

def make_signals(K,seed=42):
    random.seed(seed)
    return [[random.gauss(0,1) for _ in range(D)] for _ in range(K)]

def gen_perms(K,n_perm=8,seed=42):
    random.seed(seed);idxs=list(range(K));perms=[]
    for _ in range(n_perm): p=idxs[:];random.shuffle(p);perms.append(p)
    return perms

def run_seq(org,perm,sigs,rs,trial=0,n_steps=50):
    rng=random.Random(rs*100+trial)
    xs=[[rng.gauss(0,0.5) for _ in range(D)] for _ in range(NC)]
    for idx in perm:
        sig=sigs[idx]
        for _ in range(n_steps): xs=org.step(xs,signal=sig if org.alive else None)
    return xs

def measure_gap(org,perm_list,sigs,seed,n_trials=6):
    fpp=[]
    for pi,perm in enumerate(perm_list):
        tf=[]
        for t in range(n_trials): tf.append([r[:] for r in run_seq(org,perm,sigs,seed+pi*1000,t)])
        fpp.append(tf)
    within=[];between=[]
    for i in range(len(perm_list)):
        fi=fpp[i]
        for a in range(len(fi)):
            for b in range(a+1,len(fi)):
                for ci in range(NC): within.append(vcosine(fi[a][ci],fi[b][ci]))
        for j in range(i+1,len(perm_list)):
            fj=fpp[j]
            for ta in fi:
                for tb in fj:
                    for ci in range(NC): between.append(vcosine(ta[ci],tb[ci]))
    w=sum(within)/len(within) if within else 0.0
    b=sum(between)/len(between) if between else 0.0
    return w-b

def eval_cond(il,ih,cl,ch,seeds,ks,np_=8,nt=6):
    gaps=[];bms=[];bss=[]
    for seed in seeds:
        org=Organism(seed=seed,alive=True,init_lo=il,init_hi=ih,clip_lo=cl,clip_hi=ch)
        bms.append(org.bm);bss.append(org.bs)
        sg=[]
        for K in ks:
            sigs=make_signals(K,seed=seed+500)
            perms=gen_perms(K,n_perm=np_,seed=seed+300)
            sg.append(measure_gap(org,perms,sigs,seed,n_trials=nt))
        gaps.append(sum(sg)/len(sg))
    return gaps,bms,bss

def mean(v): return sum(v)/len(v)
def std(v):
    m=mean(v);return math.sqrt(sum((x-m)**2 for x in v)/len(v))
def ncdf(z): return 0.5*(1.0+math.erf(z/math.sqrt(2.0)))
def pval(a,b):
    n=len(a);diffs=[a[i]-b[i] for i in range(n)];md=mean(diffs)
    if n<2: return 1.0
    vd=sum((d-md)**2 for d in diffs)/(n-1);se=math.sqrt(vd/n)+1e-15
    t=md/se;return 2.0*(1.0-ncdf(abs(t)))
def cohd(a,b):
    diffs=[a[i]-b[i] for i in range(len(a))];md=mean(diffs)
    sd=math.sqrt(sum((d-md)**2 for d in diffs)/max(len(diffs)-1,1))
    return md/(sd+1e-15)

print(chr(61)*W)
print("  STAGE 4: BIRTH CONFOUND DISAMBIGUATION")
print("  VERY_NARROW [0.7,1.3] -- init vs ongoing clip separation")
print(chr(61)*W)
SEEDS=[42,137,2024,999,7];KS=[4,6,8,10];NP=8;NT=6
print(f"Seeds: {SEEDS}  K={KS}  n_perm={NP}  n_trials={NT}")
print()
print("Design:")
print("  A) Canonical:   init=[0.3,1.8] clip=[0.3,1.8] -- BASELINE")
print("  B) VN-Full:     init=[0.7,1.3] clip=[0.7,1.3] -- ORIGINAL (Entry 042)")
print("  C) VN-InitOnly: init=[0.7,1.3] clip=[0.3,1.8] -- BIRTH CONFOUND ONLY")
print("  D) VN-ClipOnly: init=[0.3,1.8] clip=[0.7,1.3] -- GENUINE BINDING ONLY")
print()
print("Logic: C>A => birth confound; D>A => genuine binding")
print()
conds=[
    ("A-Canonical",   0.3,1.8,0.3,1.8),
    ("B-VN-Full",     0.7,1.3,0.7,1.3),
    ("C-VN-InitOnly", 0.7,1.3,0.3,1.8),
    ("D-VN-ClipOnly", 0.3,1.8,0.7,1.3),
]
all_gaps={};all_births={}
for name,il,ih,cl,ch in conds:
    print(f"  [{name}] init=[{il},{ih}] clip=[{cl},{ch}]... ",end="",flush=True)
    gaps,bms,bss=eval_cond(il,ih,cl,ch,SEEDS,KS,NP,NT)
    all_gaps[name]=gaps;all_births[name]=(bms,bss)
    print(f"mean={mean(gaps):+.4f} std={std(gaps):.4f} birth_mean={mean(bms):.4f} birth_std={mean(bss):.4f}")
print()
print(chr(61)*W);print("  BIRTH DISTRIBUTIONS");print(chr(61)*W)
for name,il,ih,cl,ch in conds:
    bms,bss=all_births[name]
    tag="VN-init" if il==0.7 else "Natural-init"
    print(f"  {name:<16} birth_mean={mean(bms):.4f} birth_std={mean(bss):.4f}  [{tag}]")
can=all_gaps["A-Canonical"];cm=mean(can)
print()
print(chr(61)*W);print("  MI GAP TABLE");print(chr(61)*W)
for name,il,ih,cl,ch in conds:
    g=all_gaps[name];gm=mean(g);gs=std(g)
    if name=="A-Canonical":
        ps=[f"{x:+.4f}" for x in g]
        print(f"  {name:<16} mean={gm:+.4f} std={gs:.4f} [BASELINE] per-seed: {ps}")
    else:
        d=cohd(g,can);p=pval(g,can)
        sig="SIG" if p<0.05 else ("bdr" if p<0.15 else "ns ")
        ps=[f"{x:+.4f}" for x in g]
        print(f"  {name:<16} mean={gm:+.4f} std={gs:.4f} diff={gm-cm:+.4f} d={d:+.3f} p={p:.3f} [{sig}] per-seed: {ps}")
print()
print(chr(61)*W);print("  KEY COMPARISONS");print(chr(61)*W)
comps=[
    ("B-VN-Full","A-Canonical","B vs A (full VN vs Canonical, Entry042 replication)"),
    ("C-VN-InitOnly","A-Canonical","C vs A (birth confound isolation)"),
    ("D-VN-ClipOnly","A-Canonical","D vs A (genuine clip binding isolation)"),
    ("D-VN-ClipOnly","C-VN-InitOnly","D vs C (clip-only vs init-only)"),
    ("B-VN-Full","C-VN-InitOnly","B vs C (full vs init-only)"),
]
for ca,cb,lbl in comps:
    ga=all_gaps[ca];gb=all_gaps[cb]
    diff=mean(ga)-mean(gb);d=cohd(ga,gb);p=pval(ga,gb)
    sig="SIGNIFICANT" if p<0.05 else ("borderline" if p<0.15 else "ns")
    print(f"  {lbl:<55} diff={diff:+.4f} d={d:+.3f} p={p:.3f} [{sig}]")
print()
print(chr(61)*W);print("  VERDICT");print(chr(61)*W)
vf=all_gaps["B-VN-Full"];vi=all_gaps["C-VN-InitOnly"];vc=all_gaps["D-VN-ClipOnly"]
df=cohd(vf,can);pf=pval(vf,can)
di=cohd(vi,can);pi=pval(vi,can)
dc=cohd(vc,can);pc=pval(vc,can)
print(f"  Replicated Entry 042 (B vs A): d={df:+.3f}, p={pf:.3f}")
print(f"  Birth confound only (C vs A):  d={di:+.3f}, p={pi:.3f}")
print(f"  Genuine clip binding (D vs A): d={dc:+.3f}, p={pc:.3f}")
print()
if not pf<0.05:
    print("  CONCLUSION: Entry 042 FAILED TO REPLICATE. Clip bounds NON-BINDING.")
elif pi<0.05 and not pc<0.05:
    print(f"  CONCLUSION: BIRTH CONFOUND. Init drives effect (d={di:+.3f}).")
    print("  Ongoing clip is NOT binding. Alpha init range is the true variable.")
elif pc<0.05 and not pi<0.05:
    print(f"  CONCLUSION: GENUINE BINDING. Ongoing clip drives effect (d={dc:+.3f}).")
    print("  Clip bounds are a genuine Stage 4 adaptive target.")
elif pi<0.05 and pc<0.05:
    if abs(di)>abs(dc)*1.5:
        print(f"  CONCLUSION: PRIMARILY BIRTH CONFOUND (d_init={di:+.3f} >> d_clip={dc:+.3f}).")
    elif abs(dc)>abs(di)*1.5:
        print(f"  CONCLUSION: PRIMARILY GENUINE BINDING (d_clip={dc:+.3f} >> d_init={di:+.3f}).")
    else:
        print(f"  CONCLUSION: MIXED -- d_init={di:+.3f}, d_clip={dc:+.3f}.")
else:
    print("  CONCLUSION: AMBIGUOUS -- effect needs both init+clip simultaneously.")
print(chr(61)*W)
