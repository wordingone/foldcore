#!/usr/bin/env python3
import math, random, time
D = 12
NC = 6

def vcosine(a, b):
    dot = na2 = nb2 = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi; na2 += ai * ai; nb2 += bi * bi
    na = math.sqrt(na2 + 1e-15); nb = math.sqrt(nb2 + 1e-15)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))

def vnorm(v): return math.sqrt(sum(vi*vi for vi in v) + 1e-15)

class Organism:
    def __init__(self, seed=42, alive=False, eta=0.0003, clip_lo=0.3, clip_hi=1.8):
        self.beta = 0.5; self.gamma = 0.9; self.eps = 0.15
        self.tau = 0.3; self.delta = 0.35; self.noise = 0.005
        self.clip = 4.0; self.seed = seed; self.alive = alive; self.eta = eta
        self.aclo = clip_lo; self.achi = clip_hi
        self.total_alpha_shift = 0.0
        random.seed(seed)
        self.alpha = [[1.1 + 0.7*(random.random()*2-1) for _ in range(D)] for _ in range(NC)]
        for i in range(NC):
            for k in range(D):
                self.alpha[i][k] = max(self.aclo, min(self.achi, self.alpha[i][k]))

    def alpha_flat(self): return [a for row in self.alpha for a in row]

    def step(self, xs, signal=None):
        beta, gamma = self.beta, self.gamma
        phi_bare = []
        for i in range(NC):
            row = []
            for k in range(D):
                kp = (k+1)%D; km = (k-1)%D
                row.append(math.tanh(self.alpha[i][k]*xs[i][k] + beta*xs[i][kp]*xs[i][km]))
            phi_bare.append(row)
        if signal:
            phi_sig = []
            for i in range(NC):
                row = []
                for k in range(D):
                    kp = (k+1)%D; km = (k-1)%D
                    row.append(math.tanh(self.alpha[i][k]*xs[i][k] + beta*(xs[i][kp]+gamma*signal[kp])*(xs[i][km]+gamma*signal[km])))
                phi_sig.append(row)
        else:
            phi_sig = phi_bare
        if self.alive and signal:
            response = [[abs(phi_sig[i][k]-phi_bare[i][k]) for k in range(D)] for i in range(NC)]
            all_resp = [response[i][k] for i in range(NC) for k in range(D)]
            om = sum(all_resp)/len(all_resp)
            os_ = math.sqrt(sum((r-om)**2 for r in all_resp)/len(all_resp)) + 1e-10
            for i in range(NC):
                for k in range(D):
                    rz = (response[i][k]-om)/os_
                    col_mean = sum(self.alpha[j][k] for j in range(NC))/NC
                    dev = self.alpha[i][k] - col_mean
                    if abs(dev) < 0.01:
                        push = self.eta * 0.3 * random.gauss(0, 1.0)
                    elif rz > 0:
                        direction = 1.0 if dev > 0 else -1.0
                        push = self.eta * math.tanh(rz) * direction * 0.5
                    else:
                        push = self.eta * 0.1 * random.gauss(0, 1.0)
                    old = self.alpha[i][k]
                    self.alpha[i][k] = max(self.aclo, min(self.achi, self.alpha[i][k] + push))
                    self.total_alpha_shift += abs(self.alpha[i][k] - old)
        weights = []
        for i in range(NC):
            raw = []
            for j in range(NC):
                if i==j: raw.append(-1e10)
                else:
                    d = sum(xs[i][k]*xs[j][k] for k in range(D))
                    raw.append(d/(D*self.tau))
            mx = max(raw)
            exps = [math.exp(min(v-mx, 50)) for v in raw]
            s = sum(exps)+1e-15
            weights.append([e/s for e in exps])
        new = []
        for i in range(NC):
            p = list(phi_sig[i])
            bare_diff = [phi_bare[i][k]-xs[i][k] for k in range(D)]
            fp_d = vnorm(bare_diff)/max(vnorm(xs[i]), 1.0)
            plast = math.exp(-(fp_d*fp_d)/0.0225)
            if plast > 0.01 and self.eps > 0:
                pull = [0.0]*D
                for j in range(NC):
                    if i==j or weights[i][j] < 1e-8: continue
                    for k in range(D): pull[k] += weights[i][j]*(phi_bare[j][k]-phi_bare[i][k])
                p = [p[k]+plast*self.eps*pull[k] for k in range(D)]
            nx = []
            for k in range(D):
                v = (1-self.delta)*xs[i][k] + self.delta*p[k]
                v += random.gauss(0, self.noise)
                v = max(-self.clip, min(self.clip, v))
                nx.append(v)
            new.append(nx)
        return new

    def centroid(self, xs): return [sum(xs[i][k] for i in range(NC))/NC for k in range(D)]

def make_signals(k, seed):
    random.seed(seed)
    sigs = {}
    for i in range(k):
        s = [random.gauss(0, 0.5) for _ in range(D)]
        nm = math.sqrt(sum(v*v for v in s)+1e-15)
        sigs[i] = [v*0.8/nm for v in s]
    return sigs

def gen_perms(k, n_perm, seed):
    random.seed(seed)
    base = list(range(k))
    perms = [tuple(base), tuple(reversed(base))]
    seen = set(perms)
    att = 0
    while len(perms) < n_perm and att < n_perm*50:
        p = base[:]; random.shuffle(p); t = tuple(p)
        if t not in seen: perms.append(t); seen.add(t)
        att += 1
    return perms

def run_sequence(org, order, signals, base_seed, trial, n_org=300, n_per_sig=60, n_settle=30, n_final=60):
    random.seed(base_seed)
    xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]
    for _ in range(n_org): xs = org.step(xs)
    for idx, sid in enumerate(order):
        random.seed(base_seed*1000+sid*100+idx*10+trial)
        sig = [signals[sid][k]+random.gauss(0, 0.05) for k in range(D)]
        for _ in range(n_per_sig): xs = org.step(xs, sig)
        for _ in range(n_settle): xs = org.step(xs)
    for _ in range(n_final): xs = org.step(xs)
    return org.centroid(xs), xs

def measure_gap(org, signals, k, seed, n_perm=8, n_trials=6):
    perms = gen_perms(k, n_perm, seed=seed*10+k)
    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            c, _ = run_sequence(org, perm, signals, seed, trial)
            trials.append(c)
        endpoints[pi] = trials
    within = []
    between = []
    pis = sorted(endpoints.keys())
    for pi in pis:
        cs = endpoints[pi]
        for i in range(len(cs)):
            for j in range(i+1, len(cs)): within.append(vcosine(cs[i], cs[j]))
    for i in range(len(pis)):
        for j in range(i+1, len(pis)):
            for c1 in endpoints[pis[i]]:
                for c2 in endpoints[pis[j]]: between.append(vcosine(c1, c2))
    avg_w = sum(within)/max(len(within), 1)
    avg_b = sum(between)/max(len(between), 1)
    return avg_w - avg_b

def mean(xs): return sum(xs)/len(xs)
def std(xs):
    m = mean(xs); return math.sqrt(sum((x-m)**2 for x in xs)/len(xs)+1e-15)

def ncdf(z):
    t = 1.0/(1.0+0.2316419*abs(z))
    poly = t*(0.319381530+t*(-0.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))))
    p = 1.0 - (1.0/math.sqrt(2*math.pi))*math.exp(-0.5*z*z)*poly
    return p if z >= 0 else 1.0 - p

def pval(diffs):
    n = len(diffs); m = mean(diffs)
    s = math.sqrt(sum((d-m)**2 for d in diffs)/(n-1)+1e-15)
    t = m/(s/math.sqrt(n)+1e-15)
    p2 = 2.0*(1.0-ncdf(abs(t)))
    return p2

def cohd_paired(xs, ys):
    diffs = [y-x for x,y in zip(xs,ys)]
    m = mean(diffs)
    s = math.sqrt(sum((d-m)**2 for d in diffs)/(len(diffs)-1)+1e-15)
    return m/(s+1e-15)

def eval_condition(cl, ch, seeds, ks, eta=0.0003, n_perm=8, n_trials=6):
    alive_gaps = []; still_gaps = []
    for seed in seeds:
        k_alive = []; k_still = []
        for k in ks:
            sigs = make_signals(k, seed=seed*1000+k*7+3)
            alive = Organism(seed=seed, alive=True, eta=eta, clip_lo=cl, clip_hi=ch)
            still = Organism(seed=seed, alive=False, eta=eta, clip_lo=cl, clip_hi=ch)
            ag = measure_gap(alive, sigs, k, seed, n_perm=n_perm, n_trials=n_trials)
            sg = measure_gap(still, sigs, k, seed, n_perm=n_perm, n_trials=n_trials)
            k_alive.append(ag); k_still.append(sg)
        alive_gaps.append(mean(k_alive))
        still_gaps.append(mean(k_still))
    return alive_gaps, still_gaps

def main():
    print("=" * 72)
    print("  TASK 7: MI-GROUND TRUTH CORRELATION ACROSS CLIP CONDITIONS")
    print("  Ground truth = ALIVE MI gap > STILL MI gap")
    print("  Protocol: 10 seeds, K=[4,6,8,10], n_perm=8, n_trials=6")
    print("=" * 72)

    t0 = time.time()
    seeds = list(range(10, 20))
    ks = [4, 6, 8, 10]
    eta = 0.0003

    conditions = [
        ("CANONICAL",    0.3,  1.8),
        ("VERY_NARROW",  0.7,  1.3),
        ("NARROW_TIGHT", 0.5,  1.6),
        ("NARROW_LO",    0.5,  1.8),
        ("WIDE_LO",      0.1,  1.8),
        ("ASYMM_LO",     0.05, 1.8),
        ("ASYMM_HI",     0.3,  2.8),
    ]

    results = {}
    for name, cl, ch in conditions:
        print("\nRunning %s [%s,%s] ..." % (name, cl, ch), flush=True)
        alive_gaps, still_gaps = eval_condition(cl, ch, seeds, ks, eta=eta)
        diffs = [a-s for a,s in zip(alive_gaps, still_gaps)]
        n_alive_wins = sum(1 for d in diffs if d > 0)
        alive_mean = mean(alive_gaps)
        still_mean = mean(still_gaps)
        gap_delta = alive_mean - still_mean
        d = cohd_paired(still_gaps, alive_gaps)
        p = pval(diffs)
        gt_pass = n_alive_wins >= 7
        results[name] = {"alive_mean": alive_mean, "still_mean": still_mean,
                         "gap_delta": gap_delta, "d": d, "p": p,
                         "n_wins": n_alive_wins, "gt_pass": gt_pass, "cl": cl, "ch": ch}
        gt_sym = "#" if gt_pass else "o"
        print(f"  {gt_sym} ALIVE={alive_mean:+.4f} STILL={still_mean:+.4f} delta={gap_delta:+.4f} d={d:+.3f} p={p:.4f} wins={n_alive_wins}/10", flush=True)

    print("\n" + "=" * 72)
    print("  SUMMARY TABLE")
    print("=" * 72)
    hdr = "  Condition       Clip          ALIVE    STILL    Delta      d       p     GT"
    print(hdr)
    print("  " + "-"*70)
    for name, cl, ch in conditions:
        r = results[name]
        gt_sym = "PASS" if r["gt_pass"] else "FAIL"
        cl_s = str(cl) + "," + str(ch)
        am = r["alive_mean"]; sm = r["still_mean"]; gd = r["gap_delta"]
        d_ = r["d"]; p_ = r["p"]
        line = "  " + name.ljust(15) + ("["+cl_s+"]").ljust(14)
        line += "%+8.4f" % am + "%+8.4f" % sm + "%+8.4f" % gd
        line += "%+7.3f" % d_ + "%7.4f" % p_ + "%6s" % gt_sym
        print(line)

    print("\n  Ground truth = ALIVE wins >= 7/10 seeds")
    n_gt_pass = sum(1 for n,_,_ in conditions if results[n]["gt_pass"])
    print("  GT pass rate: %d/%d conditions pass" % (n_gt_pass, len(conditions)))

    alive_vals = [results[n]["alive_mean"] for n,_,_ in conditions]
    delta_vals = [results[n]["gap_delta"] for n,_,_ in conditions]
    print("\n  Correlation analysis (ALIVE MI gap vs GT delta):")
    m_a = mean(alive_vals); m_d = mean(delta_vals)
    num = sum((a-m_a)*(d-m_d) for a,d in zip(alive_vals, delta_vals))
    den = math.sqrt(sum((a-m_a)**2 for a in alive_vals)*sum((d-m_d)**2 for d in delta_vals)+1e-15)
    r_corr = num/den
    print("  Pearson r(ALIVE_gap, ALIVE_vs_STILL_delta) = %+.4f" % r_corr)

    elapsed = time.time() - t0
    print("\n  Elapsed: %.1fs" % elapsed)
    print("=" * 72)

main()
