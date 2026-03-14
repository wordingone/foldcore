#!/usr/bin/env python3
"""
RK — Reflexive Kernel  (v2)

v1 died. The pure quadratic self-application Φ(M) = tanh(γ·M²/k)
is contractive at zero. Everything collapsed to nothing.

The fix is mathematically precise:

  Φ(M) = tanh(α·M + β·M·M/k)

  α > 1  →  zero is UNSTABLE (supercritical bifurcation)
  β > 0  →  self-interaction creates structure

  The system MUST leave zero. It MUST find non-trivial fixed points.

  For diagonal element m: eigenform means m = tanh(α·m + β·m²/k).
  At m=0, slope = α > 1 → unstable.
  At m ≈ 0.8 (for α=1.2, β=0.8, k=4) → stable fixed point.

  The non-trivial eigenforms exist and are ATTRACTORS.

Everything else carries forward: state IS transformation,
eigenform IS objective, no loss function, no weight/activation split.

Zero dependencies. Pure Python. The code IS the math.
"""

import math
import random
import sys

# ═══════════════════════════════════════════════════════════════
# §1  MATRIX ARITHMETIC
# ═══════════════════════════════════════════════════════════════

def mzero(k):     return [[0.0]*k for _ in range(k)]
def meye(k):
    m = mzero(k)
    for i in range(k): m[i][i] = 1.0
    return m

def mrand(k, s=1.0):
    return [[random.gauss(0, s/math.sqrt(k)) for _ in range(k)]
            for _ in range(k)]

def madd(A, B):   return [[A[i][j]+B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
def msub(A, B):   return [[A[i][j]-B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
def mscale(A, s): return [[a*s for a in row] for row in A]

def mmul(A, B):
    k = len(A)
    C = mzero(k)
    for i in range(k):
        for j in range(k):
            for p in range(k): C[i][j] += A[i][p]*B[p][j]
    return C

def mtanh(A):     return [[math.tanh(a) for a in row] for row in A]
def mtrans(A):    return [[A[j][i] for j in range(len(A))] for i in range(len(A))]
def trace(A):     return sum(A[i][i] for i in range(len(A)))
def frob(A):      return math.sqrt(sum(a*a for row in A for a in row) + 1e-15)

def mclip(A, mx):
    n = frob(A)
    return mscale(A, mx/n) if n > mx else A

def inner(A, B):  return sum(A[i][j]*B[i][j] for i in range(len(A)) for j in range(len(A[0])))

def mcosine(A, B):
    na, nb = frob(A), frob(B)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return max(-1.0, min(1.0, inner(A,B)/(na*nb)))

def dominant_eigen(M, iters=30):
    k = len(M)
    v = [random.gauss(0,1) for _ in range(k)]
    n = math.sqrt(sum(x*x for x in v)+1e-15)
    v = [x/n for x in v]
    for _ in range(iters):
        w = [sum(M[i][j]*v[j] for j in range(k)) for i in range(k)]
        n = math.sqrt(sum(x*x for x in w)+1e-15)
        v = [x/n for x in w]
    Mv = [sum(M[i][j]*v[j] for j in range(k)) for i in range(k)]
    return sum(v[i]*Mv[i] for i in range(k)), v

def mat_apply(M, v):
    return [sum(M[i][j]*v[j] for j in range(len(M))) for i in range(len(M))]

def vnorm(v): return math.sqrt(sum(x*x for x in v)+1e-15)


# ═══════════════════════════════════════════════════════════════
# §2  REFLEXIVE CELL
# ═══════════════════════════════════════════════════════════════

class Cell:
    """
    State M ∈ ℝ^(k×k) IS the transformation.

    Self-application with supercritical bifurcation:
      Φ(M) = tanh(α·M + β·M·M/k)

    α > 1 → zero is unstable → non-trivial eigenforms are forced.
    β > 0 → quadratic self-interaction creates structural richness.

    Eigenform: Φ(M) = M.
    """
    __slots__ = ('k', 'alpha', 'beta', 'M', 'dM', 'age')

    def __init__(self, k, alpha=1.2, beta=0.8):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.M = mrand(k, 0.8)
        self.dM = mzero(k)
        self.age = 0

    def self_apply(self):
        """Φ(M) = tanh(α·M + β·M²/k)"""
        linear = mscale(self.M, self.alpha)
        quadratic = mscale(mmul(self.M, self.M), self.beta / self.k)
        return mtanh(madd(linear, quadratic))

    def eigenform_distance(self):
        """‖Φ(M) - M‖_F / max(‖M‖_F, 1)"""
        d = frob(msub(self.self_apply(), self.M))
        return d / max(frob(self.M), 1.0)

    def autonomy(self, sigma=0.3):
        d = self.eigenform_distance()
        return math.exp(-(d*d)/(sigma*sigma))

    def spectral_summary(self):
        lam, _ = dominant_eigen(self.M)
        return lam


# ═══════════════════════════════════════════════════════════════
# §3  REFLEXIVE KERNEL
# ═══════════════════════════════════════════════════════════════

class Kernel:
    """
    N cells coupled through structural alignment.

    ΔM_i = α_i · [Φ(M_i) - M_i]                                  (eigenform drive)
          + (1-α_i) · Σ_j w_ij · [Ψ(M_i,M_j) - M_i]             (coupling)
          + (1-α_i) · λ · signal                                   (environment)

    Ψ(M_i,M_j) = tanh(α·(M_i+M_j)/2 + β·M_i·M_j/k)             (cross-application)
    w_ij = softmax_j(alignment_ij / τ)                              (topology)

    α_i = exp(-ef_dist² / σ²)                                      (autonomy)
    """

    def __init__(self, n=8, k=4, alpha=1.2, beta=0.8, seed=None):
        if seed is not None:
            random.seed(seed)
        self.n, self.k = n, k
        self.alpha, self.beta = alpha, beta
        self.cells = [Cell(k, alpha, beta) for _ in range(n)]
        self.dt = 0.03
        self.tau = 0.3
        self.noise_scale = 0.01
        self.max_norm = 3.0
        self.t = 0.0
        self.step_count = 0

    def alignment(self, i, j):
        return mcosine(self.cells[i].M, self.cells[j].M)

    def cross_apply(self, Mi, Mj):
        """Ψ(Mi,Mj) = tanh(α·(Mi+Mj)/2 + β·Mi·Mj/k)"""
        avg = mscale(madd(Mi, Mj), self.alpha / 2.0)
        prod = mscale(mmul(Mi, Mj), self.beta / self.k)
        return mtanh(madd(avg, prod))

    def step(self, signal=None):
        n, k, dt = self.n, self.k, self.dt
        cells = self.cells

        alphas = [c.autonomy() for c in cells]

        # Coupling weights
        weights = []
        for i in range(n):
            raw = []
            for j in range(n):
                if i == j: raw.append(-1e10)
                else: raw.append(self.alignment(i,j) / self.tau)
            mx = max(raw)
            exps = [math.exp(min(r-mx, 50)) for r in raw]
            s = sum(exps) + 1e-15
            weights.append([e/s for e in exps])

        for i in range(n):
            ci = cells[i]
            ai = alphas[i]

            # Eigenform drive
            phi = ci.self_apply()
            ef_drive = msub(phi, ci.M)

            # Coupling drive
            cp_drive = mzero(k)
            for j in range(n):
                if i == j: continue
                w = weights[i][j]
                if w < 1e-8: continue
                psi = self.cross_apply(ci.M, cells[j].M)
                cp_drive = madd(cp_drive, mscale(msub(psi, ci.M), w))

            # Signal
            sig_drive = mzero(k)
            if signal is not None:
                sig_drive = mscale(signal, (1.0 - ai) * 0.3)

            # Gentle dissipation to prevent runaway
            norm_M = frob(ci.M)
            dissipation = mscale(ci.M, -0.01 * max(0, norm_M - 1.5))

            ci.dM = madd(madd(madd(
                mscale(ef_drive, ai),
                mscale(cp_drive, 1.0 - ai)),
                sig_drive),
                dissipation)

            ci.dM = madd(ci.dM, mrand(k, self.noise_scale))

        for ci in cells:
            ci.M = madd(ci.M, mscale(ci.dM, dt))
            ci.M = mclip(ci.M, self.max_norm)
            ci.age += 1

        self.t += dt
        self.step_count += 1

    # ── Observables ──────────────────────────────────────────

    def mean_ef_dist(self):
        return sum(c.eigenform_distance() for c in self.cells) / self.n

    def mean_autonomy(self):
        return sum(c.autonomy() for c in self.cells) / self.n

    def mean_energy(self):
        return sum(frob(c.dM)**2 for c in self.cells) / self.n

    def mean_norm(self):
        return sum(frob(c.M) for c in self.cells) / self.n

    def alignment_matrix(self):
        return [[self.alignment(i,j) for j in range(self.n)] for i in range(self.n)]

    def composite(self):
        R = meye(self.k)
        for c in self.cells:
            R = mmul(R, c.M)
        return R

    def process_signal(self, signal, inject=40, settle=40):
        before = self.composite()
        for _ in range(inject): self.step(signal)
        for _ in range(settle): self.step()
        return msub(self.composite(), before)

    def measure(self):
        return dict(t=round(self.t,3),
                    ef=round(self.mean_ef_dist(),4),
                    au=round(self.mean_autonomy(),4),
                    E=round(self.mean_energy(),6),
                    nrm=round(self.mean_norm(),4))


# ═══════════════════════════════════════════════════════════════
# §4  EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

def bar(v, w=20, lo=0.0, hi=1.0):
    f = max(0.0, min(1.0, (v-lo)/(hi-lo+1e-10)))
    n = int(f*w)
    return '█'*n + '░'*(w-n)

def fmt(x, w=8): return f"{x:>{w}.4f}"

def run():
    W = 72
    print("="*W)
    print("  RK — Reflexive Kernel v2")
    print("  Φ(M) = tanh(α·M + β·M²/k),  α=1.2 > 1 → zero unstable")
    print("  State IS transformation. Eigenform IS objective.")
    print("="*W)

    K = Kernel(n=8, k=4, alpha=1.2, beta=0.8, seed=42)

    # ── §1: Eigenform convergence ────────────────────────────

    print(f"\n{'─'*W}")
    print("  §1  EIGENFORM CONVERGENCE")
    print(f"      α > 1 destabilizes zero. Cells MUST find non-trivial fixed points.")
    print(f"{'─'*W}\n")

    hdr = f"  {'step':>5}  {'ef_dist':>8}  {'auton':>8}  {'‖M‖':>8}  {'energy':>10}  convergence"
    print(hdr)
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*20}")

    for epoch in range(25):
        for _ in range(80): K.step()
        m = K.measure()
        print(f"  {K.step_count:>5}  {fmt(m['ef'])}  {fmt(m['au'])}"
              f"  {fmt(m['nrm'])}  {m['E']:>10.6f}  {bar(1-m['ef'], 20)}")

    print(f"\n  Per-cell at t={K.t:.1f}:")
    print(f"  {'cell':>4}  {'ef_dist':>8}  {'auton':>7}  {'‖M‖':>7}  {'λ_dom':>8}  state")
    for i, c in enumerate(K.cells):
        d = c.eigenform_distance()
        a = c.autonomy()
        lam = c.spectral_summary()
        nrm = frob(c.M)
        st = "EIGENFORM" if d < 0.08 else "near" if d < 0.2 else "seeking" if d < 0.5 else "coupled"
        print(f"  {i:>4}  {d:>8.4f}  {a:>7.3f}  {nrm:>7.3f}  {lam:>+8.4f}  {st}")

    ef_final = K.mean_ef_dist()
    au_final = K.mean_autonomy()

    # ── §2: Perturbation response ────────────────────────────

    print(f"\n{'─'*W}")
    print("  §2  PERTURBATION & RECOVERY")
    print(f"{'─'*W}\n")

    pre = K.measure()
    signal = mzero(4)
    signal[0][0] = 3.0; signal[0][1] = -2.0
    signal[1][0] = 1.5; signal[2][3] = -2.0

    print(f"  {'phase':<12} {'step':>5}  {'ef_dist':>8}  {'auton':>8}  {'‖M‖':>8}  {'energy':>10}")
    print(f"  {'─'*12} {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")

    for ep in range(6):
        for _ in range(50): K.step(signal)
        m = K.measure()
        print(f"  {'PERTURBED':<12} {K.step_count:>5}  {fmt(m['ef'])}  "
              f"{fmt(m['au'])}  {fmt(m['nrm'])}  {m['E']:>10.6f}")

    print()
    for ep in range(8):
        for _ in range(50): K.step()
        m = K.measure()
        print(f"  {'recovering':<12} {K.step_count:>5}  {fmt(m['ef'])}  "
              f"{fmt(m['au'])}  {fmt(m['nrm'])}  {m['E']:>10.6f}")

    post = K.measure()
    delta_ef = abs(post['ef'] - pre['ef'])
    result = "HOMEOSTASIS" if delta_ef < 0.08 else "REORGANIZED"
    print(f"\n  → {result} (Δef_dist = {delta_ef:.4f})")

    # ── §3: History-dependent computation ────────────────────

    print(f"\n{'─'*W}")
    print("  §3  HISTORY-DEPENDENT COMPUTATION")
    print(f"{'─'*W}\n")

    def conditioned(cond_sig):
        rk = Kernel(n=8, k=4, alpha=1.2, beta=0.8, seed=100)
        for _ in range(800): rk.step()
        for _ in range(500): rk.step(cond_sig)
        for _ in range(400): rk.step()
        return rk

    ca = mzero(4); ca[0][0]=2.0; ca[1][1]=2.0
    cb = mzero(4); cb[2][2]=2.0; cb[3][3]=2.0

    print("  Conditioning A: amplify subspace {0,1}")
    ka = conditioned(ca)
    print(f"    ef={ka.mean_ef_dist():.4f}  au={ka.mean_autonomy():.4f}  ‖M‖={ka.mean_norm():.4f}")

    print("  Conditioning B: amplify subspace {2,3}")
    kb = conditioned(cb)
    print(f"    ef={kb.mean_ef_dist():.4f}  au={kb.mean_autonomy():.4f}  ‖M‖={kb.mean_norm():.4f}")

    Ca, Cb = ka.composite(), kb.composite()
    cos_comp = mcosine(Ca, Cb)
    print(f"\n  Composite similarity: {cos_comp:+.4f}")
    print(f"  ‖C_A‖={frob(Ca):.4f}  ‖C_B‖={frob(Cb):.4f}  ‖C_A-C_B‖={frob(msub(Ca,Cb)):.4f}")

    random.seed(999)
    test = mrand(4, 1.5)
    ra = ka.process_signal(test)
    rb = kb.process_signal(test)
    cos_r = mcosine(ra, rb)
    print(f"\n  Same signal → response similarity: {cos_r:+.4f}")
    print(f"  ‖resp_A‖={frob(ra):.4f}  ‖resp_B‖={frob(rb):.4f}")

    history_ok = abs(cos_comp) < 0.9 or abs(cos_r) < 0.9
    print(f"\n  → {'HISTORY SHAPES COMPUTATION' if history_ok else 'Histories converged'}")

    # ── §4: Emergent computation ─────────────────────────────

    print(f"\n{'─'*W}")
    print("  §4  EMERGENT COMPUTATION")
    print(f"{'─'*W}\n")

    rk = Kernel(n=8, k=4, alpha=1.2, beta=0.8, seed=7)

    print(f"  {'step':>5}  {'λ_dom':>8}  {'‖C‖':>8}  {'tr(C)':>8}  structure")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*20}")

    for epoch in range(15):
        for _ in range(120): rk.step()
        C = rk.composite()
        lam, _ = dominant_eigen(C)
        tr = trace(C); nrm = frob(C)
        r = abs(lam)/(nrm+1e-10)
        st = "crystallized" if r > 0.4 else "structuring" if r > 0.15 else "fluid"
        print(f"  {rk.step_count:>5}  {lam:>+8.4f}  {nrm:>8.4f}  {tr:>+8.4f}  {st}")

    print(f"\n  Composite applied to test vectors:")
    C = rk.composite()
    random.seed(42)
    for t in range(5):
        vi = [random.gauss(0,1) for _ in range(4)]
        vo = mat_apply(C, vi)
        ni, no = vnorm(vi), vnorm(vo)
        vin = [x/ni for x in vi]; von = [x/no for x in vo] if no > 1e-10 else [0]*4
        cos = sum(a*b for a,b in zip(vin,von))
        gain = no/(ni+1e-10)
        angle = math.acos(max(-1,min(1,cos)))*180/math.pi
        print(f"    v{t}: gain={gain:.3f}  rotation={angle:.1f}°")

    # ── §5: Eigenform taxonomy ───────────────────────────────

    print(f"\n{'─'*W}")
    print("  §5  EIGENFORM TAXONOMY (isolated cells, pure self-application)")
    print(f"{'─'*W}\n")

    random.seed(0)
    found = []
    N_trials = 50
    for trial in range(N_trials):
        c = Cell(4, alpha=1.2, beta=0.8)
        c.M = mrand(4, 0.8)
        for _ in range(4000):
            phi = c.self_apply()
            c.M = madd(c.M, mscale(msub(phi, c.M), 0.03))
            c.M = madd(c.M, mrand(4, 0.005))
            c.M = mclip(c.M, 3.0)
        d = c.eigenform_distance()
        if d < 0.1:
            lam, _ = dominant_eigen(c.M)
            found.append((d, lam, frob(c.M), trace(c.M), c.M))

    print(f"  {len(found)}/{N_trials} converged to eigenform (d < 0.1)\n")
    if found:
        print(f"  {'ef_dist':>8}  {'λ_dom':>8}  {'‖M‖':>8}  {'tr(M)':>8}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
        for d, lam, nrm, tr, _ in sorted(found)[:15]:
            print(f"  {d:>8.4f}  {lam:>+8.4f}  {nrm:>8.4f}  {tr:>+8.4f}")

        if len(found) >= 2:
            lams = [f[1] for f in found]; nrms = [f[2] for f in found]
            lam_mean = sum(lams)/len(lams); nrm_mean = sum(nrms)/len(nrms)
            lam_var = sum((l-lam_mean)**2 for l in lams)/len(lams)
            nrm_var = sum((n-nrm_mean)**2 for n in nrms)/len(nrms)

            cos_vals = []
            for i in range(min(len(found),10)):
                for j in range(i+1, min(len(found),10)):
                    cos_vals.append(abs(mcosine(found[i][4], found[j][4])))
            avg_cos = sum(cos_vals)/len(cos_vals) if cos_vals else 1.0

            print(f"\n  λ_dom variance:  {lam_var:.4f}")
            print(f"  ‖M‖ variance:    {nrm_var:.4f}")
            print(f"  Avg |cos| between eigenforms: {avg_cos:.4f}")
            diversity = "DIVERSE" if avg_cos < 0.8 else "SIMILAR"
            print(f"  → Eigenform landscape: {diversity}")

    # ── §6: Alignment evolution ──────────────────────────────

    print(f"\n{'─'*W}")
    print("  §6  TOPOLOGY SELF-ORGANIZATION")
    print(f"{'─'*W}\n")

    rk2 = Kernel(n=6, k=4, alpha=1.2, beta=0.8, seed=33)
    for epoch in range(8):
        for _ in range(200): rk2.step()
        A = rk2.alignment_matrix()
        print(f"  t={rk2.t:>6.1f}  ", end="")
        vals = []
        for i in range(6):
            for j in range(i+1, 6):
                vals.append(A[i][j])
        avg_a = sum(abs(v) for v in vals)/len(vals) if vals else 0
        hi_a = sum(1 for v in vals if abs(v) > 0.5)
        print(f"avg|align|={avg_a:.3f}  strong_links={hi_a}  "
              f"ef={rk2.mean_ef_dist():.3f}  au={rk2.mean_autonomy():.3f}")

    # ── Summary ──────────────────────────────────────────────

    print(f"\n{'═'*W}")
    print("  RESULTS")
    print(f"{'═'*W}\n")

    eigenform_ok = ef_final < 0.2
    autonomy_ok = au_final > 0.15
    taxonomy_ok = len(found) > N_trials * 0.3

    checks = [
        ("Eigenform convergence (ef_dist < 0.2)", eigenform_ok, f"{ef_final:.4f}"),
        ("Autonomy emergence (auton > 0.15)",     autonomy_ok,  f"{au_final:.4f}"),
        ("History-dependent computation",          history_ok,   f"cos={cos_comp:+.3f}"),
        ("Eigenform diversity (>30% converge)",    taxonomy_ok,  f"{len(found)}/{N_trials}"),
    ]

    for name, ok, detail in checks:
        sym = "■" if ok else "□"
        print(f"  {sym} {name:<45} {detail}")

    n_pass = sum(1 for _,ok,_ in checks if ok)
    print(f"\n  {n_pass}/4 criteria met.")

    if n_pass >= 3:
        print("""
  The reflexive kernel works. Cells find non-trivial eigenforms—
  fixed points of their own self-application. The state IS the
  transformation that produces it. No loss function was specified.
  Self-consistency under self-application is the only organizer.
  """)
    elif n_pass >= 1:
        print("""
  Partial success. The supercritical bifurcation prevents collapse
  to zero, but full eigenform convergence needs further tuning.
  The principle is sound. The dynamics need refinement.
  """)
    else:
        print("""
  The system did not converge. Back to the mathematics.
  """)

    print(f"{'─'*W}")
    print("""
  KEY INSIGHT FROM v1's FAILURE:

  Pure self-multiplication M² is contractive at zero.
  tanh(M²) → 0 for small M. Zero is a black hole.

  Adding α·M with α > 1 makes zero UNSTABLE:
    d/dM[tanh(α·M + β·M²/k)] at M=0  =  α > 1

  The system is FORCED out of the trivial fixed point.
  Non-trivial eigenforms become the only attractors.

  This is the computational analogue of metabolism:
  you need energy input (α > 1) to maintain organized structure.
  A purely dissipative system (α ≤ 1) dies.

  v1 was a dead universe. v2 has metabolism.
""")
    print(f"{'─'*W}")


if __name__ == '__main__':
    run()
