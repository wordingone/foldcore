class S:

    def __init__(self, na):
        self.A = na
        self.T = {}
        self.G = {}
        self.R = {}
        self.mu = None
        self.d = 0
        self.n = 0
        self.p = None
        self.k = 1

    def __call__(self, x):
        D = len(x)
        if not self.mu:
            self.mu = [0.0] * D
            self.d = D
        self.n += 1
        z = [x[i] - self.mu[i] for i in range(D)]
        r = 1.0 / self.n
        for i in range(D):
            self.mu[i] += r * (x[i] - self.mu[i])
        c = self._map(z)
        if self.p:
            pc, pa, pz = self.p
            e = self.G.setdefault((pc, pa), {})
            e[c] = e.get(c, 0) + 1
            t = self.R.setdefault(pc, {}).setdefault((pa, c), [[0.0] * D, 0])
            t[1] += 1
            for i in range(D):
                t[0][i] += (pz[i] - t[0][i]) / t[1]
            self._split(pc)
            c = self._map(z)
        a = self._act(c)
        self.p = (c, a, z)
        return a

    def _map(self, z):
        c = 0
        while c in self.T:
            d, v, l, r = self.T[c]
            c = l if z[d] < v else r
        return c

    def _act(self, c):
        b, bn = 0, -1
        for a in range(self.A):
            n = sum(self.G.get((c, a), {}).values())
            if bn < 0 or n < bn:
                b, bn = a, n
        return b

    def _split(self, c):
        if c in self.T or c not in self.R:
            return
        pairs = [(v[1], v[0]) for v in self.R[c].values() if v[1] >= 4]
        tn = sum(p[0] for p in pairs)
        if tn < 32 or len(pairs) < 2:
            return
        pairs.sort(key=lambda p: p[0], reverse=True)
        n0, m0 = pairs[0]
        n1, m1 = pairs[1]
        bd, bv, bs = 0, 0.0, 0.0
        for i in range(self.d):
            s = abs(m1[i] - m0[i])
            if s > bs:
                bd, bv, bs = i, (m0[i] * n0 + m1[i] * n1) / (n0 + n1), s
        if bs < 1e-9:
            return
        l, r = self.k, self.k + 1
        self.k += 2
        self.T[c] = (bd, bv, l, r)


if __name__ == '__main__':
    from random import seed as sd, randint as ri, gauss as gs, random as rn
    N, D, A = 8, 64, 4
    DX = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for seed in range(10):
        sd(seed)
        s = S(A)
        p = [ri(0, N - 1), ri(0, N - 1)]
        v = set()
        for t in range(50000):
            x = [1.0 / (1 + (i // N - p[0]) ** 2 + (i % N - p[1]) ** 2) + gs(0, .01) for i in range(D)]
            a = s(x)
            dr, dc = DX[a]
            p = [max(0, min(N - 1, p[0] + dr)), max(0, min(N - 1, p[1] + dc))]
            v.add(tuple(p))
            if rn() < .005:
                p = [ri(0, N - 1), ri(0, N - 1)]
            if len(v) == N * N:
                print(f"{seed}: WIN@{t} cells={1 + len(s.T)}")
                break
        else:
            print(f"{seed}: {len(v)}/{N*N} cells={1 + len(s.T)}")
