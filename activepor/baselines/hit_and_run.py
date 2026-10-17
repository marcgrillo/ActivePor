"""
Hit-and-run sampler for the uniform distribution on
    { w in R^d : w >= 0, 1^T w = 1, A w >= b },
used to estimate pairwise winning indices (PWI) and rank acceptability indices (RAI) as in
Ciomek et al. (2017) (they report 10,000 draws with thinning (d-1)^3, Hit-And-Run of
Tervonen et al. 2013).  Directions are drawn uniformly on the unit sphere of the sum-zero
subspace, so 1^T w = 1 is preserved exactly; the chord is cut by w >= 0 and by A w >= b.

The inner loop is compiled with numba.  `thinning` and `burn` are parameters; the values used
for Dvf are set in rules.DvfRule.
"""

import numpy as np

try:
    from numba import njit
except ImportError:                     # pragma: no cover
    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco if not (args and callable(args[0])) else args[0]


@njit(cache=True)
def _har_core(w0, A, b, n_samples, thinning, burn, seed):
    np.random.seed(seed)
    d = w0.shape[0]
    t = A.shape[0]
    w = w0.copy()
    Aw = np.zeros(t)
    for k in range(t):
        s = 0.0
        for j in range(d):
            s += A[k, j] * w[j]
        Aw[k] = s
    out = np.empty((n_samples, d))
    dirv = np.empty(d)
    Ad = np.empty(t)
    total = burn + n_samples * thinning
    kept = 0
    step = 0
    while kept < n_samples and step < 50 * total + 1000:
        step += 1
        # direction uniform on the sphere of the sum-zero subspace
        m = 0.0
        for j in range(d):
            dirv[j] = np.random.standard_normal()
            m += dirv[j]
        m /= d
        nrm = 0.0
        for j in range(d):
            dirv[j] -= m
            nrm += dirv[j] * dirv[j]
        nrm = np.sqrt(nrm)
        if nrm < 1e-12:
            continue
        for j in range(d):
            dirv[j] /= nrm
        # chord from w >= 0
        tlo = -1e300
        thi = 1e300
        for j in range(d):
            if dirv[j] > 1e-14:
                v = -w[j] / dirv[j]
                if v > tlo:
                    tlo = v
            elif dirv[j] < -1e-14:
                v = -w[j] / dirv[j]
                if v < thi:
                    thi = v
        # chord from A w >= b
        for k in range(t):
            s = 0.0
            for j in range(d):
                s += A[k, j] * dirv[j]
            Ad[k] = s
            sl = Aw[k] - b[k]          # current slack (>= 0 inside)
            if s > 1e-14:
                v = -sl / s
                if v > tlo:
                    tlo = v
            elif s < -1e-14:
                v = -sl / s
                if v < thi:
                    thi = v
        if thi - tlo <= 1e-15:
            continue
        tt = tlo + np.random.random() * (thi - tlo)
        for j in range(d):
            w[j] += tt * dirv[j]
            if w[j] < 0.0:
                w[j] = 0.0
        for k in range(t):
            Aw[k] += tt * Ad[k]
        if step % 512 == 0:
            # numerical hygiene: renormalise and recompute A w
            s = 0.0
            for j in range(d):
                s += w[j]
            for j in range(d):
                w[j] /= s
            for k in range(t):
                s = 0.0
                for j in range(d):
                    s += A[k, j] * w[j]
                Aw[k] = s
        if step > burn and (step - burn) % thinning == 0:
            for j in range(d):
                out[kept, j] = w[j]
            kept += 1
    return out[:kept]


def hit_and_run(w0, A, b, n_samples, thinning, burn, seed):
    """Uniform samples from {w >= 0, sum w = 1, A w >= b}, starting at the interior point w0."""
    w0 = np.ascontiguousarray(np.asarray(w0, dtype=np.float64))
    A = np.ascontiguousarray(np.asarray(A, dtype=np.float64).reshape(-1, w0.shape[0]))
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float64).reshape(-1))
    out = _har_core(w0, A, b, int(n_samples), int(max(1, thinning)), int(max(0, burn)), int(seed) % (2 ** 31 - 1))
    if len(out) < n_samples:
        raise RuntimeError(f'hit-and-run produced only {len(out)}/{n_samples} samples (degenerate polytope?)')
    return out


def pairwise_winning_index(X, samples):
    """PWI[i, k] = share of samples with U(a_i) > U(a_k); diagonal 0."""
    U = samples @ X.T                                   # (S, n)
    n = X.shape[0]
    pwi = np.zeros((n, n))
    for i in range(n):
        pwi[i] = np.mean(U[:, i][:, None] > U, axis=0)
    np.fill_diagonal(pwi, 0.0)
    return pwi


def rank_acceptability_index(X, samples):
    """RAI[i, k] = share of samples giving alternative i rank k+1 (rank 1 = best)."""
    U = samples @ X.T
    n = X.shape[0]
    ranks = np.argsort(np.argsort(-U, axis=1), axis=1)   # 0 = best
    rai = np.zeros((n, n))
    for i in range(n):
        rai[i] = np.bincount(ranks[:, i], minlength=n) / len(U)
    return rai
