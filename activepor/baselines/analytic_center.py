"""
Analytic centre and Sonnevend-ellipsoid longest axis of a polyhedron in the augmented
equality form used by Toubia, Simester, Hauser & Dahan (2003) and Toubia, Hauser &
Simester (2004):

    P = { u in R^N : X u = a, u >= 0 }.

* LP1  (2003 A1 / 2004 A1): interior point.
      max sum_j y_j  s.t.  X u = theta a,  theta >= 1,  u >= y >= 0,  y <= 1.
      u0 = u*/theta* is interior when y* > 0; coordinates with y_j* = 0 are identically zero
      on P and are removed (2004 p. 130).
* OPT1 (2003 A1 / 2004 A2): analytic centre  max sum_j ln u_j  s.t.  X u = a, u > 0,
      by Newton's method (Freund 1993) with direction
      d = u - U^2 X^T (X U^2 X^T)^{-1} a  and step alpha = 1 if ||U^{-1} d|| < 1/4, else an
      exact line search (2003 A2).  The stopping rule is not given by Toubia et al.; we stop
      when ||U^{-1} d|| < 1e-8 (or after 200 iterations).
* OPT3 (2003 A2-A3 / 2004 A5): longest axis of E = { X u = a, (u-ubar)^T Ubar^{-2} (u-ubar) <= 1 }
      = eigenvector of the smallest POSITIVE eigenvalue of M = Ubar^{-2} - X^T (X X^T)^{-1} X Ubar^{-2}.
      Computed equivalently (and without an eigenvalue threshold) as g = Z h with Z an
      orthonormal basis of null(X) and h the eigenvector of the smallest eigenvalue of the
      SPD matrix Z^T Ubar^{-2} Z.
"""

import numpy as np
from scipy.optimize import linprog
from scipy import linalg


def interior_point_lp1(X, a):
    """Toubia LP1. Returns (u0, zero_mask) where zero_mask marks coordinates identically 0 on P,
    or (None, None) if P is empty."""
    m, N = X.shape
    # variables z = [u (N), y (N), theta]
    c = np.zeros(2 * N + 1)
    c[N:2 * N] = -1.0                       # maximise sum y
    A_eq = np.hstack([X, np.zeros((m, N)), -np.asarray(a, dtype=float).reshape(-1, 1)])
    b_eq = np.zeros(m)
    A_ub = np.hstack([-np.eye(N), np.eye(N), np.zeros((N, 1))])   # y - u <= 0
    b_ub = np.zeros(N)
    bounds = [(0, None)] * N + [(0, 1.0)] * N + [(1.0, None)]
    res = None
    # the theta-scaled LP can be numerically awkward when P is (nearly) a point: retry with the
    # interior-point and dual-simplex codes of HiGHS before giving up
    for method, opts in (('highs', {}), ('highs-ipm', {}), ('highs-ds', {'presolve': False})):
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, options=opts)
        if res.status in (0, 2):
            break
    if res.status == 2:
        return None, None
    if res.status != 0:
        # fallback: unscaled LP (the normalisation row fixes the scale in our system) and an explicit
        # max u_j test for every coordinate the unscaled LP leaves at zero
        c2 = np.r_[np.zeros(N), -np.ones(N)]
        A_eq2 = np.hstack([X, np.zeros((m, N))])
        A_ub2 = np.hstack([-np.eye(N), np.eye(N)])
        res2 = linprog(c2, A_ub=A_ub2, b_ub=np.zeros(N), A_eq=A_eq2, b_eq=np.asarray(a, dtype=float),
                       bounds=[(0, None)] * N + [(0, 1.0)] * N, method='highs-ipm')
        if res2.status == 2:
            return None, None
        if res2.status != 0:
            raise RuntimeError(f'LP1 failed: {res.message} / {res2.message}')
        u = res2.x[:N].copy()
        zero_mask = np.zeros(N, dtype=bool)
        for j in np.where(res2.x[N:] <= 1e-9)[0]:
            cj = np.zeros(N)
            cj[j] = -1.0
            rj = linprog(cj, A_eq=X, b_eq=np.asarray(a, dtype=float), bounds=[(0, None)] * N, method='highs-ipm')
            if rj.status == 0 and -rj.fun > 1e-9:
                u = 0.5 * (u + rj.x)          # move towards a point with u_j > 0 (convexity keeps feasibility)
            else:
                zero_mask[j] = True
        u[zero_mask] = 0.0
        return u, zero_mask
    u = res.x[:N]
    y = res.x[N:2 * N]
    theta = res.x[-1]
    zero_mask = y <= 1e-9
    u0 = u / theta
    return u0, zero_mask


def _newton_direction(u, X, a):
    """Newton direction of OPT1 at u.  With r = X u - a the KKT system X d = -r gives
    d = u - U^2 X^T (X U^2 X^T)^{-1} (2 X u - a), which is Toubia et al.'s
    d = u - U^2 X^T (X U^2 X^T)^{-1} a whenever X u = a holds exactly; the residual-correcting
    form keeps round-off in X u - a from accumulating over the iterations."""
    U2 = u * u
    XU2 = X * U2[None, :]                    # X U^2
    G = XU2 @ X.T                            # X U^2 X^T
    rhs = 2.0 * (X @ u) - np.asarray(a, dtype=float)
    try:
        z = linalg.solve(G, rhs, assume_a='sym')
    except linalg.LinAlgError:
        z = linalg.lstsq(G, rhs)[0]
    return u - U2 * (X.T @ z)


def analytic_center(X, a, u0, tol=1e-8, max_iter=200):
    """Newton iterations for OPT1 from the interior point u0 (X u0 = a, u0 > 0).
    Stops when the Newton decrement ||U^{-1} d|| < tol (unspecified by Toubia et al.) or when it stops
    decreasing below 1e-6 (numerical floor); returns the iterate with the smallest decrement."""
    u = np.array(u0, dtype=float)
    best_u, best_nd = u.copy(), np.inf
    for _ in range(max_iter):
        d = _newton_direction(u, X, a)
        nd = np.linalg.norm(d / u)
        if not np.isfinite(nd):
            break
        if nd < best_nd:
            best_u, best_nd = u.copy(), nd
        elif best_nd < 1e-6:
            break
        if nd < tol:
            break
        if nd < 0.25:
            alpha = 1.0
        else:
            # exact line search for max_alpha sum ln(u + alpha d) on (0, alpha_max)
            neg = d < 0
            alpha_max = np.min(-u[neg] / d[neg]) if np.any(neg) else np.inf
            hi = alpha_max if np.isfinite(alpha_max) else 1e6
            lo = 0.0
            # f'(alpha) = sum d_j / (u_j + alpha d_j) is decreasing; bisection on its root
            for _k in range(200):
                mid = 0.5 * (lo + hi)
                fp = np.sum(d / (u + mid * d))
                if fp > 0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-12 * max(1.0, hi):
                    break
            alpha = 0.5 * (lo + hi)
            if alpha <= 0 or not np.isfinite(alpha):
                alpha = 0.5 * min(1.0, hi)
        u = u + alpha * d
        if not np.all(np.isfinite(u)) or np.any(u <= 0):
            break
    return best_u


def longest_axis(X, ubar):
    """Direction g of the longest axis of the Sonnevend ellipsoid at the analytic centre ubar
    (eigenvector of the smallest positive eigenvalue of M = P Ubar^{-2}), unit norm."""
    Z = linalg.null_space(X)
    if Z.shape[1] == 0:
        return np.zeros_like(ubar)
    W = 1.0 / (ubar * ubar)
    K = Z.T @ (W[:, None] * Z)
    K = 0.5 * (K + K.T)
    vals, vecs = linalg.eigh(K)
    h = vecs[:, 0]
    g = Z @ h
    return g / np.linalg.norm(g)


def analytic_center_and_axis(X, a):
    """Full Toubia pipeline: LP1 -> remove identically-zero coordinates -> Newton -> axis.
    Returns dict(ubar, g, active) with vectors in the ORIGINAL coordinates (zeros where removed),
    or None if P is empty."""
    X = np.asarray(X, dtype=float)
    a = np.asarray(a, dtype=float)
    N = X.shape[1]
    u0, zero_mask = interior_point_lp1(X, a)
    if u0 is None:
        return None
    active = ~zero_mask
    Xa = X[:, active]
    # drop rows that became identically zero after removing columns
    row_ok = np.linalg.norm(Xa, axis=1) > 1e-12
    Xa, aa = Xa[row_ok], a[row_ok]
    if not np.all(row_ok) and np.any(np.abs(a[~row_ok]) > 1e-9):
        return None
    # 'make X full rank by removing redundant rows' (2004 p. 129; 2003 A2): rank-revealing QR of X^T
    if Xa.shape[0] > 0:
        _, R, piv = linalg.qr(Xa.T, mode='economic', pivoting=True)
        diag = np.abs(np.diag(R))
        rank = int(np.sum(diag > 1e-10 * max(diag[0], 1e-300))) if diag.size else 0
        keep = np.sort(piv[:rank])
        Xa, aa = Xa[keep], aa[keep]
    ubar = np.zeros(N)
    g = np.zeros(N)
    if Xa.shape[0] >= Xa.shape[1]:
        # the (relaxed) polyhedron is a single point: it is its own analytic centre, no axis
        ubar[active] = u0[active]
        return {'ubar': ubar, 'g': g, 'active': active, 'point': True}
    ubar_a = analytic_center(Xa, aa, u0[active])
    g_a = longest_axis(Xa, ubar_a)
    if not (np.all(np.isfinite(ubar_a)) and np.all(np.isfinite(g_a))):
        ubar[active] = u0[active]
        return {'ubar': ubar, 'g': g, 'active': active, 'point': True}
    ubar[active] = ubar_a
    g[active] = g_a
    return {'ubar': ubar, 'g': g, 'active': active, 'point': False}
