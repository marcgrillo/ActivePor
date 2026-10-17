"""Restoring a consistent answer set when the observed answers contradict each other.

The four literature rules assume a consistent DM; an inverted answer can make the compatible set
empty, and then every LP of the rules is infeasible.

* `min_cardinality_retained(C, eps)`: the remedy of Mousseau et al. (2003), used by Dvf, Cheb and
  Css: remove a minimum number of answers so that the others are consistent, by the MILP
      min sum_k z_k  s.t.  C_k w + M z_k >= eps + delta,  z_k in {0, 1},  w in the simplex,
  with M = 1 + eps + delta and a 20 s time limit.  The buffer delta = 1e-5 keeps an integer variable
  that the solver accepts as 0 (within its tolerance) from cancelling the margin eps.  When the time
  limit is reached, the UTA-style subset of minimal total error (an LP) is used instead.
* `uniform_linf_slack(C, eps)`: the relaxation used by Poly (Toubia et al. 2003, OPT4): the
  smallest common slack s such that {C w + s >= eps} is non-empty.
"""
import os
import sys
import contextlib
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds


@contextlib.contextmanager
def _silence_stdout():
    """HiGHS prints diagnostic lines to the C-level stdout during MILP solves; redirect fd 1."""
    try:
        sys.stdout.flush()
        fd = sys.stdout.fileno()
    except Exception:
        yield
        return
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)
        os.close(devnull)


def uniform_linf_slack(C, eps=0.0):
    """s* = min { s >= 0 : exists w in the simplex with C w + s >= eps }."""
    t, d = C.shape
    if t == 0:
        return 0.0
    c = np.zeros(d + 1)
    c[-1] = 1.0
    A_ub = np.hstack([-C, -np.ones((t, 1))])
    b_ub = -np.full(t, eps)
    A_eq = np.r_[np.ones(d), 0.0][None, :]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * (d + 1), method='highs')
    if res.status != 0:
        raise RuntimeError(f'uniform slack LP failed: {res.message}')
    return max(0.0, float(res.x[-1]))


def is_consistent(C, eps=0.0, mask=None):
    """Feasibility of {w in the simplex : C[mask] w >= eps}."""
    Cm = C if mask is None else C[mask]
    t, d = Cm.shape
    if t == 0:
        return True
    res = linprog(np.zeros(d), A_ub=-Cm, b_ub=-np.full(t, eps), A_eq=np.ones((1, d)), b_eq=[1.0],
                  bounds=[(0, None)] * d, method='highs')
    return res.status == 0


MILP_BUFFER = 1e-5         # margin added in the MILP above the rule's own margin eps


def min_cardinality_retained(C, eps=0.0, time_limit=20.0):
    """Boolean mask of a maximum-cardinality consistent subset of the answer rows, and the method
    used ('none', 'milp' or 'l1'; with 'l1' the subset is the one of minimal total error, which can
    be smaller)."""
    t, d = C.shape
    if t == 0:
        return np.ones(0, dtype=bool), 'none'
    if is_consistent(C, eps):
        return np.ones(t, dtype=bool), 'none'
    target = eps + MILP_BUFFER
    M = 1.0 + target                                        # |C w| <= 1 on the simplex
    c = np.r_[np.zeros(d), np.ones(t)]                       # variables [w (d), z (t)]
    A = np.hstack([C, M * np.eye(t)])                       # C w + M z >= eps + delta
    cons = [LinearConstraint(A, lb=np.full(t, target), ub=np.full(t, np.inf)),
            LinearConstraint(np.r_[np.ones(d), np.zeros(t)][None, :], lb=[1.0], ub=[1.0])]
    integrality = np.r_[np.zeros(d), np.ones(t)]
    bounds = Bounds(lb=np.zeros(d + t), ub=np.r_[np.full(d, np.inf), np.ones(t)])
    with _silence_stdout():
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds,
                   options={'time_limit': float(time_limit), 'disp': False})
    if res.status == 0 and res.x is not None:
        mask = res.x[d:] < 0.5
        if is_consistent(C, eps, mask):
            return mask, 'milp'
    # fallback: L1 minimal total error, drop the rows with a positive error
    c1 = np.r_[np.zeros(d), np.ones(t)]
    A_ub = np.hstack([-C, -np.eye(t)])                      # -C w - e <= -eps
    res1 = linprog(c1, A_ub=A_ub, b_ub=-np.full(t, eps), A_eq=np.r_[np.ones(d), np.zeros(t)][None, :],
                   b_eq=[1.0], bounds=[(0, None)] * (d + t), method='highs')
    mask = res1.x[d:] <= 1e-9
    while not is_consistent(C, eps, mask):                  # repair numerical residue greedily
        idx = np.where(mask)[0]
        viol = eps - C[idx] @ res1.x[:d]
        mask[idx[np.argmax(viol)]] = False
    return mask, 'l1'
