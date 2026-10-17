"""Set of normalised value functions compatible with the answers, with exact
linear-programming queries (HiGHS through scipy.optimize.linprog):

    W(eps) = { w in R^d : w >= 0, 1^T w = 1, (v(winner) - v(loser))^T w >= eps for every answer }

`eps` is the strictness margin a rule uses for an answer constraint (0 for non-strict constraints,
Liang et al.'s epsilon for Cheb).
"""
import numpy as np
from scipy.optimize import linprog

TOL = 1e-9          # numerical tolerance for slacks / LP values
TIE_TOL = 1e-7      # tolerance used to detect ties between LP values (PMR in [-1, 1])


def _lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')


class AnswerPolytope:
    """Simplex intersected with one half-space per answer."""

    def __init__(self, X, prefs, eps=0.0):
        self.X = np.asarray(X, dtype=float)
        self.n, self.d = self.X.shape
        prefs = np.asarray(prefs, dtype=int).reshape(-1, 2)
        self.prefs = prefs
        self.t = len(prefs)
        self.C = (self.X[prefs[:, 0]] - self.X[prefs[:, 1]]) if self.t else np.zeros((0, self.d))
        self.eps = float(eps)
        self.slack = 0.0

    @property
    def rhs(self):
        return np.full(self.t, self.eps - self.slack)

    def _ineq(self, extra_C=None, extra_rhs=None):
        """(A_ub, b_ub) for scipy (A_ub w <= b_ub) from C w >= rhs plus optional rows."""
        C, r = self.C, self.rhs
        if extra_C is not None and len(extra_C):
            C = np.vstack([C, np.atleast_2d(extra_C)])
            r = np.concatenate([r, np.atleast_1d(extra_rhs)])
        if len(C) == 0:
            return None, None
        return -C, -r

    def max_linear(self, c, extra_C=None, extra_rhs=None):
        """max c^T w over the polytope; (value, w), or (None, None) if it is empty."""
        A_ub, b_ub = self._ineq(extra_C, extra_rhs)
        res = _lp(-np.asarray(c, dtype=float), A_ub, b_ub, np.ones((1, self.d)), [1.0],
                  bounds=[(0, None)] * self.d)
        if res.status == 2:
            return None, None
        if res.status != 0:
            raise RuntimeError(f'max_linear LP failed: {res.message}')
        return float(-res.fun), res.x

    def pmr(self, i, j):
        """Pairwise max regret of choosing a_i instead of a_j: max_w (v(a_j) - v(a_i))^T w."""
        if i == j:
            return 0.0
        val, _ = self.max_linear(self.X[j] - self.X[i])
        return val

    def necessary(self, i, j, mode='fixed'):
        """Is a_i >=^N a_j, i.e. U(a_i) >= U(a_j) for every compatible w?

        mode='fixed'  (Liang et al. 2025, LP 11): max e' s.t. (v_j - v_i)^T w >= e', C w >= rhs;
                      necessary iff infeasible or e'* <= 0.
        mode='shared' (Greco et al. 2008, used by Ciomek et al. 2017): max e' s.t.
                      (v_j - v_i)^T w >= e', C w >= e' (the same e' in every row);
                      necessary iff infeasible or e'* <= 0."""
        d = self.d
        c = np.zeros(d + 1)
        c[-1] = -1.0                                   # maximise e'
        rows = [np.r_[-(self.X[j] - self.X[i]), 1.0]]  # -(v_j - v_i)^T w + e' <= 0
        rhs_rows = [0.0]
        if self.t:
            if mode == 'fixed':
                rows.append(np.hstack([-self.C, np.zeros((self.t, 1))]))
                rhs_rows.append(-self.rhs)
            elif mode == 'shared':
                rows.append(np.hstack([-self.C, np.ones((self.t, 1))]))   # -C w + e' <= 0
                rhs_rows.append(np.full(self.t, self.slack))
            else:
                raise ValueError(mode)
        A_ub = np.vstack([np.atleast_2d(r) for r in rows])
        b_ub = np.concatenate([np.atleast_1d(r) for r in rhs_rows])
        A_eq = np.r_[np.ones(d), 0.0][None, :]
        res = _lp(c, A_ub, b_ub, A_eq, [1.0], bounds=[(0, None)] * d + [(None, 1.0)])
        if res.status == 2:
            return True
        if res.status != 0:
            raise RuntimeError(f'necessary LP failed: {res.message}')
        return float(res.x[-1]) <= TOL

    def chebyshev_center(self, extra_C=None, row_rhs=None):
        """Chebyshev centre, LP 22 of Liang et al. (2025): max r s.t. C_k w - ||C_k||_2 r >= rhs_k,
        w_j - r >= 0, r >= 0, 1^T w = 1.  `row_rhs` overrides the right-hand sides of the answer rows
        (the printed LP uses 0, i.e. no epsilon).  Returns (w, r) or (None, None) if infeasible."""
        C = self.C
        if extra_C is not None and len(extra_C):
            C = np.vstack([C, np.atleast_2d(extra_C)])
        t = len(C)
        d = self.d
        rhs = np.full(t, -self.slack) if row_rhs is None else np.broadcast_to(row_rhs, (t,))
        c = np.zeros(d + 1)
        c[-1] = -1.0
        blocks, rhs_blocks = [], []
        if t:
            norms = np.linalg.norm(C, axis=1)
            blocks.append(np.hstack([-C, norms[:, None]]))      # -C w + ||C|| r <= -rhs
            rhs_blocks.append(-np.asarray(rhs, dtype=float))
        blocks.append(np.hstack([-np.eye(d), np.ones((d, 1))]))  # -w_j + r <= 0
        rhs_blocks.append(np.zeros(d))
        A_ub = np.vstack(blocks)
        b_ub = np.concatenate(rhs_blocks)
        A_eq = np.r_[np.ones(d), 0.0][None, :]
        res = _lp(c, A_ub, b_ub, A_eq, [1.0], bounds=[(0, None)] * (d + 1))
        if res.status == 2:
            return None, None
        if res.status != 0:
            raise RuntimeError(f'chebyshev LP failed: {res.message}')
        return res.x[:d], float(res.x[-1])

    def constraint_rows(self):
        """(A, b) with A w >= b for the answer rows (input of the hit-and-run sampler)."""
        return self.C.copy(), self.rhs.copy()
