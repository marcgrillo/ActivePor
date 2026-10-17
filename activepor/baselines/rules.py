"""Four question-selection rules from the literature.  Each rule sees only the answers, never the
POR learner:

    rule = DvfRule(V, rng)                   # V: (n, d) characteristic vectors
    pair, info = rule.select(prefs, unasked) # prefs: (t, 2) [winner, loser]; unasked: [(i, j), i < j]

`select` returns one of the unasked pairs; `info['fallback']` is set when the rule has no pair to
ask (its own termination condition) and a uniformly random unasked pair is returned.

POLY  polyhedral question design, Toubia, Hauser & Simester (2004), Steps 1-3 on a fixed pool.
DVF   heuristic H_DVF of Ciomek, Kadzinski & Tervonen (2017), Eq. 21.
CHEB  estimate-then-select with Chebyshev centres, Liang, Zhang & Su (2025).
CSS   current-solution strategy of minimax-regret elicitation (Boutilier et al. 2006; Benabbou &
      Perny 2015), ranking by iterated choices (Benabbou, Perny & Viappiani 2017, Algorithm 2).
"""
import numpy as np

from .polytope import AnswerPolytope, TIE_TOL, TOL
from .consistency import is_consistent, min_cardinality_retained, uniform_linf_slack
from .hit_and_run import hit_and_run
from .analytic_center import analytic_center_and_axis


def _key(i, j):
    return (i, j) if i < j else (j, i)


class Rule:
    name = 'BASE'

    def __init__(self, X, rng):
        self.X = np.asarray(X, dtype=float)
        self.n, self.d = self.X.shape
        self.rng = rng
        self.rounds = 0

    def _fallback(self, unasked, info, reason):
        """The rule has no pair to ask: return a uniformly random unasked pair."""
        k = int(self.rng.integers(len(unasked)))
        info['fallback'] = reason
        return tuple(unasked[k]), info

    def _answer_rows(self, prefs):
        prefs = np.asarray(prefs, dtype=int).reshape(-1, 2)
        C = (self.X[prefs[:, 0]] - self.X[prefs[:, 1]]) if len(prefs) else np.zeros((0, self.d))
        return prefs, C

    def select(self, prefs, unasked):
        raise NotImplementedError


class RetainedAnswers:
    """Maximum-cardinality consistent subset of the answers, used by Dvf, Cheb and Css.

    With consistent answers it is the set of all answers.  When a new answer makes it infeasible,
    the subset is recomputed over all answers (consistency.min_cardinality_retained); `version`
    changes whenever the retained set changes, which invalidates cached necessary relations."""

    def __init__(self, eps):
        self.eps = eps
        self.mask = np.ones(0, dtype=bool)
        self.version = 0
        self.method = 'none'
        self.n_dropped = 0

    def update(self, C):
        t = len(C)
        if t == len(self.mask):
            return
        if t < len(self.mask):
            self.mask = np.ones(0, dtype=bool)
            self.version += 1
        new_mask = np.r_[self.mask, np.ones(t - len(self.mask), dtype=bool)]
        if is_consistent(C, self.eps, new_mask):
            self.mask = new_mask
            return
        self.mask, self.method = min_cardinality_retained(C, self.eps)
        self.n_dropped = int((~self.mask).sum())
        self.version += 1

    def describe(self, info):
        if not np.all(self.mask):
            info['exact'] = False
            info['dropped'] = int((~self.mask).sum())
            info['consistency'] = self.method


# ============================================================================== Css
class CssRule(Rule):
    """Compare the minimax-regret alternative x* with its adversary y* = argmax_j PMR(x*, j)
    (adversaries restricted to pairs not yet asked).  PMR(i, j) = max_w (v_j - v_i)^T w by one LP
    per ordered pair, MR(i) = max_j PMR(i, j), MMR = min_i MR(i); non-strict answer constraints.
    When MMR = 0, x* is certified, placed at the next rank and removed, and the procedure goes on
    with the remaining alternatives; the rule stops when every alternative is placed.
    Ties: smallest index."""
    name = 'CSS'

    def __init__(self, X, rng, strict_margin=1e-6):
        super().__init__(X, rng)
        self.retained = RetainedAnswers(eps=strict_margin)

    @staticmethod
    def choose(PMR, unasked_set, n):
        """The Css choice given the matrix PMR[i, j]; returns (pair or None, info)."""
        pool = list(range(n))
        certified = []
        info = {}
        while len(pool) >= 2:
            MR = {i: max(PMR[i, j] for j in pool) for i in pool}      # includes j = i (0)
            mmr = min(MR.values())
            xstar = min(i for i in pool if MR[i] <= mmr + TIE_TOL)
            if mmr <= TIE_TOL:
                certified.append(xstar)
                pool.remove(xstar)
                continue
            cand = [(PMR[xstar, j], j) for j in pool if j != xstar and _key(xstar, j) in unasked_set]
            if not cand:
                certified.append(xstar)
                pool.remove(xstar)
                info['adversaries_exhausted'] = info.get('adversaries_exhausted', 0) + 1
                continue
            best = max(v for v, _ in cand)
            ystar = min(j for v, j in cand if v >= best - TIE_TOL)
            info.update({'mmr': float(mmr), 'certified': len(certified), 'xstar': int(xstar), 'ystar': int(ystar)})
            return _key(xstar, ystar), info
        info['certified'] = len(certified)
        return None, info

    def select(self, prefs, unasked):
        info = {'rule': self.name, 'exact': True}
        prefs, C = self._answer_rows(prefs)
        self.retained.update(C)
        self.retained.describe(info)
        P = AnswerPolytope(self.X, prefs[self.retained.mask], eps=0.0)
        n = self.n
        PMR = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    PMR[i, j] = P.pmr(i, j)
        pair, cinfo = self.choose(PMR, set(_key(*p) for p in unasked), n)
        info.update(cinfo)
        if pair is None:
            return self._fallback(unasked, info, 'pool_exhausted')
        self.rounds += 1
        return pair, info


# ============================================================================= Cheb
class ChebRule(Rule):
    """Candidates: pairs with no necessary relation in either direction (LP 11 with the margin
    eps_D = 1e-3 of Liang et al.).  For each candidate, the Chebyshev centres u+ and u- of the two sets
    obtained by adding either hypothetical answer (LP 22, no margin); d+ = u+ . (v_i - v_j),
    d- = u- . (v_j - v_i); ask the pair minimising max(d+/d-, d-/d+).  Ties: first pair in
    enumeration order.  Stops when no candidate is left."""
    name = 'CHEB'

    def __init__(self, X, rng, eps_D=1e-3):
        super().__init__(X, rng)
        self.eps_D = eps_D
        self.retained = RetainedAnswers(eps=eps_D)
        self.nec_cache = {}
        self.cache_version = 0

    def select(self, prefs, unasked):
        info = {'rule': self.name, 'exact': True}
        prefs, C = self._answer_rows(prefs)
        self.retained.update(C)
        self.retained.describe(info)
        if self.retained.version != self.cache_version:
            self.nec_cache = {}
            self.cache_version = self.retained.version
        P = AnswerPolytope(self.X, prefs[self.retained.mask], eps=self.eps_D)
        X = self.X
        Q = [(i, j) for (i, j) in unasked if not (self._nec(P, i, j) or self._nec(P, j, i))]
        info['n_candidates'] = len(Q)
        if not Q:
            return self._fallback(unasked, info, 'candidate_set_empty')
        best_ratio, best_pair = np.inf, None
        for (i, j) in Q:
            row = (X[i] - X[j])[None, :]
            u_plus, _ = P.chebyshev_center(extra_C=row, row_rhs=0.0)
            u_minus, _ = P.chebyshev_center(extra_C=-row, row_rhs=0.0)
            if u_plus is None or u_minus is None:
                continue
            d_plus = float(u_plus @ (X[i] - X[j]))
            d_minus = float(u_minus @ (X[j] - X[i]))
            ratio = np.inf if (d_plus <= 0 or d_minus <= 0) else max(d_plus / d_minus, d_minus / d_plus)
            if ratio < best_ratio - 1e-12:
                best_ratio, best_pair = ratio, (i, j)
        if best_pair is None:
            return self._fallback(unasked, info, 'all_ratios_undefined')
        info['ratio'] = float(best_ratio)
        self.rounds += 1
        return best_pair, info

    def _nec(self, P, i, j):
        if (i, j) in self.nec_cache:
            return True
        nec = P.necessary(i, j, mode='fixed')
        if nec:
            self.nec_cache[(i, j)] = True      # monotone while the retained set does not change
        return nec


# ============================================================================== Dvf
class DvfRule(Rule):
    """Among the pairs with no necessary relation (Greco et al. 2008 LP), ask the pair maximising
    min{PWI(i, k), PWI(k, i)}, with the pairwise winning indices estimated from `n_samples`
    hit-and-run draws of the compatible set; exact ties broken uniformly at random.  Thinning
    (d - 1)^3 as in Ciomek et al. when it is at most 200, 10 d otherwise; burn-in 2000 draws."""
    name = 'DVF'

    def __init__(self, X, rng, n_samples=10000, thinning=None, burn=None, strict_margin=1e-6):
        super().__init__(X, rng)
        self.n_samples = int(n_samples)
        if thinning is None:
            cubic = (self.d - 1) ** 3
            thinning = cubic if cubic <= 200 else 10 * self.d
        self.thinning = int(thinning)
        self.burn = int(burn) if burn is not None else 2000
        self.retained = RetainedAnswers(eps=strict_margin)
        self.nec_cache = {}
        self.cache_version = 0

    def select(self, prefs, unasked):
        info = {'rule': self.name, 'exact': True}
        prefs, C = self._answer_rows(prefs)
        self.retained.update(C)
        self.retained.describe(info)
        if self.retained.version != self.cache_version:
            self.nec_cache = {}
            self.cache_version = self.retained.version
        P = AnswerPolytope(self.X, prefs[self.retained.mask], eps=0.0)
        Q = [(i, j) for (i, j) in unasked if not (self._nec(P, i, j) or self._nec(P, j, i))]
        info['n_candidates'] = len(Q)
        if not Q:
            return self._fallback(unasked, info, 'candidate_set_empty')
        w0, r0 = P.chebyshev_center()
        if w0 is None or r0 <= 1e-10:
            return self._fallback(unasked, info, 'degenerate_polytope')
        A, b = P.constraint_rows()
        seed = int(self.rng.integers(2 ** 31 - 1))
        S = hit_and_run(w0, A, b, self.n_samples, self.thinning, self.burn, seed)
        U = S @ self.X.T
        scores = np.empty(len(Q))
        for q, (i, j) in enumerate(Q):
            scores[q] = min(np.mean(U[:, i] > U[:, j]), np.mean(U[:, j] > U[:, i]))
        best = scores.max()
        tied = np.where(scores >= best)[0]
        k = tied[int(self.rng.integers(len(tied)))] if len(tied) > 1 else tied[0]
        info['score'] = float(best)
        info['n_tied'] = int(len(tied))
        self.rounds += 1
        return Q[k], info

    def _nec(self, P, i, j):
        if (i, j) in self.nec_cache:
            return True
        nec = P.necessary(i, j, mode='shared')
        if nec:
            self.nec_cache[(i, j)] = True
        return nec


# ============================================================================= Poly
class PolyRule(Rule):
    """Steps 1-3 of Toubia, Hauser & Simester (2004) on a fixed pool, J = 2.

    Step 1: analytic centre ubar and longest axis g of the polyhedron {X u = a, u >= 0} with
            u = (w, s), one slack per answer, and the normalisation 1^T w = 1.
    Step 2: extreme points u_1 = ubar + alpha_1 g, u_2 = ubar - alpha_2 g where the line leaves the
            polyhedron.
    Step 3: for j = 1, 2 the alternative of largest value under u_j subject to the budget
            v(a)^T ubar <= M, M ~ U[0, 0.5]; M is redrawn (up to 30 times) until the two
            alternatives differ and have not been compared; otherwise the rule stops.
    Inconsistent answers: every constraint is relaxed by the smallest common slack that makes the
    set non-empty (OPT4 of Toubia et al. 2003), and the rule asks the pair whose values are closest
    under the analytic centre (permanently, from the first contradiction on)."""
    name = 'POLY'

    def __init__(self, X, rng, budget_max=0.5, max_redraws=30):
        super().__init__(X, rng)
        self.budget_max = float(budget_max)
        self.max_redraws = int(max_redraws)
        self.infeasible_since = None

    def _augmented(self, C, slack):
        t, d = C.shape
        Xaug = np.zeros((t + 1, d + t))          # rows [C_k, -e_k] u = -slack ; [1^T, 0] u = 1
        Xaug[:t, :d] = C
        Xaug[:t, d:] = -np.eye(t)
        Xaug[t, :d] = 1.0
        a = np.r_[np.full(t, -slack), 1.0]
        return Xaug, a

    def _center_and_axis(self, C, slack):
        """Analytic centre and axis; retried with a tiny extra slack if round-off makes LP1 infeasible."""
        for extra in (0.0, 1e-9, 1e-7):
            Xaug, a = self._augmented(C, slack + extra) if len(C) else self._augmented(C, 0.0)
            res = analytic_center_and_axis(Xaug, a)
            if res is not None:
                return res
        return None

    def select(self, prefs, unasked):
        info = {'rule': self.name, 'exact': True}
        prefs, C = self._answer_rows(prefs)
        X = self.X
        slack = 0.0
        if len(C):
            slack = uniform_linf_slack(C, 0.0)
            if slack <= TOL:
                slack = 0.0
        if slack > 0 and self.infeasible_since is None:
            self.infeasible_since = len(prefs)
        res = self._center_and_axis(C, slack)
        if res is None:
            return self._fallback(unasked, info, 'no_interior_point')
        ubar, g = res['ubar'], res['g']
        w_hat = ubar[:self.d]
        if self.infeasible_since is not None or res.get('point') or np.linalg.norm(g) < 1e-12:
            diffs = np.array([X[i] - X[j] for (i, j) in unasked])
            bal = np.abs(diffs @ w_hat)
            k = int(np.argmin(bal))
            info.update({'stage': 2 if self.infeasible_since is not None else 3, 'exact': False,
                         'slack': float(slack), 'balance': float(bal[k])})
            self.rounds += 1
            return tuple(unasked[k]), info
        neg, pos = g < -1e-15, g > 1e-15
        alpha1 = np.min(-ubar[neg] / g[neg]) if np.any(neg) else 0.0
        alpha2 = np.min(ubar[pos] / g[pos]) if np.any(pos) else 0.0
        u1 = (ubar + alpha1 * g)[:self.d]
        u2 = (ubar - alpha2 * g)[:self.d]
        util_c, util_1, util_2 = X @ w_hat, X @ u1, X @ u2
        unasked_set = set(_key(*p) for p in unasked)
        for draw in range(self.max_redraws):
            M = float(self.rng.uniform(0.0, self.budget_max))
            feas = np.where(util_c <= M + 1e-12)[0]
            if len(feas) < 2:
                continue
            z1 = int(feas[np.argmax(util_1[feas])])
            z2 = int(feas[np.argmax(util_2[feas])])
            if z1 == z2 or _key(z1, z2) not in unasked_set:
                continue
            info.update({'stage': 1, 'budget': M, 'draws': draw + 1,
                         'balance': float(abs(util_c[z1] - util_c[z2]))})
            self.rounds += 1
            return _key(z1, z2), info
        return self._fallback(unasked, info, 'no_distinct_profiles')


RULES = {'POLY': PolyRule, 'DVF': DvfRule, 'CHEB': ChebRule, 'CSS': CssRule}
