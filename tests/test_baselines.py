"""Exactness of the baseline rules against the worked examples printed in the source papers:
Liang et al. (2025) Tables 2-4 (Cheb), the regret example of Boutilier et al. (2006) (Css), the necessary
relation, pairwise winning indices and rank acceptabilities of Ciomek et al. (2017) (Dvf), and the analytic
centre and longest axis of Toubia et al. (2003, 2004) checked against generic solvers (Poly).

    python -m pytest tests
"""
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from activepor.baselines.polytope import AnswerPolytope
from activepor.baselines.rules import ChebRule, CssRule, DvfRule, PolyRule
from activepor.baselines.hit_and_run import hit_and_run, pairwise_winning_index, rank_acceptability_index
from activepor.baselines.analytic_center import (analytic_center_and_axis, interior_point_lp1, analytic_center,
                                                  longest_axis)


# --------------------------------------------------------------------------- shared data
# Liang et al. (2025) Table 1 == Ciomek et al. (2017) Table 1: six alternatives, three gain criteria.
PERF = np.array([[10, 50, 70],
                 [34, 56, 84],
                 [40, 90, 45],
                 [30, 10, 70],
                 [60, 80, 45],
                 [49, 56, 54]], dtype=float)


def linear_features(perf):
    mn, mx = perf.min(0), perf.max(0)
    return (perf - mn) / (mx - mn)


X6 = linear_features(PERF)


# ================================================================== Liang et al. (2025)
def test_liang_worked_example():
    """Tables 2-4 of Liang et al. (2025): candidate sets, ratios, selected pairs; D(0) = {(a3,a1),(a5,a2),(a1,a4)}."""
    rng = np.random.default_rng(0)
    rule = ChebRule(X6, rng, eps_D=1e-3)
    prefs = [(2, 0), (4, 1), (0, 3)]                     # 0-based
    expected = [
        ({(1, 2): 2.995, (2, 4): 1.706, (2, 5): 42.419, (1, 5): 1.248, (0, 5): 15.129}, (1, 5), (1, 5)),
        ({(1, 2): 3.352, (2, 4): 1.371, (2, 5): 104.213, (0, 5): 5.371}, (2, 4), (4, 2)),
        ({(1, 2): 1.646, (2, 5): 55.314}, (1, 2), (2, 1)),
    ]
    all_pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    for it, (table, sel, answer) in enumerate(expected):
        asked = set(frozenset(p) for p in prefs)
        unasked = [p for p in all_pairs if frozenset(p) not in asked]
        P = AnswerPolytope(X6, np.array(prefs), eps=1e-3)
        Q = [p for p in unasked if not (P.necessary(p[0], p[1]) or P.necessary(p[1], p[0]))]
        assert set(Q) == set(table.keys()), (it, Q, table.keys())
        ratios = {}
        for (i, j) in Q:
            row = (X6[i] - X6[j])[None, :]
            up, _ = P.chebyshev_center(extra_C=row, row_rhs=0.0)
            um, _ = P.chebyshev_center(extra_C=-row, row_rhs=0.0)
            dp = up @ (X6[i] - X6[j])
            dm = um @ (X6[j] - X6[i])
            ratios[(i, j)] = max(dp / dm, dm / dp)
        for k, v in table.items():
            assert abs(ratios[k] - v) / v < 2.5e-3, (it, k, ratios[k], v)
        pair, info = rule.select(np.array(prefs), unasked)
        assert pair == sel, (it, pair, sel)
        prefs.append(answer)
    # after the third answer the candidate set is empty (f_UNP = 0) -> fallback
    P = AnswerPolytope(X6, np.array(prefs), eps=1e-3)
    asked = set(frozenset(p) for p in prefs)
    unasked = [p for p in all_pairs if frozenset(p) not in asked]
    Q = [p for p in unasked if not (P.necessary(p[0], p[1]) or P.necessary(p[1], p[0]))]
    assert Q == []
    # final ranking a5 > a3 > a2 > a6 > a1 > a4 is the necessary relation
    order = [4, 2, 1, 5, 0, 3]
    for a in range(6):
        for b in range(a + 1, 6):
            assert P.necessary(order[a], order[b])


# ======================================================= Boutilier et al. (2006) p. 13
def test_css_boutilier_box_example():
    """Two GAI factors with interval bounds; PMR over the box; MMR = 7, x* = AB~C, witness A~BC."""
    lo = np.array([9, 5, 10, 0, 2, 0, 4, 0], dtype=float)
    hi = np.array([17, 8, 11, 3, 4, 10, 8, 5], dtype=float)
    # items as selection vectors: (AB-factor index, BC-factor index)
    items = {'ABC': (0, 4), 'AB~C': (0, 5), 'A~BC': (1, 6), '~ABC': (2, 4), '~AB~C': (2, 5)}
    names = list(items.keys())
    Xb = np.zeros((5, 8))
    for k, nm in enumerate(names):
        Xb[k, items[nm][0]] = 1
        Xb[k, items[nm][1]] = 1
    n = 5
    PMR = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                c = Xb[j] - Xb[i]
                res = linprog(-c, bounds=list(zip(lo, hi)), method='highs')
                PMR[i, j] = -res.fun
    expected = np.array([[0, 8, 5, 2, 10],
                         [4, 0, 7, 6, 2],
                         [12, 18, 0, 6, 12],
                         [7, 15, 4, 0, 8],
                         [11, 7, 6, 4, 0]], dtype=float)
    assert np.allclose(PMR, expected), PMR
    MR = PMR.max(1)
    assert np.allclose(MR, [10, 7, 18, 15, 11])
    unasked = set((i, j) for i in range(n) for j in range(i + 1, n))
    pair, info = CssRule.choose(PMR, unasked, n)
    assert names[info['xstar']] == 'AB~C' and names[info['ystar']] == 'A~BC'
    assert abs(info['mmr'] - 7) < 1e-9
    assert pair == (1, 2)


# ============================================================ Ciomek et al. (2017)
def test_ciomek_necessary_relations_and_indices():
    """Illustrative example: S1 = {a3 > a1, a5 > a2}; Hasse of the necessary relation (Fig. 2a),
    PWIs (Fig. 2b) and RAIs (Table 2a) within Monte-Carlo error."""
    prefs = np.array([(2, 0), (4, 1)])
    P = AnswerPolytope(X6, prefs, eps=0.0)
    hasse = {(4, 5), (4, 1), (2, 0), (5, 3), (1, 3), (1, 0)}
    nec = {(i, j) for i in range(6) for j in range(6) if i != j and P.necessary(i, j, mode='shared')}
    # transitive closure of the Hasse diagram
    clos = set(hasse)
    changed = True
    while changed:
        changed = False
        for (a, b) in list(clos):
            for (c, d) in list(clos):
                if b == c and (a, d) not in clos:
                    clos.add((a, d))
                    changed = True
    assert nec == clos, (nec ^ clos)
    Q = [(i, j) for i in range(6) for j in range(i + 1, 6) if (i, j) not in nec and (j, i) not in nec]
    assert len(Q) == 7
    w0, r0 = P.chebyshev_center()
    A, b = P.constraint_rows()
    S = hit_and_run(w0, A, b, 10000, thinning=(3 - 1) ** 3, burn=1000, seed=1)
    pwi = pairwise_winning_index(X6, S)
    # Fig. 2(b) of Ciomek et al. (2017) (labels at the arc heads), values checked on the polytope
    # with 2e6 uniform samples: PWI(a5,a3)=0.792, PWI(a3,a2)=0.732, PWI(a3,a6)=0.599,
    # PWI(a3,a4)=0.980, PWI(a6,a2)=0.568, PWI(a6,a1)=0.991, PWI(a1,a4)=0.517.
    fig2b = {(4, 2): 0.792, (2, 1): 0.732, (2, 5): 0.599, (2, 3): 0.980, (5, 1): 0.568, (5, 0): 0.991, (0, 3): 0.517}
    for (i, j), v in fig2b.items():
        assert abs(pwi[i, j] - v) < 0.02, ((i, j), pwi[i, j], v)
    rai = rank_acceptability_index(X6, S)
    table2a = np.array([[0.00, 0.00, 0.00, 0.01, 0.50, 0.49],
                        [0.00, 0.16, 0.38, 0.46, 0.00, 0.00],
                        [0.20, 0.34, 0.24, 0.20, 0.02, 0.00],
                        [0.00, 0.00, 0.00, 0.02, 0.47, 0.51],
                        [0.80, 0.20, 0.00, 0.00, 0.00, 0.00],
                        [0.00, 0.30, 0.38, 0.31, 0.01, 0.00]])
    assert np.max(np.abs(rai - table2a)) < 0.04, np.round(rai, 2)
    # measures of Table 3 (f_PWI, f_RAI) from the sampled indices
    def h(p):
        return -p * np.log2(p) if p > 0 else 0.0
    f_pwi = sum(h(pwi[i, j]) for i in range(6) for j in range(6) if i != j) / 30
    f_rai = sum(h(rai[i, k]) for i in range(6) for k in range(6)) / 6
    assert abs(f_pwi - 0.1584) < 0.02 and abs(f_rai - 1.3481) < 0.08, (f_pwi, f_rai)
    # H_DVF on S1 picks the pair with the most balanced PWI: (a1, a4) with 0.51/0.49
    rule = DvfRule(X6, np.random.default_rng(3), n_samples=10000, thinning=8, burn=1000)
    all_pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    unasked = [p for p in all_pairs if frozenset(p) not in {frozenset(x) for x in prefs}]
    pair, info = rule.select(prefs, unasked)
    assert info['n_candidates'] == 7
    assert pair == (0, 3), (pair, info)


# ============================================================ Toubia et al. (2003/2004)
def test_toubia_analytic_center_and_axis():
    """Analytic centre = maximiser of sum ln u_j on {X u = a, u > 0} (checked against a generic
    solver), the longest axis solves OPT3 (checked by random search on the ellipsoid), and on a
    symmetric box the centre is the box centre."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(7)
    d, t = 5, 4
    Xf = rng.random((12, d))
    prefs = []
    w_true = rng.dirichlet(np.ones(d))
    while len(prefs) < t:
        i, j = rng.choice(12, 2, replace=False)
        prefs.append((i, j) if Xf[i] @ w_true > Xf[j] @ w_true else (j, i))
    prefs = np.array(prefs)
    C = Xf[prefs[:, 0]] - Xf[prefs[:, 1]]
    rule = PolyRule(Xf, rng)
    Xaug, a = rule._augmented(C, 0.0)
    res = analytic_center_and_axis(Xaug, a)
    ubar, g = res['ubar'], res['g']
    assert np.allclose(Xaug @ ubar, a, atol=1e-8) and np.all(ubar > 0)
    # generic solver for max sum ln u s.t. X u = a
    N = Xaug.shape[1]
    cons = {'type': 'eq', 'fun': lambda u: Xaug @ u - a}
    sol = minimize(lambda u: -np.sum(np.log(np.maximum(u, 1e-300))), ubar * (1 + 0.1 * rng.random(N)),
                   constraints=[cons], bounds=[(1e-9, None)] * N, method='SLSQP', options={'ftol': 1e-14, 'maxiter': 2000})
    assert np.sum(np.log(ubar)) >= np.sum(np.log(sol.x)) - 1e-6, (np.sum(np.log(ubar)), np.sum(np.log(sol.x)))
    # OPT3: g maximises g^T g subject to g^T Ubar^-2 g <= 1, X g = 0
    from scipy import linalg
    Z = linalg.null_space(Xaug)
    Wt = 1 / ubar ** 2
    def length(h):
        gg = Z @ h
        return (gg @ gg) / (gg @ (Wt * gg))
    best = max(length(rng.standard_normal(Z.shape[1])) for _ in range(20000))
    assert length(Z.T @ g) >= best - 1e-6
    assert np.allclose(Xaug @ g, 0, atol=1e-9)
    # symmetric box {0 <= u <= 1}^3 in equality form: centre at 0.5
    Xb = np.hstack([np.eye(3), np.eye(3)])          # u + s = 1
    ab = np.ones(3)
    u0, zm = interior_point_lp1(Xb, ab)
    ub = analytic_center(Xb, ab, u0)
    assert np.allclose(ub, 0.5, atol=1e-7)


def test_polytope_consistency_handling():
    """Contradictory answers: minimum-cardinality removal restores a consistent set."""
    from activepor.baselines.consistency import min_cardinality_retained, is_consistent
    rng = np.random.default_rng(11)
    Xf = rng.random((10, 4))
    w_true = rng.dirichlet(np.ones(4))
    pairs = [(i, j) for i in range(10) for j in range(i + 1, 10)]
    rng.shuffle(pairs)
    prefs, flipped = [], []
    for k, (i, j) in enumerate(pairs):
        win, lose = (i, j) if Xf[i] @ w_true > Xf[j] @ w_true else (j, i)
        if k % 5 == 0:
            win, lose = lose, win
            flipped.append(k)
        prefs.append((win, lose))
    prefs = np.array(prefs)
    C = Xf[prefs[:, 0]] - Xf[prefs[:, 1]]
    assert not is_consistent(C, 1e-6)
    mask, method = min_cardinality_retained(C, 1e-6)
    assert is_consistent(C, 1e-6, mask)
    assert (~mask).sum() <= len(flipped)
