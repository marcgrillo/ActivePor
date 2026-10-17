"""Query-selection scores.

US       predictive entropy h(p_t(1|q))
BALD     mutual information I(Y_q; W | D_{t-1})
RANKUNC  -|POI_t(q) - 1/2| on the posterior draws             (Bayesian learners only)

Bayesian learners use Monte Carlo estimators on their posterior draws.  FTRL learners use
Laplace-Taylor scores: second-order expansions of the response probability under the Gaussian
(Laplace) approximation around the MAP estimate.  Every rule asks the candidate with the largest
score; ties go to the first pair of the enumeration order of `unasked_pairs`.
"""
import itertools
import numpy as np
from scipy.special import expit

from .model import EPS, entropy, response_probability
from .learners import simplex_interior

RULES = ('US', 'BALD', 'RANKUNC')


def unasked_pairs(n, answers):
    """Candidate pool P_t: pairs (i, j), i < j, not asked yet (one orientation per pair)."""
    asked = set()
    for a, b in answers:
        asked.add((int(a), int(b)))
        asked.add((int(b), int(a)))
    return [p for p in itertools.combinations(range(n), 2) if p not in asked]


def scores(learner, pairs, rule):
    """Score of every candidate pair (i, j) for `rule` ('US', 'BALD' or 'RANKUNC')."""
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    R = learner.V[pairs[:, 0]] - learner.V[pairs[:, 1]]
    if learner.kind == 'BAYES':
        return mc_scores(R, learner.samples, learner.link, rule)
    if rule == 'RANKUNC':
        raise ValueError('RankUnc needs posterior draws: it exists for the Bayesian learners only')
    return lt_scores(R, learner.w, learner.Sigma, learner.link, rule)


def mc_scores(R, W, link, rule):
    """Monte Carlo scores from posterior draws W (S x d)."""
    Z = R @ W.T                                            # value differences, (pairs, S)
    if rule == 'RANKUNC':
        poi = np.mean(Z > 0.0, axis=1)
        return -np.abs(poi - 0.5)
    P = np.clip(response_probability(Z, link), EPS, 1.0 - EPS)
    H = entropy(P.mean(axis=1))                            # h(p_hat_t)
    if rule == 'US':
        return H
    if rule == 'BALD':
        return H - entropy(P).mean(axis=1)                 # minus the mean conditional entropy
    raise ValueError(f'unknown rule {rule!r}')


def lt_scores(R, w, Sigma, link, rule):
    """Laplace-Taylor scores around the MAP estimate w with covariance Sigma (in w)."""
    if link == 'BT':
        eta = expit(R @ w)
        var_z = np.einsum('ij,jk,ik->i', R, Sigma, R)     # r^T Sigma r
        if rule == 'US':                                    # h(eta + tr(G_q Sigma) / 2)
            return entropy(eta + 0.5 * eta * (1.0 - eta) * (1.0 - 2.0 * eta) * var_z)
        if rule == 'BALD':                                  # Var(eta) / (2 eta (1 - eta))
            p = np.clip(eta, EPS, 1.0 - EPS)
            return 0.5 * p * (1.0 - p) * var_z
        raise ValueError(f'unknown rule {rule!r}')
    w = simplex_interior(w)
    p = np.clip(0.5 * (1.0 + R @ w), EPS, 1.0 - EPS)
    if rule == 'US':                                        # eta is affine in w (G_q = 0)
        return entropy(p)
    if rule == 'BALD':                                      # Var(eta) / (2 eta (1 - eta))
        A = 0.5 * R                                         # gradient of eta (= x_q on the tangent space)
        var_p = np.einsum('ij,jk,ik->i', A, Sigma, A)
        return 0.5 * var_p / (p * (1.0 - p))
    raise ValueError(f'unknown rule {rule!r}')
