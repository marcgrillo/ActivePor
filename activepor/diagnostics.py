"""Monitoring quantities computed during an elicitation.

* Information-gain residual: before the answer to the selected query is observed, the hypothetical
  surprises S_t(1), S_t(0) and the expected surprise B_t (the BALD score of the query); after it,
  the residual xi_t = S_t(y_t) - B_t.  `LinkResidual` accumulates
  E_link(T) = |sum xi_t| / (sum B_t + 1e-6).
* Curvature ratio eps^Lap of the FTRL objective at its minimiser: the largest ratio between the
  cubic and the quadratic term of its Taylor expansion at one-standard-deviation displacements.
* Residual information R_t = sum of the BALD scores of the unasked pairs; the one-bit stopping rule
  stops when R_t < log 2.
* Split-R-hat of the posterior draws of the Bayesian learners.
"""
import warnings

import numpy as np
from scipy.special import expit
from scipy.stats import norm, rankdata

from .model import EPS, response_probability
from .learners import ALPHA0, KAPPA, laplace, map_estimate, simplex_interior, tangent_basis
from .acquisition import scores

ONE_BIT = np.log(2.0)


# ----------------------------------------------------------------- information-gain residual
def hypothetical_surprises(learner, pair):
    """(p_t(1), S_t(1), S_t(0), B_t) for the query pair = (i, j), y = 1 meaning a_i preferred.

    Bayesian learners: self-normalised reweighting of the posterior draws by the likelihood of each
    answer.  FTRL learners: refit with each hypothetical answer and take the Kullback-Leibler
    divergence between the updated and the current Laplace Gaussians (in the coordinates of the
    chart), averaged with the plug-in probabilities."""
    i, j = int(pair[0]), int(pair[1])
    r = learner.V[i] - learner.V[j]
    if learner.kind == 'BAYES':
        p1 = np.clip(response_probability(learner.samples @ r, learner.link), EPS, 1.0 - EPS)
        p0 = 1.0 - p1
        m1, m0 = p1.mean(), p0.mean()
        s1 = np.sum(p1 / p1.sum() * np.log(p1 / m1))
        s0 = np.sum(p0 / p0.sum() * np.log(p0 / m0))
        return m1, s1, s0, m1 * s1 + m0 * s0
    m1 = float(np.clip(response_probability(learner.w @ r, learner.link), EPS, 1.0 - EPS))
    m0 = 1.0 - m1
    mu, cov = _chart_mean(learner.w, learner.link), learner.Sigma_chart
    surprise = {}
    for y, answer in ((1, (i, j)), (0, (j, i))):
        R = learner.diffs(learner.answers + [answer])
        w_y = map_estimate(R, learner.link)
        cov_y, _ = laplace(R, w_y, learner.link)
        surprise[y] = gaussian_kl(_chart_mean(w_y, learner.link), cov_y, mu, cov)
    return m1, surprise[1], surprise[0], m1 * surprise[1] + m0 * surprise[0]


def _chart_mean(w, link):
    """Coordinates of w in the chart of the Laplace approximation: w itself for the Bradley-Terry
    link, Psi^T w in the affine chart of the Linear link (only differences between means matter)."""
    w = np.asarray(w, dtype=float)
    if link == 'BT':
        return w
    return tangent_basis(len(w)).T @ w


def gaussian_kl(mu1, S1, mu0, S0, ridge=1e-8):
    """KL(N(mu1, S1) || N(mu0, S0)), clipped at 0."""
    k = len(mu1)
    S0 = S0 + np.eye(k) * ridge
    S1 = S1 + np.eye(k) * ridge
    try:
        diff = mu0 - mu1
        tr = np.trace(np.linalg.solve(S0, S1))
        maha = diff @ np.linalg.solve(S0, diff)
        logdet = np.linalg.slogdet(S0)[1] - np.linalg.slogdet(S1)[1]
        return max(0.0, 0.5 * (tr + maha - k + logdet))
    except np.linalg.LinAlgError:
        warnings.warn('Gaussian KL: singular covariance, surprise set to 0', RuntimeWarning)
        return 0.0


class LinkResidual:
    """Running information-gain residual E_link(T) = |sum xi_t| / (sum B_t + eps)."""

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.sum_xi = 0.0
        self.sum_b = 0.0

    def update(self, surprise_realized, expected):
        xi = surprise_realized - expected
        self.sum_xi += xi
        self.sum_b += expected
        return xi, abs(self.sum_xi) / (self.sum_b + self.eps)


# ----------------------------------------------------------------------------- curvature ratio
def curvature_ratio(learner):
    """eps^Lap of an FTRL learner at its current estimate.

    With L the lower Cholesky factor of the Laplace covariance in the chart coordinates, the
    displacements delta_j = +-L e_j have quadratic term Q(delta_j) = 1/2, so the ratio
    |C(delta_j)| / Q(delta_j) is |D_j| / 3, with D_j the third derivative of the objective along
    L e_j, computed in closed form.  For the Linear link the direction in w is Psi L e_j (affine
    chart)."""
    try:
        L = np.linalg.cholesky(learner.Sigma_chart)
    except np.linalg.LinAlgError:
        warnings.warn('curvature ratio: the Laplace covariance is not positive definite', RuntimeWarning)
        return np.nan
    R = learner.diffs()
    D = []
    if learner.link == 'BT':
        w = learner.w
        p = expit(R @ w)
        for v in L.T:
            D.append(np.sum(p * (1 - p) * (1 - 2 * p) * (R @ v) ** 3) - 2.0 * (ALPHA0 - 1.0) * np.sum((v / w) ** 3))
    else:
        w = simplex_interior(learner.w)
        a = np.clip(0.5 * (1.0 + R @ w), 1e-12, None)          # probability of each observed answer
        for v in (tangent_basis(len(w)) @ L).T:
            b = 0.5 * (R @ v)
            D.append(-2.0 * np.sum((b / a) ** 3) - 2.0 * KAPPA * np.sum((v / w) ** 3))
    return float(np.max(np.abs(D)) / 3.0)


# ------------------------------------------------------------------------ residual information
def residual_information(learner, pool):
    """R_t = sum over the unasked pairs of the BALD score."""
    if len(pool) == 0:
        return np.nan
    return float(np.sum(scores(learner, pool, 'BALD')))


# --------------------------------------------------------------------------------- split-R-hat
def split_rhat(draws):
    """Rank-normalised split-R-hat (Vehtari et al., 2021) of one chain of draws (S x d).

    The chain is split into two halves; for every coordinate, R-hat is computed on the normal
    scores of the ranks (bulk) and of the ranks of the distances to the median (tails), and the
    largest of all these values is returned.  Values close to 1 indicate that the two halves agree."""
    x = np.asarray(draws, dtype=float)
    half = x.shape[0] // 2
    if half < 2:
        return np.nan
    x = x[:2 * half]

    def normal_scores(z):
        u = (rankdata(z, axis=0) - 0.375) / (z.shape[0] + 0.25)
        return norm.ppf(u).reshape(2, half, -1)

    def rhat(z):
        means = z.mean(axis=1)
        W = z.var(axis=1, ddof=1).mean(axis=0)
        B = half * means.var(axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.sqrt(((half - 1) / half * W + B / half) / W)
        return np.where(W > 0, r, np.nan)

    values = np.concatenate([rhat(normal_scores(x)), rhat(normal_scores(np.abs(x - np.median(x, axis=0))))])
    return float(np.nanmax(values)) if np.any(np.isfinite(values)) else np.nan
