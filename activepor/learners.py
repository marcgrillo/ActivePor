"""The four POR learners.

BAYES-LIN  Dirichlet(0.5) prior on the simplex, random-walk Metropolis-Hastings with a
           multiplicative log-normal proposal renormalised onto the simplex.
BAYES-BT   independent Gamma(shape 1, scale 1) priors on R^d_+, Hamiltonian Monte Carlo in log w.
FTRL-LIN   MAP with the regulariser -kappa sum log w (kappa = 0.1) on the simplex (SLSQP).
FTRL-BT    MAP with the regulariser sum [beta0 w - (alpha0 - 1) log w] (alpha0 = 2, beta0 = 1) on
           R^d_+ (SCS through CVXPY).

Both samplers adapt their step during `n_tune` tuning iterations (discarded) to a target
acceptance rate (0.65 for HMC, 0.3 for MH) and start from the posterior mean of the previous round.
The FTRL learners also return the Laplace approximation of their objective at the minimiser.  For
the Linear link the simplex constraint is removed by the affine chart w = w_hat + Psi nu, with Psi an
orthonormal basis of the tangent space {u : 1^T u = 0} (`tangent_basis`): `Sigma_chart` is the
covariance in nu, (Psi^T H Psi)^{-1}, and `Sigma` = Psi Sigma_chart Psi^T its image in w.
"""
import warnings

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from scipy.special import expit

DIRICHLET_CONCENTRATION = 0.5          # BAYES-LIN prior
GAMMA_SHAPE, GAMMA_SCALE = 1.0, 1.0    # BAYES-BT prior
KAPPA = 0.1                            # FTRL-LIN regulariser
ALPHA0, BETA0 = 2.0, 1.0               # FTRL-BT regulariser
RIDGE = 1e-8                           # added to the Hessian before it is inverted


class Learner:
    """Common bookkeeping: the characteristic vectors V and the answers received so far."""
    kind = None

    def __init__(self, V, link):
        if link not in ('LIN', 'BT'):
            raise ValueError(f'unknown link {link!r}')
        self.V = np.asarray(V, dtype=float)
        self.link = link
        self.d = self.V.shape[1]
        self.answers = []                       # [(winner, loser), ...] in the order received

    @property
    def name(self):
        return f'{self.kind}-{self.link}'

    def observe(self, winner, loser):
        self.answers.append((int(winner), int(loser)))

    def diffs(self, answers=None):
        """Rows r = v(winner) - v(loser): every observed answer has outcome y = 1 in this orientation."""
        answers = self.answers if answers is None else answers
        if len(answers) == 0:
            return np.empty((0, self.d))
        a = np.asarray(answers, dtype=int)
        return self.V[a[:, 0]] - self.V[a[:, 1]]


# ============================================================================ Bayesian learners
class BayesianLearner(Learner):
    """Posterior represented by `n_samples` MCMC draws."""
    kind = 'BAYES'

    def __init__(self, V, link, n_samples=2000, n_tune=500, rng=None):
        super().__init__(V, link)
        self.n_samples = int(n_samples)
        self.n_tune = int(n_tune)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.samples = None                     # (S, d) after fit()

    def fit(self):
        start = None if self.samples is None else self.samples.mean(axis=0)
        sampler = hmc_bt if self.link == 'BT' else mh_lin
        self.samples = sampler(self.diffs(), self.n_samples, self.n_tune, self.rng, start)
        return self

    def draws(self):
        return self.samples


def hmc_bt(R, n_samples, n_tune, rng, start=None, n_leapfrog=10, step_size=0.05, target_accept=0.65):
    """HMC for BAYES-BT in theta = log w (Gamma prior plus the Jacobian of the log transform).

    The step size is adapted during `n_tune` iterations by the stochastic approximation
    eps <- eps * exp((a_t - target) / sqrt(t + 1)); the chain starts at log(start) (the posterior
    mean of the previous round) or at the prior mean w = 1."""
    d = R.shape[1]
    alpha, scale = GAMMA_SHAPE, GAMMA_SCALE

    def energy(theta):
        w = np.exp(theta)
        loglik = np.sum(-np.logaddexp(0.0, -(R @ w)))
        logprior = np.sum((alpha - 1.0) * theta - w / scale + theta)
        return -(loglik + logprior)

    def grad(theta):
        w = np.exp(theta)
        g_w = R.T @ expit(-(R @ w))
        return -(g_w * w + alpha - w / scale)

    def proposal(theta, p, eps):
        theta, p = theta.copy(), p.copy()
        g = grad(theta)
        p -= 0.5 * eps * g
        for i in range(n_leapfrog):
            theta += eps * p
            g = grad(theta)
            if i != n_leapfrog - 1:
                p -= eps * g
        p -= 0.5 * eps * g
        return theta, p

    def acceptance(theta, p, theta_new, p_new):
        log_a = energy(theta) - energy(theta_new) + 0.5 * np.sum(p ** 2) - 0.5 * np.sum(p_new ** 2)
        if np.isnan(log_a):
            return 0.0                    # a proposal whose energy cannot be evaluated is rejected
        return float(np.exp(min(log_a, 0.0)))

    theta = np.zeros(d) if start is None else np.log(np.clip(start, 1e-6, None))
    eps = step_size
    for t in range(n_tune):
        p = rng.normal(0.0, 1.0, d)
        theta_new, p_new = proposal(theta, p, eps)
        a = acceptance(theta, p, theta_new, p_new)
        if rng.uniform() < a:
            theta = theta_new
        eps = float(np.clip(eps * np.exp((a - target_accept) / np.sqrt(t + 1)), 1e-4, 0.5))
    samples = np.empty((n_samples, d))
    for s in range(n_samples):
        p = rng.normal(0.0, 1.0, d)
        theta_new, p_new = proposal(theta, p, eps)
        if rng.uniform() < acceptance(theta, p, theta_new, p_new):
            theta = theta_new
        samples[s] = np.exp(theta)
    return samples


def mh_lin(R, n_samples, n_tune, rng, start=None, sigma=0.1, target_accept=0.3):
    """Random-walk Metropolis-Hastings for BAYES-LIN on the simplex.

    Proposal w' = normalise(w * exp(N(0, sigma^2 I))); the Hastings correction of this
    multiplicative proposal is sum log w' - sum log w.  sigma is adapted during `n_tune`
    iterations as in hmc_bt; the chain starts at `start` (posterior mean of the previous round)
    or at the centre of the simplex."""
    d = R.shape[1]
    alpha = DIRICHLET_CONCENTRATION

    def log_target(w):
        p = 0.5 * (1.0 + R @ w)
        if np.any(p <= 0):
            return -np.inf
        return np.sum(np.log(p)) + np.sum((alpha - 1.0) * np.log(w + 1e-12))

    if start is None:
        w = np.full(d, 1.0 / d)
    else:
        w = np.clip(start, 1e-6, 1.0)
        w = w / w.sum()
    current = log_target(w)

    def step(w, current, sigma):
        prop = w * np.exp(rng.normal(0.0, sigma, d))
        prop /= prop.sum()
        target = log_target(prop)
        if not np.isfinite(target):
            return prop, target, -np.inf
        return prop, target, target - current + np.sum(np.log(prop) - np.log(w))

    for t in range(n_tune):
        prop, target, log_ratio = step(w, current, sigma)
        a = min(1.0, np.exp(log_ratio))
        if rng.uniform() < a:
            w, current = prop, target
        sigma = sigma * np.exp((a - target_accept) / np.sqrt(t + 1))
    samples = np.empty((n_samples, d))
    for s in range(n_samples):
        prop, target, log_ratio = step(w, current, sigma)
        if np.log(rng.uniform()) < log_ratio:
            w, current = prop, target
        samples[s] = w
    return samples


# ============================================================================== FTRL learners
class FTRLLearner(Learner):
    """Regularised MAP estimate and its Laplace approximation."""
    kind = 'FTRL'

    def __init__(self, V, link):
        super().__init__(V, link)
        self.w = None            # MAP estimate
        self.Sigma = None        # Laplace covariance in w
        self.Sigma_chart = None  # covariance in the coordinates of the chart (= Sigma for BT)

    def fit(self):
        R = self.diffs()
        self.w = map_estimate(R, self.link)
        self.Sigma_chart, self.Sigma = laplace(R, self.w, self.link)
        return self

    def draws(self):
        return self.w[None, :]


def map_estimate(R, link):
    """Minimiser of F(w) = -sum log p(y_s | w, q_s) + Omega(w)."""
    d = R.shape[1]
    if link == 'BT':
        w = cp.Variable(d)
        z = R @ w
        loglik = cp.sum(z - cp.logistic(z))                       # sum log sigma(z)
        logprior = cp.sum((ALPHA0 - 1) * cp.log(w) - BETA0 * w)    # -Omega(w)
        problem = cp.Problem(cp.Minimize(-loglik - logprior), [w >= 1e-12])
        try:
            problem.solve(solver=cp.SCS, verbose=False)
            if w.value is None:
                raise ValueError('SCS returned no solution')
            return w.value
        except Exception as exc:
            warnings.warn(f'FTRL-BT: the MAP problem was not solved ({exc}); the uniform vector is used',
                          RuntimeWarning)
            return np.ones(d) / d

    def objective(w):
        p = 0.5 * (1.0 + R @ w)
        loglik = np.sum(np.log(p)) if np.all(p > 0) else -np.inf
        return -(loglik + KAPPA * np.sum(np.log(w + 1e-12)))

    res = minimize(objective, np.ones(d) / d, method='SLSQP', bounds=[(0, 1)] * d,
                   constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},))
    return res.x


def tangent_basis(d):
    """Orthonormal basis Psi (d x (d-1)) of the tangent space {u : 1^T u = 0} of the simplex
    (Helmert contrasts)."""
    Psi = np.zeros((d, d - 1))
    for k in range(1, d):
        Psi[:k, k - 1] = 1.0
        Psi[k, k - 1] = -float(k)
        Psi[:, k - 1] /= np.sqrt(k * (k + 1.0))
    return Psi


def simplex_interior(w):
    w = np.asarray(w, dtype=float) + 1e-12
    return w / w.sum()


def laplace(R, w, link):
    """Laplace covariance of the FTRL objective at its minimiser, the inverse Hessian.  Returns
    (covariance in the chart coordinates, covariance in w); for the Linear link the chart is the affine
    chart w = w_hat + Psi nu and the covariance in w is Psi (Psi^T H Psi)^{-1} Psi^T."""
    if link == 'BT':
        p = np.clip(expit(R @ w), 1e-12, 1 - 1e-12)
        H = np.einsum('i,ij,ik->jk', p * (1.0 - p), R, R) + np.diag((ALPHA0 - 1.0) / w ** 2)
        H += np.eye(len(w)) * RIDGE
        Sigma = _inverse(H)
        return Sigma, Sigma
    w = simplex_interior(w)
    X = 0.5 * (1.0 + R)                                   # eta = w^T x_q with x_q = (1 + r_q) / 2
    p = np.clip(X @ w, 1e-12, 1 - 1e-12)
    H = np.einsum('i,ij,ik->jk', 1.0 / p ** 2, X, X) + np.diag(KAPPA / w ** 2)
    Psi = tangent_basis(len(w))
    H_nu = Psi.T @ H @ Psi + np.eye(len(w) - 1) * RIDGE
    try:
        L = np.linalg.cholesky(H_nu)
        Sigma_nu = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(w) - 1)))
    except np.linalg.LinAlgError:
        Sigma_nu = _inverse(H_nu)
    return Sigma_nu, Psi @ Sigma_nu @ Psi.T


def _inverse(M, ridge=1e-6):
    try:
        return np.linalg.inv(M + np.eye(M.shape[0]) * ridge)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def make_learner(name, V, n_samples=2000, rng=None):
    """'BAYES-LIN', 'BAYES-BT', 'FTRL-LIN' or 'FTRL-BT'."""
    kind, link = name.split('-')
    if kind == 'BAYES':
        return BayesianLearner(V, link, n_samples=n_samples, rng=rng)
    if kind == 'FTRL':
        return FTRLLearner(V, link)
    raise ValueError(f'unknown learner {name!r}')


def bayes_bt_samples(n_alternatives, n_criteria):
    """Number of posterior draws of BAYES-BT for n alternatives and m criteria: max(2000, 500 m, 200 n)."""
    return max(2000, 500 * n_criteria, 200 * n_alternatives)
