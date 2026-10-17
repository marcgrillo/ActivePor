"""Additive value model and probabilistic response links.

Each alternative a is described by its characteristic vector v(a), so that its value is
U(a) = v(a)^T w with non-negative increments w.  A pairwise query q = (a_i, a_j) has the feature
difference r_q = v(a_i) - v(a_j), and the response link maps the value difference w^T r_q to the
probability that the decision maker (DM) prefers a_i.
"""
import numpy as np
from scipy.special import expit

LINKS = ('LIN', 'BT')

# Probabilities are clipped to [EPS, 1 - EPS] before entropies are taken.
EPS = 1e-9


def characteristic_points(perf, n_pieces):
    """gamma + 1 equally spaced characteristic points between the worst and the best performance
    observed on each criterion (all criteria of gain type)."""
    perf = np.asarray(perf, dtype=float)
    lo, hi = perf.min(axis=0), perf.max(axis=0)
    return [np.linspace(lo[j], hi[j], n_pieces + 1) for j in range(perf.shape[1])]


def characteristic_vectors(perf, n_pieces=3):
    """Matrix V (n x d), d = m * n_pieces, whose row a is the characteristic vector v(a).

    Component k of criterion j is 1 above the characteristic point c_j^k, 0 below c_j^{k-1} and
    linear in between, so that U(a) = V[a] @ w with w >= 0 is non-decreasing in every criterion."""
    perf = np.asarray(perf, dtype=float)
    n, m = perf.shape
    V = np.zeros((n, m * n_pieces))
    for j, c in enumerate(characteristic_points(perf, n_pieces)):
        for k in range(n_pieces):
            width = c[k + 1] - c[k]
            if width > 0:
                V[:, j * n_pieces + k] = np.clip((perf[:, j] - c[k]) / width, 0.0, 1.0)
    return V


def response_probability(z, link):
    """eta = P(first alternative preferred) as a function of the value difference z = w^T r_q:
    Linear link (1 + z) / 2 (w on the simplex), Bradley-Terry link sigma(z)."""
    if link == 'BT':
        return expit(z)
    if link == 'LIN':
        return 0.5 * (1.0 + z)
    raise ValueError(f'unknown link {link!r}')


def entropy(p):
    """Binary entropy h(p) in nats, with p clipped to [EPS, 1 - EPS]."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
