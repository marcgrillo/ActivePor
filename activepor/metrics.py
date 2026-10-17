"""Acceptability indices and ranking-quality measures.

`W` is either the S x d matrix of posterior draws of a Bayesian learner or the MAP estimate of an
FTRL learner as a 1 x d matrix (S = 1).  Rankings are 0-based: rank 0 is the best alternative.
"""
import numpy as np


def values(V, W):
    """Comprehensive values U (S x n) of the n alternatives under every draw."""
    return (np.asarray(V) @ np.atleast_2d(W).T).T


def ranks(V, W):
    """Rank of every alternative under every draw (S x n), 0 = best."""
    order = np.argsort(-values(V, W), axis=1)
    return np.argsort(order, axis=1)


def rai(V, W):
    """Rank acceptability indices: RAI[i, r] = share of draws placing a_i at rank r."""
    rk = ranks(V, W)
    n = rk.shape[1]
    out = np.zeros((n, n))
    for i in range(n):
        out[i] = np.bincount(rk[:, i], minlength=n) / rk.shape[0]
    return out


def poi(V, W):
    """Pairwise outranking indices: POI[i, j] = share of draws with U(a_i) >= U(a_j)."""
    U = values(V, W)
    return np.mean(U[:, :, None] >= U[:, None, :], axis=0)


def asrs(V, W, ranking):
    """ASRS: mean support of the true rank of each alternative."""
    return float(np.mean(ranks(V, W) == np.asarray(ranking)[None, :]))


def asps(V, W, ranking):
    """ASPS: mean support of the true order of each pair."""
    U = values(V, W)
    ranking = np.asarray(ranking)
    better, worse = np.nonzero(ranking[:, None] < ranking[None, :])
    n = len(ranking)
    return float(np.sum(np.mean(U[:, better] >= U[:, worse], axis=0)) * 2.0 / (n * (n - 1)))


# ------------------------------------------------------------------- trajectory statistics
def aulc(curve):
    """Normalised area under a learning curve: mean of the metric over the rounds."""
    return float(np.nanmean(curve))


def first_reach(curve, theta):
    """T_theta: first round (1-based) with curve >= theta; the budget if never."""
    hit = np.nonzero(np.asarray(curve) >= theta)[0]
    return int(hit[0]) + 1 if len(hit) else len(curve)


def final_quality(curve):
    """ASRS_final: mean over the last 5% of the budget (at least one round)."""
    curve = np.asarray(curve, dtype=float)
    return float(np.nanmean(curve[-max(1, len(curve) // 20):]))


def one_bit_stop(residual, budget, threshold=np.log(2.0)):
    """Number of answers at which the residual-information rule stops:
    the first t with R_t < threshold, R_t computed after t answers; the budget if never."""
    residual = np.asarray(residual, dtype=float)
    hit = np.nonzero(np.isfinite(residual) & (residual < threshold))[0]
    return int(hit[0]) + 1 if len(hit) else int(budget)


def stays_above(curve, level=0.95):
    """b_95: fraction of the budget after which the curve stays at or above `level` x its final value."""
    curve = np.asarray(curve, dtype=float)
    below = np.nonzero(curve < level * final_quality(curve))[0]
    return min(1.0, (below[-1] + 2) / len(curve)) if len(below) else 1.0 / len(curve)
