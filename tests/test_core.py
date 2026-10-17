"""Checks of the core package.

    python -m pytest tests
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from activepor import data as D
from activepor.acquisition import lt_scores, mc_scores, scores, unasked_pairs
from activepor.baselines.consistency import is_consistent, min_cardinality_retained
from activepor.diagnostics import curvature_ratio, hypothetical_surprises, residual_information, split_rhat
from activepor.elicitation import SimulatedDM, make_strategy, run
from activepor.learners import (ALPHA0, BETA0, KAPPA, BayesianLearner, FTRLLearner, laplace, simplex_interior,
                                tangent_basis)
from activepor.metrics import asps, asrs, one_bit_stop, poi, rai
from activepor.model import characteristic_vectors, entropy


@pytest.fixture(scope='module')
def tmp(tmp_path_factory):
    return str(tmp_path_factory.mktemp('data'))


def small_instance(tmp, n=8, m=3, seed=0, rule='consistent'):
    folders = D.generate(os.path.join(tmp, f'gen{seed}_{rule}'), sizes=[n], criteria=[m], n_instances=1,
                         utility='piecewise_linear', answer_rules=[rule], seed=seed)
    return D.instances(folders[rule], n, m)[0]


def test_characteristic_vectors_reproduce_piecewise_linear_values():
    rng = np.random.default_rng(1)
    perf = rng.random((12, 3))
    V = characteristic_vectors(perf, 3)
    assert V.shape == (12, 9) and V.min() >= 0 and V.max() <= 1
    # the worst performance on a criterion has value 0, the best has all its increments
    for j in range(3):
        assert np.all(V[np.argmin(perf[:, j]), 3 * j:3 * j + 3] == 0)
        assert np.all(V[np.argmax(perf[:, j]), 3 * j:3 * j + 3] == 1)
    # U = V w is non-decreasing in every criterion for w >= 0
    w = rng.random(9)
    for j in range(3):
        order = np.argsort(perf[:, j])
        assert np.all(np.diff((V[:, 3 * j:3 * j + 3] @ w[3 * j:3 * j + 3])[order]) >= -1e-12)


def test_mc_scores_decomposition():
    rng = np.random.default_rng(2)
    R = rng.normal(size=(20, 5))
    W = rng.gamma(2.0, 1.0, size=(3000, 5))
    us, bald = mc_scores(R, W, 'BT', 'US'), mc_scores(R, W, 'BT', 'BALD')
    assert np.all(bald >= -1e-12) and np.all(bald <= us + 1e-12)     # 0 <= I(Y; W) <= H(Y)
    assert np.all(us <= np.log(2) + 1e-12)


def test_laplace_taylor_bt_scores_match_monte_carlo_for_a_concentrated_posterior():
    rng = np.random.default_rng(3)
    d = 4
    w = rng.gamma(3.0, 1.0, size=d)
    A = rng.normal(size=(d, d))
    Sigma = 1e-3 * (A @ A.T + d * np.eye(d))
    R = rng.normal(size=(30, d)) * 0.5
    W = rng.multivariate_normal(w, Sigma, size=200000)
    mc_b, lt_b = mc_scores(R, W, 'BT', 'BALD'), lt_scores(R, w, Sigma, 'BT', 'BALD')
    mc_u, lt_u = mc_scores(R, W, 'BT', 'US'), lt_scores(R, w, Sigma, 'BT', 'US')
    assert np.allclose(lt_b, mc_b, rtol=0.1, atol=2e-5)
    assert np.allclose(lt_u, mc_u, atol=2e-4)


def test_linear_laplace_covariance_is_the_constrained_covariance(tmp):
    """Psi (Psi^T H Psi)^{-1} Psi^T equals the inverse Hessian conditioned on 1^T delta = 0."""
    inst = small_instance(tmp, seed=4, rule='linear')
    V = inst.features(3)
    learner = FTRLLearner(V, 'LIN')
    for a, b in inst.answers[:12]:
        learner.observe(a, b)
    learner.fit()
    w = simplex_interior(learner.w)
    X = 0.5 * (1.0 + learner.diffs())
    p = X @ w
    H = (X / p[:, None] ** 2).T @ X + np.diag(KAPPA / w ** 2)
    Sig = np.linalg.inv(H)
    one = np.ones(len(w))
    constrained = Sig - np.outer(Sig @ one, Sig @ one) / (one @ Sig @ one)
    assert np.allclose(learner.Sigma, constrained, atol=1e-6 * np.abs(constrained).max())
    Psi = tangent_basis(len(w))
    assert np.allclose(Psi.T @ Psi, np.eye(len(w) - 1)) and np.allclose(Psi.sum(axis=0), 0.0)


def test_linear_curvature_ratio_is_the_exact_third_derivative_in_the_affine_chart(tmp):
    inst = small_instance(tmp, seed=9, rule='linear')
    V = inst.features(3)
    learner = FTRLLearner(V, 'LIN')
    for a, b in inst.answers[:12]:
        learner.observe(a, b)
    learner.fit()
    R, w = learner.diffs(), simplex_interior(learner.w)
    X = 0.5 * (1.0 + R)

    def second(x, v):                    # second derivative of the objective along v at x
        return np.sum((X @ v) ** 2 / (X @ x) ** 2) + KAPPA * np.sum(v ** 2 / x ** 2)
    D3 = []
    for v in (tangent_basis(len(w)) @ np.linalg.cholesky(learner.Sigma_chart)).T:
        h = 1e-6 * np.min(w / np.maximum(np.abs(v), 1e-300))
        D3.append(abs(second(w + h * v, v) - second(w - h * v, v)) / (2 * h))
    assert np.isclose(curvature_ratio(learner), max(D3) / 3.0, rtol=1e-4)


def test_bt_curvature_ratio_uses_the_exact_third_derivative(tmp):
    inst = small_instance(tmp, seed=5)
    V = inst.features(3)
    learner = FTRLLearner(V, 'BT')
    for a, b in inst.answers[:10]:
        learner.observe(a, b)
    learner.fit()
    R, w = learner.diffs(), learner.w

    def F(x):
        return np.sum(np.logaddexp(0.0, -(R @ x))) + np.sum(BETA0 * x - (ALPHA0 - 1.0) * np.log(x))
    L = np.linalg.cholesky(learner.Sigma)
    h = 1e-3
    D3 = [abs((F(w + 2 * h * v) - 2 * F(w + h * v) + 2 * F(w - h * v) - F(w - 2 * h * v)) / (2 * h ** 3))
          for v in L.T]
    assert np.isclose(curvature_ratio(learner), max(D3) / 3.0, rtol=1e-3)


def test_samplers_and_link_residual(tmp):
    inst = small_instance(tmp, seed=6, rule='bt')
    V = inst.features(3)
    for link in ('BT', 'LIN'):
        learner = BayesianLearner(V, link, n_samples=500, n_tune=200, rng=np.random.default_rng(0))
        for a, b in inst.answers[:8]:
            learner.observe(a, b)
        learner.fit()
        S = learner.samples
        assert S.shape == (500, V.shape[1]) and np.all(S > 0)
        if link == 'LIN':
            assert np.allclose(S.sum(axis=1), 1.0)
        p1, s1, s0, b = hypothetical_surprises(learner, inst.answers[8])
        assert np.isclose(b, p1 * s1 + (1 - p1) * s0) and b >= 0
        # the expected surprise is the BALD score of the query
        assert np.isclose(b, scores(learner, [tuple(inst.answers[8])], 'BALD')[0], rtol=1e-6, atol=1e-9)


def test_metrics_of_the_true_ranking():
    rng = np.random.default_rng(7)
    V = rng.random((6, 4))
    w = rng.random(4)
    ranking = np.empty(6, dtype=int)
    ranking[np.argsort(-(V @ w))] = np.arange(6)
    assert asrs(V, w[None], ranking) == 1.0 and asps(V, w[None], ranking) == 1.0
    W = rng.random((50, 4))
    R, P = rai(V, W), poi(V, W)
    assert np.allclose(R.sum(0), 1) and np.allclose(R.sum(1), 1)
    assert np.allclose(P + P.T, 1 + np.eye(6))
    assert np.isclose(asrs(V, W, ranking), np.mean([R[i, ranking[i]] for i in range(6)]))
    assert one_bit_stop([5.0, 2.0, 0.5, 0.1], 4) == 3 and one_bit_stop([5.0, 2.0, 1.0, np.nan], 4) == 4


def test_generator_answer_rules_share_instances(tmp):
    folders = D.generate(os.path.join(tmp, 'rules'), sizes=[10], criteria=[2], n_instances=5,
                         utility='exponential', answer_rules=['consistent', 'bt', 'linear'], seed=3)
    z = {r: D.load_config(f, 10, 2) for r, f in folders.items()}
    for r in ('bt', 'linear'):
        assert np.array_equal(z[r]['performance'], z['consistent']['performance'])
        same = np.all(np.sort(z[r]['answers'], 2) == np.sort(z['consistent']['answers'], 2))
        assert same                                              # same questions in the same order
    inverted = lambda c: np.mean([[c['ranking'][h][a] > c['ranking'][h][b] for a, b in c['answers'][h]]
                                  for h in range(5)])
    assert inverted(z['consistent']) == 0.0
    meta = D.load_meta(folders['bt'])
    U, beta = z['bt']['utility'], meta['beta']['n10_m2']
    du = np.array([[abs(U[h][a] - U[h][b]) for a, b in z['bt']['answers'][h]] for h in range(5)])
    assert np.isclose(np.mean(1.0 / (1.0 + np.exp(beta * du))), 0.2)    # expected inversion rate
    assert abs(meta['inversion_rate']['n10_m2'] - 0.2) < 0.05            # realized on 225 answers
    assert np.isclose(inverted(z['bt']), meta['inversion_rate']['n10_m2'])


def test_a_full_run_with_every_rule(tmp):
    inst = small_instance(tmp, n=7, m=2, seed=8, rule='bt')
    V, dm = inst.features(3), inst.dm()
    T = 21
    for name in ('RANDOM', 'US', 'BALD'):
        out = run(V, dm, FTRLLearner(V, 'BT'), make_strategy(name, dm), diagnostics=True)
        pairs = {frozenset(p) for p in out['answers']}
        assert len(pairs) == T and np.all((out['asrs'] >= 0) & (out['asrs'] <= 1))
        assert np.all(np.isfinite(out['residual_info'][:-1])) and np.isnan(out['residual_info'][-1])
        assert np.all(out['eps_lap'][1:] > 0)
    learner = BayesianLearner(V, 'BT', n_samples=300, n_tune=100, rng=np.random.default_rng(1))
    out = run(V, dm, learner, make_strategy('RANKUNC', dm), budget=6, diagnostics=True)
    assert len(out['asrs']) == 6 and np.all(np.isfinite(out['rhat'])) and np.all(out['rhat'] > 0.9)
    learner = FTRLLearner(V, 'BT')
    learner.observe(*dm.order[0]); learner.fit()
    pool = unasked_pairs(7, learner.answers)
    assert np.isclose(residual_information(learner, pool), np.sum(scores(learner, pool, 'BALD')))
    assert np.isclose(entropy(0.5), np.log(2))


def test_split_rhat():
    rng = np.random.default_rng(10)
    assert split_rhat(rng.normal(size=(2000, 3))) < 1.01
    drifting = rng.normal(size=(2000, 3)) + np.linspace(0.0, 3.0, 2000)[:, None]
    assert split_rhat(drifting) > 1.1


def test_consistency_repair_keeps_a_consistent_subset_with_the_strict_margin(tmp):
    """The MILP subset passes the exact check at the margin 1e-6 used by Dvf and Css."""
    for seed in (11, 12, 13):
        inst = small_instance(tmp, n=10, m=2, seed=seed, rule='bt')
        V = inst.features(3)
        A = np.asarray(inst.answers[:30])
        C = V[A[:, 0]] - V[A[:, 1]]
        if is_consistent(C, 1e-6):
            continue
        mask, method = min_cardinality_retained(C, 1e-6)
        assert method == 'milp' and is_consistent(C, 1e-6, mask) and 0 < (~mask).sum() < len(C)
