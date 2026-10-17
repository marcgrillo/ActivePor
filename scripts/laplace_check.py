"""Reliability of the Laplace-Taylor scores of the FTRL learners.

For the FTRL learners, the Laplace-Taylor (LT) US and BALD scores of every unasked pair are compared
with the same scores computed on the exact FTRL posterior, approximated by importance sampling:
40,000 value functions drawn from the prior implied by the regulariser (Gamma(alpha0, beta0) for
FTRL-BT; a flat Dirichlet, re-weighted by the regulariser, for FTRL-LIN) and weighted by the likelihood
of the answers received so far.  States: the FTRL estimate after t answers of the DM's random question
order, t in {1, 3, 5, 8, 11, 14, 18, 22, 27, 33}; states whose effective sample size is below 2% of the
draws are discarded.  Agreement: Kendall's tau between the two orderings of the candidates, and the
share of candidates the reference ranks above the pair selected by the LT score.

    python scripts/laplace_check.py [--workers 6]       ->  results/laplace_check.csv
    python scripts/make_tables.py laplace               ->  the table body and the correlations

Defaults: FTRL-BT on consistent DMs with piecewise-linear truths (data/synthetic_pwl/consistent) and
FTRL-LIN on DMs answering by the linear rule (data/link/linear), so that the link of the learner is
right in both cases; n = 30, m in {2, 6, 10}, 20 instances each (see the README for the commands that
generate these datasets).  --seed makes the reference draws repeatable.
"""
import argparse
import csv
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_var] = '1'                    # one BLAS thread per worker; set before numpy is loaded
import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import kendalltau

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

ROUNDS = [1, 3, 5, 8, 11, 14, 18, 22, 27, 33]


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def reference_scores(learner, Rc, n_draws, rng):
    """Importance-sampling US and BALD scores of the candidates (rows of Rc) and the relative ESS."""
    from activepor.learners import ALPHA0, BETA0, KAPPA
    R = learner.diffs()
    d = learner.d
    if learner.link == 'LIN':
        W = rng.dirichlet(np.ones(d), size=n_draws)
        p = np.clip(0.5 * (1.0 + R @ W.T), 1e-12, 1 - 1e-12)
        logw = np.sum(np.log(p), axis=0) + KAPPA * np.sum(np.log(W + 1e-12), axis=1)
    else:
        W = rng.gamma(shape=ALPHA0, scale=1.0 / BETA0, size=(n_draws, d))
        logw = np.sum(-np.logaddexp(0.0, -(R @ W.T)), axis=0)
    w = np.exp(logw - logsumexp(logw))
    ess = 1.0 / np.sum(w ** 2) / n_draws
    Z = Rc @ W.T
    P = expit(Z) if learner.link == 'BT' else np.clip(0.5 * (1.0 + Z), 1e-12, 1 - 1e-12)
    us = entropy(P @ w)
    return us, us - entropy(P) @ w, ess


def check_instance(job):
    from activepor import data as D
    from activepor.learners import FTRLLearner
    from activepor.acquisition import lt_scores, unasked_pairs
    from activepor.diagnostics import curvature_ratio
    data_dir, name, link, n, m, h, n_draws, min_ess, seed = job
    rng = np.random.default_rng(None if seed is None else [seed, zlib.crc32(name.encode()), m, h])
    inst = D.instances(data_dir, n, m)[h]
    V = inst.features(3)
    learner = FTRLLearner(V, link)
    rows = []
    for t in range(1, max(ROUNDS) + 1):
        learner.observe(*inst.answers[t - 1])
        if t not in ROUNDS:
            continue
        learner.fit()
        pool = unasked_pairs(n, learner.answers)
        if len(pool) < 5:
            continue
        P = np.array(pool)
        Rc = V[P[:, 0]] - V[P[:, 1]]
        us_ref, bald_ref, ess = reference_scores(learner, Rc, n_draws, rng)
        row = {'learner': name, 'm': m, 'instance': h, 't': t, 'ratio': curvature_ratio(learner), 'ess': ess,
               'usable': ess >= min_ess}
        for rule, ref in (('bald', bald_ref), ('us', us_ref)):
            lt = lt_scores(Rc, learner.w, learner.Sigma, link, rule.upper())
            usable = row['usable']
            row[f'tau_{rule}'] = kendalltau(lt, ref).correlation if usable else np.nan
            row[f'rank_{rule}'] = np.sum(ref > ref[np.argmax(lt)]) / len(pool) if usable else np.nan
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bt-data', default=os.path.join(ROOT, 'data', 'synthetic_pwl', 'consistent'))
    ap.add_argument('--lin-data', default=os.path.join(ROOT, 'data', 'link', 'linear'))
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--criteria', nargs='+', type=int, default=[2, 6, 10])
    ap.add_argument('--instances', type=int, default=20)
    ap.add_argument('--draws', type=int, default=40000)
    ap.add_argument('--min-ess', type=float, default=0.02)
    ap.add_argument('--seed', type=int, help='non-negative seed that makes the reference draws repeatable')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'laplace_check.csv'))
    a = ap.parse_args()
    jobs = [(data_dir, name, link, a.n, m, h, a.draws, a.min_ess, a.seed)
            for data_dir, name, link in ((a.bt_data, 'FTRL-BT', 'BT'), (a.lin_data, 'FTRL-LIN', 'LIN'))
            for m in a.criteria for h in range(a.instances)]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(check_instance, jobs):
            rows += r
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    dropped = {L: sum(1 for r in rows if r['learner'] == L and not r['usable']) for L in ('FTRL-BT', 'FTRL-LIN')}
    print(f"wrote {a.out}: {sum(r['usable'] for r in rows)} usable states; "
          f"discarded (ESS < {100 * a.min_ess:.0f}%): {dropped}")


if __name__ == '__main__':
    main()
