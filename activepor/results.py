"""Storage of the runs and the statistics computed from them.

One run (learner x strategy x dataset x configuration x instance) is stored as
    <root>/<dataset>/<learner>/<strategy>/n{n}_m{m}/{instance:02d}.npz
with the arrays returned by activepor.elicitation.run (entry t-1 = round t), `seed`, and `own_stop`
(baseline rules: answers collected when the rule first had no pair to ask; -1 for the other strategies).

Stopping points are counted in answers collected: a run that stops with s answers uses the share
b = s / T_max of the budget and is scored by its ASRS after those s answers.  For the one-bit rule, s is
the first number of answers after which the residual information of the unasked pairs is below log 2.
"""
import os

import numpy as np
from scipy.stats import wilcoxon

from .metrics import final_quality, first_reach, one_bit_stop, stays_above

SCORE_RULES = ('US', 'BALD', 'RANKUNC')        # stopped by the residual-information rule
BASELINES = ('DVF', 'CHEB', 'CSS', 'POLY')     # stopped by their own termination condition


def run_file(root, dataset, learner, strategy, n, m, instance):
    return os.path.join(root, dataset, learner, strategy, f'n{n}_m{m}', f'{instance:02d}.npz')


def save_run(path, out, own_stop=-1, seed=-1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, own_stop=own_stop, seed=seed, **out)


def load_runs(root, dataset, learner, strategy, n, m, instances):
    """List of run dicts (None where the file is missing)."""
    out = []
    for h in range(instances):
        p = run_file(root, dataset, learner, strategy, n, m, h)
        if os.path.exists(p):
            with np.load(p) as z:
                out.append({k: z[k] for k in z.files})
        else:
            out.append(None)
    return out


def curves(runs, key='asrs'):
    """(T, H) matrix of a per-round quantity; runs that are missing are left out.

    The matrix is C-contiguous: with S = 1 (FTRL) the mean curves are multiples of 1/(nH) and can
    equal theta exactly, so the summation order of the means decides T_theta in such ties."""
    cols = [r[key] for r in runs if r is not None]
    return np.ascontiguousarray(np.array(cols).T) if cols else np.empty((0, 0))


def paired_curves(act_runs, rnd_runs, key='asrs'):
    """(T, H) matrices of a strategy and of random elicitation on the instances where both runs exist."""
    keep = [(a, r) for a, r in zip(act_runs, rnd_runs) if a is not None and r is not None]
    return curves([a for a, _ in keep], key), curves([r for _, r in keep], key)


def stops(runs, strategy, threshold=np.log(2.0)):
    """Answers collected at the stop of every run: one-bit residual-information rule for US, BALD
    and RankUnc, the rule's own termination condition for the baselines."""
    out = []
    for r in runs:
        if r is None:
            continue
        T = len(r['asrs'])
        if strategy in SCORE_RULES:
            out.append(one_bit_stop(r['residual_info'], T, threshold))
        else:
            own = int(r['own_stop'])
            out.append(T if own < 0 else own)
    return np.array(out, dtype=int)


def sem2(v):
    """Two standard errors of the mean."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return 2.0 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def wilcoxon_p(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    return float(wilcoxon(d).pvalue) if len(d) and np.any(d != 0) else 1.0


# ------------------------------------------------------------------------ sample efficiency
def efficiency(groups, frac=0.75):
    """AULC, T_theta and saving pooled over configurations.

    `groups` is a list of (active curves, random curves) pairs, one per configuration, each of shape
    (T, H) with the same instances in the same columns (see paired_curves).  Per configuration, theta = frac x the largest value of the mean
    random curve, and T_theta is the first round at which the mean curve reaches it; T_theta is
    averaged over the configurations, the saving is 1 - T_theta / T_theta(Random), and AULC is the
    mean over runs of the per-run mean over the rounds (+- 2 SEM over the pooled runs)."""
    A, Ar, T, Tr = [], [], [], []
    for act, rnd in groups:
        k = min(act.shape[1], rnd.shape[1])
        act, rnd = act[:, :k], rnd[:, :k]
        A += np.nanmean(act, axis=0).tolist()
        Ar += np.nanmean(rnd, axis=0).tolist()
        m_act, m_rnd = np.nanmean(act, axis=1), np.nanmean(rnd, axis=1)
        theta = frac * np.nanmax(m_rnd)
        T.append(first_reach(m_act, theta))
        Tr.append(first_reach(m_rnd, theta))
    A, Ar = np.array(A), np.array(Ar)
    return {'aulc': A.mean(), 'aulc_ci': sem2(A), 'aulc_rand': Ar.mean(), 'aulc_rand_ci': sem2(Ar),
            'T': float(np.mean(T)), 'T_rand': float(np.mean(Tr)),
            'saving': 1.0 - np.mean(T) / np.mean(Tr), 'n': len(A)}


# ------------------------------------------------------------------- stopping (Secs. 6.3-6.4)
def at_stop(act_runs, rnd_runs, strategy, threshold=np.log(2.0)):
    """Per-run statistics at the stop: answers collected (`stop`), b, r = ASRS at stop / final ASRS,
    ASRS at stop, b95, and the gain over random elicitation after the same number of answers (paired by
    instance)."""
    rows = []
    for ra, rr in zip(act_runs, rnd_runs):
        if ra is None or rr is None:
            continue
        T = len(ra['asrs'])
        s = int(stops([ra], strategy, threshold)[0])
        s = min(max(s, 1), T)
        a = ra['asrs'][s - 1]
        final = final_quality(ra['asrs'])
        rows.append({'stop': s, 'b': s / T, 'asrs': a, 'r': a / final if final > 0 else np.nan,
                     'd_rand': a - rr['asrs'][s - 1], 'b95': stays_above(ra['asrs'])})
    return rows


def summarize_stops(rows):
    """Means over runs; r (and its 10th percentile) leaves out runs whose final ASRS is 0
    (`r_excluded` counts them)."""
    g = lambda k: np.array([x[k] for x in rows], dtype=float)
    r = g('r')
    return {'n': len(rows), 'stop': g('stop').mean(), 'b': g('b').mean(), 'b95': g('b95').mean(),
            'r': np.nanmean(r), 'r_p10': np.nanpercentile(r, 10), 'r_excluded': int(np.sum(np.isnan(r))),
            'asrs': g('asrs').mean(), 'asrs_ci': sem2(g('asrs')),
            'd_rand': g('d_rand').mean(), 'd_rand_ci': sem2(g('d_rand'))}


def frozen_curve(asrs, stop, grid):
    """ASRS of one run on a grid of budget fractions, held constant after its stopping round."""
    T = len(asrs)
    y = np.interp(grid, np.arange(1, T + 1) / T, asrs)
    return np.where(grid <= stop / T, y, asrs[stop - 1])
