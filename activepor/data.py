"""Datasets of simulated decision makers and the synthetic generator.

A dataset is a folder with `meta.json` (how it was made) and one `n{n}_m{m}.npz` file per
configuration (n alternatives, m criteria), holding for every instance h:

    performance  (H, n, m)  performances, all criteria of gain type
    ranking      (H, n)     the DM's ranking, 0 = best
    answers      (H, K, 2)  the DM's answer to each of the K = n(n-1)/2 pairs, as [winner, loser],
                            listed in the DM's random question order (= the Random strategy)
    utility      (H, n)     true value of each alternative (NaN for real DMs)

The answer to a pair is fixed in advance, so every strategy asked the same pair gets the same
answer, and the answers are inconsistent with the ranking exactly where the answer rule
inverted them.
"""
import json
import os

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from .model import characteristic_vectors
from .elicitation import SimulatedDM

UTILITIES = ('exponential', 'piecewise_linear')
ANSWER_RULES = ('consistent', 'bt', 'linear')


# ------------------------------------------------------------------------------------- I/O
def config_file(folder, n, m):
    return os.path.join(folder, f'n{n}_m{m}.npz')


def save_config(folder, n, m, performance, ranking, answers, utility=None):
    os.makedirs(folder, exist_ok=True)
    performance = np.asarray(performance, dtype=float)
    if utility is None:
        utility = np.full(performance.shape[:2], np.nan)
    np.savez_compressed(config_file(folder, n, m), performance=performance,
                        ranking=np.asarray(ranking, dtype=int), answers=np.asarray(answers, dtype=int),
                        utility=np.asarray(utility, dtype=float))


def load_config(folder, n, m):
    with np.load(config_file(folder, n, m)) as z:
        return {k: z[k] for k in z.files}


def save_meta(folder, meta):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def load_meta(folder):
    with open(os.path.join(folder, 'meta.json')) as f:
        return json.load(f)


def configurations(folder):
    """(n, m) of every configuration file in the dataset folder."""
    out = []
    for name in sorted(os.listdir(folder)):
        if name.startswith('n') and name.endswith('.npz'):
            n, m = name[1:-4].split('_m')
            out.append((int(n), int(m)))
    return sorted(out)


class Instance:
    """One performance matrix with its simulated DM."""

    def __init__(self, performance, ranking, answers, utility=None):
        self.performance = np.asarray(performance, dtype=float)
        self.ranking = np.asarray(ranking, dtype=int)
        self.answers = np.asarray(answers, dtype=int)
        self.utility = utility

    def features(self, n_pieces=3):
        """Characteristic vectors of the learners' piecewise-linear model (gamma_j = n_pieces)."""
        return characteristic_vectors(self.performance, n_pieces)

    def dm(self):
        return SimulatedDM(self.answers, self.ranking)


def instances(folder, n, m, count=None):
    z = load_config(folder, n, m)
    H = len(z['ranking']) if count is None else min(count, len(z['ranking']))
    return [Instance(z['performance'][h], z['ranking'][h], z['answers'][h], z['utility'][h]) for h in range(H)]


# ------------------------------------------------------------------------------- generator
def exponential_utility(perf, rng):
    """Exponential marginals u_j(x) = omega_j (1 - exp(-zeta_j x)) / (1 - exp(-zeta_j)), with
    omega ~ Dirichlet(1, ..., 1) and zeta_j ~ U[-10, 10] (zeta_j != 0)."""
    m = perf.shape[1]
    omega = rng.dirichlet(np.ones(m))
    zeta = rng.uniform(-10.0, 10.0, m)
    zeta[np.abs(zeta) < 1e-5] = 1e-5
    return np.sum(omega * (1.0 - np.exp(-zeta * perf)) / (1.0 - np.exp(-zeta)), axis=1)


def piecewise_linear_utility(perf, rng, n_pieces=3):
    """A value function of the learners' own class: increments w ~ Dirichlet(1, ..., 1) on the
    characteristic vectors with n_pieces equal sub-intervals per criterion."""
    V = characteristic_vectors(perf, n_pieces)
    return V @ rng.dirichlet(np.ones(V.shape[1]))


def ranking_of(utility):
    rank = np.empty(len(utility), dtype=int)
    rank[np.argsort(-utility)] = np.arange(len(utility))
    return rank


def tune_beta(du, target):
    """beta such that the mean over the pairs of 1 - sigma(beta du) equals `target`."""
    f = lambda beta: float(np.mean(1.0 - expit(beta * du))) - target
    if f(1e-6) < 0:
        return 1e-6
    if f(1e6) > 0:
        return 1e6
    return float(brentq(f, 1e-6, 1e6, xtol=1e-10))


def generate(out_dir, sizes=(10, 30), criteria=(2, 6, 10), n_instances=20, utility='exponential',
             answer_rules=('consistent', 'bt'), inconsistency=0.2, n_pieces=3, seed=None):
    """Generate synthetic instances and write one dataset folder per answer rule.

    For every configuration (n, m) and instance: performances ~ U[0, 1]; a true value function
    (`utility` = 'exponential' (exponential_utility), or 'piecewise_linear' with `n_pieces`
    pieces, the learners' own class); a random question order of all n(n-1)/2 pairs; and one
    uniform random number per question, shared by all answer rules (common random numbers, so two
    rules answer differently only where their probabilities disagree).  Answer rules:

      consistent  the order of the true values;
      bt          inverted with probability 1 - sigma(beta du), du = |U(a_i) - U(a_j)|, with beta
                  tuned per configuration so that the mean inversion probability over all pairs and
                  instances equals `inconsistency`;
      linear      inverted with probability 1 - (1 + du) / 2.

    The folders are out_dir/consistent, out_dir/bt<100*inconsistency>, out_dir/linear.  `seed`
    (default: fresh randomness) is recorded in meta.json, so the data can be drawn again.  Returns
    {rule: folder}."""
    if utility not in UTILITIES:
        raise ValueError(f'utility must be one of {UTILITIES}')
    for rule in answer_rules:
        if rule not in ANSWER_RULES:
            raise ValueError(f'answer rules must be among {ANSWER_RULES}')
    label = {'consistent': 'consistent', 'bt': f'bt{int(round(100 * inconsistency))}', 'linear': 'linear'}
    folders = {r: os.path.join(out_dir, label[r]) for r in answer_rules}
    seed = int(np.random.SeedSequence(seed).entropy)
    metas = {r: {'utility': utility, 'answer_rule': r, 'seed': seed, 'instances': n_instances,
                 'target_inconsistency': {'consistent': 0.0, 'bt': inconsistency, 'linear': None}[r],
                 'beta': {}, 'inversion_rate': {},
                 'shares_instances_with': [label[o] for o in answer_rules if o != r]}
             for r in answer_rules}
    if utility == 'piecewise_linear':
        for r in answer_rules:
            metas[r]['utility_pieces'] = n_pieces
    for n in sizes:
        pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
        K = len(pairs)
        for m in criteria:
            rng = np.random.default_rng([seed, n, m])
            perf = rng.random((n_instances, n, m))
            U = np.empty((n_instances, n))
            orders = np.empty((n_instances, K), dtype=int)
            u = np.empty((n_instances, K))
            for h in range(n_instances):
                U[h] = (exponential_utility(perf[h], rng) if utility == 'exponential'
                        else piecewise_linear_utility(perf[h], rng, n_pieces))
                orders[h] = rng.permutation(K)
                u[h] = rng.random(K)
            ranking = np.array([ranking_of(U[h]) for h in range(n_instances)])
            q = pairs[orders]                                    # (H, K, 2), question order
            hi = np.take_along_axis(U, q[:, :, 0], 1) > np.take_along_axis(U, q[:, :, 1], 1)
            win = np.where(hi, q[:, :, 0], q[:, :, 1])
            los = np.where(hi, q[:, :, 1], q[:, :, 0])
            du = np.take_along_axis(U, win, 1) - np.take_along_axis(U, los, 1)
            for r in answer_rules:
                if r == 'consistent':
                    p_keep, beta = np.ones_like(du), None
                elif r == 'bt':
                    beta = tune_beta(du.ravel(), inconsistency) if inconsistency > 0 else None
                    p_keep = expit(beta * du) if beta is not None else np.ones_like(du)
                else:
                    p_keep, beta = np.clip(0.5 * (1.0 + du), 0.0, 1.0), None
                flip = u > p_keep
                answers = np.stack([np.where(flip, los, win), np.where(flip, win, los)], axis=2)
                save_config(folders[r], n, m, perf, ranking, answers, U)
                metas[r]['beta'][f'n{n}_m{m}'] = beta
                metas[r]['inversion_rate'][f'n{n}_m{m}'] = float(flip.mean())
    for r in answer_rules:
        save_meta(folders[r], metas[r])
    return folders
