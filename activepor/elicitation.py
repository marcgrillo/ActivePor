"""One elicitation run: a learner, a query-selection strategy and a simulated decision maker.

Protocol: at round t the strategy picks a pair from the pool of pairs not yet asked, the DM
answers, and the learner is refitted on all answers.  Round 1 asks the first pair of the DM's random
question order for every strategy (there is no answer to learn from yet); from round 2 on the
strategy chooses.  After every answer the run records ASRS and ASPS against the DM's ranking and the
residual information R_t of the remaining pool; with `diagnostics=True` it also records the
information-gain residual (all learners), the curvature ratio (FTRL learners) and the split-R-hat of
the posterior draws (Bayesian learners).
"""
import numpy as np

from . import metrics
from .acquisition import scores, unasked_pairs
from .diagnostics import LinkResidual, curvature_ratio, hypothetical_surprises, split_rhat


class SimulatedDM:
    """A DM given by the answers it gives to every pair, listed in its random question order.

    `answers` is a (K, 2) array of [winner, loser] rows; `ranking` the DM's ranking (0 = best)."""

    def __init__(self, answers, ranking):
        self.order = [(int(a), int(b)) for a, b in np.asarray(answers, dtype=int)]
        self.ranking = np.asarray(ranking, dtype=int)
        self._answer = {frozenset(p): p for p in self.order}

    def answer(self, i, j):
        return self._answer[frozenset((int(i), int(j)))]


# ------------------------------------------------------------------------------- strategies
class RandomOrder:
    """Random elicitation: the DM's random question order."""
    name = 'RANDOM'

    def __init__(self, dm):
        self.dm = dm

    def select(self, t, learner, pool, bald):
        return self.dm.order[t - 1]


class Acquisition:
    """US, BALD or RANKUNC: the unasked pair with the largest score."""

    def __init__(self, rule):
        self.name = rule

    def select(self, t, learner, pool, bald):
        s = bald if (self.name == 'BALD' and bald is not None) else scores(learner, pool, self.name)
        return pool[int(np.argmax(s))]


class Replay:
    """A fixed question sequence, e.g. the one produced by a baseline rule (activepor.baselines)."""

    def __init__(self, name, answers):
        self.name = name
        self.sequence = [(int(a), int(b)) for a, b in np.asarray(answers, dtype=int)]

    def select(self, t, learner, pool, bald):
        return self.sequence[t - 1]


def make_strategy(name, dm, sequence=None):
    if name == 'RANDOM':
        return RandomOrder(dm)
    if name in ('US', 'BALD', 'RANKUNC'):
        return Acquisition(name)
    if sequence is None:
        raise ValueError(f'strategy {name!r} needs its question sequence (activepor.baselines.run_rule)')
    return Replay(name, sequence)


# ------------------------------------------------------------------------------------- run
def run(V, dm, learner, strategy, budget=None, residual_info=True, diagnostics=False, keep_states=False):
    """Elicit up to `budget` answers (default: all n(n-1)/2 pairs).  Returns a dict of arrays with
    entry t-1 referring to round t:

      answers        (T, 2) the answer of round t as [preferred, other]
      asrs, asps     ranking quality after t answers
      residual_info  residual information R of the pairs still unasked after t answers (NaN once the
                     pool is empty); the one-bit rule stops after the first t with R < log 2
    and, with `diagnostics=True` (from round 2 on, when there is a learner state to diagnose):
      expected_gain  B_t(q_t), the information the model expects from the answer to round t's query
      surprise       S_t(y_t), the surprise of the answer actually given
      xi, e_link     the information-gain residual xi_t and the running E_link after round t
      eps_lap        the curvature ratio of the FTRL state used to choose round t's query
      rhat           the split-R-hat of the Bayesian posterior draws after t answers (from round 1)"""
    n = V.shape[0]
    T = n * (n - 1) // 2 if budget is None else int(budget)
    out = {'answers': np.zeros((T, 2), dtype=int),
           'asrs': np.full(T, np.nan), 'asps': np.full(T, np.nan),
           'residual_info': np.full(T, np.nan)}
    if diagnostics:
        for key in ('expected_gain', 'surprise', 'xi', 'e_link', 'eps_lap', 'rhat'):
            out[key] = np.full(T, np.nan)
        link = LinkResidual()
    states = []
    bald = None                                    # BALD scores of the current pool, if computed
    for t in range(1, T + 1):
        pool = unasked_pairs(n, learner.answers)
        pair = dm.order[0] if t == 1 else tuple(strategy.select(t, learner, pool, bald))
        if diagnostics and t > 1:
            p1, s1, s0, b = hypothetical_surprises(learner, pair)
            if learner.kind == 'FTRL':
                out['eps_lap'][t - 1] = curvature_ratio(learner)
        winner, loser = dm.answer(*pair)
        if diagnostics and t > 1:
            s = s1 if winner == pair[0] else s0
            out['expected_gain'][t - 1], out['surprise'][t - 1] = b, s
            out['xi'][t - 1], out['e_link'][t - 1] = link.update(s, b)
        learner.observe(winner, loser)
        learner.fit()
        if diagnostics and learner.kind == 'BAYES':
            out['rhat'][t - 1] = split_rhat(learner.samples)
        out['answers'][t - 1] = (winner, loser)
        W = learner.draws()
        out['asrs'][t - 1] = metrics.asrs(V, W, dm.ranking)
        out['asps'][t - 1] = metrics.asps(V, W, dm.ranking)
        if keep_states:
            states.append(np.array(W if learner.kind == 'BAYES' else learner.w))
        bald = None
        next_pool = unasked_pairs(n, learner.answers)
        if next_pool and (residual_info or (strategy.name == 'BALD' and t < T)):
            bald = scores(learner, next_pool, 'BALD')
            if residual_info:
                out['residual_info'][t - 1] = float(np.sum(bald))
    if keep_states:
        out['states'] = np.array(states)
    return out
