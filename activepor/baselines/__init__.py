"""Literature baseline rules (Poly, Dvf, Cheb, Css) and the question sequences they produce.

The rules use only the answers, so a rule asks the same questions whatever learner is later fed
with them: `run_rule` produces the sequence once, and every learner is evaluated on it with
`activepor.elicitation.Replay`.
"""
import numpy as np

from ..acquisition import unasked_pairs
from .rules import RULES

BASELINES = tuple(RULES)


def make_rule(name, V, rng, **options):
    return RULES[name](V, rng, **options)


def run_rule(name, V, dm, budget=None, seed=None, **options):
    """Question sequence of rule `name` on the simulated DM `dm`.

    Round 1 asks the DM's first pair; from round 2 on the rule chooses.  On a round where the
    rule has no pair to ask, a uniformly random unasked pair is asked instead; the
    number of answers at the first such round is the rule's own stopping point.

    Returns a dict with `answers` ((T, 2) [winner, loser]), `own_stop` (number of answers when the
    rule first had no pair, T if never), `fallback` (per round, True when the pair was random),
    `dropped` (per round, number of answers the consistency repair left out) and `repair` (per round,
    the repair behind the answers kept by Dvf, Cheb and Css: '' while every answer is kept, 'milp'
    for the mixed-integer program, 'l1' for the minimal-total-error LP that replaces it when it hits
    its time limit or its solution fails the exact consistency check)."""
    n = V.shape[0]
    T = n * (n - 1) // 2 if budget is None else int(budget)
    rule = make_rule(name, V, np.random.default_rng(seed), **options)
    answers = [dm.order[0]]
    fallback = np.zeros(T, dtype=bool)
    dropped = np.zeros(T, dtype=int)
    repair = np.full(T, '', dtype='<U4')
    for t in range(2, T + 1):
        pair, info = rule.select(np.array(answers), unasked_pairs(n, answers))
        fallback[t - 1] = bool(info.get('fallback'))
        dropped[t - 1] = int(info.get('dropped', 0))
        repair[t - 1] = info.get('consistency', '')
        answers.append(dm.answer(*pair))
    hits = np.nonzero(fallback)[0]
    own_stop = int(hits[0]) if len(hits) else T          # answers collected when the rule stopped
    return {'answers': np.array(answers, dtype=int), 'own_stop': own_stop,
            'fallback': fallback, 'dropped': dropped, 'repair': repair}
