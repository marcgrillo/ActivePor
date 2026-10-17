"""Generate synthetic instances.

    python scripts/generate_data.py --out data/my_study --utility exponential
    python scripts/generate_data.py --out data/my_study_pwl --utility piecewise_linear --pieces 3

Every instance has performances drawn uniformly from [0, 1], a true additive value function, and a
random order of all n(n-1)/2 questions.  The true value function is either

    exponential       u_j(x) = w_j (1 - exp(-z_j x)) / (1 - exp(-z_j)),  w ~ Dirichlet(1), z_j ~ U[-10, 10],
    piecewise_linear  the learners' own model with --pieces equal sub-intervals per criterion and
                      increments drawn from Dirichlet(1).

One dataset folder is written per answer rule (--answers), all sharing the instances, the question order
and the random numbers that decide inversions:

    consistent        the order of the true values;
    bt                Bradley-Terry rule, beta tuned on the generated pairs of each configuration so that the
                      expected share of inverted answers is --inconsistency;
    linear            the linear rule P(a_i > a_j) = (1 + U(a_i) - U(a_j)) / 2.

Every run draws fresh random numbers; the seed used is recorded in meta.json, and --seed sets it.
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from activepor import data as D           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='output folder (one sub-folder per answer rule)')
    ap.add_argument('--utility', choices=D.UTILITIES, required=True)
    ap.add_argument('--pieces', type=int, default=3, help='pieces per criterion of a piecewise-linear truth')
    ap.add_argument('--sizes', nargs='+', type=int, default=[10, 30], help='numbers of alternatives n')
    ap.add_argument('--criteria', nargs='+', type=int, default=[2, 6, 10], help='numbers of criteria m')
    ap.add_argument('--instances', type=int, default=20, help='performance matrices per configuration')
    ap.add_argument('--answers', nargs='+', choices=D.ANSWER_RULES, default=['consistent', 'bt'])
    ap.add_argument('--inconsistency', type=float, default=0.2, help='target inversion rate of the bt rule')
    ap.add_argument('--seed', type=int, help='non-negative seed (default: fresh randomness)')
    a = ap.parse_args()
    folders = D.generate(a.out, a.sizes, a.criteria, a.instances, a.utility, a.answers, a.inconsistency,
                         a.pieces, a.seed)
    for rule, folder in folders.items():
        meta = D.load_meta(folder)
        rates = ', '.join(f'{k}: {100 * v:.1f}%' for k, v in meta['inversion_rate'].items())
        print(f'{folder}  ({rule}; inverted answers {rates})')


if __name__ == '__main__':
    main()
