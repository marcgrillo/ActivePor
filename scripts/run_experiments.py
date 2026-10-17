"""Run elicitation experiments on the datasets in data/ (or on any dataset in the same format).

    python scripts/run_experiments.py synthetic       # data/synthetic: exponential true value functions
    python scripts/run_experiments.py synthetic-pwl   # data/synthetic_pwl: piecewise-linear true value functions
    python scripts/run_experiments.py link            # data/link: BAYES-LIN on DMs with three answer rules
    python scripts/run_experiments.py mobile          # data/mobile (see the README); --pieces sets the model

The datasets are not included: the README gives the scripts/generate_data.py commands that create them.

Every run (dataset x learner x strategy x configuration x instance) is written to
results/<dataset>/<learner>/<strategy>/n{n}_m{m}/{instance}.npz and skipped if that file exists, so an
interrupted sweep resumes.  The baseline rules (POLY, DVF, CHEB, CSS) use only the answers: their question
sequences are computed once (results/<dataset>/sequences/) and every learner is evaluated on them.

Options restrict a preset: --learners, --strategies, --sizes, --criteria, --instances.  --datasets runs the
preset on other datasets, given by name under data/ (e.g. my_study/consistent) or by path; their results go
to results/<group>/<rule>.  Every run draws fresh random numbers; with --seed the sweep is repeatable (the
seed of every run is derived from it and from the run's identifiers).  The seed of every run is saved in
its file.  --workers sets the number of parallel processes; --dry-run lists the runs without executing them.
"""
import argparse
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

STRATEGIES = ['RANDOM', 'US', 'BALD', 'RANKUNC', 'POLY', 'DVF', 'CHEB', 'CSS']
RULES = ['POLY', 'DVF', 'CHEB', 'CSS']                     # baseline rules, whose sequences are computed once
SYNTHETIC = {'FTRL-BT': ['RANDOM', 'US', 'BALD', 'POLY', 'DVF', 'CHEB', 'CSS'],
             'BAYES-BT': ['RANDOM', 'US', 'BALD', 'RANKUNC', 'POLY', 'DVF', 'CHEB', 'CSS']}
# `diagnostics` lists, per learner, the strategies whose runs also record the monitoring quantities
# (information-gain residual, curvature ratio, split-R-hat) at every round.
EXPERIMENTS = {
    'synthetic': dict(datasets=['synthetic/consistent', 'synthetic/bt20'], sizes=[10, 30], criteria=[2, 6, 10],
                      instances=20, strategies=SYNTHETIC, pieces=3, samples='adaptive', budget=None,
                      diagnostics=SYNTHETIC),
    'synthetic-pwl': dict(datasets=['synthetic_pwl/consistent', 'synthetic_pwl/bt20'], sizes=[10, 30],
                          criteria=[2, 6, 10], instances=20, strategies=SYNTHETIC, pieces=3, samples='adaptive',
                          budget=None, diagnostics=SYNTHETIC),
    'link': dict(datasets=['link/linear', 'link/bt20', 'link/consistent'], sizes=[30], criteria=[2, 6, 10],
                 instances=20, strategies={'BAYES-LIN': ['RANDOM', 'US', 'BALD']}, pieces=3, samples=2000,
                 budget=50, diagnostics={'BAYES-LIN': ['RANDOM', 'US', 'BALD']}),
    'mobile': dict(datasets=['mobile'], sizes=[10], criteria=[4], instances=94, strategies=SYNTHETIC, pieces=5,
                   samples=2000, budget=None, diagnostics={}),
}


def results_name(dataset, pieces, default_pieces):
    return dataset if pieces == default_pieces else f'{dataset}_p{pieces}'


def locate(dataset):
    """Folder of a dataset given by name (relative to data/) or by path, and the name of its results:
    the path under data/, or the last two folders (<group>/<rule>) of a dataset elsewhere."""
    data = os.path.join(ROOT, 'data')
    for folder in (os.path.join(data, dataset), dataset):
        if os.path.isdir(folder):
            folder = os.path.abspath(folder)
            if os.path.normcase(folder).startswith(os.path.normcase(data) + os.sep):
                name = os.path.relpath(folder, data)
            else:
                name = os.path.join(*folder.split(os.sep)[-2:])
            return folder, name.replace(os.sep, '/')
    sys.exit(f'missing dataset {dataset}: generate it with scripts/generate_data.py (names are relative to '
             'data/; see the README)')


def run_seed(base, *identifiers):
    """Seed of one run: fresh randomness, or derived from `base` and the run's identifiers."""
    import numpy as np
    if base is None:
        return int(np.random.default_rng().integers(2 ** 63 - 1))
    key = [zlib.crc32(str(x).encode()) for x in identifiers]
    return int(np.random.SeedSequence([int(base)] + key).generate_state(1, np.uint64)[0] % (2 ** 63 - 1))


def sequence_task(job):
    """Question sequence of a baseline rule on one instance."""
    import numpy as np
    from activepor import data as D
    from activepor.baselines import run_rule
    from activepor.results import run_file
    data_dir, name, rule, n, m, h, pieces, budget, results, base = job
    path = run_file(results, name, 'sequences', rule, n, m, h)
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    inst = D.instances(data_dir, n, m)[h]
    seed = run_seed(base, name, 'sequence', rule, n, m, h)
    seq = run_rule(rule, inst.features(pieces), inst.dm(), budget=budget, seed=seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, seed=seed, **seq)
    return path, time.time() - t0


def learner_task(job):
    """One elicitation run of a learner with a strategy on one instance."""
    import numpy as np
    from activepor import data as D
    from activepor.learners import bayes_bt_samples, make_learner
    from activepor.elicitation import make_strategy, run
    from activepor.results import run_file, save_run
    data_dir, name, learner, strategy, n, m, h, pieces, budget, samples, diag, results, base = job
    path = run_file(results, name, learner, strategy, n, m, h)
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    inst = D.instances(data_dir, n, m)[h]
    V = inst.features(pieces)
    dm = inst.dm()
    seed = run_seed(base, name, learner, strategy, n, m, h)
    S = bayes_bt_samples(n, m) if samples == 'adaptive' else int(samples)
    model = make_learner(learner, V, n_samples=S, rng=np.random.default_rng(seed))
    sequence, own_stop = None, -1
    if strategy in RULES:
        with np.load(run_file(results, name, 'sequences', strategy, n, m, h)) as z:
            sequence, own_stop = z['answers'], int(z['own_stop'])
    out = run(V, dm, model, make_strategy(strategy, dm, sequence), budget=budget, diagnostics=diag)
    save_run(path, out, own_stop=own_stop, seed=seed)
    return path, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('experiment', choices=list(EXPERIMENTS))
    ap.add_argument('--learners', nargs='+', help='a subset of the learners of the preset')
    ap.add_argument('--strategies', nargs='+', help='a subset of the strategies of the preset')
    ap.add_argument('--datasets', nargs='+', help='datasets by name under data/ or by path (default of the preset)')
    ap.add_argument('--sizes', nargs='+', type=int, help='numbers of alternatives n')
    ap.add_argument('--criteria', nargs='+', type=int, help='numbers of criteria m')
    ap.add_argument('--instances', type=int, help='the first H instances of every configuration')
    ap.add_argument('--pieces', type=int, help='linear pieces per criterion of the learners (default of the preset)')
    ap.add_argument('--seed', type=int, help='non-negative base seed that makes the sweep repeatable')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--results', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.seed is not None and a.seed < 0:
        sys.exit('--seed must be non-negative')

    cfg = EXPERIMENTS[a.experiment]
    valid = {'learners': list(cfg['strategies']),
             'strategies': sorted({s for v in cfg['strategies'].values() for s in v}, key=STRATEGIES.index)}
    for what, given in (('learners', a.learners), ('strategies', a.strategies)):
        unknown = [x for x in given or [] if x not in valid[what]]
        if unknown:
            sys.exit(f'{a.experiment}: unknown {what} {unknown}; choose from {valid[what]}')
    pieces = a.pieces or cfg['pieces']
    learners = {L: s for L, s in cfg['strategies'].items() if not a.learners or L in a.learners}
    configs = [(n, m) for n in (a.sizes or cfg['sizes']) for m in (a.criteria or cfg['criteria'])]
    H = a.instances or cfg['instances']
    seq_jobs, run_jobs = [], []
    for dataset in a.datasets or cfg['datasets']:
        data_dir, dataset = locate(dataset)
        name = results_name(dataset, pieces, cfg['pieces'])
        for learner, strategies in learners.items():
            for strategy in strategies:
                if a.strategies and strategy not in a.strategies:
                    continue
                diag = strategy in cfg['diagnostics'].get(learner, [])
                for n, m in configs:
                    for h in range(H):
                        if strategy in RULES:
                            job = (data_dir, name, strategy, n, m, h, pieces, cfg['budget'], a.results, a.seed)
                            if job not in seq_jobs:
                                seq_jobs.append(job)
                        run_jobs.append((data_dir, name, learner, strategy, n, m, h, pieces, cfg['budget'],
                                         cfg['samples'], diag, a.results, a.seed))
    if not run_jobs:
        sys.exit(f'{a.experiment}: nothing to run with these options')
    # the largest problems first, so that the slowest runs do not start last
    seq_jobs.sort(key=lambda j: -j[3])
    run_jobs.sort(key=lambda j: (j[2] != 'BAYES-BT' and j[2] != 'BAYES-LIN', -j[4], j[5]))
    print(f'{a.experiment}: {len(seq_jobs)} baseline sequences, {len(run_jobs)} runs, {a.workers} workers', flush=True)
    if a.dry_run:
        for j in seq_jobs:
            print('sequence', j[1], j[2], f'n={j[3]} m={j[4]} instance={j[5]}')
        for j in run_jobs:
            print('run', j[1], j[2], j[3], f'n={j[4]} m={j[5]} instance={j[6]}')
        return
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'                    # one BLAS thread per worker process
    t0 = time.time()
    for label, fn, jobs in (('sequence', sequence_task, seq_jobs), ('run', learner_task, run_jobs)):
        if not jobs:
            continue
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = {pool.submit(fn, j): j for j in jobs}
            for k, fut in enumerate(as_completed(futures), 1):
                try:
                    path, sec = fut.result()
                    if sec > 0:
                        print(f'[{k}/{len(jobs)} {label}s, {(time.time() - t0) / 60:.1f} min] '
                              f'{os.path.relpath(path, a.results)} ({sec:.0f} s)', flush=True)
                except Exception as exc:
                    print(f'FAILED {label} {futures[fut][1:7]}: {exc!r}', flush=True)
    print(f'done in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
