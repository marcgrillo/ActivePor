"""Summary tables of the runs in results/, printed as LaTeX rows followed by the underlying numbers.

    python scripts/make_tables.py efficiency   # AULC, T_theta and saving over random elicitation
    python scripts/make_tables.py stopping     # one-bit stopping rule: budget used and quality retained
    python scripts/make_tables.py baselines    # all strategies at their own stopping points
    python scripts/make_tables.py link         # information-gain residual and gains on the link runs
    python scripts/make_tables.py laplace      # Laplace-Taylor against exact scores (results/laplace_check.csv)
    python scripts/make_tables.py mobile       # efficiency and one-bit stop on the mobile runs
    python scripts/make_tables.py mobile-rai   # rank acceptability indices of one DM at the one-bit stop
    python scripts/make_tables.py all

The first three read the runs of the synthetic group given by --synthetic (default: synthetic, i.e.
data/synthetic; e.g. --synthetic synthetic_pwl or a group of generated data such as my_study).
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from activepor import results as RS                                    # noqa: E402
from activepor.metrics import one_bit_stop                             # noqa: E402

H = 20                                    # performance matrices per configuration
CRITERIA = [2, 6, 10]
LEARNERS = [('FTRL', 'FTRL-BT'), ('BAYES', 'BAYES-BT')]
NAMES = {'BALD': 'BALD', 'US': 'US', 'RANKUNC': r'\textsc{RankUnc}', 'DVF': r'\textsc{Dvf}',
         'CHEB': r'\textsc{Cheb}', 'CSS': r'\textsc{Css}', 'POLY': r'\textsc{Poly}', 'RANDOM': r'\textit{Random}'}
LIT = ['DVF', 'CHEB', 'CSS', 'POLY']
PM = r'{\scriptsize\,$\pm$\,'
BS = chr(92)
END = ' ' + BS * 2


def regimes(group):
    return [(f'{group}/consistent', '0%'), (f'{group}/bt20', '20%')]


def runs(a, dataset, learner, strategy, n, m, instances=H):
    return RS.load_runs(a.results, dataset, learner, strategy, n, m, instances)


def bold(text, fields):
    parts = text.split('&')
    for i in fields:
        parts[i] = ' ' + BS + 'textbf{' + parts[i].strip() + '} '
    return '&'.join(parts)


def unique_best(cells, key, want_max):
    """Strategy with the best value of `key`, or None when two tie on the printed value."""
    vals = [(s, round(c[key], 3 if key == 'aulc' else 6)) for s, c in cells.items()]
    if not vals:
        return None
    best = max(v for _, v in vals) if want_max else min(v for _, v in vals)
    winners = [s for s, v in vals if v == best]
    return winners[0] if len(winners) == 1 else None


# ======================================================================== sample efficiency
def efficiency_cells(a, learner, n, dataset, strategy):
    groups = []
    for m in CRITERIA:
        act, rnd = RS.paired_curves(runs(a, dataset, learner, strategy, n, m),
                                    runs(a, dataset, learner, 'RANDOM', n, m))
        if act.size:
            groups.append((act, rnd))
    return RS.efficiency(groups) if groups else None


def efficiency(a):
    for label, strategies, emphasise in (('US and BALD', ['BALD', 'US'], True),
                                         ('baseline rules, whole budget', LIT, False)):
        print(f'\n% ---- efficiency, {label} (AULC +- 2 SEM | T_theta | saving) ----')
        text = []
        for short, learner in LEARNERS:
            for si, n in enumerate((10, 30)):
                cells = {(ds, s): efficiency_cells(a, learner, n, ds, s)
                         for ds, _ in regimes(a.synthetic) for s in strategies}
                best = {}
                for ds, _ in regimes(a.synthetic):
                    sub = {s: cells[(ds, s)] for s in strategies if cells[(ds, s)]}
                    best[(ds, 'aulc')] = unique_best(sub, 'aulc', True) if emphasise else None
                    best[(ds, 'T')] = unique_best(sub, 'T', False) if emphasise else None
                rows = len(strategies) + 1
                lead = (f'{BS}multirow{{{2 * rows}}}{{*}}{{{short}}} ' if si == 0 else '') + \
                       f'& {BS}multirow{{{rows}}}{{*}}{{${n}$}} '
                for i, s in enumerate(strategies):
                    parts = []
                    for ds, _ in regimes(a.synthetic):
                        c = cells[(ds, s)]
                        if c is None:
                            parts.append('-- & -- & --')
                            continue
                        cell = f"{c['aulc']:.3f}{PM}{c['aulc_ci']:.3f}}} & {c['T']:.0f} & {100 * c['saving']:.1f}{BS}%"
                        mark = ([0] if best[(ds, 'aulc')] == s else []) + ([1, 2] if best[(ds, 'T')] == s else [])
                        parts.append(bold(cell, mark) if mark else cell)
                    print((lead if i == 0 else ' & ') + f'& {NAMES[s]} & ' + ' & '.join(parts) + END)
                    for (ds, lab), c in ((r, cells[(r[0], s)]) for r in regimes(a.synthetic)):
                        if c:
                            text.append(f"% {short:5s} n={n:2d} {lab:3s} {s:7s}: AULC={c['aulc']:.3f}+-{c['aulc_ci']:.3f} "
                                        f"T={c['T']:.1f} T_rand={c['T_rand']:.1f} saving={100 * c['saving']:.1f}% "
                                        f"(runs={c['n']})")
                parts = []
                for ds, _ in regimes(a.synthetic):
                    c = cells[(ds, strategies[0])]
                    parts.append('-- & -- & --' if c is None else
                                 f"{c['aulc_rand']:.3f}{PM}{c['aulc_rand_ci']:.3f}}} & {c['T_rand']:.0f} & --")
                print(f' & & {NAMES["RANDOM"]} & ' + ' & '.join(parts) + END)
                print(BS + 'cmidrule(lr){2-9}' if si == 0 else BS + 'midrule')
        print('\n'.join(text))


# ================================================================================ stopping
def dagger(st):
    """Mark r when runs whose final ASRS is 0 had to be left out of it."""
    return '$^' + BS + 'dagger$' if st['r_excluded'] else ''


def stop_rows(a, learner, strategy, dataset, sizes, threshold=np.log(2.0)):
    rows = []
    for n in sizes:
        for m in CRITERIA:
            rows += RS.at_stop(runs(a, dataset, learner, strategy, n, m),
                               runs(a, dataset, learner, 'RANDOM', n, m), strategy, threshold)
    return rows


def stopping(a):
    print('\n% ---- one-bit stop (F1 = 30): per regime b95 | b | r (p10), in % ----')
    for short, learner in LEARNERS:
        for j, s in enumerate(('BALD', 'US')):
            cells = []
            for ds, _ in regimes(a.synthetic):
                rows = stop_rows(a, learner, s, ds, [30])
                if not rows:
                    cells.append('-- & -- & --')
                    continue
                st = RS.summarize_stops(rows)
                cells.append(f"{100 * st['b95']:.0f} & {100 * st['b']:.0f} & {100 * st['r']:.0f} ({100 * st['r_p10']:.0f})"
                             + dagger(st))
            lead = f'{BS}multirow{{2}}{{*}}{{{short}}} & ' if j == 0 else ' & '
            print(lead + f'{s} & ' + ' & '.join(cells) + END)
        print(BS + 'midrule')
    ds = regimes(a.synthetic)[0][0]
    out = []
    for x, lab in [(4.0, '4 nats'), (1.0, '1 nat'), (np.log(2.0), '1 bit'), (np.log(2.0) / 2, 'half bit'), (0.1, '0.1 nat')]:
        rows = stop_rows(a, 'BAYES-BT', 'BALD', ds, [30], x)
        if rows:
            st = RS.summarize_stops(rows)
            out.append(f"{lab}: b={100 * st['b']:.0f}% r={100 * st['r']:.0f}%")
    if out:
        print('% BAYES-BT BALD, consistent answers, F1 = 30: ' + ', '.join(out))
    for short, learner in LEARNERS:
        for s in ('BALD', 'US'):
            for ds, lab in regimes(a.synthetic):
                rows = stop_rows(a, learner, s, ds, [10])
                if rows:
                    st = RS.summarize_stops(rows)
                    print(f"% F1=10 {short:5s} {s:4s} {lab:3s}: b={100 * st['b']:.0f}% r={100 * st['r']:.0f}% "
                          f"(p10 {100 * st['r_p10']:.0f}%) b95={100 * st['b95']:.0f}%")


# ======================================================== strategies at their stopping points
def baselines(a):
    print('\n% ---- at the stopping points: per regime b | r (p10) | dRandom +- 2 SEM, in % ----')
    text = []
    for short, learner in LEARNERS:
        rules = ['BALD', 'US'] + (['RANKUNC'] if short == 'BAYES' else []) + LIT
        print(f'{BS}multirow{{{2 * len(rules)}}}{{*}}{{{short}}}')
        for k, n in enumerate((10, 30)):
            for j, s in enumerate(rules):
                cells = []
                for ds, lab in regimes(a.synthetic):
                    rows = stop_rows(a, learner, s, ds, [n])
                    if not rows:
                        cells.append('-- & -- & --')
                        continue
                    st = RS.summarize_stops(rows)
                    cells.append(f"{100 * st['b']:.0f} & {100 * st['r']:.0f} ({100 * st['r_p10']:.0f}){dagger(st)} & "
                                 f"{st['d_rand']:+.3f}{{{BS}scriptsize$\\pm${st['d_rand_ci']:.3f}}}")
                    text.append(f"% {short:5s} F1={n:2d} {lab:3s} {s:7s} | {st['n']:3d} runs ({st['r_excluded']} without r)"
                                f" | stop {st['stop']:6.1f} "
                                f"answers, b={100 * st['b']:5.1f}% | ASRS {st['asrs']:.3f}+-{st['asrs_ci']:.3f} | "
                                f"dRandom {st['d_rand']:+.3f}+-{st['d_rand_ci']:.3f} | r={100 * st['r']:5.1f}% "
                                f"(p10 {100 * st['r_p10']:.0f}%)")
                lead = f' & {BS}multirow{{{len(rules)}}}{{*}}{{${n}$}} & ' if j == 0 else ' & & '
                print(lead + f'{NAMES[s]} & ' + ' & '.join(cells) + END)
            print(BS + 'cmidrule(lr){2-9}' if k == 0 else BS + 'midrule')
    print('\n'.join(text))


# ==================================================================================== link
LINK = [('link/linear', 'linear rule, correct'), ('link/bt20', r'BT rule $20\%$, wrong'),
        ('link/consistent', r'consistent $0\%$, wrong')]
ROUNDS = 50


def stats(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    return {'mean': float(d.mean()), 'sem2': RS.sem2(d), 'p': RS.wilcoxon_p(d), 'n': len(d)}


def fmt_p(p):
    if p >= 0.995:
        return '$1.0$'
    if p >= 0.01:
        return f'${p:.2f}$'
    e = int(np.floor(np.log10(p)))
    mant = int(round(p / 10 ** e))
    if mant == 10:
        mant, e = 1, e + 1
    return f'${mant}{BS}times10^{{{e}}}$'


def link_values(a, dataset, strategy):
    """Per instance: E_link after 50 rounds, OLS slope of E_link over rounds 2-50, AULC over 50 rounds."""
    level, slope, aulc = [], [], []
    for m in CRITERIA:
        for r in runs(a, dataset, 'BAYES-LIN', strategy, 30, m):
            if r is None:
                level.append(np.nan); slope.append(np.nan); aulc.append(np.nan)
                continue
            e = r['e_link'][1:ROUNDS]                          # rounds 2..50
            x = np.arange(2, 2 + len(e), dtype=float)
            xc = x - x.mean()
            level.append(e[-1])
            slope.append(float((e - e.mean()) @ xc / (xc @ xc)))
            aulc.append(float(np.nanmean(r['asrs'][:ROUNDS])))
    return np.array(level), np.array(slope), np.array(aulc)


def link(a):
    V = {(ds, s): link_values(a, ds, s) for ds, _ in LINK for s in ('BALD', 'RANDOM', 'US')}
    print('\n% ---- E_link: slope x1e3 (mean +- 2 SEM) | p vs the linear-rule DM | E_link at 50 ----')
    for s, name in (('BALD', BS + 'ac{BALD}'), ('RANDOM', 'Random'), ('US', BS + 'ac{US}')):
        print(f'{BS}multirow{{3}}{{*}}{{{name}}}')
        for ds, lab in LINK:
            lev, slo, _ = V[(ds, s)]
            sl, lv = stats(slo * 1e3), stats(lev)
            p = '--' if ds == 'link/linear' else fmt_p(stats((slo - V[('link/linear', s)][1]) * 1e3)['p'])
            print(f" & {lab} & ${sl['mean']:+.2f}\\pm{sl['sem2']:.2f}$ & {p} & ${lv['mean']:.3f}\\pm{lv['sem2']:.3f}$" + END)
        if s != 'US':
            print(BS + 'midrule')
    print('\n% ---- gains: AULC (rounds 1-50) gain over random | BALD - US (mean +- 2 SEM, p) ----')
    for ds, lab in LINK:
        rnd = V[(ds, 'RANDOM')][2]
        g = {s: V[(ds, s)][2] - rnd for s in ('US', 'BALD')}

        def gain(x):
            st = stats(x)
            return f"${st['mean']:+.3f}\\pm{st['sem2']:.3f}{'' if st['p'] < 0.05 else '^{' + BS + 'dagger}'}$"
        bu = stats(g['BALD'] - g['US'])
        print(f"{lab} & {gain(g['US'])} & {gain(g['BALD'])} & ${bu['mean']:+.3f}\\pm{bu['sem2']:.3f}$ & {fmt_p(bu['p'])}" + END)


# ===================================================================== Laplace reliability
def laplace(a):
    import pandas as pd
    path = os.path.join(a.results, 'laplace_check.csv')
    if not os.path.exists(path):
        print(f'% {path} missing: run scripts/laplace_check.py first')
        return
    df = pd.read_csv(path)
    dropped = df[~df.usable].groupby('learner').size().to_dict()
    df = df[df.usable]
    print(f'\n% ---- Laplace-Taylor vs exact scores ({len(df)} usable rounds; tau = Kendall tau) ----')
    groups = [('FTRL-BT', 'all ($<1$)', df[df.learner == 'FTRL-BT'])]
    lin = df[df.learner == 'FTRL-LIN']
    for lo, hi, lab in ((-np.inf, 1, '$<1$'), (1, 1.5, '$1$--$1.5$'), (1.5, 2, '$1.5$--$2$'), (2, np.inf, '$>2$')):
        groups.append(('FTRL-LIN', lab, lin[(lin.ratio >= lo) & (lin.ratio < hi)]))
    first = True
    for learner, lab, g in groups:
        lead = learner if learner == 'FTRL-BT' else (f'{BS}multirow{{4}}{{*}}{{FTRL-LIN}}' if first else '')
        if learner == 'FTRL-LIN':
            first = False
        cells = [f'{len(g)}', f'{g.ratio.mean():.2f}']
        for s in ('bald', 'us'):
            cells.append(f"${g[f'tau_{s}'].mean():.3f}\\pm{RS.sem2(g[f'tau_{s}']):.3f}$")
        for s in ('bald', 'us'):
            cells.append(f"{100 * g[f'rank_{s}'].mean():.1f}")
        print(f'{lead} & {lab} & ' + ' & '.join(cells) + END)
        if learner == 'FTRL-BT':
            print(BS + 'midrule')
    for name, sub in (('all rounds', df), ('FTRL-LIN', lin)):
        for s in ('bald', 'us'):
            r, p = pearsonr(sub.ratio, sub[f'tau_{s}'])
            rs, ps = spearmanr(sub.ratio, sub[f'tau_{s}'])
            print(f'% {name:10s} {s.upper():4s}: Pearson(ratio, tau) {r:+.2f} (p={p:.1e}), Spearman {rs:+.2f} (p={ps:.1e})')
    bt = df[df.learner == 'FTRL-BT']
    print(f'% FTRL-BT ratio range {bt.ratio.min():.2f}-{bt.ratio.max():.2f}; FTRL-LIN mean ratio by round: '
          + ', '.join(f't={t}: {g.ratio.mean():.2f}' for t, g in lin.groupby('t')))
    print(f'% rounds discarded because the effective sample size is below the threshold: {dropped}')


# ================================================================================== mobile
SCORE_RULES = {'BAYES-BT': ['BALD', 'US', 'RANKUNC'], 'FTRL-BT': ['BALD', 'US']}


def mobile_stats(a, name, learner, strategy):
    act = runs(a, name, learner, strategy, 10, 4, 94)
    rnd = runs(a, name, learner, 'RANDOM', 10, 4, 94)
    A, Rn = RS.paired_curves(act, rnd)
    if not A.size:
        return None
    e = RS.efficiency([(A, Rn)])
    a_hm, r_hm = np.nanmean(A, 0), np.nanmean(Rn, 0)
    d = a_hm - r_hm
    e.update({'d': d, 'better': float(np.mean(d > 0)), 'p': RS.wilcoxon_p(d),
              'theta': 0.75 * np.nanmax(np.nanmean(Rn, 1)), 'A': A, 'R': Rn})
    rows = RS.at_stop(act, rnd, strategy)
    e['at_stop'] = RS.summarize_stops(rows) if rows else None
    e['stops'] = np.array([x['stop'] for x in rows]) if rows else None
    if rows:
        lvl = e['at_stop']['asrs']
        e['T_rand_at_stop'] = int(np.argmax(np.nanmean(Rn, 1) >= lvl)) + 1 if np.any(np.nanmean(Rn, 1) >= lvl) else None
    return e


def mobile(a, name='mobile'):
    print(f'\n% ---- efficiency and one-bit stop ({name}): AULC +- 2 SEM | T_theta | saving | questions | r (p10) ----')
    for short, learner in (('BAYES', 'BAYES-BT'), ('FTRL', 'FTRL-BT')):
        strategies = SCORE_RULES[learner] + LIT
        cells = {s: mobile_stats(a, name, learner, s) for s in strategies}
        cells = {s: c for s, c in cells.items() if c}
        if not cells:
            continue
        best_a = unique_best({s: {'aulc': c['aulc']} for s, c in cells.items()}, 'aulc', True)
        best_t = unique_best({s: {'T': c['T']} for s, c in cells.items()}, 'T', False)
        print(f'{BS}multirow{{{len(cells) + 1}}}{{*}}{{{short}}}')
        for s, c in cells.items():
            cell = f"{c['aulc']:.3f}{PM}{c['aulc_ci']:.3f}}} & {c['T']:.0f} & {100 * c['saving']:.1f}{BS}%"
            cell = bold(cell, ([0] if best_a == s else []) + ([1, 2] if best_t == s else []))
            stop = (f"{c['at_stop']['stop']:.1f} & {100 * c['at_stop']['r']:.0f} ({100 * c['at_stop']['r_p10']:.0f})"
                    if s in SCORE_RULES[learner] else '-- & --')
            print(f' & {NAMES[s]} & {cell} & {stop}' + END)
        ref = cells.get('BALD') or next(iter(cells.values()))
        print(f" & {NAMES['RANDOM']} & {ref['aulc_rand']:.3f}{PM}{ref['aulc_rand_ci']:.3f}}} & {ref['T_rand']:.0f} & -- & -- & --"
              + END)
        print(BS + 'midrule')
        for s, c in cells.items():
            line = (f"% {short:5s} {s:7s}: theta={c['theta']:.3f} T={c['T']:.0f} T_rand={c['T_rand']:.0f} "
                    f"saving={100 * c['saving']:.1f}% dAULC>0 for {100 * c['better']:.0f}% of DMs (p={c['p']:.1e})")
            if c['at_stop']:
                st = c['at_stop']
                line += (f" | stop {st['stop']:.1f} questions (min {c['stops'].min()}, max {c['stops'].max()}), "
                         f"ASRS at stop {st['asrs']:.3f}, reached by random after {c['T_rand_at_stop']} questions, "
                         f"r={100 * st['r']:.0f}% (p10 {100 * st['r_p10']:.0f}%)")
            print(line)
        print(f"% {short:5s} final ASRS with all 45 answers: {np.nanmean(ref['A'][-1]):.3f}")
        if 'BALD' in cells and 'US' in cells:                # paired by DM
            B, U = RS.paired_curves(runs(a, name, learner, 'BALD', 10, 4, 94), runs(a, name, learner, 'US', 10, 4, 94))
            for lo, hi, lab in ((0, 15, 'first 15'), (0, 45, 'all 45'), (15, 45, 'rounds 16-45')):
                d = np.nanmean(B[lo:hi], 0) - np.nanmean(U[lo:hi], 0)
                print(f'% {short:5s} BALD - US, AULC over {lab:12s}: {d.mean():+.3f}+-{RS.sem2(d):.3f} (p={RS.wilcoxon_p(d):.1e})')


def mobile_rai(a, subject=0, name='mobile', pieces=5):
    """RAI of one DM at the one-bit stop of BAYES-BT with BALD (and random elicitation after as many
    questions): the stored run gives the stop, a re-run with the run's own seed gives the posterior draws."""
    from activepor import data as D
    from activepor.learners import make_learner
    from activepor.elicitation import make_strategy, run
    from activepor.metrics import rai
    data_dir = os.path.join(ROOT, 'data', 'mobile')
    inst = D.instances(data_dir, 10, 4)[subject]
    V, dm = inst.features(pieces), inst.dm()
    stored = runs(a, name, 'BAYES-BT', 'BALD', 10, 4, subject + 1)[subject]
    t = one_bit_stop(stored['residual_info'], 45)
    order = np.argsort(inst.ranking)
    print(f'\n% ---- rank acceptability indices: DM {subject}, ranking ' + ' > '.join(f'a{i + 1}' for i in order)
          + f', {t} questions (one-bit stop of BALD) ----')
    for strategy in ('BALD', 'RANDOM'):
        seed = int(RS.load_runs(a.results, name, 'BAYES-BT', strategy, 10, 4, subject + 1)[subject]['seed'])
        learner = make_learner('BAYES-BT', V, n_samples=2000, rng=np.random.default_rng(seed))
        run(V, dm, learner, make_strategy(strategy, dm), budget=t, residual_info=False)
        R = rai(V, learner.samples)
        asrs = np.mean([R[i, inst.ranking[i]] for i in range(10)])
        print(f'% {strategy}: ASRS {asrs:.3f}')
        for i in order:
            row = ' & '.join((f'{BS}textbf{{{100 * R[i, r]:.1f}}}' if r == inst.ranking[i] else f'{100 * R[i, r]:.1f}')
                             if R[i, r] >= 0.005 else '' for r in range(10))
            print(('' if strategy == 'BALD' else '% ') + f'$a_{{{i + 1}}}$ & {row}' + END)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('table', choices=['efficiency', 'stopping', 'baselines', 'link', 'laplace', 'mobile',
                                      'mobile-rai', 'all'])
    ap.add_argument('--results', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--synthetic', default='synthetic', help='group of the synthetic runs (efficiency, stopping, baselines)')
    ap.add_argument('--subject', type=int, default=0, help='mobile-rai: the DM')
    a = ap.parse_args()
    todo = ['efficiency', 'stopping', 'baselines', 'link', 'laplace', 'mobile'] if a.table == 'all' else [a.table]
    for t in todo:
        if t == 'mobile':
            for name in ('mobile', 'mobile_p2', 'mobile_p3'):
                if os.path.isdir(os.path.join(a.results, name)):
                    mobile(a, name)
        elif t == 'mobile-rai':
            mobile_rai(a, a.subject)
        else:
            globals()[t](a)


if __name__ == '__main__':
    main()
