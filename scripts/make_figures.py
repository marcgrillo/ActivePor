"""Figures of the runs in results/ (written to results/figures/).

    python scripts/make_figures.py sensitivity     # AULC gains over random against F1 and F2 (FTRL-BT, BAYES-BT)
    python scripts/make_figures.py frontier        # budget used against quality retained, for many stopping thresholds
    python scripts/make_figures.py curves          # learning curves of all strategies, frozen at their stopping points
    python scripts/make_figures.py mobile          # learning curves on the mobile runs
    python scripts/make_figures.py all

The first three read the runs of the synthetic group given by --synthetic (default: synthetic); for
another group the figures go to results/figures/<group>/.  The values behind the bars and markers are
printed as well.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from activepor import results as RS                              # noqa: E402

H = 20
CRITERIA = [2, 6, 10]
SIZES = [10, 30]
COL = {'US': 'tab:orange', 'BALD': 'tab:red'}
STYLE = {'BALD': ('o', 'tab:red'), 'US': ('s', 'tab:orange'), 'RANKUNC': ('D', 'tab:purple'),
         'DVF': ('^', 'tab:green'), 'CHEB': ('v', 'tab:blue'), 'CSS': ('P', 'tab:brown'), 'POLY': ('X', 'tab:olive')}
LABEL = {'BALD': 'BALD', 'US': 'US', 'RANKUNC': 'RankUnc', 'DVF': 'Dvf', 'CHEB': 'Cheb', 'CSS': 'Css', 'POLY': 'Poly'}
ZMARK = {'RANKUNC': 5.0, 'DVF': 5.2, 'CHEB': 5.3, 'CSS': 5.4, 'POLY': 5.5, 'US': 5.6, 'BALD': 5.8}
LEGEND_ORDER = ['Random', 'BALD', 'US', 'RankUnc', 'Dvf', 'Cheb', 'Css', 'Poly']
RC = {'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 16, 'xtick.labelsize': 14,
      'ytick.labelsize': 14, 'legend.fontsize': 13}


def regimes(group):
    return [(f'{group}/consistent', 'consistent'), (f'{group}/bt20', '20% inconsistency')]


def runs(a, dataset, learner, strategy, n, m, instances=H):
    return RS.load_runs(a.results, dataset, learner, strategy, n, m, instances)


def style(rc):
    """Matplotlib defaults plus the figure's own sizes (each figure starts from a clean state)."""
    plt.rcdefaults()
    plt.rcParams.update(rc)


def save(fig, a, name):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('% wrote', path)


def mean_band(Y):
    """Mean curve and +-2 standard errors (no band with a single run)."""
    if len(Y) < 2:
        return Y.mean(0), np.zeros(Y.shape[1])
    return Y.mean(0), 2 * Y.std(0, ddof=1) / np.sqrt(len(Y))


# ============================================================================== sensitivity
def gains(a, learner, strategy, metric, n_list, m_list):
    """Paired AULC gains over random elicitation, pooled over configurations, regimes and matrices."""
    g = []
    for ds, _ in regimes(a.synthetic):
        for n in n_list:
            for m in m_list:
                act, rnd = RS.paired_curves(runs(a, ds, learner, strategy, n, m),
                                            runs(a, ds, learner, 'RANDOM', n, m), metric)
                if act.size:
                    g += (np.nanmean(act, 0) - np.nanmean(rnd, 0)).tolist()
    g = np.array(g)
    return (g.mean(), RS.sem2(g), len(g)) if len(g) else (np.nan, 0.0, 0)


def sensitivity(a):
    style({'font.size': 17, 'axes.titlesize': 18, 'axes.labelsize': 17, 'xtick.labelsize': 15,
          'ytick.labelsize': 15, 'legend.fontsize': 15})
    for learner, name in (('FTRL-BT', 'fig_sensitivity_ftrl.png'), ('BAYES-BT', 'fig_sensitivity_bayes.png')):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))
        width = 0.34
        for ri, (metric, mlab) in enumerate((('asrs', 'ASRS'), ('asps', 'ASPS'))):
            for ci, (factor, values) in enumerate((('F1', SIZES), ('F2', CRITERIA))):
                ax = axes[ri][ci]
                xpos = np.arange(len(values))
                top = 0.0
                for k, s in enumerate(('US', 'BALD')):
                    means, errs = [], []
                    for v in values:
                        mu, e, cnt = (gains(a, learner, s, metric, [v], CRITERIA) if factor == 'F1'
                                      else gains(a, learner, s, metric, SIZES, [v]))
                        means.append(mu); errs.append(e)
                        print(f'% {learner} {mlab} {s:4s} {factor}={v:2d}: dAULC={mu:+.4f} +- {e:.4f} (n={cnt})')
                    ax.bar(xpos + (k - 0.5) * width, means, width, yerr=errs, capsize=4, color=COL[s], label=s,
                           edgecolor='white', linewidth=0.6)
                    top = max(top, np.nanmax(np.array(means) + np.array(errs), initial=0.0))
                ax.axhline(0, color='gray', lw=0.9)
                ax.set_xticks(xpos)
                ax.set_xticklabels([f'${factor[0]}_{factor[1]}={v}$' for v in values])
                if ci == 0:
                    ax.set_ylabel(rf'{mlab}: $\Delta$AULC (vs Random)')
                ax.grid(alpha=0.3, axis='y')
                if ri == 0 and ci == 0:
                    if np.isfinite(top) and top > 0:
                        ax.set_ylim(top=1.22 * top)       # room for the legend above the bars
                    ax.legend(loc='upper left', ncol=2, framealpha=0.95)
        fig.tight_layout()
        save(fig, a, name)


# ======================================================================== stopping frontier
THRESHOLDS = np.logspace(np.log10(30), np.log10(0.003), 60)
ANNOTATE = {('BAYES-BT', '4 nats'): (8, 88.5), ('BAYES-BT', '0.1 nats'): (26, 101.2),
            ('FTRL-BT', '4 nats'): (19, 87), ('FTRL-BT', '0.1 nats'): (32, 93)}


def frontier(a):
    style(RC)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    styles = {0: '-', 1: '--'}
    for ax, learner in zip(axes, ('BAYES-BT', 'FTRL-BT')):
        for ri, (ds, rlab) in enumerate(regimes(a.synthetic)):
            for s in ('US', 'BALD'):
                pairs = [(ra, rr) for m in CRITERIA for ra, rr in zip(runs(a, ds, learner, s, 30, m),
                                                                       runs(a, ds, learner, 'RANDOM', 30, m))]
                act = [p[0] for p in pairs]
                rnd = [p[1] for p in pairs]
                if not any(r is not None and q is not None for r, q in pairs):
                    continue

                def point(x):
                    st = RS.summarize_stops(RS.at_stop(act, rnd, s, x))
                    return 100 * st['b'], 100 * st['r']
                cur = np.array([point(x) for x in THRESHOLDS])
                ax.plot(cur[:, 0], cur[:, 1], styles[ri], color=COL[s], lw=(1.7 if s == 'US' else 2.4),
                        zorder=(4 if s == 'US' else 3))
                b1, r1 = point(np.log(2.0))
                if s == 'US':
                    ax.plot(b1, r1, 'o', mfc='white', mec='k', mew=1.8, ms=10, zorder=7)
                else:
                    ax.plot(b1, r1, 'o', color='k', mec='k', ms=9, zorder=6)
                line = f'% {learner} {s:4s} {rlab:18s}: one bit b={b1:.1f} r={r1:.1f}'
                for x, lab in ((4.0, '4 nats'), (0.1, '0.1 nats')):
                    bx, rx = point(x)
                    line += f' | {lab}: b={bx:.1f} r={rx:.1f}'
                    if s == 'BALD' and ri == 0:
                        ax.plot(bx, rx, 'o', ms=6, mfc='white', mec=COL[s], mew=1.3, zorder=5)
                        ax.annotate(lab, (bx, rx), xytext=ANNOTATE[(learner, lab)], textcoords='data', fontsize=13,
                                    color=COL[s], ha='center', va='center', zorder=7,
                                    arrowprops=dict(arrowstyle='-', color=COL[s], lw=1.0, shrinkA=2, shrinkB=4))
                print(line)
        ax.axhline(95, ls='--', c='gray', lw=1, zorder=1)
        ax.set_title(learner)
        ax.grid(alpha=.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(30, 103)
        ax.set_xlabel('fraction of the budget used at stop (%)')
    axes[0].set_ylabel('final ASRS retained (%)')
    handles = [Line2D([], [], color=COL[s], ls=styles[ri], lw=2.4, label=f'{s}, {lab}')
               for s in ('US', 'BALD') for ri, lab in enumerate(('consistent', '20% inc.'))]
    handles.append(Line2D([], [], color='k', marker='o', mec='k', ms=9, ls='none', label='one bit (BALD)'))
    handles.append(Line2D([], [], mfc='white', marker='o', mec='k', mew=1.8, ms=10, ls='none', label='one bit (US)'))
    axes[1].legend(handles=handles, loc='lower right', framealpha=0.95)
    fig.tight_layout()
    save(fig, a, 'fig_stopping_frontier.png')


# ============================================== learning curves frozen at the stopping points
def filled(y):
    """A learning curve with missing rounds (NaN) interpolated from the neighbouring rounds."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    return y if ok.all() else np.interp(np.arange(len(y)), np.flatnonzero(ok), y[ok])


def frozen(run_list, strategy, grid):
    """Curves on the grid of budget fractions, each frozen after its own stop; stop fractions."""
    Y, B = [], []
    for r in run_list:
        if r is None:
            continue
        T = len(r['asrs'])
        s = int(min(max(RS.stops([r], strategy)[0], 1), T))
        Y.append(RS.frozen_curve(filled(r['asrs']), s, grid))
        B.append(s / T)
    return np.array(Y), np.array(B)


def random_curves(run_list, grid):
    out = []
    for r in run_list:
        if r is not None:
            T = len(r['asrs'])
            out.append(np.interp(grid, np.arange(1, T + 1) / T, filled(r['asrs'])))
    return np.array(out)


def truncate(grid, m, s, bmax):
    keep = grid < bmax
    return (np.concatenate([grid[keep], [bmax]]), np.concatenate([m[keep], [np.interp(bmax, grid, m)]]),
            np.concatenate([s[keep], [np.interp(bmax, grid, s)]]))


def legend_ymin(fig, ax, leg, lows, pad=0.03):
    """y-axis minimum for which a lower-right legend lies below every band it overlaps (None: fits)."""
    fig.canvas.draw()
    bb, axb = leg.get_window_extent(), ax.get_window_extent()
    g = (bb.y1 - axb.y0) / axb.height
    x0 = ax.transData.inverted().transform((bb.x0, bb.y0))[0]
    ymin, ymax = ax.get_ylim()
    under = [float(np.nanmin(low[x >= x0 - 1.0])) for x, low in lows
             if np.any(np.isfinite(low[x >= x0 - 1.0]))]
    if not under:
        return None
    need = (min(under) - pad * (ymax - ymin) - g * ymax) / (1.0 - g)
    return None if not np.isfinite(need) or ymin <= need else need


def place_legend(fig, axes, lows):
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    order = [labels.index(l) for l in LEGEND_ORDER if l in labels]
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    kw = dict(loc='lower right', ncol=4, handlelength=1.6, columnspacing=1.0, handletextpad=0.5,
              borderaxespad=0.4, framealpha=0.95)
    best = None
    for ci, ax in enumerate(axes):
        leg = ax.legend(handles, labels, **kw)
        ymin, ymax = ax.get_ylim()
        need = legend_ymin(fig, ax, leg, lows[ci])
        cost = 0.0 if need is None else (ymin - need) / (ymax - ymin)
        leg.remove()
        if best is None or cost < best[1] - 1e-9:
            best = (ci, cost, need)
    ci, cost, need = best
    axes[ci].legend(handles, labels, **kw)
    if need is not None:
        axes[ci].set_ylim(bottom=need)


def strategies_of(learner):
    return ['BALD', 'US'] + (['RANKUNC'] if learner == 'BAYES-BT' else []) + ['DVF', 'CHEB', 'CSS', 'POLY']


def curves(a):
    style(RC)
    grid = np.linspace(0.0, 1.0, 201)[1:]
    for learner, name in (('BAYES-BT', 'fig_learning_curves_bayes.png'), ('FTRL-BT', 'fig_learning_curves_ftrl.png')):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        lows = {0: [], 1: []}
        print(f'% {learner}: rule | runs | mean and max stop (% of budget) | ASRS at the marker')
        for ci, (ds, title) in enumerate(regimes(a.synthetic)):
            ax = axes[ci]
            every = lambda s: [r for n in SIZES for m in CRITERIA for r in runs(a, ds, learner, s, n, m)]
            Y = random_curves(every('RANDOM'), grid)
            if len(Y):
                m_, s_ = mean_band(Y)
                ax.plot(100 * grid, m_, '--', color='0.25', lw=2.2, label='Random')
                ax.fill_between(100 * grid, m_ - s_, m_ + s_, color='0.35', alpha=0.15, lw=0)
                lows[ci].append((100 * grid, m_ - s_))
            for s in strategies_of(learner):
                Y, B = frozen(every(s), s, grid)
                if not len(Y):
                    continue
                m_, s_ = mean_band(Y)
                x, mm, ss = truncate(grid, m_, s_, float(B.max()))
                mk, col = STYLE[s]
                ax.plot(100 * x, mm, '-', color=col, lw=2, label=LABEL[s])
                ax.fill_between(100 * x, mm - ss, mm + ss, color=col, alpha=0.15, lw=0)
                lows[ci].append((100 * x, mm - ss))
                yb = float(np.interp(B.mean(), grid, m_))
                ax.plot(100 * B.mean(), yb, mk, color=col, ms=10, mec='k', zorder=ZMARK[s])
                print(f'% {title:18s} {s:7s} {len(Y):4d} runs | stop {100 * B.mean():5.1f} / {100 * B.max():5.1f} | {yb:.3f}')
            ax.set_title(title)
            ax.grid(alpha=.3)
            ax.set_xlim(0, 100)
            ax.set_xlabel('fraction of the budget used (%)')
            if ci == 0:
                ax.set_ylabel('ASRS')
        fig.tight_layout()
        place_legend(fig, axes, lows)
        save(fig, a, name)


# =================================================================================== mobile
def mobile(a, name='mobile'):
    T = 45
    x = np.arange(1, T + 1)
    learners = ('BAYES-BT', 'FTRL-BT')
    load = lambda learner, s: runs(a, name, learner, s, 10, 4, 94)
    style({'font.size': 12, 'axes.titlesize': 13, 'legend.fontsize': 10.5})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, learner in zip(axes, learners):
        for s, col, lab in (('BALD', 'tab:red', 'BALD'), ('US', 'tab:orange', 'US'), ('RANDOM', 'tab:gray', 'Random')):
            rl = load(learner, s)
            C = RS.curves(rl)
            if not C.size:
                continue
            mu = np.nanmean(C, 1)
            ci = np.array([RS.sem2(C[t]) for t in range(T)])
            ax.plot(x, mu, '-', color=col, lw=2.2, label=lab)
            ax.fill_between(x, mu - ci, mu + ci, color=col, alpha=0.15, lw=0)
            if s != 'RANDOM':
                stop = RS.stops(rl, s).mean()
                ax.plot(stop, np.interp(stop, x, mu), 'o', color=col, ms=9, mec='k', zorder=5)
                print(f'% {name} {learner} {s}: mean one-bit stop {stop:.1f} questions')
        ax.set_title(learner)
        ax.set_xlabel('number of questions')
        ax.grid(alpha=.3)
        ax.set_xlim(1, T)
    axes[0].set_ylabel('ASRS')
    axes[0].legend(loc='lower right')
    fig.tight_layout()
    save(fig, a, f'fig_{name}_curves.png')

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    grid = x / T
    for ax, learner in zip(axes, learners):
        Y = random_curves(load(learner, 'RANDOM'), grid)
        if len(Y):
            m_, s_ = mean_band(Y)
            ax.plot(x, m_, '--', color='0.25', lw=2, label='Random')
            ax.fill_between(x, m_ - s_, m_ + s_, color='0.35', alpha=0.12, lw=0)
        for s in strategies_of(learner):
            Y, B = frozen(load(learner, s), s, grid)
            if not len(Y):
                continue
            m_, s_ = mean_band(Y)
            stop = np.rint(B * T)
            keep = x <= stop.max()
            mk, col = STYLE[s]
            ax.plot(x[keep], m_[keep], '-', color=col, lw=2, label=LABEL[s])
            ax.fill_between(x[keep], (m_ - s_)[keep], (m_ + s_)[keep], color=col, alpha=0.12, lw=0)
            yb = float(np.interp(stop.mean(), x, m_))
            ax.plot(stop.mean(), yb, mk, color=col, ms=9, mec='k', zorder=ZMARK[s])
            print(f'% {name} {learner} {s:7s}: stop {stop.mean():.1f} (max {stop.max():.0f}) questions, ASRS {yb:.3f}')
        ax.set_title(learner)
        ax.set_xlabel('number of questions')
        ax.grid(alpha=.3)
        ax.set_xlim(1, T)
    axes[0].set_ylabel('ASRS')
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    order = [labels.index(l) for l in LEGEND_ORDER if l in labels]
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order], loc='upper left', ncol=4,
                   handlelength=1.6, columnspacing=1.0, handletextpad=0.5, framealpha=0.95)
    fig.tight_layout()
    save(fig, a, f'fig_{name}_curves_all.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('figure', choices=['sensitivity', 'frontier', 'curves', 'mobile', 'all'])
    ap.add_argument('--results', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--out', default=None,
                    help='default: <results>/figures, or <results>/figures/<synthetic> for another data group')
    ap.add_argument('--synthetic', default='synthetic', help='group of the synthetic runs')
    a = ap.parse_args()
    a.out = a.out or os.path.join(a.results, 'figures', *([] if a.synthetic == 'synthetic' else [a.synthetic]))
    for f in (['sensitivity', 'frontier', 'curves', 'mobile'] if a.figure == 'all' else [a.figure]):
        globals()[f](a)


if __name__ == '__main__':
    main()
