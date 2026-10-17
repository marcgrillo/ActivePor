# activepor: active learning for Probabilistic Ordinal Regression

A decision maker (DM) compares pairs of alternatives. A Probabilistic Ordinal Regression (POR) learner
turns the answers into a distribution over additive value functions and ranks the alternatives. This
package chooses the next pair to ask with **Uncertainty Sampling (US)** or **Bayesian Active Learning by
Disagreement (BALD)**, monitors the elicitation, and stops it when the information left in the unasked
pairs falls below one bit. It also contains four question-selection rules from the literature, a
generator of simulated DMs, and scripts that run experiments and summarize them.

| Folder | Contents |
|---|---|
| `activepor/` | the package |
| `scripts/` | data generation, experiments, tables and figures |
| `data/` | datasets, created by `scripts/generate_data.py` (not included) |
| `tests/` | unit tests, including the worked examples of the literature rules |

## Installation

Python 3.10 or newer:

```
git clone https://github.com/marcgrillo/ActivePor.git
cd ActivePor
pip install -r requirements.txt
pip install -e .                # makes the package `activepor` importable from anywhere
python -m pytest tests          # about 10 seconds
```

The scripts also run without `pip install -e .`, since they add the repository root to the Python path.

## Quick start

Rank a set of alternatives by asking a DM the most informative questions, and stop at one bit:

```python
import numpy as np
from activepor.model import characteristic_vectors
from activepor.learners import BayesianLearner
from activepor.acquisition import scores, unasked_pairs
from activepor.metrics import rai

perf = np.random.default_rng().random((8, 3))      # 8 alternatives x 3 criteria, all to be maximised
V = characteristic_vectors(perf, n_pieces=3)       # piecewise-linear value model, 3 pieces per criterion
learner = BayesianLearner(V, 'BT', n_samples=2000) # BAYES-BT; FTRLLearner(V, 'BT') gives FTRL-BT

def ask(i, j):                                     # replace by a real DM: return (preferred, other)
    return (i, j) if perf[i].sum() > perf[j].sum() else (j, i)

learner.observe(*ask(0, 1))
learner.fit()
while True:
    pool = unasked_pairs(len(perf), learner.answers)
    bald = scores(learner, pool, 'BALD')
    if len(pool) == 0 or bald.sum() < np.log(2):   # residual information below one bit: stop
        break
    learner.observe(*ask(*pool[int(np.argmax(bald))]))
    learner.fit()

print(len(learner.answers), 'questions asked')
print(np.round(rai(V, learner.draws()), 2))        # rank acceptability indices of the recommendation
```

## Components

| Module | Contents |
|---|---|
| `model` | `characteristic_vectors` (piecewise-linear value model), `response_probability` (Linear and Bradley-Terry links) |
| `learners` | `BayesianLearner` (BAYES-LIN, BAYES-BT), `FTRLLearner` (FTRL-LIN, FTRL-BT), `make_learner` |
| `acquisition` | `scores(learner, pool, rule)` for US, BALD and RankUnc; `unasked_pairs` |
| `diagnostics` | information-gain residual (`hypothetical_surprises`, `LinkResidual`), `curvature_ratio`, `residual_information`, `split_rhat` |
| `metrics` | `rai`, `poi`, `asrs`, `asps`, `aulc`, `first_reach`, `one_bit_stop` |
| `elicitation` | `SimulatedDM`, `make_strategy`, `run` (one elicitation with a simulated DM) |
| `baselines` | `run_rule` for Poly, Dvf, Cheb and Css |
| `data` | dataset format, `generate`, `instances` |
| `results` | storage of runs and the statistics computed from them |

**Learners.**

* BAYES-LIN: Dirichlet(0.5) prior on the simplex, random-walk Metropolis-Hastings with a multiplicative
  log-normal proposal.
* BAYES-BT: independent Gamma(1, 1) priors, Hamiltonian Monte Carlo in log w with 10 leapfrog steps.
* Both samplers adapt their step during 500 tuning iterations (acceptance rate 0.3 for
  Metropolis-Hastings, 0.65 for HMC), discard them, and start from the posterior mean of the previous
  round. `n_samples` draws are kept (2000 by default; the synthetic presets of `run_experiments.py`
  keep max(2000, 500 m, 200 n) for BAYES-BT).
* FTRL-LIN: MAP estimate with the regularizer -0.1 sum log w on the simplex (SLSQP).
* FTRL-BT: MAP estimate with the regularizer sum (w - log w) (SCS through CVXPY).
* The FTRL learners use the Laplace approximation at the MAP estimate. For FTRL-LIN it is computed in
  the affine chart w = w_hat + Psi nu, with Psi an orthonormal basis of the tangent space of the simplex.

**Scores.** The Bayesian learners use Monte Carlo estimates on their posterior draws. The FTRL learners
use Laplace-Taylor scores, which are second-order expansions of the response probability under the
Laplace approximation. RankUnc (`-|POI - 1/2|`) needs posterior draws and is available for the Bayesian
learners only.

**Monitoring.**

* The information-gain residual compares the information that the model expects from the answer
  with the information the answer actually brings. The surprise is a KL divergence: between
  reweighted posterior draws for the Bayesian learners, between Laplace Gaussians for the FTRL learners.
* The curvature ratio compares the cubic and the quadratic terms of the FTRL objective at one standard
  deviation along the columns of the Cholesky factor of the Laplace covariance. The cubic term is the
  exact third derivative.
* The residual information is the sum of the BALD scores of the unasked pairs. The one-bit rule stops
  after the first answer that brings it below log 2. `split_rhat` checks the posterior draws of the
  Bayesian learners.

**Implementation notes.**

* Round 1 asks the first pair of the DM's random question order, for every strategy; the strategy
  chooses from round 2 on.
* Ties between scores go to the first pair in the order of `itertools.combinations`.
* Probabilities are clipped to [1e-9, 1 - 1e-9].
* A ridge of 1e-8 is added to the Hessian before it is inverted, plus 1e-6 in the inverse for FTRL-BT.
* Stopping points are counted in answers collected: a run that stops with s answers uses s / T_max of
  the budget, and its recommendation is the one after those s answers.

**Literature rules.**

* Poly is the polyhedral question design of Toubia, Hauser and Simester (2004).
* Dvf is the heuristic of Ciomek, Kadziński and Tervonen (2017) that splits the set of compatible value
  functions most evenly.
* Cheb is the Chebyshev-centre rule of Liang, Zhang and Su (2025).
* Css is the current-solution strategy of minimax-regret elicitation (Boutilier et al. 2006; Benabbou,
  Perny and Viappiani 2017).
* A rule sees only the answers, so its question sequence is computed once and every learner is fed
  with it. When the rule has no pair to ask (its own termination condition), a uniformly random
  unasked pair is asked, and the rule's stopping point is the number of answers collected before that
  round.
* With contradictory answers, Poly relaxes all answer constraints by the smallest common slack.
* With contradictory answers, Dvf, Cheb and Css keep a largest consistent subset of the answers, found
  by a mixed-integer program with a 20-second time limit. When the limit is reached, the subset of
  minimal total error is used instead.
* Poly draws its budget uniformly from [0, 0.5] and redraws it up to 30 times.
* Dvf uses 10,000 hit-and-run draws after 2,000 burn-in steps, with thinning (d - 1)^3 when that is at
  most 200 and 10 d otherwise.

## Data

A dataset is a folder with `meta.json` (how it was made) and one `n{n}_m{m}.npz` file per configuration
of `n` alternatives and `m` criteria. Each file holds H instances, with these arrays:

| Array | Shape | Contents |
|---|---|---|
| `performance` | (H, n, m) | performances, all criteria of gain type |
| `ranking` | (H, n) | the DM's ranking, 0 = best |
| `answers` | (H, n(n-1)/2, 2) | the DM's answer to every pair as [preferred, other], in the DM's random question order |
| `utility` | (H, n) | true value of every alternative (NaN for real DMs) |

The answer to every pair is fixed in advance, so any strategy that asks a pair gets the same answer.

No datasets are included; `scripts/generate_data.py` creates them. These three commands create the
datasets that the experiment presets below expect (a few seconds each):

```
python scripts/generate_data.py --out data/synthetic --utility exponential
python scripts/generate_data.py --out data/synthetic_pwl --utility piecewise_linear --pieces 3
python scripts/generate_data.py --out data/link --utility piecewise_linear --pieces 3 --sizes 30 --answers consistent bt linear
```

| Command | Writes | True value functions | Answers | n | m |
|---|---|---|---|---|---|
| first | `data/synthetic/consistent`, `data/synthetic/bt20` | exponential marginals | consistent; Bradley-Terry rule with 20% inverted answers | 10, 30 | 2, 6, 10 |
| second | `data/synthetic_pwl/consistent`, `data/synthetic_pwl/bt20` | piecewise linear, 3 pieces | as above | 10, 30 | 2, 6, 10 |
| third | `data/link/consistent`, `data/link/bt20`, `data/link/linear` | piecewise linear, 3 pieces | as above, and the linear rule | 30 | 2, 6, 10 |

One folder per answer rule is written. The rules share the instances, the question order and the
random numbers that decide the inversions. The Bradley-Terry scale is tuned on the generated pairs, so
that the target share of inversions is expected. `--sizes`, `--criteria`, `--instances` and
`--inconsistency` change the design (defaults: n = 10, 30; m = 2, 6, 10; 20 instances; 20%
inversions). Every call draws fresh random numbers; the seed it used is recorded in `meta.json`, and
`--seed` sets it. Other designs can be written to other folders (e.g. `--out data/my_study`) and run
with `--datasets` (below).

**Real data.** `scripts/build_mobile_dataset.py` builds `data/mobile` from the spreadsheet of the
mobile-phone experiment of Ciomek, Kadziński and Tervonen (Omega, 2017): 10 contracts, 4 criteria and
the rankings stated by 94 DMs. The spreadsheet is not included here; it is available from the authors
of that study.

```
python scripts/build_mobile_dataset.py path/to/mobile-phone-experiment.xlsx
```

## Running experiments

Each preset reads the datasets created above (`mobile` reads `data/mobile`, built from the spreadsheet).

```
python scripts/run_experiments.py synthetic                # data/synthetic: FTRL-BT and BAYES-BT, all strategies
python scripts/run_experiments.py synthetic-pwl            # the same on data/synthetic_pwl
python scripts/run_experiments.py link                     # data/link: BAYES-LIN with US, BALD and random, 50 questions
python scripts/run_experiments.py mobile                   # data/mobile, 5 pieces per criterion (--pieces to change)
python scripts/run_experiments.py synthetic --datasets my_study/consistent my_study/bt20   # generated data
```

* Every run is written to `results/<dataset>/<learner>/<strategy>/n{n}_m{m}/<instance>.npz`, with the
  per-round arrays described in `activepor/elicitation.py`: answers, ASRS, ASPS, residual
  information and, where computed, the monitoring quantities.
* A run that already exists is skipped, so an interrupted sweep can be restarted with the same command.
* `--learners`, `--strategies`, `--sizes`, `--criteria` and `--instances` select a part of the sweep.
* `--workers` sets the number of parallel processes.
* `--dry-run` lists the runs without executing them.
* Dataset names are relative to `data/`; a path to a dataset elsewhere also works.
* Every run draws fresh random numbers and stores its seed. `--seed` makes a whole sweep repeatable.
* Runtimes with 16 workers:
  * `link`: about 10 minutes.
  * `mobile`: 1-2 hours.
  * `synthetic`: about a day, most of it BAYES-BT and the Css and Dvf sequences at n = 30.

`scripts/laplace_check.py` compares the Laplace-Taylor scores of FTRL-BT and FTRL-LIN with exact
scores. The exact scores are computed by importance sampling with 40,000 draws, at 10 early rounds of
every instance of `data/synthetic_pwl/consistent` (FTRL-BT) and `data/link/linear` (FTRL-LIN) with
n = 30. The comparison is written to `results/laplace_check.csv` (about 20 minutes).

## Tables and figures

The tables are printed as LaTeX rows, followed by the numbers behind them. The figures are written to
`results/figures/`.

| Command | Output |
|---|---|
| `python scripts/make_tables.py efficiency` | AULC, T_theta (questions to reach 75% of the best mean ASRS of random elicitation) and saving over random elicitation |
| `python scripts/make_tables.py stopping` | one-bit rule: share b of the budget used, share r of the final ASRS retained, and other thresholds |
| `python scripts/make_tables.py baselines` | every strategy at its own stopping point, against random elicitation after as many questions |
| `python scripts/make_tables.py link` | trend of the information-gain residual, and gains over random elicitation, on the `link` runs |
| `python scripts/make_tables.py laplace` | agreement between Laplace-Taylor and exact scores by range of the curvature ratio |
| `python scripts/make_tables.py mobile`, `mobile-rai` | efficiency and one-bit stop on the `mobile` runs; rank acceptability indices of one DM (`--subject`) |
| `python scripts/make_figures.py sensitivity` | AULC gains over random elicitation against n and m |
| `python scripts/make_figures.py frontier` | budget used against quality retained, for a range of stopping thresholds |
| `python scripts/make_figures.py curves` | learning curves of all strategies, each run frozen at its stopping point |
| `python scripts/make_figures.py mobile` | learning curves on the `mobile` runs |

`--synthetic <group>` makes the efficiency, stopping and baseline tables and figures from another group
of synthetic runs: `synthetic_pwl`, or generated data such as `my_study`. The tables and figures read
the folders `consistent` and `bt20` of the group, and the figures go to `results/figures/<group>/`.

## License

MIT, see `LICENSE`.
