"""Active learning for Probabilistic Ordinal Regression (POR) in multiple criteria ranking.

Modules:

    model         piecewise-linear value model and response links
    learners      BAYES-LIN, BAYES-BT, FTRL-LIN, FTRL-BT
    acquisition   US, BALD and RankUnc scores
    diagnostics   information-gain residual, curvature ratio, residual information, split-R-hat
    metrics       RAI, POI, ASRS, ASPS and learning-curve statistics
    elicitation   one elicitation run with a simulated decision maker
    baselines     the question-selection rules Poly, Dvf, Cheb and Css from the literature
    data          dataset format and the synthetic generator
    results       storage of runs and the statistics computed from them
"""
__version__ = '1.0.0'
