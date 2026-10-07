# QubeResearch: a hierarchical Bayesian model for next-day portfolio returns

[![CI](https://github.com/MarcoGaloppo/QubeResearch/actions/workflows/ci.yml/badge.svg)](https://github.com/MarcoGaloppo/QubeResearch/actions/workflows/ci.yml)

My solution to the **QRT Challenge Data #167, "Asset Allocation Performance forecasting"**
([challengedata.ens.fr/challenges/167](https://challengedata.ens.fr/challenges/167)).
Each row holds a portfolio's last 20 daily returns and signed volumes. The task is to predict the
**sign** of the next day's return, and the score is plain accuracy.

The full analysis, from the raw data to the submission file, is in
[`BaysianHierarchicalApproach.ipynb`](BaysianHierarchicalApproach.ipynb).  The reasoning has been written out cell by cell to allow for a better reading.

## Results

These come from a honest cross-validation with 5 folds, holding out whole reconstructed stretches of time, with everything refitted inside each fold. 

| model | CV accuracy |
|---|---|
| always predict "up" (base rate) | 0.5072 |
| sign of yesterday's return | 0.5189 |
| pooled OLS (one set of coefficients for all portfolios) | 0.5195 |
| **hierarchical Bayesian model (HBM)** | **0.5234** |

- **Edge over the base rate:** +2.2 points, z = 9.5 (per-date paired test).
- **Value of the hierarchy over pooled OLS:** +0.46 points, **z = 3.4** with a block bootstrap
  over stretches, which accounts for serial correlation. 

The edge is small, as we would expect.

## Approach

1. **We rebuild the date order.** The dates are shuffled in the data, and yet consecutive 20-day   windows overlap. We thus use a nearest-neighbour matching of normalized windows, together with a majority vote across portfolios, and a **5σ rank-correlation check on every link** to rebuild
the training dates into stretches of consecutive days. 
2. **We implement honest cross validation.** With overlapping windows, random-date CV would just leak targets into training. This would inflate the score. On the other hand, holding out whole stretches removes this would-be leak. Anyway, the test set overlaps no training date.
3. **We use inputs in volatility units.** We divide yesterday's return and the mean of days 2–5 by the row's own 20-day vol, which removes most of the volatility-mixing part of the fat tails (kurtosis drops from about 7.7 to 1.9).
4. **The model.** We adopt a student-t regression with a per-group tail index ν, coupled with portfolio coefficients pooled towards group and global centres. To wit,

   $$y_i = Z_i\cdot\theta_p + \sigma_p\,\epsilon_i,\quad \epsilon_i\sim t_{\nu_g},\qquad
   \theta_p\sim\mathcal N(\theta_g,\ \mathrm{diag}\,\tau^2),\quad \theta_g\sim\mathcal N(\theta_0,\ \mathrm{diag}\,\omega^2)\,.$$

   This is fitted with a hand-written **Gibbs sampler** (numpy/scipy only). To do so, the Student-t is written as a scale mixture of Gaussians, and ν is updated with a collapsed Metropolis step. Four chains from different starting values converge rapidly (R̂ < 1.01).

## Extensions tested and rejected

- **Signed volume (VHBM).** Although signed volume shows a clear reversal effect in two groups
  (about $4\sigma$) in-sample, it is too small to flip sign calls. In the end the CV gain is about 0 points. Dropped.
- **Time-varying coefficients (THBM).** The plan was to use "pseudo-rows" built from the test
  windows' own history. They turned out to be structurally biased against real outcomes, and with a non-trivially structured bias at that. A first calibrated correction gained only +0.08 points at 1.7σ, so it is not worth the extra assumptions. Dropped.

## How to run

```bash
pip install -r requirements.txt
jupyter lab BaysianHierarchicalApproach.ipynb
```

**Data.** The challenge data is **not included** in this repository. Register on Challenge Data,
download `X_train.csv`, `y_train.csv`, `X_test.csv` and `sample_submission.csv` from the challenge
page, and place them next to the notebook. A full run takes about an hour on a laptop, mostly for
the MCMC fits and the cross-validation.

## Continuous integration

The notebook can't be executed in CI without the (non-redistributable) data. On every push and
pull request, GitHub Actions instead runs static checks
([`.github/scripts/check_notebook.py`](.github/scripts/check_notebook.py)):

- the notebook is a valid Jupyter notebook;
- every code cell is valid Python;
- all imported libraries install from `requirements.txt`;
- no challenge data files, and no files over 50 MB, are committed.

## Repository structure

```
BaysianHierarchicalApproach.ipynb   the full analysis
requirements.txt                    Python dependencies
.github/workflows/ci.yml            CI workflow
.github/scripts/check_notebook.py   CI checks
LICENSE                             MIT
```

## License

Code: [MIT](LICENSE). The challenge data belongs to its provider and is subject to the
Challenge Data terms of use; it is not redistributed here.
