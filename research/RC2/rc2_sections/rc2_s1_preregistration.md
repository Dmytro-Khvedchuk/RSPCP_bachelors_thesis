# RC2 Section 1: Pre-Registration & Decision Rules

## Methodology

Section 1 defines all mechanical decision criteria *before* examining any data,
following the pre-registration paradigm advocated by Nosek et al. (2018) and
Chambers & Tzavella (2022). The goal is to convert exploratory analysis into
confirmatory analysis, reducing researcher degrees of freedom. Any post-hoc
deviation is counted as a trial for the Deflated Sharpe Ratio (Bailey & Lopez de
Prado, 2014).

## Guiding Principles

1. **Mechanical rules over human judgment.** Every keep/drop/proceed decision follows
   a pre-specified threshold.
2. **Negative results are valid.** If the data says "no predictable structure," this
   is documented honestly.
3. **Economic significance over statistical significance** (Ziliak & McCloskey, 2008).
   A p-value below 0.05 says nothing about whether an effect is large enough to profit
   from after transaction costs.
4. **Trial counting is honest.** Each post-hoc decision increments the trial counter
   for the DSR.

## Pre-Registered Parameters (Executable)

The notebook instantiates all thresholds in a single `RC2PreRegistration` Pydantic
model (frozen=True) to prevent mutation. Key parameters:

### Temporal Partitions
- Feature selection: 2020-01-01 to 2023-01-01
- Model development: 2020-01-01 to 2024-01-01
- Final holdout: 2024-01-01 onwards

### Feature Validation (Rules F1-F4)
- MI permutations: 1,000 with block size 50
- Ridge DA permutations: 500
- Significance: alpha = 0.05 with Benjamini-Hochberg correction
- Stability: MI significant in >= 50% of 4 temporal windows
- Minimum features fallback (F2): keep top 5 if < 5 pass F1
- VIF warning: > 10.0 (diagnostic, no auto-drop)

### Bar Type Classification (Rule B1)
- Tier A (>= 2,000 bars): Full ML pipeline
- Tier B (500-2,000): Restricted to Ridge, logistic, gradient boosting
- Tier C (< 500): Statistical profiling only, no modeling

### Economic Thresholds
- Round-trip cost: 20 bps (Binance spot)
- Min DA excess over break-even: 0.5 pp (Rule DA3)
- Max break-even DA for viability: 55% (Rule H2)

### Model Complexity (Rules M1-M2)
- Linear-first: BDS must reject i.i.d. to justify nonlinear models
- DL gate: BDS + N_eff >= 2,000 + >= 3 features

### Trial Counting
- Pre-registered combos: 4 assets x 5 bar_types x 3 horizons = **60**
- Classifiers, regressors, and recommender configs add ~21 more
- Post-hoc deviations start at 0

## Go/No-Go Decision Matrix

Seven criteria (G1-G7) are defined, with G1/G2/G4 as blockers and G3/G5/G6/G7 as
informational. The decision is fully mechanical -- Section 8 computes each criterion
programmatically and fills in the table.

## Connection to Lopez de Prado Methodology

The pre-registration framework directly addresses Lopez de Prado's (2018) warnings
about backtest overfitting and the "Deflated Sharpe Ratio." By committing to
thresholds before seeing data, the researcher's degrees of freedom are constrained.
The trial counter enables honest DSR computation in Phase 14, penalizing the number
of configurations tested.

## Recommendations for Next Steps

- Maintain the 0 post-hoc deviation count. Every future change to feature selection,
  thresholds, or bar types must be documented as a deviation.
- The DSR threshold of SR_obs > 2.0 (at ~80-100 trials and crypto kurtosis 5-15)
  is demanding. The pre-registration at 60 trials provides headroom.
