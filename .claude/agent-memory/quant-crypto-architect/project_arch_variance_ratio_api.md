---
name: arch.unitroot.VarianceRatio expects price levels not returns
description: Critical API detail -- arch.unitroot.VarianceRatio takes cumulative price levels (cumsum of returns), not raw returns, as input
type: project
---

`arch.unitroot.VarianceRatio(y, lags=q, robust=True)` expects **price levels** (cumulative sum of returns), not raw returns. It internally computes first differences. Passing raw returns gives VR values near 0 instead of near 1 under the null.

**Why:** Discovered during Phase 5B implementation when variance ratio tests on white noise returned VR ~ 0.04 instead of ~1.0. The fix is `prices = np.cumsum(returns_array)` before passing to VarianceRatio.

**How to apply:** Any code using `arch.unitroot.VarianceRatio` must pass `np.cumsum(returns)` not `returns`. This applies to the `SerialDependenceAnalyzer` in `src/app/profiling/application/serial_dependence.py`.
