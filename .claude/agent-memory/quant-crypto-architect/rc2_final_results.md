---
name: RC2 Final Results Summary
description: Complete RC2 Go/No-Go results -- GO decision with weak signal, volatility features dominate, 0 post-hoc deviations
type: project
---

RC2 overall decision: **GO** (all 3 blockers pass). Date: 2026-03-24.

**Why:** The data exhibits structure beyond random walk (PE < 0.98 on imbalance bars, BDS rejects i.i.d. for all 4 assets), features carry statistically detectable MI (8/23 pass BH-corrected MI), and sample sizes are adequate (BTCUSDT dollar N_eff = 5,286).

**How to apply:** Signal is very weak -- no single feature exceeds break-even DA (57.23% on dollar bars). Best feature ret_zscore_24 at 51.81% (-5.42 pp below break-even). The project must rely on multi-feature ensemble, regime conditioning, and meta-labeling. Volatility features dominate (rv_12, rv_24, rv_48, amihud_24, bbwidth_20_2.0).

Key numbers:
- Kept features (fallback F2): amihud_24, bbwidth_20_2.0, rv_12, rv_24, rv_48
- Asset universe: BTC (confirmed), ETH (confirmed), SOL (marginal N_eff=808), LTC (excluded -- only 199 dollar bars)
- PE(d=5): dollar 0.9977, vol_imbalance 0.9740 (structure), time_1h 0.9992
- BDS: all 4 assets reject i.i.d. -> DL gate OPEN
- Kendall tau cross-asset: 0.571 (p<0.0001) -> shared features
- BTC Granger-causes ETH/LTC/SOL at lag 1
- Buy-and-hold Sharpe: 0.576 (hurdle)
- GARCH persistence: 1.000 (near-IGARCH) on all assets
- Post-hoc deviations: 0, trial count: 60
- Regression arm: FEASIBLE
- All 3 horizons (fwd_logret_1, _4, _24) confirmed
