---
name: RC3 Pivot Decision and Thesis Narrative Pivot
description: RC3 gating test results -- PIVOT confirmed, all 5 models at coin-flip, Phases 10-12 abandoned, thesis pivots to methodology + negative result narrative
type: project
---

RC3 gating test completed (2026-03-28): **PIVOT confirmed.** All direction classifiers produce coin-flip results.

**Why:** Five model families (Ridge, ElasticNet, LightGBM, ExtraTrees, MLP) across 3 bar types (dollar N=5141, dollar_imbalance N=423, volume N=3118) with CPCV all fail to exceed majority class baseline. Best DA=50.64% vs majority=51.12% (p=0.517). GARCH(1,1) unanimously beats ML for volatility (Wilcoxon W=0.0). Abstention filter DA=75.79% vs majority=76.09% (p=0.871). Zero post-hoc deviations.

**How to apply:**
- Phases 10-12 (full classification/regression/recommendation modeling) are ABANDONED per pre-registration PIVOT rule
- Thesis narrative pivots to: "Empirical evaluation of Lopez de Prado framework -- evidence for market efficiency"
- Writing focus should be on methodology description (Clean Architecture, CPCV, pre-registration, stationarity policy) and the strength of the negative result
- The software system itself (engineering quality, Clean Architecture) becomes a primary contribution alongside the scientific finding
- Key thesis chapters: methodology (system design), data adequacy (RC1/RC2/RC7), predictability analysis (PE/VR/BDS/GARCH/classifier failure), discussion (market efficiency argument)
- The CPCV splitter, backtest engine (Phase 8/9), and all RC1-RC7 analysis are retained as thesis content
