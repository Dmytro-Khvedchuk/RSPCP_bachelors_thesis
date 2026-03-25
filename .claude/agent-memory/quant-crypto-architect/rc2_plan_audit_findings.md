---
name: RC2 Plan Audit Findings — Resolved
description: All 20 audit findings from RC2 plan audit have been addressed in IMPLEMENTATION_PLAN.md (2026-03-25)
type: project
---

RC2 plan audit completed 2026-03-25. Verdict: PASS WITH CONCERNS. **All findings now addressed in IMPLEMENTATION_PLAN.md.**

**Resolution summary (2026-03-25):**

1. **A1 (CRITICAL) Trial count:** Two-tier DSR framework added to Quick Reference + Phase 14E. Tier 1 = 60 pre-registered (sensitivity), Tier 2 = exhaustive (primary). TrialCounter Pydantic model tracks across phases.

2. **C1 (MAJOR) Ensemble DA untested:** Phase 9-pre gating test added with three outcomes (FULL GO / CONDITIONAL GO / PIVOT). Pre-registered thresholds defined.

3. **A2 (MAJOR) LTCUSDT / 4 assets:** Updated all references: Phase 9G (pooling), Phase 12A (labels: 1350 not 1800), Phase 13A (2/3 not 3/4), Asset Universe table, dependency diagram.

4. **D1 (MAJOR) Phase 10 before 9:** Phase 10D explicitly split into STANDALONE (available immediately) and PIPELINE (after Phase 9, in RC3). HAR-RV, QLIKE, Mincer-Zarnowitz added.

5. **C2 (MAJOR) Vol forecasting:** Explicit forward_rv_h target in 10A, HAR-RV model in 10B, QLIKE + Mincer-Zarnowitz R2 + MAE on log-vol metrics defined.

6. **E1 (MAJOR) Ensemble DA hand-waved:** Addressed by Phase 9-pre gating test (pre-registered, mechanically computed).

7-20. All minor/gap findings addressed: stationarity policy (5-close.3), atr_14/rsi_14 (5-close.2), cost sensitivity at 5 levels (5-close.1), temporal instability (12A rolling MI proxy), feature set hierarchy (Quick Reference), conditional break-even DA (5-close.6), BTC-lagged features (9B), SOLUSDT Tier B protocol (5-close.7), NMI fix (5-close.4), label overlap fwd_logret_24 (9F), R5 confrontation paragraph (14G), expected negative results (Quick Reference), canonical sample size table (Quick Reference).

**Why:** Comprehensive resolution ensures the plan is audit-clean before Phase 7 begins.

**How to apply:** The plan is now consecutive and delegatable. Current work = Phase 5-close (7 sub-tasks). After 5-close, proceed to Phase 7. Phase 9-pre gating test is the critical decision point before Phases 9/10.
