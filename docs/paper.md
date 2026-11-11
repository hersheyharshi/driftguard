# DriftGuard: Cause-Attributing Data Quality for Long-Lived Ingestion Pipelines

**Harshita Pandita** · draft, November 2026

## Abstract

Data pipelines that ingest the same source for years face two kinds of change. Schemas evolve: columns get renamed, re-encoded, added or dropped. Values go wrong in ways that pass simple row-level rules, such as unit changes, timezone shifts, replayed deliveries and swapped columns. Existing tools mostly *detect* such problems. They rarely *explain* them, and they never measure what the problems cost downstream. We present DriftGuard, a pipeline that (i) maps arbitrary header layouts onto a canonical schema using name similarity, statistical column fingerprints, a globally optimal assignment and cross-field invariants; (ii) tests drifted batches against specific, repairable hypotheses, with self-calibrated thresholds; (iii) repairs only attributed faults, verifies each repair by re-inspection, and holds everything else. On a synthetic NYC-taxi benchmark with known ground truth (six historical header eras, twelve fault types, five seeds), DriftGuard detected all 100 injected detectable faults, named the correct cause every time, and raised false alarms on 1.2% of clean batches. It identified 97% of columns from values alone under meaningless headers, with zero silent mis-mappings. In a day-ahead demand-forecasting task at a 40% batch fault rate, it prevented 94% of the accuracy loss that a naive pipeline suffered.

## 1 Introduction

My motivation comes from industrial work. Reconciling five undocumented schema variants across 2,000+ motor-test-bench files showed me that most data errors are *silent*: a file loads, the numbers look plausible, and the model trained on them is quietly worse. The NYC TLC yellow-taxi records (2009–present, >1B trips) are a public dataset with the same problems. Column names change across eras (`Trip_Pickup_DateTime` → `pickup_datetime` → `tpep_pickup_datetime`). Categorical encodings change (`CASH` → `CSH` → `2`). Coordinates were replaced by zone IDs in mid-2016. New fees appear as new columns.

Research questions:
- **RQ1.** How accurately can columns be identified when headers are unfamiliar or meaningless, and what does each type of evidence contribute?
- **RQ2.** Can batch-level faults be attributed to a *specific cause* reliably enough to repair them automatically?
- **RQ3.** How much does bad ingestion degrade a downstream model, and how much of that damage does the pipeline prevent?

## 2 Related work

*To be completed.* Schema matching (e.g. COMA, Valentine benchmark); data validation in ML pipelines (TFX Data Validation, Great Expectations, Deequ); drift detection (KS / PSI tests, label-shift literature); data lineage. Positioning: DriftGuard combines instance-based schema matching with hypothesis-driven drift *attribution* and closes the loop with automatic repair, verification, and a downstream evaluation.

## 3 Method

**3.1 Schema registry.** For each source column *s* and canonical column *c*, a name score N(s,c) ∈ [0,1] is the maximum over *c*'s aliases of exact-match (1.0) or 0.6·token-Jaccard + 0.4·sequence similarity after abbreviation expansion. A statistical score T(s,c) compares *s*'s fingerprint with a reference profile of *c*: signed-log quantiles, cardinality, integer, zero and negative shares, and value-frequency vectors (for categoricals with a vocabulary map, coverage of the vocabulary plus total-variation distance). The two are blended as S = 0.6·N + 0.4·T. If a source header is uninformative (max N < 0.35), the registry falls back to S = T. The assignment maximising ΣS is found with the Hungarian algorithm. A near-tie (two canonical columns within 0.06 for the same source, in both directions) triggers invariant testing: both assignments are applied to a sample, and the one with a higher pass rate on cross-field invariants (dropoff > pickup, speed ≤ 80 mph, totals equal the sum of components) is kept if it wins by >0.15. Otherwise the pair is flagged. Approved mappings are stored under a fingerprint of the file's layout.

**3.2 Sentinel.** For each monitored numeric column, a batch is compared with a rolling reference (last 14 accepted batches) using the KS statistic and a mean-log location shift. Thresholds are calibrated per column as mean + 5σ of leave-one-out statistics over a trusted warm-up window, with floors. Drifted columns are tested against hypotheses:
- *unit change*: argmin over k ∈ {1.609, 0.621, 100, 0.01, 1000, 0.001, 60, 1/60} of KS(x/k, ref), accepted if it falls below 1.5× the threshold and halves the raw KS;
- *column swap*: KS(xₐ, ref_b) and KS(x_b, refₐ) both below threshold;
- *clock shift*: the circular lag L that minimises the L1 distance between hour-of-day histograms, accepted if it halves the unshifted distance.

Further checks cover replays (row-hash overlap > 90% → reject; > 0.5% → drop rows), null bursts, volume, vocabulary drift, missing columns, invariant pass rates and late arrivals. Unattributed drift above 3× the threshold holds the batch.

**3.3 Repair and verification.** Attributed faults are inverted (divide by k, shift by −L, swap back, drop duplicate rows), derived columns are recomputed, and the batch is re-inspected. Any residual attributed drift, or a post-repair replay, holds the batch. Row contracts then quarantine invalid rows with reasons, and every step writes lineage (row counts, content digests, timings, decisions).

## 4 Experimental setup

A generator produces NYC-like trips with known truth: 60 zones with Zipf popularity and zone-specific daily rhythms, weekday and weekend profiles, distance, speed, fare and tip models, airport trips, and surcharges. It can render each day in six historical header eras. Faults are injected into raw batches with labels: header synonyms, meaningless headers, km distances, cents, whole-batch replay, 20% row replay, 35% null burst, +5 h clock shift, `$1,234.50` text numbers, tip/tolls swap, an internally consistent 8% fare increase, and a dropped column.

- **E1:** 6 eras × {named, meaningless} headers × 4 evidence ablations × 5 seeds.
- **E2:** 14 warm-up days, then each fault injected twice per seed on non-adjacent days, with 32 clean days per seed; 5 seeds.
- **E3:** 91 days at fault rates {0, 10, 25, 40}%, 3 seeds. A HistGradientBoosting (Poisson) model forecasts hourly pickups per zone from lags 24/48/168 h, a 24 h rolling mean, lagged mean distance and fare, hour, weekday and zone. Each pipeline's model trains on its own warehouse, and errors are scored against ground truth over the last 21 days. Held partitions are marked unknown (NaN); the naive pipeline silently loses failed batches.

## 5 Results

**RQ1 (Table 1).** With real names, every configuration that uses names reaches 100%. Under meaningless headers, names contribute nothing, and statistics alone recover 97% of columns. The 3% missed are entirely `improvement_surcharge`, a constant whose value differs between eras (0.30 vs 1.00). These are reported as unmapped, never mis-mapped. Invariants did not change accuracy but cut flagged columns from 136 to 80 (−41%), mostly by ordering pickup and dropoff timestamps. **Silent mis-mappings: 0 in all 240 configurations.**

| headers | evidence | correct | flagged | silent wrong |
|---|---|---|---|---|
| named | name only | 100% | 0 | 0 |
| named | name + stats + invariants | 100% | 0 | 0 |
| meaningless | name only | 0% | 0 | 0 |
| meaningless | stats only | 97% | 136 | 0 |
| meaningless | name + stats + invariants | 97% | 80 | 0 |

**RQ2 (Table 2).** All 10 detectable fault types were detected 10/10 times with the correct cause. Unit changes, clock shifts, swaps and row replays were repaired automatically, and the repaired data matched ground truth to within a KS distance of 0.006. Whole-batch replays were rejected. The consistent 8% fare increase was always flagged as *unexplained* drift and never auto-repaired, the intended behaviour, since it can't be told apart from a real tariff change. False alarms: 2 of 160 clean batches (1.2%). The sentinel compares random samples of each batch; across different sampling seeds this ranged from 1 to 4 of 160 (0.6–2.5%), so the alarm rate is sensitive to sample size, a parameter worth studying. Under meaningless headers, mappings were 100% correct, but 70% of those batches were conservatively held for review over a residual near-tie.

**RQ3 (Table 3).** Day-ahead WAPE, mean over 3 seeds:

| fault rate | naive | DriftGuard | oracle |
|---|---|---|---|
| 0% | 37.4% | 37.4% | 37.4% |
| 10% | 39.0% | 38.1% | 37.4% |
| 25% | 41.8% | 38.8% | 37.4% |
| 40% | 44.9% | 37.8% | 37.4% |

The naive pipeline degrades steadily with the fault rate (+7.5 pp at 40%). DriftGuard stays within 1.4 pp of the oracle and prevents 94% of the damage at 40%. The non-monotonic DriftGuard curve reflects seed variance (3 seeds) and the varying number of held days.

## 5b Real data (NYC TLC 2023, historical layouts 2009–2021)

*To be filled from `scripts/run_real_study.py` output.*
- **A:** for each historical layout, the observed headers, name-based mapping, and agreement of value-only (scrambled-header) mapping with it; checked against the official data dictionaries.
- **B:** contract-violation rates on real rows; every real day the sentinel flagged, each with a human verdict (real event vs data problem).
- **C:** detection, correct-cause rate and clean-day alarm rate with faults injected into real batches.
- **D:** forecast WAPE for the naive pipeline, DriftGuard, and a clean run, over 265 zones.

## 6 Discussion and limitations

Attribution turns monitoring into action: a named cause can be inverted and verified, while an anonymous alarm can only page someone. The main costs are conservatism (holds on ambiguous layouts) and a dependence on a trustworthy warm-up window. The rolling reference can absorb slow decay. Schedule-dependent columns get loose thresholds because weekdays and weekends are pooled. Section 5 is synthetic; Section 5b reports the same measurements on real TLC data.

## 7 Use of AI tools

Generative AI tools were used substantially to write code and documentation for this project. The author reviewed the generated material, ran and validated all experiments, and takes responsibility for the methods, results and conclusions.

## 8 Reproducibility

`pip install -e ".[dev]" && python -m driftguard bench && python -m driftguard report` regenerates every number and figure above. Seeds are fixed, and the code, configuration and tests are in the repository.
