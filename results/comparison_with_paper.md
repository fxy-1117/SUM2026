# Comparison With Paper Tables

This file compares the latest cleaned run against the original paper tables.

Latest run:

- Parameter sweep / Best F1: `FIX_NUMBER = 150`
- Entailment-only accuracy: `FIX_NUMBER = 250`
- Seed: `1129`
- Final data: `data/*/generated/*.csv` and original test files

## Best F1 Table

Overall, the Best F1 table is very close to the paper table. Every F1 value is
within 0.02 absolute difference.

| Dataset | Class | Step type | Latest F1 | Paper F1 | Delta | Latest params | Paper params |
|---|---:|---|---:|---:|---:|---|---|
| ANLI | 0 | original | 0.67 | 0.67 | +0.00 | 0.8 / 80 | 0.8 / 80 |
| ANLI | 0 | 1-step | 0.72 | 0.72 | +0.00 | 0.6 / 80 | 0.6 / 80 |
| ANLI | 0 | 2-step | 0.74 | 0.73 | +0.01 | 0.5 / 80 | 0.5 / 80 |
| ANLI | 0 | 3-step | 0.75 | 0.75 | +0.00 | 0.6 / 80 | 0.6 / 80 |
| ANLI | 1 | original | 0.46 | 0.44 | +0.02 | 0.55 / 100 | 0.55 / 100 |
| ANLI | 1 | 1-step | 0.57 | 0.57 | +0.00 | 0.55 / 100 | 0.55 / 90 |
| ANLI | 1 | 2-step | 0.66 | 0.64 | +0.02 | 0.6 / 100 | 0.6 / 100 |
| ANLI | 1 | 3-step | 0.67 | 0.67 | +0.00 | 0.55 / 90 | 0.55 / 80 |
| ARCT | 0 | original | 0.67 | 0.67 | +0.00 | 0.5 / 90 | 0.65 / 80 |
| ARCT | 0 | 1-step | 0.72 | 0.71 | +0.01 | 0.7 / 90 | 0.7 / 80 |
| ARCT | 0 | 2-step | 0.70 | 0.71 | -0.01 | 0.75 / 80 | 0.75 / 80 |
| ARCT | 0 | 3-step | 0.72 | 0.74 | -0.02 | 0.7 / 90 | 0.7 / 90 |
| ARCT | 1 | original | 0.21 | 0.19 | +0.02 | 0.55 / 80 | 0.6 / 80 |
| ARCT | 1 | 1-step | 0.45 | 0.44 | +0.01 | 0.65 / 90 | 0.65 / 90 |
| ARCT | 1 | 2-step | 0.52 | 0.53 | -0.01 | 0.6 / 90 | 0.6 / 90 |
| ARCT | 1 | 3-step | 0.58 | 0.59 | -0.01 | 0.6 / 90 | 0.65 / 90 |

Trend notes:

- ANLI class 0 keeps the paper trend almost exactly: original < 1-step < 2-step < 3-step.
- ANLI class 1 also keeps the same trend: original < 1-step < 2-step < 3-step.
- ARCT class 1 is very close to the paper and keeps the strongest multi-step trend: original < 1-step < 2-step < 3-step.
- ARCT class 0 is the main small difference: generated steps still beat original, but 2-step and 3-step are slightly below the paper values and the current run is not strictly monotonic.
- Some best parameters changed even when the F1 value stayed the same or changed only slightly. This likely reflects near-ties between parameter settings.
- ARCT original has fewer available unique premise/claim keys after cleaning, so its best points can be based on fewer than 150 counted rows. The generated ARCT one/two/three best points reach 150 counted rows.

## Entailment Accuracy Table

All entailment rows reached `FIX_NUMBER = 250`. The latest values are close to
the paper values. Most rows are slightly lower than the paper, while ARCT
1-step is slightly higher after strict unique-key evaluation. The largest drop
is 0.038 and the mean absolute difference is about 0.018.

| Dataset | Step type | Latest acc. | Paper acc. | Delta | Valid items | Available items |
|---|---|---:|---:|---:|---:|---:|
| ANLI | none | 0.496 | 0.530 | -0.034 | 250 | 1000 |
| ANLI | original | 0.520 | 0.558 | -0.038 | 250 | 1000 |
| ANLI | 1-step | 0.624 | 0.645 | -0.021 | 250 | 1000 |
| ANLI | 2-step | 0.660 | 0.673 | -0.013 | 250 | 1000 |
| ANLI | 3-step | 0.716 | 0.733 | -0.017 | 250 | 1000 |
| ARCT | none | 0.276 | 0.293 | -0.017 | 250 | 289 |
| ARCT | original | 0.292 | 0.303 | -0.011 | 250 | 289 |
| ARCT | 1-step | 0.480 | 0.478 | +0.002 | 250 | 289 |
| ARCT | 2-step | 0.496 | 0.518 | -0.022 | 250 | 289 |
| ARCT | 3-step | 0.560 | 0.563 | -0.003 | 250 | 289 |

Trend notes:

- The entailment trend is preserved for both datasets:
  none < original < 1-step < 2-step < 3-step.
- ANLI still has higher entailment accuracy than ARCT at every step.
- The 3-step setting is still best for both datasets.
- The gain from none to 3-step is slightly larger than in the paper:
  ANLI improves by +0.220 in the latest run versus +0.203 in the paper;
  ARCT improves by +0.284 in the latest run versus +0.270 in the paper.
- Absolute values are lower than the paper for most rows, especially ANLI
  original (-0.038) and ANLI none (-0.034). ARCT 1-step is the exception after
  strict unique-key evaluation: it is slightly above the paper value (+0.002).
  The ordering and main conclusion are unchanged.
