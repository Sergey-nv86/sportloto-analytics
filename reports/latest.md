# Sportloto 6/45 — full-history analysis

Draws: **18849** | range: **1 → 18930**
Holdout: **500 draws** | independent model batches: **4**

## Walk-forward / holdout

| Strategy | Development | Holdout | Δ vs 0.8 |
|---|---:|---:|---:|
| LEARNED | 0.797 | 0.848 | +0.048 |
| RANDOM | 0.791 | 0.846 | +0.046 |
| MOMENTUM | 0.801 | 0.830 | +0.030 |
| ENSEMBLE | 0.826 | 0.826 | +0.026 |
| GAP | 0.799 | 0.822 | +0.022 |
| HOT | 0.809 | 0.816 | +0.016 |
| PAIRS | 0.813 | 0.808 | +0.008 |
| SELFLAG | 0.812 | 0.804 | +0.004 |
| RECENT30 | 0.802 | 0.784 | -0.016 |
| COLD | 0.800 | 0.780 | -0.020 |
| CROSSLAG | 0.811 | 0.774 | -0.026 |

## Multiple-model test

- Random expectation: **0.800 hits** per ticket.
- Monte Carlo: **1,200 simulations** against the maximum across 11 strategies.
- Max-over-strategies p-value: **0.5895**.

## Interpretation

The four calculation batches are independent at the workflow level and are combined only after chronological out-of-sample predictions are produced. Ensemble is constructed from the batch predictions rather than re-running all models in one long process.

This is statistical research, not a guarantee of future lottery outcomes.
