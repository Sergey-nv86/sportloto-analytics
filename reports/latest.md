# Sportloto 6/45 — full-history analysis

Draws: **18849** | range: **1 → 18930**
Holdout: **2500 draws** | independent model batches: **4**

## Walk-forward / holdout

| Strategy | Development | Holdout | Δ vs 0.8 |
|---|---:|---:|---:|
| SELFLAG | 0.807 | 0.840 | +0.040 |
| PAIRS | 0.811 | 0.820 | +0.020 |
| HOT | 0.808 | 0.818 | +0.018 |
| CROSSLAG | 0.810 | 0.810 | +0.010 |
| COLD | 0.800 | 0.794 | -0.006 |
| MOMENTUM | 0.803 | 0.792 | -0.008 |
| LEARNED_EWMA | 0.798 | 0.790 | -0.010 |
| RANDOM | 0.793 | 0.789 | -0.011 |
| ENSEMBLE | 0.788 | 0.788 | -0.012 |
| GAP | 0.802 | 0.782 | -0.018 |
| RECENT30 | 0.805 | 0.780 | -0.020 |
| LEARNED | 0.802 | 0.778 | -0.022 |

## Rolling holdout blocks

| Block | RANDOM | HOT | COLD | RECENT30 | MOMENTUM | GAP | SELFLAG | CROSSLAG | PAIRS | LEARNED | LEARNED_EWMA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-500 | 0.772 | 0.878 | 0.830 | 0.768 | 0.746 | 0.762 | 0.862 | 0.822 | 0.866 | 0.782 | 0.820 |
| 501-1000 | 0.792 | 0.766 | 0.846 | 0.782 | 0.824 | 0.778 | 0.802 | 0.820 | 0.794 | 0.754 | 0.848 |
| 1001-1500 | 0.760 | 0.798 | 0.744 | 0.794 | 0.792 | 0.736 | 0.872 | 0.790 | 0.826 | 0.762 | 0.756 |
| 1501-2000 | 0.776 | 0.830 | 0.770 | 0.772 | 0.770 | 0.814 | 0.860 | 0.844 | 0.804 | 0.752 | 0.770 |
| 2001-2500 | 0.846 | 0.816 | 0.780 | 0.784 | 0.830 | 0.822 | 0.804 | 0.774 | 0.808 | 0.838 | 0.756 |

## Multiple-model test

- Random expectation: **0.800 hits** per ticket.
- Monte Carlo: **1,200 simulations** against the maximum across 12 strategies.
- Max-over-strategies p-value: **0.0508**.

## Interpretation

The four calculation batches are independent at the workflow level and are combined only after chronological out-of-sample predictions are produced. Ensemble is constructed from the batch predictions rather than re-running all models in one long process.

This is statistical research, not a guarantee of future lottery outcomes.
