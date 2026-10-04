# Sportloto 6/45 — full-history analysis

Draws: **18882** | range: **1 → 18963**
Holdout: **2500 draws** | independent model batches: **4**

## Walk-forward / holdout

| Strategy | Development | Holdout | Δ vs 0.8 |
|---|---:|---:|---:|
| SELFLAG | 0.807 | 0.840 | +0.040 |
| HOT | 0.808 | 0.816 | +0.016 |
| PAIRS | 0.812 | 0.815 | +0.015 |
| CROSSLAG | 0.810 | 0.809 | +0.009 |
| COLD | 0.800 | 0.797 | -0.003 |
| MOMENTUM | 0.803 | 0.796 | -0.004 |
| RANDOM | 0.793 | 0.794 | -0.006 |
| LEARNED_EWMA | 0.797 | 0.791 | -0.009 |
| ENSEMBLE | 0.790 | 0.790 | -0.010 |
| RECENT30 | 0.804 | 0.786 | -0.014 |
| GAP | 0.802 | 0.782 | -0.018 |
| LEARNED | 0.802 | 0.779 | -0.021 |

## Rolling holdout blocks

| Block | RANDOM | HOT | COLD | RECENT30 | MOMENTUM | GAP | SELFLAG | CROSSLAG | PAIRS | LEARNED | LEARNED_EWMA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-500 | 0.790 | 0.872 | 0.838 | 0.786 | 0.758 | 0.750 | 0.846 | 0.822 | 0.864 | 0.790 | 0.826 |
| 501-1000 | 0.774 | 0.768 | 0.838 | 0.778 | 0.812 | 0.778 | 0.800 | 0.820 | 0.782 | 0.746 | 0.824 |
| 1001-1500 | 0.772 | 0.790 | 0.756 | 0.780 | 0.780 | 0.740 | 0.882 | 0.790 | 0.844 | 0.768 | 0.778 |
| 1501-2000 | 0.770 | 0.834 | 0.784 | 0.784 | 0.788 | 0.824 | 0.856 | 0.830 | 0.774 | 0.742 | 0.778 |
| 2001-2500 | 0.866 | 0.814 | 0.768 | 0.800 | 0.840 | 0.816 | 0.814 | 0.784 | 0.812 | 0.850 | 0.748 |

## Multiple-model test

- Random expectation: **0.800 hits** per ticket.
- Monte Carlo: **1,200 simulations** against the maximum across 12 strategies.
- Max-over-strategies p-value: **0.0674**.

## Interpretation

The four calculation batches are independent at the workflow level and are combined only after chronological out-of-sample predictions are produced. Ensemble is constructed from the batch predictions rather than re-running all models in one long process.

This is statistical research, not a guarantee of future lottery outcomes.
