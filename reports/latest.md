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

## Multiple-model test

- Random expectation: **0.800 hits** per ticket.
- Monte Carlo: **1,200 simulations** against the maximum across 12 strategies.
- Max-over-strategies p-value: **0.0674**.

## Current next recommendation

**28 · 25 · 36 · 2 · 43 · 41**

Dynamic champion ensemble; models are reweighted after each newly observed draw.

This is statistical research, not a guarantee of future lottery outcomes.
