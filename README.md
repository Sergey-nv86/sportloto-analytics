# Sportloto 6/45 Statistical Research

Automated chronological analysis of Sportloto 6/45 results.

The pipeline:
- downloads monthly archive pages;
- validates and deduplicates draws;
- runs walk-forward predictive strategies without future-data leakage;
- reserves the latest 500 draws as a final holdout;
- compares fixed strategies with the random 6/45 baseline;
- runs a Monte Carlo max-over-strategies null test;
- writes a reproducible Markdown report.

This project is statistical research, not a guarantee of future lottery outcomes.

## Latest result

See [reports/latest.md](reports/latest.md).

## Run locally

```bash
python3 scripts/run_analysis.py
```
