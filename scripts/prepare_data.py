#!/usr/bin/env python3
from scripts.run_analysis import load
rows,errors,scanned=load()
print(f"DRAWS={len(rows)} MONTHS={scanned} WARNINGS={len(errors)}")
if len(rows)<1000: raise SystemExit(f"Not enough data: {len(rows)}")
