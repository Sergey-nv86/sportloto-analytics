#!/usr/bin/env python3
import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nfrom run_analysis import load
rows,errors,scanned=load()
print(f"DRAWS={len(rows)} MONTHS={scanned} WARNINGS={len(errors)}")
if len(rows)<1000: raise SystemExit(f"Not enough data: {len(rows)}")
