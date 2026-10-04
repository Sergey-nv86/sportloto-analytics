#!/usr/bin/env python3
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from fetch_stoloto import main as fetch_stoloto
from run_analysis import load


def run():
    try:
        fetch_stoloto()
        print("SOURCE=STOLOTO_OFFICIAL_API")
        return
    except Exception as exc:
        print(f"WARNING=Stoloto official API failed: {exc}")
        print("SOURCE=LEGACY_ARCHIVE_FALLBACK")
        rows, errors, scanned = load()
        print(f"DRAWS={len(rows)} MONTHS={scanned} WARNINGS={len(errors)}")
        if len(rows) < 1000:
            raise SystemExit(f"Not enough data: {len(rows)}")


if __name__ == "__main__":
    run()
