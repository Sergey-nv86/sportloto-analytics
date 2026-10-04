#!/usr/bin/env python3
"""Fetch recent Sportloto 5/36 draws from the official Stoloto API.

The collector is incremental: it keeps the existing full archive and refreshes
the most recent days from Stoloto. If the official API is unavailable, the
caller may fall back to the historical collector.
"""
import csv
import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "results_5x36.csv"

API = "https://www.stoloto.ru/p/api/mobile/api/v35/service/draws/archive"
UA = "Mozilla/5.0 SportlotoResearch/3.0"
PARTNER = os.environ.get("STOLOTO_PARTNER", "bXMjXFRXZ3coWXh6R3s1NTdUX3dnWlBMLUxmdg")


def request_json(date_from: str, date_to: str):
    params = urllib.parse.urlencode({
        "count": "100",
        "game": "5x36plus",
        "date_from": date_from,
        "date_to": date_to,
    })
    headers = {
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ru-RU,ru;q=0.8,en-US;q=0.5",
        "Content-Type": "application/x-www-form-urlencoded",
        "Device-Type": "MOBILE",
        "Gosloto-Partner": PARTNER,
        "Referer": "https://www.stoloto.ru/5x36plus/archive",
    }
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"{API}?{params}", headers=headers)
            with urllib.request.urlopen(
                req, timeout=40, context=ssl.create_default_context()
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            if attempt < 2:
                time.sleep(2 + attempt)
    raise last


def normalize_numbers(value):
    if isinstance(value, dict):
        for key in ("numbers", "values", "balls", "combination", "winningCombination"):
            if key in value:
                return normalize_numbers(value[key])
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, dict):
                for key in ("number", "value", "ball"):
                    if key in item:
                        item = item[key]
                        break
            try:
                n = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= 36:
                out.append(n)
        return sorted(set(out))
    if isinstance(value, str):
        import re
        return [int(x) for x in re.findall(r"(?<!\\d)(?:[1-9]|[1-2]\\d|3[0-6])(?!\\d)", value)]
    return []


def parse_draws(payload):
    draws = payload.get("draws", payload) if isinstance(payload, dict) else payload
    if not isinstance(draws, list):
        return []
    out = []
    for d in draws:
        if not isinstance(d, dict):
            continue
        number = d.get("number", d.get("drawNumber", d.get("draw")))
        date = d.get("date", d.get("datetime", d.get("drawDate", "")))
        nums = normalize_numbers(d.get("winningCombination", d.get("result", d.get("numbers"))))
        try:
            number = int(number)
        except (TypeError, ValueError):
            continue
        if len(nums) < 6 or len(set(nums[:5])) != 5 or not 1 <= nums[5] <= 4:
            continue
        out.append((number, str(date), tuple(sorted(nums[:5])) + (int(nums[5]),)))
    return out


def read_existing():
    rows = {}
    if not DATA.exists():
        return rows
    with DATA.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                ns = tuple(sorted(int(r[f"n{i}"]) for i in range(1, 7)))
                rows[int(r["draw"])] = (int(r["draw"]), r["datetime"], ns)
            except (KeyError, ValueError):
                continue
    return rows


def write_rows(rows):
    DATA.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda x: (x[1], x[0]))
    with DATA.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["draw", "datetime", "n1", "n2", "n3", "n4", "n5", "n6"])
        for draw, dt, ns in ordered:
            w.writerow([draw, dt, *ns])
    return ordered


def main():
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=7)
    payload = request_json(start.isoformat(), (today + timedelta(days=1)).isoformat())
    fresh = parse_draws(payload)
    if not fresh:
        raise RuntimeError("Stoloto API returned no valid 5/36 draws")

    rows = read_existing()
    conflicts = []
    added = 0
    for row in fresh:
        old = rows.get(row[0])
        if old and old[2] != row[2]:
            conflicts.append((row[0], old[2], row[2]))
        else:
            if old is None:
                added += 1
            rows[row[0]] = row

    if conflicts:
        raise RuntimeError(f"Official Stoloto conflict for draw IDs: {[x[0] for x in conflicts]}")

    ordered = write_rows(rows)
    print(f"STOLOTO_API_OK draws={len(fresh)} added={added} total={len(ordered)}")
    print(f"STOLOTO_RANGE={fresh[0][0]}..{fresh[-1][0]}")


if __name__ == "__main__":
    main()
