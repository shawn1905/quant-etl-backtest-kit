#!/usr/bin/env python3
"""Standalone core EOD ETL runner (sanitized open-source orchestrator).

Runs the core daily data collection & validation sequence:
  1) update_contracts_openctp.py <date>
  2) derive_main_1d.py --date <date>
  3) update_1dopt_openctp.py <date>
  4) update_basic_openctp.py
  5) check_integrity.py --date <date> --core-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run_step(label: str, cmd: list[str]) -> None:
    print(f">>> {label}: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"FAILED ({res.returncode}): {label}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run sanitized core EOD ETL pipeline")
    ap.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"), help="Target trading date YYYYMMDD")
    ap.add_argument("--include-seat", action="store_true", help="Also run collect_seat.py")
    args = ap.parse_args()

    day = args.date
    run_step("1. futures all contracts 1d", [PY, os.path.join(HERE, "update_contracts_openctp.py"), day])
    run_step("2. derive continuous dominant 1d", [PY, os.path.join(HERE, "derive_main_1d.py"), "--date", day])
    run_step("3. options 1d", [PY, os.path.join(HERE, "update_1dopt_openctp.py"), day])
    run_step("4. contract specs basic tables", [PY, os.path.join(HERE, "update_basic_openctp.py")])
    if args.include_seat:
        run_step("5. exchange seat rankings", [PY, os.path.join(HERE, "collect_seat.py"), "--date", day])
    run_step("6. core integrity verification", [PY, os.path.join(HERE, "check_integrity.py"), "--date", day, "--core-only"])
    print(f"OK: core EOD ETL finished cleanly for {day}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
