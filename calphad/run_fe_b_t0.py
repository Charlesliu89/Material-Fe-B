#!/usr/bin/env python3
"""
Compute a simple T0 line for the Fe-B system (metastable equality of two phases).

Logic:
- Load default TDB (calphad/thermo/Database/COST507.tdb) unless overridden.
- Compute Gibbs energy (GM) for BCC_A2 and FCC_A1 along X_B grid for each temperature.
- Detect sign changes of ΔG = GM_BCC - GM_FCC to locate crossing (T0) by linear interpolation.
- Output a small table of T (K) vs. X_B (fraction) where ΔG crosses 0.

Usage:
    python calphad/run_fe_b_t0.py [--tdb PATH] [--tmin 500] [--tmax 1800] [--tstep 50] [--grid 51]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calphad.tasks import compute_fe_b_t0  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Fe-B T0 line (BCC vs FCC).")
    parser.add_argument("--tdb", type=Path, help="Path to TDB (default: calphad/thermo/Database/COST507.tdb).")
    parser.add_argument("--tmin", type=float, default=500.0, help="Minimum temperature (K).")
    parser.add_argument("--tmax", type=float, default=1800.0, help="Maximum temperature (K).")
    parser.add_argument("--tstep", type=float, default=50.0, help="Temperature step (K).")
    parser.add_argument("--grid", type=int, default=51, help="Number of composition points along X_B (default 51).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = compute_fe_b_t0(
        tdb_path=str(args.tdb) if args.tdb else None,
        tmin=args.tmin,
        tmax=args.tmax,
        tstep=args.tstep,
        grid=args.grid,
    )

    if df.empty:
        print("No T0 crossings found in the specified range.")
        return
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
