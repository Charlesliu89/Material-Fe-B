#!/usr/bin/env python3
"""
Run a simple CALPHAD equilibrium example using the default TDB.

Usage:
    python calphad/run_sample_equilibrium.py [--tdb PATH] [--temp K] [--grid N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calphad.tasks import run_sample_equilibrium  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sample CALPHAD equilibrium.")
    parser.add_argument(
        "--tdb",
        type=Path,
        default=Path("calphad/thermo/Database/crfeni_mie.tdb"),
        help="Path to TDB (default: calphad/thermo/Database/crfeni_mie.tdb).",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1200.0,
        help="Temperature in K (default: 1200.0).",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=21,
        help="Grid points per dimension for composition sampling (default: 21).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_sample_equilibrium(
        tdb_path=str(args.tdb),
        temperature=args.temp,
        grid=args.grid,
    )

    components = ["FE", "CR", "VA"]
    phases = sorted({str(ph) for ph in result.Phase.values.ravel()})
    print(f"Components: {components}")
    print(f"Phases ({len(phases)}): {phases[:10]}{' ...' if len(phases) > 10 else ''}")
    print("Variables:", list(result.data_vars))
    print("Dimensions:", dict(result.sizes))


if __name__ == "__main__":
    main()
