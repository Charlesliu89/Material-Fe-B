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
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calphad.calphad_core import load_database  # noqa: E402

try:
    import pycalphad  # type: ignore
    from pycalphad import calculate  # type: ignore
    from pycalphad import variables as v  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "pycalphad is required. Install optional deps: pip install -r requirements-calphad.txt"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Fe-B T0 line (BCC vs FCC).")
    parser.add_argument("--tdb", type=Path, help="Path to TDB (default: calphad/thermo/Database/COST507.tdb).")
    parser.add_argument("--tmin", type=float, default=500.0, help="Minimum temperature (K).")
    parser.add_argument("--tmax", type=float, default=1800.0, help="Maximum temperature (K).")
    parser.add_argument("--tstep", type=float, default=50.0, help="Temperature step (K).")
    parser.add_argument("--grid", type=int, default=51, help="Number of composition points along X_B (default 51).")
    return parser.parse_args()


def _find_crossing(xs: np.ndarray, diff: np.ndarray) -> Optional[Tuple[float, float]]:
    """Find x where diff changes sign; return (x_cross, diff_interp)."""
    sign = np.sign(diff)
    zero = np.where(sign == 0)[0]
    if zero.size > 0:
        idx = zero[0]
        return float(xs[idx]), float(0.0)
    changes = np.where(np.diff(sign) != 0)[0]
    if changes.size == 0:
        return None
    idx = changes[0]
    x0, x1 = xs[idx], xs[idx + 1]
    d0, d1 = diff[idx], diff[idx + 1]
    if d1 == d0:
        return float(x0), float(d0)
    x_cross = x0 + (0 - d0) * (x1 - x0) / (d1 - d0)
    return float(x_cross), float(0.0)


def main() -> None:
    args = parse_args()
    db = load_database(args.tdb) if args.tdb else load_database()

    comps = ["FE", "B", "VA"]
    phases = ["BCC_A2", "FCC_A1"]
    temps = np.arange(args.tmin, args.tmax + 1e-9, args.tstep)
    x_b = np.linspace(0, 1, args.grid)

    rows: List[dict] = []
    for t in temps:
        gm_values = {}
        for ph in phases:
            try:
                res = calculate(db, comps, ph, T=t, P=101325, points=x_b, model=None, output="GM")
            except Exception:
                gm_values[ph] = None
                continue
            gm = np.array(res.GM.squeeze())
            gm_values[ph] = gm
        if any(val is None for val in gm_values.values()):
            continue
        diff = gm_values["BCC_A2"] - gm_values["FCC_A1"]
        crossing = _find_crossing(x_b, diff)
        if crossing is None:
            continue
        x_cross, _ = crossing
        rows.append({"T_K": float(t), "X_B": x_cross})

    if not rows:
        print("No T0 crossings found in the specified range.")
        return

    df = pd.DataFrame(rows).sort_values("T_K")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
