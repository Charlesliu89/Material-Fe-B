#!/usr/bin/env python3
"""
Plot GM curves for Fe-Cr binary (BCC vs FCC) at a given temperature.

Usage:
    python calphad/run_fe_cr_plot.py [--tdb PATH] [--temp K] [--grid N] [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from pycalphad import calculate  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("pycalphad is required; install via requirements-calphad.txt") from exc

from calphad.calphad_core import load_database  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Fe-Cr GM for BCC/FCC at a given temperature.")
    parser.add_argument(
        "--tdb",
        type=Path,
        default=Path("calphad/thermo/Database/crfeni_mie.tdb"),
        help="Path to TDB (default: calphad/thermo/Database/crfeni_mie.tdb).",
    )
    parser.add_argument("--temp", type=float, default=1200.0, help="Temperature in K (default: 1200 K).")
    parser.add_argument("--grid", type=int, default=101, help="Grid points across composition (default: 101).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calphad/plots/fe_cr_gm.png"),
        help="Output PNG path (default: calphad/plots/fe_cr_gm.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db = load_database(args.tdb)
    comps = ["CR", "FE", "NI", "VA"]  # include NI (fixed to 0) and VA to activate phases

    x_cr = np.linspace(0.0, 1.0, args.grid)
    points = np.zeros((args.grid, 3))  # CR, FE, NI (VA inferred)
    points[:, 0] = x_cr
    points[:, 1] = 1.0 - x_cr
    points[:, 2] = 0.0  # fix NI = 0

    calc_bcc = calculate(db, comps, "BCC_A2", T=args.temp, P=101325, points=points, output="GM")
    calc_fcc = calculate(db, comps, "FCC_A1", T=args.temp, P=101325, points=points, output="GM")

    gm_bcc = np.array(calc_bcc.GM.squeeze())
    gm_fcc = np.array(calc_fcc.GM.squeeze())
    x_coord = x_cr

    diff = gm_bcc - gm_fcc
    # locate sign change for T0 (if any)
    x_t0 = None
    if np.any(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0):
        idx = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0][0]
        x0, x1 = x_coord[idx], x_coord[idx + 1]
        d0, d1 = diff[idx], diff[idx + 1]
        x_t0 = x0 + (0 - d0) * (x1 - x0) / (d1 - d0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_coord, gm_bcc, label="BCC_A2 GM")
    ax.plot(x_coord, gm_fcc, label="FCC_A1 GM")
    ax.set_xlabel("X(CR)")
    ax.set_ylabel("GM (J/mol-atom)")
    ax.set_title(f"Fe-Cr GM at T={args.temp:.0f} K")
    if x_t0 is not None:
        x_t0_float = float(x_t0)
        ax.axvline(x_t0_float, color="red", linestyle="--", label=f"T0 approx at X_CR≈{x_t0_float:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")
    if x_t0 is not None:
        print(f"Approximate T0 crossing at X_CR ≈ {x_t0:.4f}")


if __name__ == "__main__":
    main()
