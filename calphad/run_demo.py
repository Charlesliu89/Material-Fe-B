#!/usr/bin/env python3
"""
Unified CALPHAD demo runner.

Usage:
    python calphad/run_demo.py equilibrium [--tdb PATH] [--temp K] [--grid N]
    python calphad/run_demo.py fe-cr-gm [--tdb PATH] [--temp K] [--grid N] [--output PATH]
    python calphad/run_demo.py fe-b-t0 [--tdb PATH] [--tmin K] [--tmax K] [--tstep K] [--grid N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calphad.tasks import compute_fe_b_t0, compute_fe_cr_gm, run_sample_equilibrium  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CALPHAD demo runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eq = subparsers.add_parser("equilibrium", help="Run a sample equilibrium calculation.")
    eq.add_argument(
        "--tdb",
        type=Path,
        default=Path("calphad/thermo/Database/crfeni_mie.tdb"),
        help="Path to TDB (default: calphad/thermo/Database/crfeni_mie.tdb).",
    )
    eq.add_argument("--temp", type=float, default=1200.0, help="Temperature in K (default: 1200.0).")
    eq.add_argument("--grid", type=int, default=21, help="Grid points per dimension (default: 21).")

    fe_cr = subparsers.add_parser("fe-cr-gm", help="Plot Fe-Cr GM curves for BCC/FCC.")
    fe_cr.add_argument(
        "--tdb",
        type=Path,
        default=Path("calphad/thermo/Database/crfeni_mie.tdb"),
        help="Path to TDB (default: calphad/thermo/Database/crfeni_mie.tdb).",
    )
    fe_cr.add_argument("--temp", type=float, default=1200.0, help="Temperature in K (default: 1200 K).")
    fe_cr.add_argument("--grid", type=int, default=101, help="Grid points across composition (default: 101).")
    fe_cr.add_argument(
        "--output",
        type=Path,
        default=Path("calphad/plots/fe_cr_gm.png"),
        help="Output PNG path (default: calphad/plots/fe_cr_gm.png).",
    )

    fe_b = subparsers.add_parser("fe-b-t0", help="Compute Fe-B T0 crossings (BCC vs FCC).")
    fe_b.add_argument("--tdb", type=Path, help="Path to TDB (default: calphad/thermo/Database/COST507.tdb).")
    fe_b.add_argument("--tmin", type=float, default=500.0, help="Minimum temperature (K).")
    fe_b.add_argument("--tmax", type=float, default=1800.0, help="Maximum temperature (K).")
    fe_b.add_argument("--tstep", type=float, default=50.0, help="Temperature step (K).")
    fe_b.add_argument("--grid", type=int, default=51, help="Number of composition points along X_B (default 51).")

    return parser


def run_equilibrium(args: argparse.Namespace) -> None:
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


def run_fe_cr_gm(args: argparse.Namespace) -> None:
    result = compute_fe_cr_gm(
        tdb_path=str(args.tdb),
        temperature=args.temp,
        grid=args.grid,
    )
    x_coord = result.x_cr
    gm_bcc = result.gm_bcc
    gm_fcc = result.gm_fcc
    x_t0 = result.x_t0

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


def run_fe_b_t0(args: argparse.Namespace) -> None:
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "equilibrium":
        run_equilibrium(args)
    elif args.command == "fe-cr-gm":
        run_fe_cr_gm(args)
    else:
        run_fe_b_t0(args)


if __name__ == "__main__":
    main()
