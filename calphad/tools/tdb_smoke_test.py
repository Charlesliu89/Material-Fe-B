#!/usr/bin/env python3
"""Smoke test a TDB for basic pycalphad equilibrium."""
from __future__ import annotations

import argparse
import sys
from typing import Iterable


def _print_error(message: str, detail: str, hints: Iterable[str]) -> None:
    print(f"ERROR: {message}")
    print(f"DETAIL: {detail}")
    hint_text = ", ".join(hints) if hints else "none"
    print(f"missing_hints: {hint_text}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a CALPHAD TDB with pycalphad.")
    parser.add_argument("tdb_path", help="Path to the TDB file")
    args = parser.parse_args()

    try:
        from pycalphad import Database, equilibrium
        from pycalphad import variables as v
    except Exception as exc:  # pragma: no cover - environment dependency
        _print_error(
            "pycalphad import failed",
            str(exc),
            ["install pycalphad (see requirements-calphad.txt)"],
        )
        return 2

    try:
        db = Database(args.tdb_path)
    except Exception as exc:
        _print_error(
            "failed to load TDB",
            str(exc),
            ["TDB parse failed; check FUNCTION/PHASE/CONSTITUENT/PARAMETER lines"],
        )
        return 3

    elements = sorted(str(el) for el in db.elements)
    if "FE" not in elements or "B" not in elements:
        _print_error(
            "required elements missing",
            f"elements present = {elements}",
            ["ensure FE and B elements are defined"],
        )
        return 4

    if "LIQUID" not in db.phases:
        _print_error(
            "missing LIQUID phase",
            f"phases present = {sorted(db.phases.keys())}",
            ["define PHASE LIQUID and CONSTITUENT line"],
        )
        return 5

    comps = ["FE", "B"]
    if "VA" in elements:
        comps.append("VA")

    try:
        conditions = {v.P: 101325, v.T: 1500, v.X("B"): 0.1}
        equilibrium(db, comps, ["LIQUID"], conditions)
    except Exception as exc:
        _print_error(
            "equilibrium calculation failed",
            str(exc),
            ["check LIQUID endmember PARAMETER definitions"],
        )
        return 6

    print("OK: smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
