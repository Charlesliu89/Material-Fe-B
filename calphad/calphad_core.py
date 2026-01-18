"""
CALPHAD helpers (optional).

Requirements:
- Install deps: pip install -r requirements.txt
- Provide a TDB file under calphad/thermo/Database/ (not tracked).

This is a scaffold; fill in with actual calculations (equilibrium, property grids, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, cast

import xarray as xr

try:
    import pycalphad  # type: ignore
    from pycalphad import Database, equilibrium, variables as v  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "pycalphad is not installed. Install requirements: "
        "pip install -r requirements.txt"
    ) from exc


DEFAULT_TDB_PATH = Path(__file__).resolve().parent / "thermo" / "Database" / "COST507.tdb"


def load_database(path: Path | str = DEFAULT_TDB_PATH) -> Database:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TDB file not found: {path}")
    return Database(path)


def simple_equilibrium(
    db: Database,
    components: Sequence[str],
    phases: Sequence[str] | None,
    temperature: float,
    pressure: float = 101325.0,
    grid_points: int = 21,
) -> xr.Dataset:
    """
    Compute a basic isothermal equilibrium over composition space for given components.

    - components: e.g., ["FE", "B"] or ["FE", "B", "CR"]
    - phases: list of phase names (None to use all from DB)
    - temperature: Kelvin
    - pressure: Pa
    - grid_points: composition sampling density per dimension
    """
    conds: dict[Any, Any] = {
        v.T: temperature,
        v.P: pressure,
    }
    # Build composition grid (simple equal spacing for A..N-1, last is remainder).
    # For ternary/binary this is adequate; refine as needed for higher-order systems.
    comp_syms = [v.Species(comp) for comp in components]
    species_syms = [sym for sym in comp_syms if str(sym) != "VA"]
    if len(species_syms) < 2:
        raise ValueError("At least two components are required for equilibrium.")
    # Assign grid to (n-1) real species; remaining species (incl. VA) is the remainder.
    space = [i / (grid_points - 1) for i in range(grid_points)]
    for sym in species_syms[:-1]:
        conds[v.X(sym)] = space

    phase_list = list(phases) if phases is not None else sorted(getattr(db, "phases", {}).keys())
    result = equilibrium(db, components, phase_list, conds, verbose=False)
    return cast(xr.Dataset, result)
