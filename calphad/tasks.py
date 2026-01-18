from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from calphad.calphad_core import DEFAULT_TDB_PATH, load_database, simple_equilibrium

try:
    from pycalphad import calculate  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "pycalphad is required. Install deps: pip install -r requirements.txt"
    ) from exc


@dataclass(frozen=True)
class FeCrGmResult:
    x_cr: np.ndarray
    gm_bcc: np.ndarray
    gm_fcc: np.ndarray
    x_t0: Optional[float]


def _find_first_crossing(xs: np.ndarray, diff: np.ndarray) -> Optional[float]:
    sign = np.sign(diff)
    zero = np.where(sign == 0)[0]
    if zero.size > 0:
        return float(xs[zero[0]])
    changes = np.where(np.diff(sign) != 0)[0]
    if changes.size == 0:
        return None
    idx = changes[0]
    x0, x1 = xs[idx], xs[idx + 1]
    d0, d1 = diff[idx], diff[idx + 1]
    if d1 == d0:
        return float(x0)
    return float(x0 + (0 - d0) * (x1 - x0) / (d1 - d0))


def compute_fe_b_t0(
    *,
    tdb_path: str | Path | None = None,
    tmin: float = 500.0,
    tmax: float = 1800.0,
    tstep: float = 50.0,
    grid: int = 51,
) -> pd.DataFrame:
    db = load_database(tdb_path or DEFAULT_TDB_PATH)
    comps = ["FE", "B", "VA"]
    phases = ["BCC_A2", "FCC_A1"]
    temps = np.arange(tmin, tmax + 1e-9, tstep)
    x_b = np.linspace(0, 1, grid)

    rows: list[dict[str, float]] = []
    for t in temps:
        gm_values: dict[str, Optional[np.ndarray]] = {}
        for ph in phases:
            try:
                res = calculate(db, comps, ph, T=t, P=101325, points=x_b, model=None, output="GM")
            except Exception:
                gm_values[ph] = None
                continue
            gm_values[ph] = np.array(res.GM.squeeze())
        if any(val is None for val in gm_values.values()):
            continue
        diff = gm_values["BCC_A2"] - gm_values["FCC_A1"]
        crossing = _find_first_crossing(x_b, diff)
        if crossing is None:
            continue
        rows.append({"T_K": float(t), "X_B": crossing})

    if not rows:
        return pd.DataFrame(columns=["T_K", "X_B"])
    return pd.DataFrame(rows).sort_values("T_K")


def compute_fe_cr_gm(
    *,
    tdb_path: str | Path | None = None,
    temperature: float = 1200.0,
    grid: int = 101,
) -> FeCrGmResult:
    db = load_database(tdb_path or DEFAULT_TDB_PATH)
    comps = ["CR", "FE", "NI", "VA"]

    x_cr = np.linspace(0.0, 1.0, grid)
    points = np.zeros((grid, 3))
    points[:, 0] = x_cr
    points[:, 1] = 1.0 - x_cr
    points[:, 2] = 0.0

    calc_bcc = calculate(db, comps, "BCC_A2", T=temperature, P=101325, points=points, output="GM")
    calc_fcc = calculate(db, comps, "FCC_A1", T=temperature, P=101325, points=points, output="GM")

    gm_bcc = np.array(calc_bcc.GM.squeeze())
    gm_fcc = np.array(calc_fcc.GM.squeeze())
    x_t0 = _find_first_crossing(x_cr, gm_bcc - gm_fcc)

    return FeCrGmResult(x_cr=x_cr, gm_bcc=gm_bcc, gm_fcc=gm_fcc, x_t0=x_t0)


def run_sample_equilibrium(
    *,
    tdb_path: str | Path | None = None,
    temperature: float = 1200.0,
    grid: int = 21,
) -> xr.Dataset:
    db = load_database(tdb_path or DEFAULT_TDB_PATH)
    components = ["FE", "CR", "VA"]
    return simple_equilibrium(db, components, phases=None, temperature=temperature, grid_points=grid)
