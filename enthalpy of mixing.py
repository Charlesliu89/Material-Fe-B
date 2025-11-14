#!/usr/bin/env python3
"""
Binary alloy enthalpy of mixing calculator driven entirely by the Excel
line-input workflow: type a single line containing any number of element
symbols and their atomic percentages, and the script will look up Ω₀–Ω₃ from
the Excel matrices for every pair and sum their contributions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

OMEGA_SHEETS = ("U0", "U1", "U2", "U3")
# Default database location
DEFAULT_DATABASE_PATH = Path("Data/Element pair data base matrices.xlsx")
# Pattern to parse "Element value" pairs (handles delimiters like :, =, or whitespace)
COMPOSITION_PATTERN = re.compile(
    r"([A-Za-z]{1,2})\s*(?:[:=])?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# Supported elements (symbol -> atomic number). Order is used to select A/B.
ATOMIC_NUMBERS: Dict[str, int] = {
    "H": 1,
    "Be": 4,
    "B": 5,
    "C": 6,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "In": 49,
    "Sn": 50,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Sm": 62,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Pb": 82,
    "Th": 90,
}


@dataclass
class BinaryMixture:
    """Component names, mole fractions, and interaction coefficients."""

    names: List[str]
    mole_fractions: List[float]  # [c_A, c_B]
    omegas: List[float]  # Ω_0 ... Ω_3

    def enthalpy_of_mixing(self) -> float:
        """Compute ΔH_mix (kJ/mol)."""
        c_a, c_b = self.mole_fractions
        delta = c_a - c_b
        polynomial = sum(omega * (delta ** k) for k, omega in enumerate(self.omegas))
        return 4.0 * polynomial * c_a * c_b

    def polynomial_terms(self) -> List[float]:
        """Return Ω_k · (c_A - c_B)^k for each polynomial order."""
        c_a, c_b = self.mole_fractions
        delta = c_a - c_b
        return [omega * (delta ** k) for k, omega in enumerate(self.omegas)]


def normalize_symbol(symbol: str) -> str:
    """Return a capitalized element symbol."""
    normalized = symbol.strip()
    if not normalized:
        raise ValueError("元素符号不能为空。")
    if len(normalized) == 1:
        return normalized.upper()
    return normalized[0].upper() + normalized[1:].lower()


def load_omega_tables(path: Path) -> Dict[str, pd.DataFrame]:
    """Load Ω matrices from the Excel workbook."""
    if not path.exists():
        raise FileNotFoundError(f"未找到 Excel 文件: {path}")

    xls = pd.ExcelFile(path)
    tables: Dict[str, pd.DataFrame] = {}
    for sheet in OMEGA_SHEETS:
        df = xls.parse(sheet)
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Element"})
        df["Element"] = df["Element"].astype(str).str.strip()
        df = df.set_index("Element")
        df.columns = [str(col).strip() for col in df.columns]
        tables[sheet] = df

    elements = tables[OMEGA_SHEETS[0]].index.tolist()
    missing = [el for el in elements if el not in ATOMIC_NUMBERS]
    if missing:
        raise KeyError(f"缺少以下元素的原子序：{', '.join(missing)}")
    return tables


def ordered_pair(elem_a: str, elem_b: str) -> Tuple[str, str]:
    """Return elements ordered by atomic number (A atomic number < B)."""
    if elem_a not in ATOMIC_NUMBERS or elem_b not in ATOMIC_NUMBERS:
        raise KeyError("元素未在数据库映射中定义。")
    if elem_a == elem_b:
        raise ValueError("需要两个不同的元素。")

    if ATOMIC_NUMBERS[elem_a] <= ATOMIC_NUMBERS[elem_b]:
        return elem_a, elem_b
    return elem_b, elem_a


def lookup_omegas(
    tables: Dict[str, pd.DataFrame], elem_a: str, elem_b: str
) -> Tuple[List[float], Tuple[str, str]]:
    """Fetch Ω coefficients for the ordered pair."""
    from_ordered, to_ordered = ordered_pair(elem_a, elem_b)
    omegas: List[float] = []
    for sheet in OMEGA_SHEETS:
        df = tables[sheet]
        try:
            value = df.at[from_ordered, to_ordered]
        except KeyError as exc:
            raise KeyError(
                f"未在工作表 {sheet} 中找到元素对 {from_ordered}-{to_ordered}。"
            ) from exc
        if pd.isna(value):
            raise KeyError(
                f"元素对 {from_ordered}-{to_ordered} 在工作表 {sheet} 中没有数值。"
            )
        omegas.append(float(value))
    return omegas, (from_ordered, to_ordered)


def parse_composition_line(line: str) -> List[Tuple[str, float]]:
    """Parse a single line like 'Fe 20 Al 30 Ni 50' into normalized fractions."""
    matches = COMPOSITION_PATTERN.findall(line)
    if not matches:
        raise ValueError("无法解析输入，请按 'Fe 20 Al 80' 的格式输入。")

    order: List[str] = []
    totals: Dict[str, float] = {}
    for symbol_raw, value_raw in matches:
        symbol = normalize_symbol(symbol_raw)
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"暂不支持元素 {symbol}。")
        value = float(value_raw)
        if value < 0:
            raise ValueError("百分比不能为负数。")
        if symbol not in totals:
            order.append(symbol)
            totals[symbol] = 0.0
        totals[symbol] += value

    if len(order) < 2:
        raise ValueError("至少需要两个不同元素。")

    total = sum(totals.values())
    if total <= 0:
        raise ValueError("百分比总和必须为正。")

    return [(symbol, totals[symbol] / total) for symbol in order]


def compute_multi_component_enthalpy(
    tables: Dict[str, pd.DataFrame],
    composition: List[Tuple[str, float]],
) -> Tuple[float, List[Dict[str, object]]]:
    """Sum ΔH_mix contributions over every unique pair in the composition."""
    total_enthalpy = 0.0
    details: List[Dict[str, object]] = []

    for (sym_a, frac_a), (sym_b, frac_b) in combinations(composition, 2):
        try:
            omegas, ordered = lookup_omegas(tables, sym_a, sym_b)
        except KeyError as exc:
            raise KeyError(f"元素对 {sym_a}-{sym_b}: {exc}") from exc

        fraction_map = {sym_a: frac_a, sym_b: frac_b}
        ordered_fractions = [fraction_map[name] for name in ordered]
        mixture = BinaryMixture(list(ordered), ordered_fractions, omegas)
        delta_h = mixture.enthalpy_of_mixing()

        details.append(
            {
                "pair": ordered,
                "fractions": ordered_fractions,
                "delta_h": delta_h,
            }
        )
        total_enthalpy += delta_h

    return total_enthalpy, details


def run_excel_line_calculator() -> None:
    """Excel-driven calculator that accepts a single-line multi-element input."""
    print("\n--- Excel 多组元计算器：单行输入元素 + 原子百分比 ---")
    default_prompt = f"{DEFAULT_DATABASE_PATH}"
    path_str = input(f"Excel 路径 (留空默认 {default_prompt}): ").strip()
    path = Path(path_str) if path_str else DEFAULT_DATABASE_PATH

    try:
        tables = load_omega_tables(path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"无法加载数据库: {exc}")
        return

    print("示例：Fe 20 Al 30 Ni 50  或  Fe=20,Al=30,Ni=50  (输入空行可退出)")
    while True:
        try:
            line = input("输入元素及原子百分比: ").strip()
        except EOFError:
            print("\n检测到 EOF，退出 Excel 模式。\n")
            return
        if not line:
            print("退出 Excel 模式。\n")
            return

        try:
            composition = parse_composition_line(line)
        except ValueError as exc:
            print(f"输入错误: {exc}")
            continue

        try:
            total_enthalpy, details = compute_multi_component_enthalpy(
                tables, composition
            )
        except KeyError as exc:
            print(exc)
            continue

        print("\n归一化原子分数:")
        for symbol, fraction in composition:
            print(f"{symbol:<5s} x = {fraction:.4f} ({fraction * 100:.2f}%)")

        print("\n各元素对贡献 (kJ/mol):")
        for detail in details:
            pair_label = f"{detail['pair'][0]}-{detail['pair'][1]}"
            c_a, c_b = detail["fractions"]
            print(
                f"{pair_label:<10s} ΔH = {detail['delta_h']:>10.5f} "
                f"(c_A={c_a:.4f}, c_B={c_b:.4f})"
            )

def main() -> None:
    """Entry point; run the Excel calculator once."""
    try:
        run_excel_line_calculator()
    except KeyboardInterrupt:
        print("\n已中断，退出。")


if __name__ == "__main__":
    main()
