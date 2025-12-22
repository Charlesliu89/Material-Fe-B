#!/usr/bin/env python3
"""
Batch enthalpy plot generator.

Splits responsibilities across:
- enthalpy_config: constants (fonts, paths, steps)
- enthalpy_core: data loading, composition utilities, enthalpy computation
- enthalpy_plot: Plotly figure builders and export helpers
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from enthalpy_config import (
    BINARY_STEP,
    COLORBAR_LABEL_CONFIG,
    DEFAULT_DATABASE_PATH,
    PLOTLY_ELEMENT_FONT,
    OMEGA_SHEETS,
    QUATERNARY_MIN_STEP,
    QUATERNARY_STEP,
    TERNARY_STEP,
)
from enthalpy_core import (
    barycentric_to_cartesian,
    build_fraction_vectors,
    build_quaternary_points,
    build_ternary_points,
    compute_multi_component_enthalpy,
    fractions_from_vector,
    load_omega_tables,
    lookup_omegas,
    normalize_step,
    normalize_symbol,
)
from enthalpy_plot import (
    add_plotly_colorbar_label,
    apply_plotly_base_style,
    build_binary_figure,
    build_quaternary_figure,
    build_ternary_figure,
    write_plotly_image,
)

# --------------------------------------------------------------------------- #
# Composition helpers
# --------------------------------------------------------------------------- #


def combo_supported(calculator, tables, combo: Sequence[str]) -> bool:
    lookup = getattr(calculator, "lookup_omegas", lookup_omegas)
    try:
        for elem_a, elem_b in combinations(combo, 2):
            lookup(tables, elem_a, elem_b)
        return True
    except KeyError:
        return False


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_or_html(fig, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_plotly_image(fig, target)
        print(f"Saved PNG to {target}")
    except Exception as exc:  # pylint: disable=broad-except
        alt_path = target.with_suffix(".html")
        fig.write_html(str(alt_path))
        print(f"PNG export failed ({exc}); saved interactive HTML to {alt_path}")


# --------------------------------------------------------------------------- #
# Worker helpers
# --------------------------------------------------------------------------- #

_WORKER_CALCULATOR = None
_WORKER_TABLES: Dict[str, object] | None = None


def _parallel_initializer(calculator_path: str, tables: Dict[str, object]) -> None:
    global _WORKER_CALCULATOR, _WORKER_TABLES
    _WORKER_CALCULATOR = load_calculator_module(Path(calculator_path))
    _WORKER_TABLES = tables


def _get_worker_state():
    if _WORKER_CALCULATOR is None or _WORKER_TABLES is None:
        raise RuntimeError("Worker state not initialized.")
    return _WORKER_CALCULATOR, _WORKER_TABLES


def _binary_worker_task(combo: Sequence[str], total_units: int):
    calculator, tables = _get_worker_state()
    fractions = [i / total_units for i in range(total_units + 1)]
    enthalpies: List[float] = []
    for frac_a in fractions:
        frac_b = 1.0 - frac_a
        composition = [(combo[0], frac_a), (combo[1], frac_b)]
        total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)
        enthalpies.append(total_enthalpy)
    return combo, fractions, enthalpies


def _ternary_worker_task(combo: Sequence[str], vectors: Sequence[Tuple[int, ...]], total_units: int):
    calculator, tables = _get_worker_state()
    a_vals: List[float] = []
    b_vals: List[float] = []
    c_vals: List[float] = []
    enthalpies: List[float] = []
    for vector in vectors:
        fractions = fractions_from_vector(vector, total_units)
        composition = list(zip(combo, fractions))
        total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)
        a_vals.append(fractions[0] * 100)
        b_vals.append(fractions[1] * 100)
        c_vals.append(fractions[2] * 100)
        enthalpies.append(total_enthalpy)
    return combo, a_vals, b_vals, c_vals, enthalpies


# --------------------------------------------------------------------------- #
# Plot builders using extracted modules
# --------------------------------------------------------------------------- #


def _save_binary_figure(combo: Sequence[str], fractions, enthalpies, output_dir: Path) -> None:
    fig = build_binary_figure(combo, fractions, enthalpies)
    target = ensure_directory(output_dir) / f"{combo[0]}-{combo[1]}.png"
    write_plotly_image(fig, target)


def _save_ternary_figure(combo: Sequence[str], a_vals, b_vals, c_vals, enthalpies, output_dir: Path) -> None:
    fig = build_ternary_figure(combo, a_vals, b_vals, c_vals, enthalpies)
    target = ensure_directory(output_dir) / f"{combo[0]}-{combo[1]}-{combo[2]}.png"
    write_plotly_image(fig, target)


# --------------------------------------------------------------------------- #
# Batch runner
# --------------------------------------------------------------------------- #


BATCH_CHUNK_SIZE = 100


def run_batch(
    calculator,
    tables: Dict[str, pd.DataFrame],
    component_count: int,
    elements: Sequence[str],
    output_path: Path,
    workers: int = 1,
    chunk_size: int = BATCH_CHUNK_SIZE,
    prompt_chunks: bool = True,
) -> None:
    if component_count not in {2, 3, 4}:
        raise ValueError("Only 2-, 3-, or 4-component alloys are supported.")

    available_elements = [el for el in tables[OMEGA_SHEETS[0]].index if el in elements]
    if len(available_elements) < component_count:
        raise ValueError("Not enough elements to form the requested combinations.")

    total_units, actual_step = normalize_step(BINARY_STEP if component_count == 2 else TERNARY_STEP)
    vectors: Optional[List[Tuple[int, ...]]] = None
    if component_count == 3:
        vectors = build_fraction_vectors(component_count, total_units)
        if not vectors:
            raise ValueError("No feasible compositions were generated with the provided step.")

    ensure_directory(output_path)
    processed = 0
    skipped = 0
    supported_combos: List[Tuple[str, ...]] = []

    for combo in combinations(available_elements, component_count):
        if not combo_supported(calculator, tables, combo):
            skipped += 1
            continue
        if component_count in {2, 3}:
            supported_combos.append(combo)
        else:
            print("[info] Use menu option 3 for quaternary preview/slices.")
            skipped += 1

    if not supported_combos:
        print(
            f"[summary] step={actual_step:.4f}, workers=0: "
            f"{processed} combinations plotted, {skipped} skipped.",
            file=sys.stderr,
        )
        return

    worker_count = min(max(1, workers), len(supported_combos))

    def process_chunk(chunk: List[Tuple[str, ...]], use_parallel: bool) -> int:
        completed = 0
        if use_parallel and len(chunk) > 1:
            calculator_path = getattr(calculator, "__file__", None)
            if not calculator_path:
                raise RuntimeError("Calculator module path is required for parallel execution.")
            init_args = (str(calculator_path), tables)
            with ProcessPoolExecutor(
                max_workers=min(worker_count, len(chunk)),
                initializer=_parallel_initializer,
                initargs=init_args,
            ) as executor:
                futures = []
                for combo in chunk:
                    if component_count == 2:
                        futures.append(executor.submit(_binary_worker_task, combo, total_units))
                    else:
                        assert vectors is not None
                        futures.append(executor.submit(_ternary_worker_task, combo, vectors, total_units))
                for future in as_completed(futures):
                    result = future.result()
                    if component_count == 2:
                        combo, fractions, enthalpies = result
                        _save_binary_figure(combo, fractions, enthalpies, output_path / "binary")
                    else:
                        combo, a_vals, b_vals, c_vals, enthalpies = result
                        _save_ternary_figure(combo, a_vals, b_vals, c_vals, enthalpies, output_path / "ternary")
                    completed += 1
            return completed

        for combo in chunk:
            if component_count == 2:
                fractions, enthalpies = _binary_worker_task(combo, total_units)
                _save_binary_figure(combo, fractions, enthalpies, output_path / "binary")
            else:
                assert vectors is not None
                a_vals, b_vals, c_vals, enthalpies = build_ternary_points(calculator, tables, combo, vectors, total_units)
                _save_ternary_figure(combo, a_vals, b_vals, c_vals, enthalpies, output_path / "ternary")
            completed += 1
        return completed

    export_all_remaining = False
    index = 0
    total = len(supported_combos)
    while index < total:
        current_chunk_size = total - index if export_all_remaining else chunk_size
        if current_chunk_size <= 0:
            current_chunk_size = total - index
        chunk = supported_combos[index : index + current_chunk_size]
        processed += process_chunk(chunk, use_parallel=worker_count > 1)
        index += len(chunk)

        if index >= total or not prompt_chunks:
            continue

        while True:
            decision = input(
                f"Exported {index}/{total} images. "
                f"Press 'c' to continue next {chunk_size}, 'a' to export all remaining, "
                "or 'q' to stop: "
            ).strip().lower()
            if decision in {"c", "", "continue"}:
                break
            if decision in {"a", "all"}:
                export_all_remaining = True
                break
            if decision in {"q", "quit", "stop"}:
                index = total
                break
            print("Please enter 'c', 'a', or 'q'.")

    print(
        f"[summary] step={actual_step:.4f}, workers={worker_count}: "
        f"{processed} combinations plotted, {skipped} skipped.",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------- #
# Custom plot handling
# --------------------------------------------------------------------------- #


def build_custom_plot(calculator, tables, elements: Sequence[str]):
    if len(elements) < 2 or len(elements) > 4:
        raise ValueError("Please provide between 2 and 4 unique elements.")
    for element in elements:
        if element not in tables[OMEGA_SHEETS[0]].index:
            raise ValueError(f"Element {element} is not available in the database.")

    component_count = len(elements)
    step_value = BINARY_STEP if component_count == 2 else TERNARY_STEP
    total_units, _ = normalize_step(step_value if component_count < 4 else QUATERNARY_STEP)
    combo = tuple(elements)

    if component_count == 2:
        fractions = [i / total_units for i in range(total_units + 1)]
        enthalpies = []
        for frac_a in fractions:
            frac_b = 1.0 - frac_a
            composition = [(combo[0], frac_a), (combo[1], frac_b)]
            total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)
            enthalpies.append(total_enthalpy)
        fig = build_binary_figure(combo, fractions, enthalpies)
        return fig, f"{combo[0]}-{combo[1]}.png"

    if component_count == 3:
        vectors = build_fraction_vectors(3, total_units)
        a_vals, b_vals, c_vals, enthalpies = build_ternary_points(calculator, tables, combo, vectors, total_units)
        fig = build_ternary_figure(combo, a_vals, b_vals, c_vals, enthalpies)
        return fig, f"{combo[0]}-{combo[1]}-{combo[2]}.png"

    if component_count == 4:
        vectors = build_fraction_vectors(4, total_units)
        if not vectors:
            raise ValueError("No feasible quaternary compositions were generated.")
        x_vals, y_vals, z_vals, enthalpies, fractions = build_quaternary_points(
            calculator, tables, combo, vectors, total_units
        )
        fig = build_quaternary_figure(combo, x_vals, y_vals, z_vals, enthalpies, fractions)
        return fig, f"{combo[0]}-{combo[1]}-{combo[2]}-{combo[3]}.png"

    raise ValueError("Unsupported component count.")


def preview_and_maybe_save(fig, default_path: Path) -> None:
    config = {"editable": True, "edits": {"annotationPosition": True}}
    fig.show(config=config)
    save = input(f"Save figure to {default_path}? (y/n): ").strip().lower()
    if save == "y":
        write_or_html(fig, default_path)


def handle_custom_plot(calculator, tables, output_dir: Path) -> None:
    prompt = (
        "Enter element symbols separated by commas (e.g., Fe,B or Fe,B,Ni)"
        " or 'b' to return to the main menu: "
    )
    custom_dir = ensure_directory(output_dir / "custom")

    while True:
        raw = input(prompt).strip()
        if not raw:
            print("No elements entered. Provide symbols or 'b' to return.")
            continue
        if raw.lower() in {"b", "back", "r", "return"}:
            print("Returning to the main menu.")
            return

        symbols = [calculator.normalize_symbol(part) for part in re.split(r"[\\s,]+", raw) if part]
        unique_elements = []
        for symbol in symbols:
            if symbol not in unique_elements:
                unique_elements.append(symbol)
        if len(unique_elements) < 2 or len(unique_elements) > 4:
            print("Please provide between 2 and 4 unique elements.")
            continue
        for element in unique_elements:
            if element not in tables[OMEGA_SHEETS[0]].index:
                print(f"Element {element} is not available in the database.")
                break
        else:
            try:
                fig, filename = build_custom_plot(calculator, tables, unique_elements)
            except ValueError as exc:
                print(exc)
                continue
            preview_and_maybe_save(fig, custom_dir / filename)


# --------------------------------------------------------------------------- #
# Ternary/quaternary helpers for interactive menu
# --------------------------------------------------------------------------- #


def handle_quaternary_preview(calculator, tables, output_dir: Path) -> None:
    prompt = "Enter four element symbols separated by commas (or 'b' to return): "
    quaternary_dir = ensure_directory(output_dir / "quaternary")

    while True:
        raw = input(prompt).strip()
        if not raw:
            print("No elements entered. Please provide four symbols or 'b' to return.")
            continue
        if raw.lower() in {"b", "back", "r", "return"}:
            print("Returning to the main menu.")
            return

        elements = [calculator.normalize_symbol(part) for part in re.split(r"[\\s,]+", raw) if part]
        unique_elements: List[str] = []
        for element in elements:
            if element not in unique_elements:
                unique_elements.append(element)

        if len(unique_elements) != 4:
            print("Please provide exactly four unique elements.")
            continue

        if not combo_supported(calculator, tables, unique_elements):
            print("Ω data is incomplete for at least one element pair; choose a different set.")
            continue

        density_raw = input("Preview step size in % (press Enter for default 1%): ").strip()
        preview_step = QUATERNARY_STEP
        if density_raw:
            try:
                preview_step = float(density_raw) / 100.0
                if preview_step <= 0:
                    raise ValueError
                if preview_step < QUATERNARY_MIN_STEP:
                    print(f"Clamping to minimum step {QUATERNARY_MIN_STEP * 100:.1f}% for stability.")
                    preview_step = QUATERNARY_MIN_STEP
            except ValueError:
                print("Invalid percentage. Using default 1% step.")
                preview_step = QUATERNARY_STEP

        total_units, actual_step = normalize_step(preview_step)
        vectors = build_fraction_vectors(4, total_units)
        if not vectors:
            print(
                f"No feasible compositions generated for step={actual_step:.3f}. "
                "Adjust QUATERNARY_STEP if needed."
            )
            continue

        print(f"Sampling {len(vectors)} compositions with {actual_step * 100:.2f}% increments.")

        combo = tuple(unique_elements)
        x_vals, y_vals, z_vals, enthalpies, fractions = build_quaternary_points(
            calculator, tables, combo, vectors, total_units
        )
        fig = build_quaternary_figure(combo, x_vals, y_vals, z_vals, enthalpies, fractions)
        fig.show(config={"displaylogo": False, "displayModeBar": True})

        while True:
            slice_raw = input(
                "Fix one element and its fraction% (e.g., Fe=25 or Fe 25). Press Enter to skip: "
            ).strip()
            if not slice_raw:
                break

            match = re.match(r"([A-Za-z]+)\s*[=\s]\s*([0-9]+(?:\.[0-9]+)?)", slice_raw)
            if not match:
                print("Invalid format. Use Element=number (percentage).")
                continue

            element = calculator.normalize_symbol(match.group(1))
            if element not in combo:
                print(f"Element must be one of the current four: {', '.join(combo)}.")
                continue

            fraction_percent = float(match.group(2))
            if fraction_percent <= 0 or fraction_percent >= 100:
                print("Fraction must be between 0 and 100 (exclusive).")
                continue

            try:
                remaining_elements, a_vals, b_vals, c_vals, enthalpy_slice = _slice_quaternary_data(
                    calculator,
                    tables,
                    combo,
                    element,
                    fraction_percent / 100.0,
                    step=BINARY_STEP,
                )
            except ValueError as exc:
                print(exc)
                continue

            if not enthalpy_slice:
                print("No compositions matched that fixed fraction.")
                continue

            slice_fig = build_ternary_figure(remaining_elements, a_vals, b_vals, c_vals, enthalpy_slice)
            slice_title = (
                f"Ternary ΔH<sub>mix</sub>: {'-'.join(combo)} "
                f"(fixed {element}={fraction_percent:.1f}%)"
            )
            slice_fig.update_layout(title=dict(text=slice_title, font=PLOTLY_ELEMENT_FONT))
            filename = f"{'-'.join(combo)}_{element}{fraction_percent:.0f}.png"
            preview_and_maybe_save(slice_fig, quaternary_dir / filename)
            break


def handle_equimolar_quinary(calculator, tables, element_pool: Sequence[str], output_dir: Path) -> None:
    available_elements = [el for el in tables[OMEGA_SHEETS[0]].index if el in element_pool]
    if len(available_elements) < 5:
        print("At least 5 elements are required in the current element pool.")
        return

    prompt = "Enter 1-5 base elements separated by commas (e.g., Fe,Co or Fe,Co,Ni,Cr,B) or 'b' to return: "
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("No elements entered. Provide 1-5 symbols or 'b' to return.")
            continue
        if raw.lower() in {"b", "back", "r", "return"}:
            print("Returning to the main menu.")
            return

        symbols = [calculator.normalize_symbol(part) for part in re.split(r"[\\s,]+", raw) if part]
        base_elements: List[str] = []
        for symbol in symbols:
            if symbol not in base_elements:
                base_elements.append(symbol)

        if len(base_elements) < 1 or len(base_elements) > 5:
            print("Please provide between 1 and 5 unique elements.")
            continue

        missing = [el for el in base_elements if el not in available_elements]
        if missing:
            print(f"Elements not in the current pool: {', '.join(missing)}.")
            continue

        if len(base_elements) == 5:
            combo = tuple(base_elements)
            if not combo_supported(calculator, tables, combo):
                print("Data is incomplete for at least one pair; choose a different set.")
                continue
            composition = [(element, 1.0 / 5.0) for element in combo]
            total_enthalpy, details = calculator.compute_multi_component_enthalpy(tables, composition)
            print(f"\nEquimolar 5-component ΔH_mix for {'-'.join(combo)} = {total_enthalpy:.5f} kJ/mol")
            if details:
                print("Pairwise contributions (kJ/mol):")
                for pair, (c_a, c_b), delta_h in details:
                    print(f"{pair:<10s} ΔH = {delta_h:>10.5f} (c_A={c_a:.4f}, c_B={c_b:.4f})")
            else:
                print("No pairwise contributions available.")
            continue

        remaining_needed = 5 - len(base_elements)
        remaining_elements = [el for el in available_elements if el not in base_elements]
        if len(remaining_elements) < remaining_needed:
            print("Not enough remaining elements to build 5-component combinations.")
            continue

        results: List[Tuple[float, Tuple[str, ...]]] = []
        skipped = 0
        for extra in combinations(remaining_elements, remaining_needed):
            combo = tuple(base_elements + list(extra))
            if not combo_supported(calculator, tables, combo):
                skipped += 1
                continue
            composition = [(element, 1.0 / 5.0) for element in combo]
            total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)
            results.append((total_enthalpy, combo))

        if not results:
            print("No valid 5-component combinations were found with complete Ω data.")
            return

        non_negative = sorted((item for item in results if item[0] >= 0), key=lambda x: x[0])
        negative = sorted((item for item in results if item[0] < 0), key=lambda x: x[0])

        print("\nEquimolar 5-component ΔH_mix (each element = 20%)")
        print(f"Base elements: {', '.join(base_elements)}")
        print(f"Element pool size: {len(available_elements)}")
        print(f"Combinations evaluated: {len(results)}; skipped: {skipped}")

        left_header = "ΔH_mix >= 0 (kJ/mol)"
        right_header = "ΔH_mix < 0 (kJ/mol)"
        index_width = max(2, len(str(len(results))))
        index_map: Dict[int, Tuple[float, Tuple[str, ...]]] = {}

        def _build_entries(items: Sequence[Tuple[float, Tuple[str, ...]]]) -> List[str]:
            entries: List[str] = []
            for value, combo in items:
                idx = len(index_map) + 1
                index_map[idx] = (value, combo)
                entries.append(f"{idx:>{index_width}d}. {value:>10.5f}  {'-'.join(combo)}")
            return entries

        left_entries = _build_entries(non_negative) if non_negative else ["None."]
        right_entries = _build_entries(negative) if negative else ["None."]

        term_width = shutil.get_terminal_size((120, 20)).columns
        print("")
        print(left_header + ":")
        for line in _format_multi_column(left_entries, term_width):
            print(line)
        print("")
        print(right_header + ":")
        for line in _format_multi_column(right_entries, term_width):
            print(line)

        if index_map:
            while True:
                detail_raw = input(
                    "Enter indices for pairwise ΔH details (e.g., 1,3-5), or press Enter to continue: "
                ).strip()
                if not detail_raw:
                    break
                selected = _parse_index_selection(detail_raw, max(index_map.keys()))
                if not selected:
                    print("No valid indices were selected.")
                    continue
                for idx in selected:
                    value, combo = index_map[idx]
                    composition = [(element, 1.0 / 5.0) for element in combo]
                    total_enthalpy, details = calculator.compute_multi_component_enthalpy(
                        tables, composition
                    )
                    print(f"\n[{idx}] {'-'.join(combo)}  ΔH_mix={total_enthalpy:.5f} kJ/mol")
                    if not details:
                        print("No pairwise contributions available.")
                        continue
                    print("Pairwise contributions (kJ/mol):")
                    for pair, (c_a, c_b), delta_h in details:
                        print(f"{pair:<10s} ΔH = {delta_h:>10.5f} (c_A={c_a:.4f}, c_B={c_b:.4f})")

        save_raw = input("Save results to Excel? (y/n): ").strip().lower()
        if save_raw in {"y", "yes"}:
            total_rows = max(len(left_entries), len(right_entries))
            rows = []
            for idx in range(total_rows):
                left_text = left_entries[idx] if idx < len(left_entries) else ""
                right_text = right_entries[idx] if idx < len(right_entries) else ""
                rows.append({left_header: left_text, right_header: right_text})

            df = pd.DataFrame(rows, columns=[left_header, right_header])
            target_dir = ensure_directory(output_dir / "quinary")
            filename = f"equimolar_5_{'-'.join(base_elements)}.xlsx"
            target = target_dir / filename
            try:
                df.to_excel(target, index=False)
                print(f"Saved Excel to {target}")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Failed to write Excel: {exc}")
        return


def _slice_quaternary_data(
    calculator,
    tables,
    combo: Sequence[str],
    fixed_element: str,
    fixed_fraction: float,
    step: float = BINARY_STEP,
) -> Tuple[Sequence[str], List[float], List[float], List[float], List[float]]:
    if fixed_fraction <= 0 or fixed_fraction >= 1:
        raise ValueError("Fixed fraction must be within (0, 1).")
    if fixed_element not in combo:
        raise ValueError(f"{fixed_element} is not part of the chosen quaternary system.")

    remaining_elements = [elem for elem in combo if elem != fixed_element]
    remainder = 1.0 - fixed_fraction
    if remainder <= 0:
        raise ValueError("Fixed fraction leaves no remaining composition to vary.")

    total_units, _ = normalize_step(step)
    vectors = build_fraction_vectors(3, total_units)
    if not vectors:
        raise ValueError("No feasible slice compositions were generated.")

    a_vals: List[float] = []
    b_vals: List[float] = []
    c_vals: List[float] = []
    enthalpy_slice: List[float] = []

    for vector in vectors:
        frac_rest = fractions_from_vector(vector, total_units)
        scaled_rest = [value * remainder for value in frac_rest]
        composition = [(fixed_element, fixed_fraction)] + list(zip(remaining_elements, scaled_rest))
        total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)

        a_vals.append(frac_rest[0] * 100)
        b_vals.append(frac_rest[1] * 100)
        c_vals.append(frac_rest[2] * 100)
        enthalpy_slice.append(total_enthalpy)

    return remaining_elements, a_vals, b_vals, c_vals, enthalpy_slice


def _format_multi_column(entries: Sequence[str], max_width: int) -> List[str]:
    if not entries:
        return ["None."]
    column_width = max(len(entry) for entry in entries) + 2
    columns = max(1, max_width // column_width)
    rows = (len(entries) + columns - 1) // columns
    lines: List[str] = []
    for row in range(rows):
        parts = []
        for col in range(columns):
            idx = row + col * rows
            if idx >= len(entries):
                continue
            parts.append(entries[idx].ljust(column_width))
        lines.append("".join(parts).rstrip())
    return lines


def _parse_index_selection(raw: str, max_index: int) -> List[int]:
    tokens = [token for token in re.split(r"[,\s]+", raw.strip()) if token]
    selected: List[int] = []
    for token in tokens:
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            try:
                start = int(start_raw)
                end = int(end_raw)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for idx in range(start, end + 1):
                if 1 <= idx <= max_index and idx not in selected:
                    selected.append(idx)
            continue
        try:
            idx = int(token)
        except ValueError:
            continue
        if 1 <= idx <= max_index and idx not in selected:
            selected.append(idx)
    return selected


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be zero or a positive integer.")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch enthalpy plot generator.")
    parser.add_argument(
        "--calculator",
        type=Path,
        default=Path(__file__).with_name("single_enthalpy_cli.py"),
        help="Path to the interactive calculator script.",
    )
    parser.add_argument(
        "--excel-db",
        type=Path,
        default=None,
        help="Path to the Omega matrices workbook (defaults to calculator's setting).",
    )
    parser.add_argument(
        "--elements",
        nargs="*",
        help="Optional whitelist of element symbols (defaults to all available).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/plots"),
        help="Directory to store generated plots (default: Data/plots).",
    )
    parser.add_argument(
        "--auto-combo",
        action="append",
        help=(
            "Non-interactive mode: provide a comma-separated element list "
            "(e.g., 'Fe,B,Ni'); can be repeated to render multiple combos then exit."
        ),
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=os.cpu_count() or 1,
        help="Number of parallel worker processes to use (default: CPU count).",
    )
    parser.add_argument(
        "--chunk-size",
        type=non_negative_int,
        default=BATCH_CHUNK_SIZE,
        help="Number of combinations to process before prompting (set 0 to disable chunking).",
    )
    parser.add_argument(
        "--chunk-auto-continue",
        action="store_true",
        help="Process all chunks without interactive prompts (useful for batch/CI).",
    )
    parser.add_argument(
        "--list-elements",
        action="store_true",
        help="List available elements from the Excel DB and exit.",
    )
    parser.add_argument(
        "--just-list-elements",
        action="store_true",
        help="Alias for --list-elements.",
    )
    return parser.parse_args()


def load_calculator_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Calculator script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("enthalpy_calculator", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load calculator module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def main() -> None:
    args = parse_args()
    calculator = load_calculator_module(args.calculator)

    excel_path = Path(args.excel_db) if args.excel_db else DEFAULT_DATABASE_PATH
    tables = calculator.load_omega_tables(excel_path)

    if args.list_elements or args.just_list_elements:
        print("Supported elements:")
        for el in list(tables[OMEGA_SHEETS[0]].index):
            print(f"- {el}")
        return

    element_pool = (
        [calculator.normalize_symbol(sym) for sym in args.elements]
        if args.elements
        else list(tables[OMEGA_SHEETS[0]].index)
    )

    if args.auto_combo:
        combos: List[List[str]] = []
        for raw in args.auto_combo:
            parts = [calculator.normalize_symbol(part) for part in re.split(r"[\\s,]+", raw) if part]
            if not parts:
                print(f"[auto] Skipping empty combo input: {raw!r}")
                continue
            unique: List[str] = []
            for symbol in parts:
                if symbol not in unique:
                    unique.append(symbol)
            combos.append(unique)

        if combos:
            for elements in combos:
                try:
                    fig, filename = build_custom_plot(calculator, tables, elements)
                except ValueError as exc:
                    print(f"[auto] Skipping {elements}: {exc}")
                    continue
                target = ensure_directory(args.output_dir / "custom") / filename
                write_or_html(fig, target)
        else:
            print("[auto] No valid combinations were provided.")
        return

    while True:
        print("\n=== Enthalpy Plot Menu ===")
        print("1) Batch binary ΔH_mix curves")
        print("2) Batch ternary ΔH_mix contour plots")
        print("3) Quaternary ΔH_mix tetrahedron (preview + slice export)")
        print("4) Custom combination plot")
        print("5) Equimolar 5-component ΔH_mix list")
        print("q) Quit")
        choice = input("Select an option: ").strip().lower()

        if choice == "1":
            try:
                run_batch(
                    calculator=calculator,
                    tables=tables,
                    component_count=2,
                    elements=element_pool,
                    output_path=args.output_dir,
                    workers=args.workers,
                    chunk_size=args.chunk_size,
                    prompt_chunks=sys.stdin.isatty() and not args.chunk_auto_continue,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Binary plotting failed: {exc}")
        elif choice == "2":
            try:
                run_batch(
                    calculator=calculator,
                    tables=tables,
                    component_count=3,
                    elements=element_pool,
                    output_path=args.output_dir,
                    workers=args.workers,
                    chunk_size=args.chunk_size,
                    prompt_chunks=sys.stdin.isatty() and not args.chunk_auto_continue,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Ternary plotting failed: {exc}")
        elif choice == "3":
            try:
                handle_quaternary_preview(calculator, tables, args.output_dir)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Quaternary preview failed: {exc}")
        elif choice == "4":
            handle_custom_plot(calculator, tables, args.output_dir)
        elif choice == "5":
            handle_equimolar_quinary(calculator, tables, element_pool, args.output_dir)
        elif choice in {"q", "quit", "exit"}:
            print("Bye.")
            break
        else:
            print("Invalid selection. Please choose 1–5 or q.")


if __name__ == "__main__":
    main()
