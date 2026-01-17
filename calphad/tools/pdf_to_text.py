#!/usr/bin/env python3
"""Convert PDFs in calphad/sources/ to text for extraction."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CALPHAD_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = CALPHAD_DIR / "sources"


def pdf_to_text(pdf_path: Path, txt_path: Path) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), str(txt_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "pdftotext failed"
        raise RuntimeError(detail)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PDFs under calphad/sources/ to .txt")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[str(DEFAULT_SOURCES)],
        help="PDF files or directories to scan (default: calphad/sources)",
    )
    args = parser.parse_args()

    try:
        subprocess.run(["pdftotext", "-h"], check=False, capture_output=True)
    except FileNotFoundError:
        print("ERROR: pdftotext is not available on PATH. Install poppler-utils.", file=sys.stderr)
        return 2

    pdfs: list[Path] = []
    for input_path in args.inputs:
        path = Path(input_path)
        if path.is_dir():
            pdfs.extend([p for p in path.rglob("*.pdf") if p.is_file()])
        else:
            if path.suffix.lower() == ".pdf":
                pdfs.append(path)

    if not pdfs:
        print("No PDF files found.")
        return 0

    for pdf_path in pdfs:
        txt_path = pdf_path.with_suffix(".txt")
        try:
            pdf_to_text(pdf_path, txt_path)
            print(f"Converted: {pdf_path} -> {txt_path}")
        except Exception as exc:
            print(f"ERROR: failed to convert {pdf_path}: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
