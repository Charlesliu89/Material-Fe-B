#!/usr/bin/env python3
"""Extract explicit TDB lines from text-like sources."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

TDB_PREFIXES = ("FUNCTION", "PHASE", "CONSTITUENT", "PARAMETER", "TYPE_DEFINITION")


def extract_from_text(text: str, source_label: str, doi: str | None) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    pages = text.split("\f")
    for page_index, page in enumerate(pages, start=1):
        for idx, line in enumerate(page.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(TDB_PREFIXES):
                results.append(
                    {
                        "line": stripped,
                        "source": source_label,
                        "doi": doi or "unknown",
                        "page": str(page_index),
                        "quote": stripped[:200],
                        "line_no": str(idx),
                    }
                )
    return results


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract explicit TDB lines from sources.")
    parser.add_argument("inputs", nargs="+", help="Files or directories to scan")
    parser.add_argument("--output", help="Write JSONL to this file")
    parser.add_argument(
        "--doi",
        default=None,
        help="DOI to associate with extracted lines (optional).",
    )
    args = parser.parse_args()

    collected: List[Dict[str, str]] = []
    for input_path in args.inputs:
        path = Path(input_path)
        if path.is_dir():
            candidates = [
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".txt", ".tdb"}
            ]
        else:
            candidates = [path]

        for file_path in candidates:
            if file_path.suffix.lower() not in {".txt", ".tdb"}:
                print(
                    f"WARN: skipping unsupported file {file_path} (only .txt/.tdb are supported)",
                    file=sys.stderr,
                )
                continue
            text = read_text_file(file_path)
            collected.extend(extract_from_text(text, str(file_path), args.doi))

    output_lines = [json.dumps(entry, ensure_ascii=False) for entry in collected]

    if args.output:
        Path(args.output).write_text("\n".join(output_lines) + ("\n" if output_lines else ""))
    else:
        print("\n".join(output_lines))

    return 0


if __name__ == "__main__":
    sys.exit(main())
