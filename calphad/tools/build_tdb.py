#!/usr/bin/env python3
"""Merge extracted TDB lines into the working Fe-B TDB."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

CALPHAD_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = CALPHAD_DIR.parent
DEFAULT_TDB = CALPHAD_DIR / "tdb" / "Fe-B.tdb"
DEFAULT_CHANGELOG = ROOT_DIR / "CHANGELOG.md"

def load_extracted(jsonl_path: Path) -> List[Tuple[str, str, str, str]]:
    lines: List[Tuple[str, str, str, str]] = []
    for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        line = payload.get("line")
        if line:
            doi = payload.get("doi") or "unknown"
            page = payload.get("page") or "n/a"
            quote = payload.get("quote") or ""
            lines.append((line, doi, page, quote))
    return lines


def _strip_existing_sourced(lines: List[str]) -> List[str]:
    if "! BEGIN SOURCED LINES\n" in lines:
        start = lines.index("! BEGIN SOURCED LINES\n")
        end = lines.index("! END SOURCED LINES\n") if "! END SOURCED LINES\n" in lines else len(lines)
        return lines[:start]
    return lines


def merge_lines(
    template_lines: List[str],
    additions: Iterable[Tuple[str, str, str, str]],
) -> List[str]:
    base = _strip_existing_sourced(template_lines)
    existing = set(line.strip() for line in base if line.strip())
    merged = list(base)
    merged.append("! BEGIN SOURCED LINES\n")
    for line, doi, page, quote in additions:
        if line.strip() and line.strip() not in existing:
            evidence = f'! SOURCE DOI:{doi} PAGE:{page} "{quote}"\n'
            merged.append(evidence)
            merged.append(line.rstrip() + "\n")
            existing.add(line.strip())
    merged.append("! END SOURCED LINES\n")
    return merged


def update_changelog(changelog: Path, message: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d")
    entry = f"- {timestamp}: {message}\n"
    if changelog.exists():
        content = changelog.read_text(encoding="utf-8")
    else:
        content = "# CHANGELOG\n\n"
    if entry not in content:
        content += entry
    changelog.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge extracted TDB lines into calphad/tdb/Fe-B.tdb")
    parser.add_argument("--template", default=str(DEFAULT_TDB), help="Base/template TDB path")
    parser.add_argument("--output", default=str(DEFAULT_TDB), help="Output TDB path")
    parser.add_argument("--extracted", required=True, help="JSONL file from extract_tdb_lines")
    parser.add_argument(
        "--changelog",
        default=str(DEFAULT_CHANGELOG),
        help="Changelog file to update",
    )
    args = parser.parse_args()

    template_path = Path(args.template)
    output_path = Path(args.output)
    extracted_path = Path(args.extracted)

    base_lines = template_path.read_text(encoding="utf-8").splitlines(keepends=True)
    additions = load_extracted(extracted_path)

    merged = merge_lines(base_lines, additions)
    output_path.write_text("".join(merged), encoding="utf-8")

    update_changelog(Path(args.changelog), f"Merged {len(additions)} extracted TDB lines into {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
