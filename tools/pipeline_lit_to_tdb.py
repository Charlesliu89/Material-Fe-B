#!/usr/bin/env python3
"""Run literature search -> fetch -> extract -> merge -> smoke test."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)


def load_candidates(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def refine_query(query: str, iteration: int) -> str:
    tweaks = [
        "supplementary tdb",
        "TDB file",
        "CALPHAD parameter",
        "PARAMETER L(",
    ]
    return f"{query} {tweaks[min(iteration, len(tweaks) - 1)]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline: search -> fetch -> extract -> build -> test")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--rows", type=int, default=10, help="Number of search results")
    parser.add_argument("--max-iter", type=int, default=3, help="Max iterations")
    parser.add_argument(
        "--candidates",
        help="Path to a JSON list of candidates (skip online search).",
    )
    args = parser.parse_args()

    candidates_path = Path("extracted/candidates.json")
    extracted_path = Path("extracted/tdb_lines.jsonl")
    extracted_path.parent.mkdir(parents=True, exist_ok=True)

    for iteration in range(args.max_iter):
        query = refine_query(args.query, iteration)
        print(f"INFO: iteration {iteration + 1} query='{query}'")

        if args.candidates:
            candidates_path = Path(args.candidates)
            candidates = load_candidates(candidates_path)
        else:
            search_cmd = [
                sys.executable,
                "tools/lit_search.py",
                "--query",
                query,
                "--rows",
                str(args.rows),
                "--json",
            ]
            result = run_step(search_cmd, timeout=120)
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
                if Path("sources/search_blocked.md").exists():
                    print("ERROR: manual search required (see sources/search_blocked.md)")
                    return 2
                return result.returncode
            candidates_path.write_text(result.stdout, encoding="utf-8")
            candidates = load_candidates(candidates_path)

        dois = [item["doi"] for item in candidates if item.get("doi")]
        if not dois:
            print("WARN: no DOIs found, refining query.")
            continue

        fetch_cmd = [sys.executable, "tools/fetch_sources.py", *dois]
        fetch = run_step(fetch_cmd, timeout=300)
        print(fetch.stdout)
        if fetch.returncode != 0:
            print(fetch.stderr)
            if Path("sources/download_queue.md").exists():
                print("ERROR: manual download required (see sources/download_queue.md)")
                return 2

        pdf_cmd = [sys.executable, "tools/pdf_to_text.py", "sources"]
        pdf = run_step(pdf_cmd, timeout=300)
        print(pdf.stdout)
        if pdf.returncode != 0:
            print(pdf.stderr)
            return pdf.returncode

        txt_inputs = ["sources"]
        extract_cmd = [sys.executable, "tools/extract_tdb_lines.py", *txt_inputs, "--output", str(extracted_path)]
        extract = run_step(extract_cmd, timeout=300)
        print(extract.stdout)
        if extract.returncode != 0:
            print(extract.stderr)
            return extract.returncode

        build_cmd = [
            sys.executable,
            "tools/build_tdb.py",
            "--extracted",
            str(extracted_path),
        ]
        build = run_step(build_cmd, timeout=120)
        print(build.stdout)
        if build.returncode != 0:
            print(build.stderr)
            return build.returncode

        smoke_cmd = [sys.executable, "tools/tdb_smoke_test.py", "tdb/Fe-B.tdb"]
        smoke = run_step(smoke_cmd, timeout=300)
        print(smoke.stdout)
        if smoke.returncode == 0:
            imported = sum(1 for line in extracted_path.read_text(encoding="utf-8").splitlines() if line.strip())
            if imported == 0:
                print("WARN: smoke test passed but no sourced lines were imported.")
                continue
            return 0
        print(smoke.stderr)

    print("ERROR: max iterations reached without adding sourced lines.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
