#!/usr/bin/env python3
"""Search for Fe-B CALPHAD assessments and possible TDB sources."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import urllib.error
import urllib.parse
import urllib.request

CALPHAD_DIR = Path(__file__).resolve().parents[1]
SOURCES_DIR = CALPHAD_DIR / "sources"

CROSSREF_URL = "https://api.crossref.org/works"
OPENALEX_URL = "https://api.openalex.org/works"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/"
USER_AGENT_BASE = "FeB-TDB-bot/0.1"


def _build_user_agent(mailto: str | None) -> str:
    if mailto:
        return f"{USER_AGENT_BASE} (mailto:{mailto})"
    return f"{USER_AGENT_BASE} (+https://example.invalid)"


def _get_json(
    url: str,
    params: Dict[str, Any],
    timeout: int = 30,
    mailto: str | None = None,
) -> Dict[str, Any]:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v})
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": _build_user_agent(mailto)},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in {401, 403}:
                raise
        except Exception as exc:
            last_error = exc
        time.sleep(1.5 * (2**attempt))
    if last_error:
        raise last_error
    raise RuntimeError("failed to fetch JSON data")


def search_crossref(query: str, rows: int, mailto: str | None) -> List[Dict[str, Any]]:
    data = _get_json(
        CROSSREF_URL,
        {"query": query, "rows": rows, "mailto": mailto},
        mailto=mailto,
    )
    items = data.get("message", {}).get("items", [])
    results = []
    for item in items:
        results.append(
            {
                "title": " ".join(item.get("title", [])),
                "doi": item.get("DOI"),
                "url": item.get("URL"),
                "score": item.get("score"),
                "issued": item.get("issued", {}).get("date-parts", [[None]])[0][0],
                "source": "crossref",
            }
        )
    return results


def _openalex_doi(raw: str | None) -> str | None:
    if not raw:
        return None
    prefix = "https://doi.org/"
    return raw[len(prefix) :] if raw.startswith(prefix) else raw


def search_openalex(
    query: str,
    rows: int,
    api_key: str | None,
    mailto: str | None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"search": query, "per-page": rows}
    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto
    data = _get_json(OPENALEX_URL, params, mailto=mailto)
    items = data.get("results", [])
    results = []
    for item in items:
        doi = _openalex_doi(item.get("doi") or item.get("ids", {}).get("doi"))
        landing_url = item.get("primary_location", {}).get("landing_page_url")
        if not landing_url and doi:
            landing_url = f"https://doi.org/{doi}"
        results.append(
            {
                "title": item.get("display_name"),
                "doi": doi,
                "url": landing_url,
                "score": item.get("relevance_score"),
                "issued": item.get("publication_year"),
                "source": "openalex",
            }
        )
    return results


def fetch_unpaywall(doi: str, email: str) -> Dict[str, Any]:
    return _get_json(f"{UNPAYWALL_URL}{doi}", {"email": email}, mailto=email)


def _write_blocked(path: str, message: str) -> None:
    dest = Path(path)
    content = (
        "# Literature search blocked\n\n"
        f"{message}\n\n"
        "Suggestion: run calphad/tools/lit_search.py locally with outbound access and set "
        "CROSSREF_MAILTO / OPENALEX_MAILTO / OPENALEX_API_KEY as needed.\n"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        handle.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Search for Fe-B CALPHAD TDB sources.")
    parser.add_argument(
        "--query",
        default="Fe B CALPHAD thermodynamic assessment TDB",
        help="Search query for Crossref",
    )
    parser.add_argument("--rows", type=int, default=10, help="Number of Crossref results")
    parser.add_argument(
        "--email",
        default=os.environ.get("UNPAYWALL_EMAIL", "test@example.com"),
        help="Email required for Unpaywall API",
    )
    parser.add_argument(
        "--crossref-mailto",
        default=os.environ.get("CROSSREF_MAILTO"),
        help="Crossref mailto (or set CROSSREF_MAILTO).",
    )
    parser.add_argument(
        "--openalex-api-key",
        default=os.environ.get("OPENALEX_API_KEY"),
        help="OpenAlex API key (or set OPENALEX_API_KEY).",
    )
    parser.add_argument(
        "--openalex-mailto",
        default=os.environ.get("OPENALEX_MAILTO") or os.environ.get("UNPAYWALL_EMAIL"),
        help="OpenAlex mailto (or set OPENALEX_MAILTO).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []
    blocked: List[Tuple[str, Exception]] = []
    try:
        results = search_crossref(args.query, args.rows, args.crossref_mailto)
    except Exception as exc:
        print(f"WARNING: crossref search failed: {exc}")
        blocked.append(("crossref", exc))

    if not results:
        try:
            results = search_openalex(
                args.query,
                args.rows,
                args.openalex_api_key,
                args.openalex_mailto,
            )
        except Exception as exc:
            print(f"ERROR: openalex search failed: {exc}")
            blocked.append(("openalex", exc))

    if not results:
        blocked_msgs = ", ".join(f"{name}: {err}" for name, err in blocked)
        _write_blocked(
            str(SOURCES_DIR / "search_blocked.md"),
            f"Both Crossref/OpenAlex searches failed or were blocked. Details: {blocked_msgs}",
        )
        print(f"ERROR: search blocked; see {SOURCES_DIR / 'search_blocked.md'}")
        return 2

    enriched = []
    for item in results:
        doi = item.get("doi")
        if doi:
            try:
                upw = fetch_unpaywall(doi, args.email)
                item["oa_status"] = upw.get("oa_status")
                item["best_oa_location"] = upw.get("best_oa_location", {}).get("url")
            except Exception:
                item["oa_status"] = None
                item["best_oa_location"] = None
        enriched.append(item)

    enriched.sort(key=lambda x: (x.get("best_oa_location") is None, -(x.get("score") or 0)))

    if args.json:
        print(json.dumps(enriched, indent=2, ensure_ascii=False))
        return 0

    for idx, item in enumerate(enriched, start=1):
        print(f"[{idx}] {item.get('title')}")
        print(f"    DOI: {item.get('doi')}")
        print(f"    URL: {item.get('url')}")
        print(f"    OA: {item.get('oa_status')} -> {item.get('best_oa_location')}")
        print(f"    Score: {item.get('score')} | Year: {item.get('issued')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
