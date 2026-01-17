#!/usr/bin/env python3
"""Download open-access sources into calphad/sources/<doi_or_slug>/"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import urllib.parse
import urllib.request
from urllib.error import HTTPError

CALPHAD_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = CALPHAD_DIR / "sources"

UNPAYWALL_URL = "https://api.unpaywall.org/v2/"
USER_AGENT = "Fe-B-TDB-Builder/1.0 (+https://example.invalid)"
LOGIN_KEYWORDS = ("login", "signin", "auth", "sso")


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def fetch_unpaywall(doi: str, email: str) -> Dict:
    query = urllib.parse.urlencode({"email": email})
    request = urllib.request.Request(
        f"{UNPAYWALL_URL}{doi}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (2**attempt))
    if last_error:
        raise last_error
    raise RuntimeError("unpaywall fetch failed")


def _looks_like_login(url: str) -> bool:
    url_lower = url.lower()
    return any(token in url_lower for token in LOGIN_KEYWORDS)


def download(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as handle:
        handle.write(response.read())
        return response.geturl()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch OA sources into calphad/sources/<slug>/")
    parser.add_argument("inputs", nargs="+", help="DOIs or URLs to fetch")
    parser.add_argument(
        "--email",
        default=os.environ.get("UNPAYWALL_EMAIL", "test@example.com"),
        help="Email for Unpaywall API",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    auth_required: List[str] = []
    fetched: List[str] = []
    queue_entries: List[str] = []

    for item in args.inputs:
        if item.startswith("http"):
            slug = slugify(item)
            target = out_dir / slug / Path(item).name
            try:
                final_url = download(item, target)
                if _looks_like_login(final_url):
                    auth_required.append(f"{item} (login required: redirected to {final_url})")
                    queue_entries.append(f"- URL: {item}\n  Expected: {target}")
                else:
                    fetched.append(str(target))
            except HTTPError as exc:
                if exc.code in {401, 403}:
                    auth_required.append(f"{item} (login required: {exc})")
                    queue_entries.append(f"- URL: {item}\n  Expected: {target}")
                else:
                    auth_required.append(f"{item} (error: {exc})")
            except Exception as exc:
                auth_required.append(f"{item} (error: {exc})")
            continue

        doi = item
        slug = slugify(doi)
        try:
            upw = fetch_unpaywall(doi, args.email)
        except Exception as exc:
            auth_required.append(f"{doi} (unpaywall error: {exc})")
            continue

        oa_url: Optional[str] = None
        if upw.get("best_oa_location"):
            oa_url = upw["best_oa_location"].get("url_for_pdf") or upw["best_oa_location"].get("url")
        if not oa_url:
            auth_required.append(f"{doi} (no OA URL)")
            continue

        filename = Path(oa_url).name or "paper.pdf"
        target = out_dir / slug / filename
        try:
            final_url = download(oa_url, target)
            if _looks_like_login(final_url):
                auth_required.append(f"{doi} (login required: redirected to {final_url})")
                queue_entries.append(f"- DOI: {doi}\n  URL: {oa_url}\n  Expected: {target}")
            else:
                fetched.append(str(target))
        except HTTPError as exc:
            if exc.code in {401, 403}:
                auth_required.append(f"{doi} (login required: {oa_url} {exc})")
                queue_entries.append(f"- DOI: {doi}\n  URL: {oa_url}\n  Expected: {target}")
            else:
                auth_required.append(f"{doi} (download error: {exc})")
        except Exception as exc:
            auth_required.append(f"{doi} (download error: {exc})")

    if queue_entries:
        queue_path = out_dir / "download_queue.md"
        queue_text = "# Manual download queue\n\n" + "\n\n".join(queue_entries) + "\n"
        queue_path.write_text(queue_text, encoding="utf-8")

    result = {"fetched": fetched, "auth_required": bool(auth_required), "missing": auth_required}
    print(json.dumps(result, indent=2))
    return 0 if not auth_required else 1


if __name__ == "__main__":
    sys.exit(main())
