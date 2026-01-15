#!/usr/bin/env python3
"""Network diagnostics for Crossref/OpenAlex/Unpaywall access."""
from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess
import sys


def _env_presence(name: str) -> str:
    return "true" if os.environ.get(name) else "false"


def _print_header(title: str) -> None:
    print(f"\n## {title}\n")


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())


def _curl_probe(url: str) -> None:
    cmd = [
        "curl",
        "-sS",
        "-L",
        "-D",
        "-",
        "-o",
        "/dev/null",
        "-w",
        "HTTP_STATUS:%{http_code}\nURL_EFFECTIVE:%{url_effective}\n",
        url,
    ]
    _run(cmd)


def main() -> int:
    _print_header("Runtime")
    print(f"time_utc: {dt.datetime.utcnow().isoformat()}Z")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")

    _print_header("Environment presence (no secrets)")
    for name in ("CROSSREF_MAILTO", "OPENALEX_MAILTO", "OPENALEX_API_KEY", "UNPAYWALL_EMAIL"):
        print(f"{name}: {_env_presence(name)}")

    crossref_mailto = os.environ.get("CROSSREF_MAILTO", "")
    openalex_mailto = os.environ.get("OPENALEX_MAILTO", "")
    openalex_key = os.environ.get("OPENALEX_API_KEY", "")

    _print_header("DNS probes")
    for host in ("api.crossref.org", "api.openalex.org", "api.unpaywall.org"):
        _run(["getent", "hosts", host])

    _print_header("Crossref probes")
    _curl_probe(f"https://api.crossref.org/works?query=fe-b&rows=1&mailto={crossref_mailto}")
    _curl_probe("https://api.crossref.org/works?query=fe-b&rows=1")

    _print_header("OpenAlex probes")
    _curl_probe(f"https://api.openalex.org/works?search=fe-b&per-page=1&mailto={openalex_mailto}")
    if openalex_key:
        _curl_probe(
            "https://api.openalex.org/works?search=fe-b&per-page=1"
            f"&mailto={openalex_mailto}&api_key={openalex_key}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
