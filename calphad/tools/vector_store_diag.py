#!/usr/bin/env python3
"""Connectivity check for OpenAI vector stores using env vars."""
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MISSING_DEPS_FILE = REPO_ROOT / ".missing_deps.txt"


def _env_presence(name: str) -> str:
    return "true" if os.environ.get(name) else "false"


def _print_header(title: str) -> None:
    print(f"\n## {title}\n")


def _load_missing_deps() -> list[str]:
    if not MISSING_DEPS_FILE.exists():
        return []
    return [
        line.strip()
        for line in MISSING_DEPS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _persist_missing_deps(deps: list[str]) -> None:
    unique = sorted(set(deps))
    MISSING_DEPS_FILE.write_text("\n".join(unique) + "\n", encoding="utf-8")


def _attempt_auto_install(deps: list[str]) -> bool:
    if not deps:
        return False
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", *deps],
        check=False,
    )
    if result.returncode == 0:
        MISSING_DEPS_FILE.unlink(missing_ok=True)
        return True
    return False


def main() -> int:
    missing_deps = _load_missing_deps()
    if missing_deps:
        _print_header("Dependency auto-install")
        if _attempt_auto_install(missing_deps):
            print("Auto-install succeeded.")
        else:
            print("Auto-install failed. Please install dependencies manually.")

    if importlib.util.find_spec("openai") is None:
        print("Missing dependency: openai")
        print("Install with: pip install -r requirements.txt")
        _persist_missing_deps(["openai"])
        print(f"Recorded missing dependencies in: {MISSING_DEPS_FILE}")
        return 2

    OpenAI = importlib.import_module("openai").OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    env_base_url = os.environ.get("OPENAI_BASE_URL")
    base_url = "https://api.openai.com/v1"
    vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")

    _print_header("Environment presence (no secrets)")
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_VECTOR_STORE_ID"):
        print(f"{name}: {_env_presence(name)}")

    _print_header("Environment values (no secrets)")
    print(f"OPENAI_BASE_URL (env): {env_base_url}")
    print(f"OPENAI_BASE_URL (effective): {base_url}")
    print(f"OPENAI_VECTOR_STORE_ID: {vector_store_id}")

    missing = [name for name, value in (
        ("OPENAI_API_KEY", api_key),
        ("OPENAI_VECTOR_STORE_ID", vector_store_id),
    ) if not value]

    if missing:
        print("\nMissing required environment variables:")
        for name in missing:
            print(f"- {name}")
        return 2

    _print_header("Vector store connectivity")
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        vector_store = client.vector_stores.retrieve(vector_store_id)
    except Exception as exc:  # noqa: BLE001 - surface API errors clearly
        print(f"Failed to retrieve vector store: {exc}")
        return 1

    print(f"vector_store_id: {vector_store.id}")
    print(f"name: {vector_store.name}")
    print(f"status: {vector_store.status}")

    all_files = []
    next_page = None
    while True:
        try:
            files = client.vector_stores.files.list(
                vector_store_id,
                limit=100,
                after=next_page,
            )
        except Exception as exc:  # noqa: BLE001 - surface API errors clearly
            print(f"Failed to list vector store files: {exc}")
            return 1

        all_files.extend(files.data)
        if not files.has_more:
            break
        next_page = files.last_id

    print(f"file_count: {len(all_files)}")
    if all_files:
        print("files:")
        for file_item in all_files:
            file_name = "<unknown>"
            try:
                file_details = client.files.retrieve(file_item.id)
            except Exception as exc:  # noqa: BLE001 - surface API errors clearly
                print(f"- {file_item.id}: failed to retrieve metadata: {exc}")
                continue
            file_name = file_details.filename or file_name
            print(f"- {file_item.id}: {file_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
