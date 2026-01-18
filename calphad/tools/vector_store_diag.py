#!/usr/bin/env python3
"""Connectivity check for OpenAI vector stores using env vars."""
from __future__ import annotations

import os
import sys

from openai import OpenAI


def _env_presence(name: str) -> str:
    return "true" if os.environ.get(name) else "false"


def _print_header(title: str) -> None:
    print(f"\n## {title}\n")


def main() -> int:
    _print_header("Environment presence (no secrets)")
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_VECTOR_STORE_ID"):
        print(f"{name}: {_env_presence(name)}")

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")

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

    try:
        files = client.vector_stores.files.list(vector_store_id)
    except Exception as exc:  # noqa: BLE001 - surface API errors clearly
        print(f"Failed to list vector store files: {exc}")
        return 1

    print(f"file_count: {len(files.data)}")
    if files.data:
        print("file_ids:")
        for file_item in files.data:
            print(f"- {file_item.id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
