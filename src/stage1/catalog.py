from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_WINE_FIELDS = (
    "slug",
    "title",
    "manufacturer",
    "region",
    "category",
    "description",
    "image_local",
    "source_url",
)


def load_catalog(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the parser JSON and return root metadata plus wine records."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Catalog root must be an object: {path}")

    wines = payload.get("wines")
    if not isinstance(wines, list):
        raise ValueError(f"Catalog field 'wines' must be an array: {path}")
    if any(not isinstance(wine, dict) for wine in wines):
        raise ValueError(f"Every catalog wine must be an object: {path}")

    metadata = {key: value for key, value in payload.items() if key != "wines"}
    return metadata, wines


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object on {path}:{line_number}")
            records.append(record)
    return records
