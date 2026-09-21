from __future__ import annotations

import argparse
import json
import mimetypes
import statistics
import time
from pathlib import Path

import httpx

from stage3.recognize import collect_paths


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequentially benchmark the Stage 6 HTTP API.")
    parser.add_argument("path", type=Path, help="Image file or directory")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--sla-ms", type=float, default=10_000.0)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = collect_paths(args.path)[: args.limit]
    if not paths:
        raise ValueError(f"No supported images found: {args.path}")
    timings: list[float] = []
    records = []
    with httpx.Client(base_url=args.base_url, timeout=args.timeout, trust_env=False) as client:
        for index, path in enumerate(paths):
            started = time.perf_counter()
            with path.open("rb") as handle:
                response = client.post(
                    "/api/v1/recognize",
                    headers={"X-Session-ID": "api-benchmark-local"},
                    files={
                        "file": (
                            path.name,
                            handle,
                            mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                        )
                    },
                )
            elapsed_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            timings.append(elapsed_ms)
            records.append(
                {
                    "index": index,
                    "image": path.as_posix(),
                    "slug": response.json()["slug"],
                    "wall_ms": elapsed_ms,
                }
            )
            print(f"{index + 1:>3}/{len(paths)}  {elapsed_ms:>8.1f} ms  {path.name}")
    report = {
        "schema_version": 1,
        "base_url": args.base_url,
        "count": len(timings),
        "sla_ms": args.sla_ms,
        "timing_ms": {
            "mean": statistics.fmean(timings),
            "p50": percentile(timings, 0.50),
            "p95": percentile(timings, 0.95),
            "max": max(timings),
        },
        "within_sla": sum(value <= args.sla_ms for value in timings),
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0 if report["within_sla"] == len(timings) else 2


if __name__ == "__main__":
    raise SystemExit(main())
