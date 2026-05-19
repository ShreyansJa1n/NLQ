"""Evaluation harness for the NL→SQL pipeline against the sample database.

Usage:
    uv run python -m eval.runner --provider anthropic
    uv run python -m eval.runner --limit 5            # smoke test 5 questions
    uv run python -m eval.runner --report path.md     # custom output

Reads `eval/dataset.yaml`, runs each NL question through the configured
pipeline, scores the result, and writes a Markdown report to
`eval/reports/<timestamp>.md`.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nl_db.config import load_settings  # noqa: E402
from nl_db.llm.registry import build_provider  # noqa: E402
from nl_db.pipeline import Pipeline  # noqa: E402
from tests.fixtures.build_sample_db import SAMPLE_DB  # noqa: E402
from tests.fixtures.build_sample_db import build as build_sample_db  # noqa: E402

DATASET = Path(__file__).parent / "dataset.yaml"
REPORTS_DIR = Path(__file__).parent / "reports"


@dataclass
class EvalCase:
    id: str
    nl: str
    expected_rows: list[list[Any]] | None = None
    ordered: bool = False
    expected_row_count: int | None = None
    expected_sql_pattern: str | None = None


@dataclass
class CaseResult:
    case: EvalCase
    generated_sql: str
    passed: bool
    reason: str
    row_count: int | None = None
    duration_s: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_cases(path: Path) -> list[EvalCase]:
    raw = yaml.safe_load(path.read_text())
    cases: list[EvalCase] = []
    for entry in raw:
        cases.append(
            EvalCase(
                id=entry["id"],
                nl=entry["nl"],
                expected_rows=entry.get("expected_rows"),
                ordered=bool(entry.get("ordered", False)),
                expected_row_count=entry.get("expected_row_count"),
                expected_sql_pattern=entry.get("expected_sql_pattern"),
            )
        )
    return cases


def _normalize_rows(rows: list[Any]) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in rows]


def _score(case: EvalCase, sql: str, rows: list[tuple[Any, ...]]) -> tuple[bool, str]:
    if case.expected_rows is not None:
        expected = _normalize_rows(case.expected_rows)
        if case.ordered:
            ok = rows == expected
            reason = "exact-row match (ordered)" if ok else (
                f"row mismatch: got {rows[:5]}{'...' if len(rows) > 5 else ''}, "
                f"expected {expected[:5]}"
            )
            return ok, reason
        ok = sorted(rows) == sorted(expected)
        reason = "exact-row match (unordered)" if ok else (
            f"row set mismatch: got {sorted(rows)[:5]}, expected {sorted(expected)[:5]}"
        )
        return ok, reason

    if case.expected_sql_pattern is not None:
        if not re.search(case.expected_sql_pattern, sql):
            return False, f"SQL pattern not matched: {case.expected_sql_pattern!r}"
        if case.expected_row_count is not None and len(rows) != case.expected_row_count:
            return False, (
                f"SQL pattern matched but row count off: "
                f"got {len(rows)}, expected {case.expected_row_count}"
            )
        return True, "SQL pattern matched"

    if case.expected_row_count is not None:
        ok = len(rows) == case.expected_row_count
        return ok, (
            f"row count match: {len(rows)}" if ok
            else f"row count mismatch: got {len(rows)}, expected {case.expected_row_count}"
        )

    return False, "case has no expectations defined"


def run_eval(
    *,
    db_path: Path,
    provider_override: str | None,
    limit: int | None,
) -> list[CaseResult]:
    if not db_path.exists():
        build_sample_db(db_path, overwrite=False)

    settings = load_settings()
    if provider_override:
        settings.provider.name = provider_override  # type: ignore[assignment]
    provider = build_provider(settings)
    pipe = Pipeline(
        provider=provider,
        db_path=db_path,
        max_rows=settings.limits.max_rows,
        timeout_s=settings.limits.timeout_s,
        paraphrase=False,  # eval doesn't need NL paraphrase
    )

    cases = _load_cases(DATASET)
    if limit is not None:
        cases = cases[:limit]

    results: list[CaseResult] = []
    for case in cases:
        start = time.monotonic()
        try:
            out = pipe.run(case.nl)
            duration = time.monotonic() - start
            assert out.result is not None
            passed, reason = _score(case, out.sql_final, out.result.rows)
            results.append(
                CaseResult(
                    case=case,
                    generated_sql=out.sql_final,
                    passed=passed,
                    reason=reason,
                    row_count=out.result.row_count,
                    duration_s=duration,
                    metadata={"provider": provider.name, "model": provider.model},
                )
            )
            status = "OK " if passed else "FAIL"
            print(f"[{status}] {case.id}  ({duration:.1f}s)  {reason}", flush=True)
        except Exception as e:  # noqa: BLE001
            duration = time.monotonic() - start
            results.append(
                CaseResult(
                    case=case,
                    generated_sql="",
                    passed=False,
                    reason=f"exception: {type(e).__name__}: {e}",
                    duration_s=duration,
                    error=str(e),
                    metadata={"provider": provider.name, "model": provider.model},
                )
            )
            print(f"[ERR ] {case.id}  ({duration:.1f}s)  {e}", flush=True)

    return results


def write_report(results: list[CaseResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pct = (passed / total * 100) if total else 0.0
    meta = results[0].metadata if results else {}

    lines = [
        f"# nl-db eval report — {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- Provider: `{meta.get('provider', '?')}`",
        f"- Model:    `{meta.get('model', '?')}`",
        f"- Score:    **{passed}/{total} ({pct:.1f}%)**",
        "",
        "## Per-question results",
        "",
        "| ID | Pass | Time (s) | Question | Generated SQL | Reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        sql_one_line = " ".join(r.generated_sql.split())
        nl_short = r.case.nl.replace("|", "\\|")
        reason = r.reason.replace("|", "\\|")
        lines.append(
            f"| {r.case.id} | "
            f"{'✅' if r.passed else '❌'} | "
            f"{r.duration_s:.1f} | "
            f"{nl_short} | "
            f"`{sql_one_line[:120]}` | "
            f"{reason} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", help="Override configured provider name.")
    parser.add_argument("--db", type=Path, default=SAMPLE_DB)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    report_path = args.report or REPORTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}.md"
    results = run_eval(
        db_path=args.db, provider_override=args.provider, limit=args.limit
    )
    write_report(results, report_path)
    print(f"\nWrote report to {report_path}")
    failures = sum(1 for r in results if not r.passed)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
