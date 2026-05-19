from __future__ import annotations

from pathlib import Path

from eval.runner import (
    CaseResult,
    EvalCase,
    _load_cases,
    _score,
    write_report,
)

DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "dataset.yaml"


def test_dataset_loads_30_well_formed_cases() -> None:
    cases = _load_cases(DATASET_PATH)
    assert len(cases) == 30
    ids = [c.id for c in cases]
    assert len(set(ids)) == 30, "case ids must be unique"

    for c in cases:
        assert c.nl, f"case {c.id} missing nl"
        has_expectation = (
            c.expected_rows is not None
            or c.expected_sql_pattern is not None
            or c.expected_row_count is not None
        )
        assert has_expectation, f"case {c.id} has no expectations"


def test_score_exact_rows_unordered() -> None:
    case = EvalCase(id="t", nl="x", expected_rows=[["a"], ["b"]])
    ok, _ = _score(case, "SELECT name FROM users", [("b",), ("a",)])
    assert ok is True


def test_score_exact_rows_ordered_fails_on_reorder() -> None:
    case = EvalCase(id="t", nl="x", expected_rows=[["a"], ["b"]], ordered=True)
    ok, reason = _score(case, "SELECT name FROM users", [("b",), ("a",)])
    assert ok is False
    assert "mismatch" in reason


def test_score_sql_pattern_match() -> None:
    case = EvalCase(id="t", nl="x", expected_sql_pattern="(?is)GROUP\\s+BY")
    ok, _ = _score(case, "select user, count(*) from t group by user", [])
    assert ok is True


def test_score_sql_pattern_with_row_count_constraint() -> None:
    case = EvalCase(
        id="t",
        nl="x",
        expected_sql_pattern="(?is)COUNT\\(",
        expected_row_count=1,
    )
    ok_one, _ = _score(case, "SELECT COUNT(*) FROM t", [(3,)])
    assert ok_one is True
    ok_zero, reason = _score(case, "SELECT COUNT(*) FROM t", [(3,), (4,)])
    assert ok_zero is False
    assert "row count" in reason


def test_score_row_count_only() -> None:
    case = EvalCase(id="t", nl="x", expected_row_count=5)
    ok_pass, _ = _score(case, "SELECT 1", [(i,) for i in range(5)])
    assert ok_pass is True
    ok_fail, _ = _score(case, "SELECT 1", [(1,)])
    assert ok_fail is False


def test_write_report_writes_markdown(tmp_path: Path) -> None:
    case = EvalCase(id="q01", nl="list users", expected_rows=[["a"]])
    result = CaseResult(
        case=case,
        generated_sql="SELECT name FROM users",
        passed=True,
        reason="exact-row match (unordered)",
        row_count=1,
        duration_s=0.5,
        metadata={"provider": "fake", "model": "fake-1"},
    )
    out = tmp_path / "report.md"
    write_report([result], out)
    text = out.read_text()
    assert "nl-db eval report" in text
    assert "1/1 (100.0%)" in text
    assert "q01" in text
    assert "✅" in text
