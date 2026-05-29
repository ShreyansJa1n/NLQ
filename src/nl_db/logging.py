from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .pipeline import PipelineOutput


def _redact(text: str, max_len: int = 200) -> str:
    """Truncate long strings; replace control chars; never log API keys verbatim."""
    if not text:
        return ""
    clean = "".join(c if c.isprintable() or c in "\n\t" else "?" for c in text)
    if len(clean) > max_len:
        return clean[:max_len] + f"... [+{len(clean) - max_len} chars]"
    return clean


def log_pipeline_run(output: PipelineOutput, log_dir: Path, provider_name: str) -> Path:
    """Append a JSONL record describing this pipeline run.

    Returns the log file path that was written to.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{time.strftime('%Y%m%d')}.jsonl"

    from .generator import CannotAnswer, Clarify

    record: dict[str, Any] = {
        "ts": time.time(),
        "provider": provider_name,
        "state": output.state,
        "question": _redact(output.question, max_len=500),
        "sql_raw": output.sql_raw,
        "sql_final": output.sql_final,
        "paraphrase": output.paraphrase,
        "auto_limit_applied": output.auto_limit_applied,
        "is_destructive": output.is_destructive,
        "confirmed": output.confirmed,
        "approx_prompt_tokens": (
            output.prompt.approx_tokens if output.prompt is not None else None
        ),
        "lazy_attempted": output.lazy_attempted,
        "lazy_iterations": (
            output.agent_run.iterations if output.agent_run is not None else None
        ),
        "lazy_fallback_reason": output.lazy_fallback_reason,
        "row_count": output.result.row_count if output.result else None,
        "truncated": output.result.truncated if output.result else None,
        "pid": os.getpid(),
    }
    if isinstance(output.outcome, CannotAnswer):
        record["cannot_answer_reason"] = output.outcome.reason
    elif isinstance(output.outcome, Clarify):
        record["clarify_question"] = output.outcome.question

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return log_path
