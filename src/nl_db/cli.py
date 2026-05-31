from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax

from .config import load_settings
from .formatter import format_as_json, format_as_table
from .llm.registry import build_provider
from .logging import log_pipeline_run
from .pipeline import Pipeline
from .schema.base import render_for_prompt

app = typer.Typer(
    help="Natural language database querying (privacy-first, provider-agnostic).",
    no_args_is_help=True,
    add_completion=False,
)

_console = Console()


def _build_pipeline(
    db: Path,
    limit: int | None,
    paraphrase: bool | None = None,
    lazy_schema: bool | None = None,
) -> tuple[Pipeline, str]:
    settings = load_settings()
    settings.db.path = db
    if limit is not None:
        settings.limits.max_rows = limit
    provider = build_provider(settings)
    gen = settings.generation
    pipe = Pipeline(
        provider=provider,
        db_path=db,
        max_rows=settings.limits.max_rows,
        timeout_s=settings.limits.timeout_s,
        max_prompt_tokens=settings.limits.max_prompt_tokens,
        paraphrase=gen.paraphrase if paraphrase is None else paraphrase,
        temperature=gen.temperature,
        max_output_tokens=gen.max_output_tokens,
        paraphrase_temperature=gen.paraphrase_temperature,
        paraphrase_max_output_tokens=gen.paraphrase_max_output_tokens,
        auto_limit=gen.auto_limit,
        num_few_shot=None if gen.num_few_shot == -1 else gen.num_few_shot,
        lazy_schema=gen.lazy_schema if lazy_schema is None else lazy_schema,
        lazy_max_iterations=gen.lazy_max_iterations,
    )
    return pipe, provider.name


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Natural language question.")],
    db: Annotated[Path, typer.Option("--db", help="Path to SQLite database.", exists=True)],
    allow_writes: Annotated[
        bool,
        typer.Option(
            "--allow-writes",
            help="Permit INSERT/UPDATE/DELETE/DROP/ALTER. Off by default.",
        ),
    ] = False,
    no_confirm: Annotated[
        bool, typer.Option("--no-confirm", help="Skip interactive confirmation.")
    ] = False,
    json_out: Annotated[
        bool, typer.Option("--json", help="Print JSON result instead of a table.")
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            help="Max rows returned. Overrides config; pass 0 to disable.",
        ),
    ] = None,
    no_paraphrase: Annotated[
        bool,
        typer.Option(
            "--no-paraphrase",
            help="Skip the NL-explain-the-SQL step (saves one LLM call).",
        ),
    ] = False,
    lazy_schema: Annotated[
        bool | None,
        typer.Option(
            "--lazy-schema/--no-lazy-schema",
            help=(
                "Use tool-calling to look up the schema on demand instead of "
                "injecting it into every prompt. Falls back to injection if "
                "the provider/model doesn't support tools. Overrides config."
            ),
        ),
    ] = None,
) -> None:
    """Ask the database a question in plain English."""
    from rich.prompt import Prompt

    from .generator import Answer, CannotAnswer, Clarify

    effective_limit = None if limit == 0 else limit
    pipe, provider_name = _build_pipeline(
        db,
        effective_limit,
        paraphrase=not no_paraphrase,
        lazy_schema=lazy_schema,
    )

    def _confirm(sql: str, paraphrase: str | None) -> bool:
        _console.print(
            Panel(
                Syntax(sql, "sql", theme="ansi_dark", word_wrap=True),
                title="[bold]Generated SQL[/bold]",
                border_style="cyan",
            )
        )
        if paraphrase:
            _console.print(
                Panel(
                    paraphrase,
                    title="[bold]In plain English[/bold]",
                    border_style="green",
                )
            )
        if no_confirm:
            return True
        return Confirm.ask("Run this query?", default=True, console=_console)

    # Allow one clarify→retry round-trip. If the model asks the same kind of
    # follow-up twice, that's likely a prompt issue, not a user issue.
    output = pipe.run(question, allow_writes=allow_writes, confirm=_confirm)
    if isinstance(output.outcome, Clarify) and not no_confirm:
        _console.print(
            Panel(
                output.outcome.question,
                title="[bold]Quick clarification[/bold]",
                border_style="yellow",
            )
        )
        clarification = Prompt.ask("Your answer", console=_console)
        combined = f"{question}\n\nClarification: {clarification}"
        output = pipe.run(combined, allow_writes=allow_writes, confirm=_confirm)

    log_path = log_pipeline_run(output, load_settings().log_dir, provider_name)
    _console.print(f"[dim]logged to {log_path}[/dim]")

    if isinstance(output.outcome, CannotAnswer):
        avail = ", ".join(output.outcome.available_tables) or "(none)"
        _console.print(
            Panel(
                f"{output.outcome.reason}\n\n[dim]Available tables: {avail}[/dim]",
                title="[bold]I can't answer that[/bold]",
                border_style="yellow",
            )
        )
        return  # successful refusal, not an error — exit 0

    if isinstance(output.outcome, Clarify):
        # Non-interactive (--no-confirm) or the user gave a clarification that
        # still came back as Clarify — surface and exit nonzero.
        _console.print(
            Panel(
                output.outcome.question,
                title="[bold]Need more information[/bold]",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=2)

    assert isinstance(output.outcome, Answer)

    if output.skipped_reason:
        _console.print(f"[yellow]{output.skipped_reason}[/yellow]")
        raise typer.Exit(code=1)

    assert output.result is not None
    if json_out:
        _console.print_json(format_as_json(output.result, indent=None))
    else:
        _console.print(format_as_table(output.result))
        if output.auto_limit_applied:
            _console.print(
                f"[dim](auto-LIMIT injected; max_rows={pipe._max_rows})[/dim]"
            )
        if output.lazy_attempted:
            if output.lazy_fallback_reason:
                _console.print(
                    f"[dim](lazy schema attempted but fell back: "
                    f"{output.lazy_fallback_reason})[/dim]"
                )
            elif output.agent_run is not None:
                trace = ", ".join(i.name for i in output.agent_run.invocations)
                _console.print(
                    f"[dim](lazy schema: {output.agent_run.iterations} "
                    f"round-trip(s) — tools: {trace or '(none)'})[/dim]"
                )


@app.command(name="schema")
def schema_cmd(
    db: Annotated[Path, typer.Option("--db", exists=True)],
) -> None:
    """Print the inferred schema for a database. Does NOT call any LLM."""
    from .schema.sqlite import SQLiteSchemaExtractor

    schema = SQLiteSchemaExtractor.from_path(db).extract()
    _console.print(Panel(render_for_prompt(schema), title=str(db), border_style="cyan"))


@app.command(name="config")
def config_cmd() -> None:
    """Print the active configuration (with secrets masked)."""
    from .config import default_config_path

    s = load_settings()
    cfg_path = default_config_path()
    cfg_present = "exists" if cfg_path.exists() else "not present (using defaults)"
    masked_key = "set" if s.provider.api_key else "unset"
    g = s.generation
    _console.print(
        Panel(
            f"[bold]config file[/bold]: {cfg_path}  [dim]({cfg_present})[/dim]\n"
            f"\n"
            f"[bold]provider[/bold]:   {s.provider.name}\n"
            f"[bold]model[/bold]:      {s.provider.model}\n"
            f"[bold]base_url[/bold]:   {s.provider.base_url or '(default)'}\n"
            f"[bold]api_key[/bold]:    {masked_key}\n"
            f"[bold]db.path[/bold]:    {s.db.path or '(unset)'}\n"
            f"[bold]limits[/bold]:     max_rows={s.limits.max_rows}, "
            f"timeout_s={s.limits.timeout_s}, max_prompt_tokens={s.limits.max_prompt_tokens}\n"
            f"[bold]generation[/bold]: temperature={g.temperature}, max_tokens={g.max_output_tokens}, "
            f"paraphrase={g.paraphrase} (t={g.paraphrase_temperature}, tok={g.paraphrase_max_output_tokens}), "
            f"auto_limit={g.auto_limit}, few_shot={g.num_few_shot}, "
            f"lazy_schema={g.lazy_schema} (max_iter={g.lazy_max_iterations})\n"
            f"[bold]log_dir[/bold]:    {s.log_dir}",
            title="[bold]nl-db config[/bold]",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    app()
