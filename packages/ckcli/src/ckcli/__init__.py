"""ckcli — command-line interface for CorridorKey."""

from __future__ import annotations

import sys

import typer

from ckcli.commands.config import app as config_app
from ckcli.commands.doctor import doctor
from ckcli.commands.init import init
from ckcli.commands.process import process
from ckcli.commands.reset import reset
from ckcli.commands.scan import scan
from ckcli.commands.wizard import wizard

app = typer.Typer(
    name="ck",
    help="CorridorKey — AI green screen keyer. Run with no arguments to launch the wizard.",
    add_completion=False,
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """Launch the interactive wizard when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        wizard()


app.command("init")(init)
app.command("doctor")(doctor)
app.command("wizard")(wizard)
app.command("process")(process)
app.command("scan")(scan)
app.command("reset")(reset)
app.add_typer(config_app, name="config")


def main() -> None:
    """Entry point for the ``ck`` console script."""
    try:
        app()
    except KeyboardInterrupt:
        from ckcli._console import console

        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
