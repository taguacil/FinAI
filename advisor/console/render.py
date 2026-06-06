"""Rich-based output helpers."""

from __future__ import annotations

from typing import Iterable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table


class Renderer:
    def __init__(self) -> None:
        self.console = Console()

    def banner(self, title: str, subtitle: str = "") -> None:
        self.console.print(Panel.fit(f"[bold cyan]{title}[/]\n{subtitle}", border_style="cyan"))

    def info(self, msg: str) -> None:
        self.console.print(f"[dim]{msg}[/]")

    def warn(self, msg: str) -> None:
        self.console.print(f"[yellow]! {msg}[/]")

    def error(self, msg: str) -> None:
        self.console.print(f"[red]✗ {msg}[/]")

    def section(self, title: str, body: str) -> None:
        self.console.print(Panel(Markdown(body or "(empty)"), title=title, border_style="blue"))

    def tools_table(self, allowed: Iterable[str], dropped: Iterable[str]) -> None:
        t = Table(title="MCP tools", show_lines=False)
        t.add_column("Status")
        t.add_column("Tool")
        for name in sorted(allowed):
            t.add_row("[green]allow[/]", name)
        for name in sorted(dropped):
            t.add_row("[red]block[/]", name)
        self.console.print(t)
