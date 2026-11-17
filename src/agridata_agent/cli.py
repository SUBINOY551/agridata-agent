from pathlib import Path
import typer
from rich.console import Console
from rich.markdown import Markdown
from agridata_agent.agent import run_agent
from agridata_agent.tools.validator import audit_dataset

app = typer.Typer(
    name="agridata",
    help="Autonomous data-engineering and visual-QA agent for agricultural CV datasets.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def audit(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to dataset directory"),
    json_out: str = typer.Option("", "--json-out", "-j", help="Optional path to save raw JSON report"),
):
    """Run deterministic dataset pairing, class-imbalance, and bounding-box geometry audit."""
    report = audit_dataset(dataset)
    if json_out:
        Path(json_out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Saved JSON audit report to {json_out}[/green]")
    console.print_json(report.model_dump_json(indent=2))


@app.command()
def run(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to dataset directory"),
    output: str = typer.Option("./splits", "--output", "-o", help="Output directory for manifests & QA grid"),
    objective: str = typer.Option(
        "Audit dataset health, generate visual QA contact sheet, and build stratified splits.",
        "--objective",
        "-q",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Run deterministic multi-tool pipeline without external LLM API calls",
    ),
    report_file: str = typer.Option("audit_report.md", "--report", "-r", help="Output Markdown report path"),
):
    """Execute the autonomous AgriData-Agent workflow."""
    console.rule("[bold green]AgriData-Agent Execution[/bold green]")
    markdown_report = run_agent(
        dataset_dir=dataset,
        output_dir=output,
        objective=objective,
        offline=offline,
    )
    Path(report_file).write_text(markdown_report, encoding="utf-8")
    console.print(Markdown(markdown_report))
    console.print(f"\n[bold cyan]Full report written to {report_file}[/bold cyan]")


if __name__ == "__main__":
    app()
