"""Typer CLI entrypoint (`unme`). Subcommands wire pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(
    name="unme",
    help="Lab UnMe — open-weight distillation pipeline (Kimi K3 → student)",
    no_args_is_help=True,
)
console = Console()

_DEFAULT_CONFIG = Path("configs/distill.yaml")
_DEFAULT_TRACES = Path("data/traces")
_DEFAULT_FILTERED = Path("data/filtered")
_DEFAULT_OUT = Path("outputs/student")
_DEFAULT_REGISTRY = Path("outputs/registry")


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open() as f:
        return yaml.safe_load(f) or {}


@app.command("generate")
def generate_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG,
    prompts: Annotated[
        Path | None, typer.Option(help="Prompt JSONL (overrides config)")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Output traces dir")] = None,
) -> None:
    """Stage 1: run teacher to emit Trace JSONL with top-k logits."""
    cfg = _load_yaml(config) if config.exists() else {}
    traces_dir = out or Path((cfg.get("data") or {}).get("traces", "data/traces"))
    prompts_path = prompts or Path(
        (cfg.get("data") or {}).get("prompts", "data/prompts/pilot.jsonl")
    )
    console.print(f"[bold]generate[/bold] prompts={prompts_path} → {traces_dir}")
    traces_dir.mkdir(parents=True, exist_ok=True)
    try:
        from unme.teacher.generate import generate as run_generate  # GLM Task 2
    except ImportError:
        console.print(
            "[yellow]teacher.generate not implemented yet (GLM Task 2). It must emit "
            "unme.schemas.Trace JSONL (top-k logits) into the traces dir.[/yellow]"
        )
        raise typer.Exit(0)
    run_generate(str(prompts_path), str(traces_dir), cfg)


@app.command("filter")
def filter_cmd(
    traces: Annotated[Path, typer.Option("--traces", "-t")] = _DEFAULT_TRACES,
    out: Annotated[Path, typer.Option("--out", "-o")] = _DEFAULT_FILTERED,
    domain: Annotated[
        str | None,
        typer.Option(help="Hint: math|code — selects default verifiers"),
    ] = None,
) -> None:
    """Stage 1b: strict synth filter → kept.jsonl + verdicts.jsonl."""
    from unme.synth.filter import filter_traces
    from unme.verify import CodeVerifier, MathVerifier

    verifiers = []
    d = (domain or "").lower()
    if d in {"math", "stem", ""}:
        verifiers.append(MathVerifier())
    if d in {"code", "cs", "coding", ""}:
        verifiers.append(CodeVerifier())

    console.print(
        f"[bold]filter[/bold] {traces} → {out} "
        f"verifiers={[type(v).__name__ for v in verifiers]}"
    )
    verdicts = filter_traces(traces, out, verifiers=verifiers)
    n_keep = sum(1 for v in verdicts if v.keep)
    console.print(f"kept={n_keep} / considered_verdicts={len(verdicts)}")


@app.command("train")
def train_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG,
    data: Annotated[
        Path | None, typer.Option(help="Filtered traces JSONL/dir")
    ] = None,
    out: Annotated[Path, typer.Option("--out")] = _DEFAULT_OUT,
) -> None:
    """Stage 2: run real distillation training (``unme.train.distill.train``)."""
    import tempfile

    import yaml

    cfg = _load_yaml(config) if config.exists() else {}
    data_cfg = cfg.setdefault("data", {})
    if data is not None:
        data_cfg["filtered"] = str(data)
    data_path = Path(data_cfg.get("filtered", "data/filtered"))
    cfg["output_dir"] = str(out)
    console.print(f"[bold]train[/bold] data={data_path} out={out}")

    kept = data_path / "kept.jsonl" if data_path.is_dir() else data_path
    if not Path(kept).exists():
        console.print(f"[red]No training data at {kept}. Run `unme filter` first.[/red]")
        raise typer.Exit(1)

    try:
        from unme.train.distill import train as run_distill
    except ImportError:
        console.print(
            "[red]torch/train stack not installed. "
            "pip install 'lab-unme[train]' to run distillation.[/red]"
        )
        raise typer.Exit(1)

    # Merge CLI overrides into a working config file for train().
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.safe_dump(cfg, tmp)
        work_cfg = Path(tmp.name)
    try:
        summary = run_distill(work_cfg)
    finally:
        work_cfg.unlink(missing_ok=True)

    hist = summary.get("loss_history") or []
    n_steps = int(summary.get("n_steps") or len(hist))
    first = hist[0] if hist else None
    last = hist[-1] if hist else None
    console.print(
        f"[green]train done[/green] n_steps={n_steps} "
        f"loss_first={first} loss_last={last} "
        f"output_dir={summary.get('output_dir')}"
    )


@app.command("eval")
def eval_cmd(
    candidate: Annotated[str, typer.Argument(help="Candidate name / checkpoint id")],
    config: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG,
    registry: Annotated[Path, typer.Option("--registry", "-r")] = _DEFAULT_REGISTRY,
) -> None:
    """Stage 0/3: evaluate candidate, write GateReport into the registry."""
    from unme.registry import Registry
    from unme.schemas import EvalResult, GateReport

    cfg = _load_yaml(config) if config.exists() else {}
    floor = float((cfg.get("eval") or {}).get("regression_floor", 0.98))

    console.print(f"[bold]eval[/bold] candidate={candidate} floor={floor}")
    report = GateReport(
        candidate=candidate,
        results=[
            EvalResult(domain="math", metric="holdout", student_score=0.0, teacher_score=1.0),
        ],
        regression_floor=floor,
    )
    reg = Registry(registry)
    path = reg.record(candidate, report)
    console.print(f"Wrote {path} passed={report.passed}")
    if not report.passed:
        console.print(
            "[yellow]Gate not passed (placeholder scores). "
            "Record real eval results before promote.[/yellow]"
        )


@app.command("promote")
def promote_cmd(
    candidate: Annotated[str, typer.Argument(help="Candidate to promote")],
    registry: Annotated[Path, typer.Option("--registry", "-r")] = _DEFAULT_REGISTRY,
) -> None:
    """Promote a candidate only if its GateReport.passed is True."""
    from unme.registry import PromotionError, Registry

    reg = Registry(registry)
    try:
        path = reg.promote(candidate)
    except PromotionError as e:
        console.print(f"[red]promote refused:[/red] {e}")
        raise typer.Exit(1) from e
    console.print(f"[green]promoted[/green] {candidate} → {path}")


def _dataset_load_smoke(config: Path, data: Path | None = None) -> None:
    """Load DistillDataset only (no training). Safe without GPU; needs torch for tensors."""
    cfg = _load_yaml(config) if config.exists() else {}
    data_path = data or Path((cfg.get("data") or {}).get("filtered", "data/filtered"))
    kept = data_path / "kept.jsonl" if data_path.is_dir() else data_path
    if not Path(kept).exists():
        console.print(f"[red]No filtered data at {kept} for dataset load.[/red]")
        raise typer.Exit(1)
    try:
        from unme.data.dataset import DistillDataset

        ds = DistillDataset(kept if Path(kept).suffix == ".jsonl" else data_path)
        console.print(f"[bold]dataset[/bold] loaded n={len(ds)} from {kept}")
    except ImportError:
        console.print(
            "[yellow]torch not installed; skip DistillDataset load "
            "(install 'lab-unme[train]' for full smoke).[/yellow]"
        )


@app.command("run")
def run_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG,
    skip_train: Annotated[
        bool,
        typer.Option("--skip-train", help="Skip train; still smoke-load DistillDataset"),
    ] = False,
    skip_generate: Annotated[
        bool,
        typer.Option(
            "--skip-generate",
            help="Skip teacher generate; reuse existing Trace JSONL under data.traces",
        ),
    ] = False,
    candidate: Annotated[
        str,
        typer.Option("--candidate", help="Name recorded under the registry"),
    ] = "unme-run",
    registry: Annotated[Path, typer.Option("--registry", "-r")] = _DEFAULT_REGISTRY,
    domain: Annotated[
        str | None,
        typer.Option(help="Filter domain hint (math|code|cs)"),
    ] = "cs",
) -> None:
    """Chain generate → filter → train → eval → promote from distill.yaml.

    With ``--skip-train``, training is skipped (no GPU required) but filtered
    traces are still loaded via DistillDataset when torch is available.

    With ``--skip-generate``, reuse existing traces (no teacher endpoint needed).
    """
    cfg = _load_yaml(config) if config.exists() else {}
    data_cfg = cfg.get("data") or {}
    traces_dir = Path(data_cfg.get("traces", "data/traces"))
    filtered_dir = Path(data_cfg.get("filtered", "data/filtered"))
    prompts_path = Path(data_cfg.get("prompts", "data/prompts/pilot.jsonl"))

    console.rule("[bold]unme run")
    console.print(
        f"config={config} skip_train={skip_train} "
        f"skip_generate={skip_generate} candidate={candidate}"
    )

    # 1) generate (unless --skip-generate)
    if skip_generate:
        console.print(
            f"[yellow]--skip-generate: reusing traces in {traces_dir}[/yellow]"
        )
        if not traces_dir.exists() or not any(traces_dir.glob("*.jsonl")):
            console.print(
                f"[red]No Trace JSONL under {traces_dir}. "
                "Run generate first or point data.traces at existing traces.[/red]"
            )
            raise typer.Exit(1)
    else:
        base_url = str((cfg.get("teacher") or {}).get("base_url") or "").strip()
        if not base_url:
            console.print(
                f"unme run needs a teacher endpoint. Set teacher.base_url in {config}\n"
                f"(see configs/distill.real.yaml), or pass --skip-generate to reuse existing\n"
                f"traces in {traces_dir}."
            )
            raise typer.Exit(1)
        generate_cmd(config=config, prompts=prompts_path, out=traces_dir)

    # 2) filter
    filter_cmd(traces=traces_dir, out=filtered_dir, domain=domain)

    # 3) train (or dataset-load only)
    if skip_train:
        console.print("[yellow]--skip-train: omitting train_cmd[/yellow]")
        _dataset_load_smoke(config, data=filtered_dir)
    else:
        train_cmd(config=config, data=filtered_dir, out=_DEFAULT_OUT)

    # 4) eval
    eval_cmd(candidate=candidate, config=config, registry=registry)

    # 5) promote (may refuse until real eval scores pass the gate)
    try:
        promote_cmd(candidate=candidate, registry=registry)
    except typer.Exit as exc:
        if getattr(exc, "exit_code", 1) not in (0, None):
            console.print(
                "[yellow]promote refused (gate). Pipeline stages above still completed.[/yellow]"
            )
            if not skip_train:
                raise
        # with --skip-train, placeholder eval often fails the floor; do not fail the chain


if __name__ == "__main__":
    app()
