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


def _resolve_student_dir(cfg: dict, override: Path | None = None) -> Path:
    """Single source of truth for the student checkpoint directory.

    Priority: explicit CLI override → ``config.output_dir`` → ``outputs/student``.
    Used by both ``train_cmd`` and ``eval_cmd`` so run→eval never disagree.
    """
    if override is not None:
        return Path(override)
    return Path(cfg.get("output_dir") or "outputs/student")


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
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Student checkpoint dir (default: config.output_dir or outputs/student)",
        ),
    ] = None,
) -> None:
    """Stage 2: run real distillation training (``unme.train.distill.train``)."""
    import tempfile

    import yaml

    cfg = _load_yaml(config) if config.exists() else {}
    data_cfg = cfg.setdefault("data", {})
    if data is not None:
        data_cfg["filtered"] = str(data)
    data_path = Path(data_cfg.get("filtered", "data/filtered"))
    student_dir = _resolve_student_dir(cfg, out)
    cfg["output_dir"] = str(student_dir)
    console.print(f"[bold]train[/bold] data={data_path} out={student_dir}")

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
    epoch_losses = summary.get("epoch_losses") or []
    n_steps = int(summary.get("n_steps") or len(hist))
    first = hist[0] if hist else None
    last = hist[-1] if hist else None
    console.print(
        f"[green]train done[/green] n_steps={n_steps} "
        f"loss_first={first} loss_last={last} "
        f"output_dir={summary.get('output_dir')}"
    )
    if epoch_losses:
        # e.g. "epochs: 3.41 → 2.98 → 2.55"
        trend = " → ".join(f"{x:.4g}" for x in epoch_losses)
        console.print(f"epochs: {trend}")


@app.command("inspect")
def inspect_cmd(
    traces: Annotated[
        Path,
        typer.Option("--traces", "-t", help="Trace JSONL file or directory"),
    ] = _DEFAULT_TRACES,
    max_positions: Annotated[
        int,
        typer.Option("--max-positions", help="Cap printed output positions"),
    ] = 64,
) -> None:
    """Pretty-print the first Trace's teacher top-k distribution (token ids + probs)."""
    from unme.inspect import inspect_traces

    console.print(f"[bold]inspect[/bold] traces={traces}")
    try:
        table = inspect_traces(traces, max_positions=max_positions)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]inspect failed:[/red] {e}")
        raise typer.Exit(1) from e
    console.print(table)


def load_student_callable(
    checkpoint: str | Path,
    *,
    max_new_tokens: int = 64,
):
    """Build a ``(prompt: str) -> str`` callable from an HF causal LM checkpoint.

    Separated for tests (monkeypatch with a stub). Requires torch/transformers.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = str(checkpoint)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    model.eval()

    def student(prompt: str) -> str:
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = out[0, enc["input_ids"].shape[1] :]
        return tok.decode(new_tokens, skip_special_tokens=True)

    return student


@app.command("eval")
def eval_cmd(
    candidate: Annotated[str, typer.Argument(help="Candidate name / checkpoint id")],
    config: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG,
    registry: Annotated[Path, typer.Option("--registry", "-r")] = _DEFAULT_REGISTRY,
    student_path: Annotated[
        Path | None,
        typer.Option(
            "--student",
            help="HF checkpoint dir (default: config output_dir or outputs/student)",
        ),
    ] = None,
) -> None:
    """Stage 0/3: score the trained student on data/eval and write a GateReport."""
    from unme.eval.harness import evaluate
    from unme.registry import Registry

    cfg = _load_yaml(config) if config.exists() else {}
    eval_cfg = cfg.get("eval") or {}
    floor = float(eval_cfg.get("regression_floor", 0.98))
    suite = Path(eval_cfg.get("suite", "data/eval"))
    teacher_scores = eval_cfg.get("teacher_scores") or {"cs": 1.0}
    # Normalize keys to str, values to float
    teacher_scores = {str(k): float(v) for k, v in dict(teacher_scores).items()}

    ckpt = _resolve_student_dir(cfg, student_path)
    console.print(
        f"[bold]eval[/bold] candidate={candidate} floor={floor} "
        f"suite={suite} student={ckpt}"
    )

    try:
        student = load_student_callable(ckpt)
    except Exception as e:
        console.print(f"[red]failed to load student from {ckpt}:[/red] {e}")
        raise typer.Exit(1) from e

    report = evaluate(
        student,
        teacher_scores,
        suite,
        floor,
        candidate=candidate,
    )
    reg = Registry(registry)
    path = reg.record(candidate, report)
    for r in report.results:
        console.print(
            f"  domain={r.domain} student={r.student_score:.4f} "
            f"teacher={r.teacher_score:.4f} ratio={r.ratio:.4f}"
        )
    console.print(f"Wrote {path} passed={report.passed}")
    if not report.passed:
        console.print(
            "[yellow]Gate not passed (real ratios below floor). "
            "Improve student or lower eval.regression_floor before promote.[/yellow]"
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

    # 3) train (or dataset-load only) — config.output_dir is the single source of truth
    if skip_train:
        console.print("[yellow]--skip-train: omitting train_cmd[/yellow]")
        _dataset_load_smoke(config, data=filtered_dir)
    else:
        train_cmd(config=config, data=filtered_dir)

    # 4) eval — same student dir resolution as train
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
