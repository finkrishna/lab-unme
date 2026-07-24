"""Distillation training loop: HF student + top-k KL distillation.

``train(config_path)`` loads ``configs/distill.yaml``, builds a HuggingFace student,
streams ``unme.data.dataset.DistillDataset`` batches, calls ``combined_distill_loss``
and runs an AdamW + grad-clip loop. One ``nn.Linear`` projection is learned per
``distill.hidden_layer_map`` pair to align student/teacher hidden dimensions when the
batch carries teacher hidden states; when the batch has none, the hidden term is 0
(the projections are still instantiated so configs stay consistent between runs).

CLI: ``python -m unme.train.distill <config_path>`` (also reachable as the dotted
callable ``unme.train.distill:train``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from unme.utils.io import load_yaml

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

try:
    from unme.data.dataset import DistillDataset, collate
    from unme.train.losses import combined_distill_loss
except ImportError:  # pragma: no cover
    if _IMPORT_ERROR is None:
        raise
    DistillDataset = None  # type: ignore[assignment]
    collate = None  # type: ignore[assignment]
    combined_distill_loss = None  # type: ignore[assignment]


def _require_torch() -> None:
    if _IMPORT_ERROR is not None:  # pragma: no cover
        raise ImportError(
            "unme.train.distill requires torch/transformers. Install: "
            "pip install 'lab-unme[train]'"
        ) from _IMPORT_ERROR


def _load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"distill config not found: {path}")
    return load_yaml(path)


def _build_student(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.train()
    return tok, model


def _layer_dims(model) -> list[int]:
    """Per-transformer-layer hidden dim, one entry per layer index.

    Supports GPT-2 style ``model.transformer.h[i]`` and Llama/Qwen style
    ``model.model.layers[i]``; falls back to the model's config hidden size for
    every layer when the block structure is unknown.
    """
    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is None:
        blocks = getattr(getattr(model, "model", None), "layers", None)
    cfg = model.config
    d = int(getattr(cfg, "hidden_size", 0) or 0)
    if d == 0:
        # best-effort: read weight shape
        x = blocks[0] if blocks else None
        x = getattr(x, "mlp", None) or getattr(x, "mlp.c_fc", None) if x is not None else None
        w = getattr(getattr(x, "c_fc", None), "weight", None) if x is not None else None
        if w is not None:
            d = int(w.shape[-1])
    n = len(blocks) if blocks is not None else 0
    if n == 0:
        n = int(getattr(cfg, "n_layer", 0) or getattr(cfg, "num_hidden_layers", 0) or 0)
    return [d] * n


def _build_projections(
    model, teacher_hidden_dim: int | None, layer_map: dict[int, int]
) -> dict[tuple[int, int], Any]:
    from torch import nn

    projs: dict[tuple[int, int], Any] = {}
    student_dims = _layer_dims(model)
    if not layer_map or teacher_hidden_dim is None or teacher_hidden_dim <= 0:
        return projs
    for s_idx, _t_idx in layer_map.items():
        d_s = student_dims[s_idx] if s_idx < len(student_dims) else student_dims[-1]
        if d_s <= 0:
            continue
        projs[(s_idx, _t_idx)] = nn.Linear(d_s, teacher_hidden_dim)
    return projs


def _slice_hidden_states(
    hidden_states: tuple,
    out_positions: torch.Tensor,
    student_indices: list[int],
) -> dict[int, torch.Tensor]:
    """Slice per-layer hidden states at the output positions for the requested student
    layer indices.

    ``hidden_states`` is the tuple returned by ``model(..., output_hidden_states=True)``
    (length ``n_layers + 1``: embeddings then one tensor per layer). Returns a dict of
    ``student_idx -> (B, T, D)`` gathered at ``out_positions``.

    This does NOT run a forward pass — it reuses the single forward already computed
    for the logits, so the loop costs ONE student forward per batch (not two).
    """
    per_layer: dict[int, torch.Tensor] = {}
    for s_idx in student_indices:
        layer = hidden_states[s_idx + 1] if s_idx + 1 < len(hidden_states) else hidden_states[-1]
        sliced = layer.gather(
            1, out_positions.unsqueeze(-1).expand(-1, out_positions.shape[1], layer.shape[-1])
        )
        per_layer[s_idx] = sliced
    return per_layer


def _is_zero_tensor(t: torch.Tensor | None) -> bool:
    """True iff ``t`` is None or all entries are exactly zero.

    Used to treat a placeholder (all-zero) teacher hidden-state tensor as ABSENT so the
    hidden-match loss isn't trained against zeros. ``abs().sum()==0`` avoids NaNs and
    works for any float dtype.
    """
    if t is None:
        return True
    return bool(t.abs().sum().item() == 0)


def train(config_path: str | Path = "configs/distill.yaml") -> dict[str, Any]:
    """Run one distillation pass per ``configs/distill.yaml``.

    Returns a small summary dict (last loss, n_steps, output_dir). Designed to be
    import-safe when torch/transformers are missing (raises a clear ImportError only
    on call), so smoke tests can skip cleanly.
    """
    _require_torch()
    cfg = _load_config(config_path)

    student_name = (cfg.get("student") or {}).get("model")
    if not student_name:
        raise ValueError("distill config: student.model is required")
    data_cfg = cfg.get("data") or {}
    traces_path = data_cfg.get("filtered") or "data/filtered"
    d_cfg = cfg.get("distill") or {}
    temperature = float(d_cfg.get("temperature", 1.0))
    alpha_kl = float(d_cfg.get("alpha_kl", 1.0))
    alpha_hidden = float(d_cfg.get("alpha_hidden", 0.5))
    alpha_ce = float(d_cfg.get("alpha_ce", 0.1))
    lr = float(d_cfg.get("lr", 2e-4))
    batch_size = int(d_cfg.get("batch_size", 1))
    epochs = int(d_cfg.get("epochs", 1))
    grad_clip = float(d_cfg.get("grad_clip", 1.0))
    layer_map: dict[int, int] = {int(k): int(v) for k, v in (d_cfg.get("hidden_layer_map") or {}).items()}

    _tok, model = _build_student(student_name)

    ds = DistillDataset(traces_path, load_hidden=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    teacher_hidden_dim = None
    if (ds and getattr(ds, "traces", None) and ds.traces[0].hidden_path):
        teacher_hidden_dim = _peek_teacher_hidden_dim(ds.traces[0])  # type: ignore[arg-type]
        if teacher_hidden_dim is not None and len(teacher_hidden_dim) != 1:
            teacher_hidden_dim = None  # multi-layer teacher hidden not supported in this loop

    projs = _build_projections(model, teacher_hidden_dim, layer_map)
    proj_params = [p for proj in projs.values() for p in proj.parameters()]
    optim = torch.optim.AdamW(list(model.parameters()) + proj_params, lr=lr)

    device = torch.device("cpu")
    model.to(device)
    for proj in projs.values():
        proj.to(device)

    history: list[float] = []
    dataset_len = len(ds)
    for _epoch in range(max(1, epochs)):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output_ids = batch["output_ids"].to(device)
            teacher_top_ids = batch["teacher_top_ids"].to(device)
            teacher_top_logprobs = batch["teacher_top_logprobs"].to(device)
            mask = batch["mask"].to(device)  # (B, T)
            batch_hidden = batch.get("hidden_states")  # (B, T, D) | None

            B, S_in = input_ids.shape
            T = output_ids.shape[1]
            # Sequence seen by the student: prompt + full target, so logits at
            # positions S_in-1 .. S_in+T-2 predict output_ids[0..T-1].
            seq = torch.cat([input_ids, output_ids], dim=1)
            atn = torch.cat([attention_mask, torch.ones_like(output_ids, dtype=attention_mask.dtype)], dim=1)
            out_positions = torch.arange(S_in, S_in + T, device=device).unsqueeze(0).expand(B, T)

            # SINGLE student forward — request hidden states once and reuse for both
            # the logits and any hidden-match slice (was previously two forwards).
            need_hidden_states = bool(projs)
            out = model(
                input_ids=seq,
                attention_mask=atn,
                output_hidden_states=need_hidden_states,
                use_cache=False,
            )
            logits = out.logits
            # shift to logits that predict output_ids: position s predicts token s+1.
            # we fed seq[0..S_in+T-1]; token output_ids[t] is at seq position S_in+t;
            # its predicting logit is at position S_in+t-1.
            pred_logits = logits[:, S_in - 1 : S_in - 1 + T, :]
            V = pred_logits.shape[-1]
            # Distillation requires a SHARED tokenizer. Teacher token ids that exceed
            # the student vocab are a silent-corruption bug (targeting the wrong id),
            # so we fail loudly instead of clamping.
            assert int(teacher_top_ids.max()) < V, (
                "student vocab < teacher token id: student must use the teacher tokenizer"
            )
            assert int(output_ids.max()) < V, (
                "student vocab < teacher token id: student must use the teacher tokenizer"
            )

            student_hidden_tensor = None
            teacher_hidden_tensor = None
            projection = None
            # Skip the hidden term when there's no teacher hidden signal: either the
            # batch carries none, OR it carries an all-zero placeholder tensor (the
            # OpenAI endpoint can't emit hiddens) — training MSE against zeros is
            # meaningless and harmful, so treat it as ABSENT.
            teacher_hidden_on_device = batch_hidden.to(device) if batch_hidden is not None else None
            if projs and not _is_zero_tensor(teacher_hidden_on_device):
                (s_idx, _t_idx), projection = next(iter(projs.items()))
                student_hidden_tensor = _slice_hidden_states(
                    out.hidden_states, out_positions, [s_idx]
                ).get(s_idx)
                teacher_hidden_tensor = teacher_hidden_on_device.to(
                    next(projection.parameters()).dtype
                )

            total, _terms = combined_distill_loss(
                pred_logits,
                output_ids,
                teacher_top_ids,
                teacher_top_logprobs,
                student_hidden=student_hidden_tensor,
                teacher_hidden=teacher_hidden_tensor,
                projection=projection,
                alpha_kl=alpha_kl,
                alpha_hidden=alpha_hidden,
                alpha_ce=alpha_ce,
                temperature=temperature,
                mask=mask,
            )

            optim.zero_grad(set_to_none=True)
            total.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if proj_params:
                    torch.nn.utils.clip_grad_norm_(proj_params, grad_clip)
            optim.step()
            history.append(float(total.detach()))
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite loss: {float(total)}")

    output_dir = Path(cfg.get("output_dir") or "outputs/student")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(str(output_dir))
    _tok.save_pretrained(str(output_dir))
    return {
        "n_steps": len(history),
        "n_examples": dataset_len,
        "loss_history": history,
        "output_dir": str(output_dir),
    }


def _peek_teacher_hidden_dim(trace) -> int | None:
    """Look at the .npz behind a Trace's hidden_path and return its feature dim."""
    import numpy as np

    path = trace.hidden_path
    if not path or not Path(path).exists():
        return None
    try:
        with np.load(path) as data:
            arr = next(iter(data.values()))
    except (OSError, ValueError, StopIteration):
        return None
    return int(arr.shape[-1]) if arr.ndim >= 2 else None


def main() -> None:
    """Console entry: ``python -m unme.train.distill [config_path]``."""
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/distill.yaml"
    summary = train(cfg)
    last = summary["loss_history"][-1] if summary["loss_history"] else None
    print(f"distill done: steps={summary['n_steps']} loss_last={last}")


if __name__ == "__main__":
    main()
