"""Torch Dataset over filtered Trace JSONL for sparse top-k distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from unme.schemas import Trace

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "unme.data.dataset requires torch. Install the train extra: pip install 'lab-unme[train]'"
    ) from e


def _load_traces(path: str | Path) -> list[Trace]:
    p = Path(path)
    files: list[Path]
    if p.is_file():
        files = [p]
    else:
        files = sorted(p.glob("*.jsonl"))
    traces: list[Trace] = []
    for fp in files:
        with fp.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                traces.append(Trace.model_validate(orjson.loads(line)))
    return traces


def _load_hidden(path: str | None) -> torch.Tensor | None:
    if not path:
        return None
    fp = Path(path)
    if not fp.exists():
        return None
    # Support .pt / .pth via torch; .npy via torch.from_numpy
    def _torch_load(p: Path):
        try:
            return torch.load(p, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(p, map_location="cpu")

    if fp.suffix in {".pt", ".pth"}:
        obj = _torch_load(fp)
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict) and "hidden" in obj:
            return obj["hidden"]
        raise ValueError(f"Unrecognized hidden payload in {fp}")
    if fp.suffix == ".npy":
        import numpy as np

        arr = np.load(fp)
        return torch.from_numpy(arr)
    try:
        obj = _torch_load(fp)
        if isinstance(obj, torch.Tensor):
            return obj
    except (OSError, RuntimeError, ValueError):
        return None
    return None


class DistillDataset(Dataset):
    """Yields per-trace tensors for GLM distill.py-style training loops.

    Item keys
    ---------
    input_ids : LongTensor [S_in]
    output_ids : LongTensor [T]
    teacher_top_ids : LongTensor [T, K]
    teacher_top_logprobs : FloatTensor [T, K]
    attention_mask : LongTensor [S_in]  (all-ones over real input tokens)
    hidden_states : FloatTensor [...] | None
    """

    def __init__(self, traces_path: str | Path, *, load_hidden: bool = True) -> None:
        self.traces = _load_traces(traces_path)
        self.load_hidden = load_hidden
        if not self.traces:
            raise ValueError(f"No traces found at {traces_path}")

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tr = self.traces[idx]
        k = tr.topk
        t = len(tr.output_ids)
        top_ids = torch.zeros(t, k, dtype=torch.long)
        top_lp = torch.zeros(t, k, dtype=torch.float32)
        for i, step in enumerate(tr.steps):
            # pad / truncate step to k
            ids = list(step.token_ids[:k]) + [0] * max(0, k - len(step.token_ids))
            lps = list(step.logprobs[:k]) + [0.0] * max(0, k - len(step.logprobs))
            top_ids[i] = torch.tensor(ids[:k], dtype=torch.long)
            top_lp[i] = torch.tensor(lps[:k], dtype=torch.float32)

        item: dict[str, Any] = {
            "input_ids": torch.tensor(tr.input_ids, dtype=torch.long),
            "output_ids": torch.tensor(tr.output_ids, dtype=torch.long),
            "teacher_top_ids": top_ids,
            "teacher_top_logprobs": top_lp,
            "attention_mask": torch.ones(len(tr.input_ids), dtype=torch.long),
            "hidden_states": None,
        }
        if self.load_hidden and tr.hidden_path:
            item["hidden_states"] = _load_hidden(tr.hidden_path)
        return item


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad a list of dataset items into a batch; build generation ``mask``.

    Returns
    -------
    input_ids : [B, S_in_max]
    attention_mask : [B, S_in_max]
    output_ids : [B, T_max]
    teacher_top_ids : [B, T_max, K]
    teacher_top_logprobs : [B, T_max, K]
    mask : [B, T_max]  — 1 on real output positions (loss positions), 0 on pad
    hidden_states : optional [B, T_max, D] if every item has hidden states
    """
    if not batch:
        raise ValueError("empty batch")

    bsz = len(batch)
    max_in = max(x["input_ids"].numel() for x in batch)
    max_t = max(x["output_ids"].numel() for x in batch)
    k = batch[0]["teacher_top_ids"].shape[-1]

    input_ids = torch.zeros(bsz, max_in, dtype=torch.long)
    attention_mask = torch.zeros(bsz, max_in, dtype=torch.long)
    output_ids = torch.zeros(bsz, max_t, dtype=torch.long)
    teacher_top_ids = torch.zeros(bsz, max_t, k, dtype=torch.long)
    teacher_top_logprobs = torch.zeros(bsz, max_t, k, dtype=torch.float32)
    mask = torch.zeros(bsz, max_t, dtype=torch.float32)

    hiddens: list[torch.Tensor | None] = []
    for i, item in enumerate(batch):
        s = item["input_ids"].numel()
        t = item["output_ids"].numel()
        input_ids[i, :s] = item["input_ids"]
        attention_mask[i, :s] = item["attention_mask"][:s]
        output_ids[i, :t] = item["output_ids"]
        teacher_top_ids[i, :t] = item["teacher_top_ids"]
        teacher_top_logprobs[i, :t] = item["teacher_top_logprobs"]
        mask[i, :t] = 1.0
        hiddens.append(item.get("hidden_states"))

    out: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_ids": output_ids,
        "teacher_top_ids": teacher_top_ids,
        "teacher_top_logprobs": teacher_top_logprobs,
        "mask": mask,
    }

    if all(h is not None for h in hiddens):
        # pad hidden on time dim
        assert hiddens[0] is not None
        d = hiddens[0].shape[-1]
        hidden_batch = torch.zeros(bsz, max_t, d, dtype=hiddens[0].dtype)
        for i, h in enumerate(hiddens):
            assert h is not None
            t = min(h.shape[0], max_t)
            # allow [T, D] or [T, L, D] — if 3D, mean over layer for collate simplicity
            if h.ndim == 3:
                h_use = h.mean(dim=1)
            else:
                h_use = h
            hidden_batch[i, :t] = h_use[:t]
        out["hidden_states"] = hidden_batch
    else:
        out["hidden_states"] = None

    return out
