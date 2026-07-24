"""JSONL / path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import orjson
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, rows: Iterable[BaseModel | dict]) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    n = 0
    with path.open("wb") as f:
        for row in rows:
            if isinstance(row, BaseModel):
                payload = row.model_dump(mode="json")
            else:
                payload = row
            f.write(orjson.dumps(payload))
            f.write(b"\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: BaseModel | dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
    with path.open("ab") as f:
        f.write(orjson.dumps(payload))
        f.write(b"\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
        yield  # pragma: no cover
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def read_models(path: str | Path, model: type[T]) -> list[T]:
    return [model.model_validate(row) for row in read_jsonl(path)]


def write_json(path: str | Path, obj: BaseModel | dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def load_yaml(path: str | Path) -> dict:
    import yaml

    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def dump_pretty(obj: BaseModel | dict) -> str:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, indent=2, default=str)
