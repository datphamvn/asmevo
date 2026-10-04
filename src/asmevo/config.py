"""Typed experiment configuration: configs/default.yaml overlaid with configs/local.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
LOCAL_CONFIG = ROOT / "configs" / "local.yaml"


@dataclass(frozen=True)
class DatasetConfig:
    cache_dir: str
    limit: int | None


@dataclass(frozen=True)
class LLMConfig:
    provider: str  # a key of asmevo.utils.llm.client.PROVIDERS
    base_url: str
    api_key_env: str  # environment variable (or .env key) holding the API key
    model: str
    reasoning_mode: str | None
    reasoning_effort: str | None
    max_output_tokens: int
    timeout_s: float
    max_retries: int
    # USD per 1k tokens. null fails fast instead of silently reporting $0; use 0.0 if free.
    cost_per_1k_input: float | None
    cost_per_1k_output: float | None


@dataclass(frozen=True)
class EvalConfig:
    cc: str
    link_flags: tuple[str, ...]
    compile_timeout_s: float
    run_timeout_s: float
    memory_limit_mb: int
    hyperfine: str
    warmup_runs: int
    timing_repeats: int
    workers: int


@dataclass(frozen=True)
class RunConfig:
    workers: int
    output_dir: str


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig
    llm: LLMConfig
    eval: EvalConfig
    run: RunConfig


_SECTIONS = {"dataset": DatasetConfig, "llm": LLMConfig, "eval": EvalConfig, "run": RunConfig}


def _read(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"{path}: unknown config sections {sorted(unknown)}")
    return raw


def _build(cls: type, data: dict[str, Any]) -> Any:
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys {sorted(unknown)}")
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})


def load_config(path: Path | None = None) -> Config:
    """``default.yaml`` overlaid key by key with ``path``, or with ``configs/local.yaml``
    when no path is given and that file exists."""
    raw = _read(DEFAULT_CONFIG)
    override = path or (LOCAL_CONFIG if LOCAL_CONFIG.is_file() else None)
    if override:
        for name, section in _read(override).items():
            raw[name] = {**raw[name], **section}
    return Config(**{name: _build(cls, raw[name]) for name, cls in _SECTIONS.items()})


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Export ``KEY=VALUE`` lines from ``path``; variables already set in the shell win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
