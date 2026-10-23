"""Typed experiment configuration: configs/default.yaml overlaid with configs/local.yaml."""

from __future__ import annotations

import math
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
    problem_ids: tuple[int, ...] | None


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
    timing_stat: str
    balanced_order: bool
    pin_cpus: bool
    workers: int

    def __post_init__(self) -> None:
        if self.timing_stat not in ("mean", "median"):
            raise ValueError(f"eval.timing_stat must be mean or median, got {self.timing_stat!r}")


@dataclass(frozen=True)
class RunConfig:
    workers: int
    output_dir: str


@dataclass(frozen=True)
class SearchConfig:
    max_candidates: int  # LLM-made candidates per (problem, seed), repairs and failures included
    population_size: int
    offspring_per_generation: int
    generations: int
    tournament_size: int
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.generations * self.offspring_per_generation > self.max_candidates:
            raise ValueError("search: generations * offspring_per_generation exceeds "
                             "max_candidates")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("search: tournament_size must be in [1, population_size]")


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    dev_fraction: float  # problems for tuning prompts/config; the rest is the test split
    heldout_fraction: float  # per problem, test inputs kept out of search for the final check

    def __post_init__(self) -> None:
        for name in ("dev_fraction", "heldout_fraction"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"split.{name} must be in (0, 1)")


@dataclass(frozen=True)
class RefinedBestOfNConfig:
    max_calls: int
    samples_per_round: int  # >= max_calls leaves one round: plain best-of-N
    min_gain: float

    def __post_init__(self) -> None:
        for key in ("max_calls", "samples_per_round"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:  # 4.0 or true would crash or skew a run
                raise ValueError(f"refined_best_of_n.{key} must be an integer of at least 1, "
                                 f"got {value!r}")
        if not (math.isfinite(self.min_gain) and self.min_gain >= 0):
            raise ValueError("refined_best_of_n.min_gain must be finite and not negative, "
                             f"got {self.min_gain!r}")


@dataclass(frozen=True)
class RefinementLoopConfig:
    rounds: int


@dataclass(frozen=True)
class AideConfig:
    max_calls: int
    num_drafts: int
    debug_prob: float
    max_debug_depth: int


@dataclass(frozen=True)
class EohConfig:
    max_calls: int
    pop_size: int
    num_parents: int
    operators: tuple[str, ...]  # applied in turn after the i1 seeds

    def __post_init__(self) -> None:
        if not self.operators or set(self.operators) - {"e1", "e2", "m1", "m2"}:
            raise ValueError("eoh.operators must be a non-empty list of e1, e2, m1, m2, "
                             f"got {list(self.operators)}")


@dataclass(frozen=True)
class AsmGraftConfig:
    max_calls: int
    max_repairs: int  # repairs in a row of a broken latest candidate (R)
    min_gain: float  # a pull is rewarded if its child beats the reference by this much (delta)
    rex_c: float  # REx constant of the node bandit (C)
    discount: float  # posterior decay of the role bandit, in (0, 1] (gamma)

    def __post_init__(self) -> None:
        if not 0 < self.discount <= 1:
            raise ValueError(f"asmgraft.discount must be in (0, 1], got {self.discount}")


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig
    llm: LLMConfig
    eval: EvalConfig
    run: RunConfig
    search: SearchConfig
    split: SplitConfig
    refined_best_of_n: RefinedBestOfNConfig
    refinement_loop: RefinementLoopConfig
    aide: AideConfig
    eoh: EohConfig
    asmgraft: AsmGraftConfig


_SECTIONS = {
    "dataset": DatasetConfig,
    "llm": LLMConfig,
    "eval": EvalConfig,
    "run": RunConfig,
    "search": SearchConfig,
    "split": SplitConfig,
    "refined_best_of_n": RefinedBestOfNConfig,
    "refinement_loop": RefinementLoopConfig,
    "aide": AideConfig,
    "eoh": EohConfig,
    "asmgraft": AsmGraftConfig,
}


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
    """``default.yaml``, overlaid key by key with ``configs/local.yaml`` if it exists, then
    with ``path``."""
    raw = _read(DEFAULT_CONFIG)
    overrides = [LOCAL_CONFIG] if LOCAL_CONFIG.is_file() else []
    if path is not None:
        overrides.append(path)
    for override in overrides:
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
