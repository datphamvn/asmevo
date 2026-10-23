"""Test set: the ``val`` split of LLM4Code/llm_superoptimizer_ds (200 programs).

Only this split is used for evaluation; ``train`` and ``fewshot`` are never downloaded.
Each problem is a C program with its gcc -O3 x86-64 baseline assembly and stdin/stdout tests.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

HF_REPO = "LLM4Code/llm_superoptimizer_ds"
HF_REVISION = "efd4d6ef011da28a83a40aa9e821d44507c7d73b"
TEST_FILE = "data/val-00000-of-00001.parquet"

_FENCE_RE = re.compile(r"^\s*```[\w+-]*\n(.*?)\n?```\s*$", re.DOTALL)


@dataclass(frozen=True)
class TestCase:
    stdin: str
    expected_stdout: str


@dataclass(frozen=True)
class Problem:
    id: int
    c_code: str
    baseline_asm: str  # gcc -O3 output; the speedup reference for this problem
    tests: tuple[TestCase, ...]
    baseline_binary: bytes  # ELF the dataset authors linked from ``baseline_asm``
    standard_prompt: str  # SuperCoder's prompt for this problem (C code + baseline_asm)


def strip_fence(text: str) -> str:
    """Remove a surrounding Markdown code fence (```c / ```assembly), if present."""
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def download_test_file(cache_dir: str) -> Path:
    """Local path of the pinned parquet, downloaded once into ``cache_dir``."""
    return Path(hf_hub_download(repo_id=HF_REPO, filename=TEST_FILE, revision=HF_REVISION,
                                repo_type="dataset", cache_dir=cache_dir))


def load_test_set(path: Path, cache_dir: str, limit: int | None = None) -> list[Problem]:
    """Problems from the parquet at ``path``, the one hashed with ``sha256_file``, so the
    manifest names the bytes loaded."""
    ds = load_dataset("parquet", data_files={"test": str(path)}, split="test",
                      cache_dir=cache_dir)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    problems = []
    for row in ds:
        info = row["extra_info"]
        problems.append(
            Problem(
                id=info["index"],
                c_code=strip_fence(info["c_code"]),
                baseline_asm=strip_fence(info["unoptimized_assembly"]),
                tests=tuple(
                    TestCase(i, o) for i, o in zip(info["inputs"], info["outputs"], strict=True)
                ),
                baseline_binary=info["unoptimized_compiled"],
                standard_prompt=info["question"],
            )
        )
    return problems
