"""``manifest.json``: what a run needs to be reproduced exactly.

The reproducible part (dataset bytes, problem ids, splits, seeds) must match when a run is
resumed or re-created from an old manifest. Toolchain versions, the git commit, and the
creation time are only recorded: they explain differences but do not block a run.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from asmevo.config import ROOT, Config
from asmevo.data.dataset import HF_REPO, HF_REVISION, TEST_FILE, Problem

MANIFEST = "manifest.json"
REPRODUCIBLE = ("dataset", "problem_ids", "splits", "split_seed", "seed")


def tool_version(cmd: str) -> str | None:
    """First line of ``<cmd> --version``, or None if the tool is missing or fails."""
    try:
        out = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=10,
                             check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return next((line.strip() for line in out.splitlines() if line.strip()), None)


def git_commit(repo: Path = ROOT) -> dict[str, Any] | None:
    """HEAD commit and whether the working tree has changes, or None outside a git checkout."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              timeout=10, check=True).stdout
    try:
        return {"commit": git("rev-parse", "HEAD").strip(),
                "dirty": bool(git("status", "--porcelain").strip())}
    except (OSError, subprocess.SubprocessError):
        return None


def build_manifest(
    problems: Sequence[Problem],
    *,
    dataset_sha256: str,
    cfg: Config,
    seed: int | None,
    splits: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """``seed`` is the search method's seed (None for single_call); ``splits`` comes from T6
    and defaults to one ``all`` split holding every problem."""
    ids = [p.id for p in problems]
    return {
        "dataset": {"repo": HF_REPO, "revision": HF_REVISION, "file": TEST_FILE,
                    "sha256": dataset_sha256},
        "problem_ids": ids,
        "splits": splits if splits is not None else {"all": ids},
        "split_seed": cfg.split.seed,
        "seed": seed,
        "toolchain": {"cc": tool_version(cfg.eval.cc),
                      "hyperfine": tool_version(cfg.eval.hyperfine)},
        "git": git_commit(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def read_manifest(path: Path) -> dict[str, Any]:
    """Read ``path``, or ``path/manifest.json`` when ``path`` is a run directory."""
    if path.is_dir():
        path = path / MANIFEST
    return json.loads(path.read_text())


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Write ``manifest`` into ``run_dir`` and return it.

    If the run already has one (a resume), keep that file and return it instead; raise if its
    reproducible part differs from ``manifest``, and warn if the git commit changed.
    """
    path = run_dir / MANIFEST
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    old = read_manifest(path)
    changed = [k for k in REPRODUCIBLE if old.get(k) != manifest.get(k)]
    if changed:
        raise ValueError(f"{path}: {', '.join(changed)} differ from the current run")
    if old.get("git") != manifest.get("git"):
        print(f"[manifest] warning: git commit changed since {path} was written: "
              f"{old.get('git')} -> {manifest.get('git')}")
    return old


def select_problems(problems: Sequence[Problem], manifest: dict[str, Any],
                    dataset_sha256: str) -> list[Problem]:
    """The manifest's problems, in its order, from a dataset with the same SHA-256."""
    expected = manifest["dataset"]["sha256"]
    if expected != dataset_sha256:
        raise ValueError(f"dataset SHA-256 {dataset_sha256} does not match the manifest's "
                         f"{expected}")
    by_id = {p.id: p for p in problems}
    missing = [i for i in manifest["problem_ids"] if i not in by_id]
    if missing:
        raise ValueError(f"problems {missing} from the manifest are not in the dataset")
    return [by_id[i] for i in manifest["problem_ids"]]
