import hashlib

import pytest

from asmevo.config import load_config
from asmevo.data import manifest as m
from asmevo.data.dataset import Problem, sha256_file
from asmevo.data.manifest import build_manifest, read_manifest, select_problems, write_manifest

SHA = "a" * 64


def make_problem(i: int) -> Problem:
    return Problem(id=i, c_code="", baseline_asm="", tests=(), baseline_binary=b"",
                   standard_prompt="")


@pytest.fixture
def problems() -> list[Problem]:
    return [make_problem(i) for i in (3, 1, 2)]


@pytest.fixture
def manifest(problems, monkeypatch) -> dict:
    monkeypatch.setattr(m, "tool_version", lambda cmd: f"{cmd} 1.0")
    monkeypatch.setattr(m, "git_commit", lambda: {"commit": "c0", "dirty": False})
    return build_manifest(problems, dataset_sha256=SHA, cfg=load_config(), seed=0)


def test_sha256_file(tmp_path):
    path = tmp_path / "f.parquet"
    path.write_bytes(b"x" * 3_000_000)  # spans several chunks
    assert sha256_file(path) == hashlib.sha256(b"x" * 3_000_000).hexdigest()


def test_build_manifest(manifest):
    assert manifest["dataset"]["sha256"] == SHA
    assert manifest["problem_ids"] == [3, 1, 2]
    assert manifest["splits"] == {"all": [3, 1, 2]}
    assert manifest["split_seed"] == load_config().split.seed
    assert manifest["seed"] == 0
    assert manifest["toolchain"] == {"cc": "gcc 1.0", "hyperfine": "hyperfine 1.0"}
    assert manifest["git"] == {"commit": "c0", "dirty": False}


def test_build_manifest_without_tools(problems, monkeypatch):
    monkeypatch.setattr(m, "tool_version", lambda cmd: None)
    monkeypatch.setattr(m, "git_commit", lambda: None)
    out = build_manifest(problems, dataset_sha256=SHA, cfg=load_config(), seed=None,
                         splits={"dev": [3], "test": [1, 2]})
    assert out["toolchain"] == {"cc": None, "hyperfine": None}
    assert out["git"] is None and out["seed"] is None
    assert out["splits"] == {"dev": [3], "test": [1, 2]}


def test_tool_version_missing():
    assert m.tool_version("asmevo-no-such-tool") is None


def test_write_manifest_resume(tmp_path, manifest):
    assert write_manifest(tmp_path, manifest) == manifest
    assert read_manifest(tmp_path) == read_manifest(tmp_path / m.MANIFEST) == manifest
    # Same reproducible part on a resume: the original file is kept.
    resumed = {**manifest, "git": {"commit": "c1", "dirty": True}, "created_at": "later"}
    assert write_manifest(tmp_path, resumed) == manifest


@pytest.mark.parametrize("key, value", [
    ("problem_ids", [1, 2, 3]),
    ("seed", 1),
    ("splits", {"all": [3, 1]}),
    ("dataset", {"sha256": "b" * 64}),
])
def test_write_manifest_rejects_changes(tmp_path, manifest, key, value):
    write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match=key):
        write_manifest(tmp_path, {**manifest, key: value})


def test_select_problems_keeps_manifest_order(problems, manifest):
    selected = select_problems(sorted(problems, key=lambda p: p.id), manifest, SHA)
    assert [p.id for p in selected] == [3, 1, 2]


def test_select_problems_rejects_other_dataset(problems, manifest):
    with pytest.raises(ValueError, match="SHA-256"):
        select_problems(problems, manifest, "b" * 64)


def test_select_problems_rejects_missing_id(problems, manifest):
    with pytest.raises(ValueError, match=r"\[2\]"):
        select_problems(problems[:2], manifest, SHA)
