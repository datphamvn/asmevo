import json
from dataclasses import replace

import pytest

from asmevo.config import load_config
from asmevo.data import manifest as m
from asmevo.data.dataset import Problem
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


def test_build_manifest(manifest):
    assert manifest["dataset"]["sha256"] == SHA
    assert manifest["problem_ids"] == [3, 1, 2]
    assert manifest["splits"] == {"all": [3, 1, 2]}
    assert manifest["split_seed"] == load_config().split.seed
    assert manifest["seed"] == 0
    assert manifest["toolchain"] == {"cc": "gcc 1.0", "hyperfine": "hyperfine 1.0"}
    assert manifest["git"] == {"commit": "c0", "dirty": False}


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


def test_select_problems_rejects_other_dataset(problems, manifest):
    with pytest.raises(ValueError, match="SHA-256"):
        select_problems(problems, manifest, "b" * 64)


@pytest.fixture
def fake_generate(monkeypatch):
    """``main.generate`` with a fake parquet of 5 problems and an LLM that returns no asm."""
    from asmevo import main
    from asmevo.schemas import Generation

    state = {"sha": SHA}
    monkeypatch.setattr(m, "tool_version", lambda cmd: None)
    monkeypatch.setattr(main, "make_client", lambda cfg: None)
    monkeypatch.setattr(main, "download_test_file", lambda cache_dir: "parquet")
    monkeypatch.setattr(main, "sha256_file", lambda path: state["sha"])
    monkeypatch.setattr(main, "load_test_set", lambda cache_dir, path, limit=None:
                        [make_problem(i) for i in range(5)][:limit])
    monkeypatch.setitem(main.METHODS, "single_call",
                        lambda p, llm, cfg: Generation(p.id, "single_call", None, ()))
    return main, state


def test_generate_reuses_manifest(tmp_path, fake_generate):
    main, state = fake_generate
    cfg = load_config()
    cfg_3 = replace(cfg, dataset=replace(cfg.dataset, limit=3))
    main.generate(cfg_3, "single_call", tmp_path / "a")
    assert read_manifest(tmp_path / "a")["problem_ids"] == [0, 1, 2]
    main.generate(cfg_3, "single_call", tmp_path / "a")  # resume

    # Another limit, same manifest: same problems.
    main.generate(cfg, "single_call", tmp_path / "b", tmp_path / "a" / m.MANIFEST)
    b = read_manifest(tmp_path / "b")
    assert b["problem_ids"] == [0, 1, 2]
    assert len(main.load_generations(tmp_path / "b")) == 3

    # A hand-edited manifest that drops problems the run already generated.
    edited = {**read_manifest(tmp_path / "a"), "problem_ids": [0], "splits": {"all": [0]}}
    (tmp_path / "a" / m.MANIFEST).write_text(json.dumps(edited))
    with pytest.raises(SystemExit, match=r"\[1, 2\]"):
        main.generate(cfg_3, "single_call", tmp_path / "a")

    state["sha"] = "b" * 64
    with pytest.raises(SystemExit, match="SHA-256"):
        main.generate(cfg, "single_call", tmp_path / "b")
