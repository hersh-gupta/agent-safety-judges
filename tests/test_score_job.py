"""judge/score_job.py, the script a job runs, on a tiny random model.

The tiny model is built by clef-tune's own test helpers, which are not part of its installed
package. Point CLEF_TUNE_CHECKOUT at a clef-tune checkout to run these; without it they are
skipped. They also need torch (clef-tune's `train` extra).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from judge import score_job

CHECKOUT = os.environ.get("CLEF_TUNE_CHECKOUT")
pytestmark = pytest.mark.skipif(
    not CHECKOUT, reason="set CLEF_TUNE_CHECKOUT to a clef-tune checkout to build the tiny model"
)


@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("torch")
    path = Path(CHECKOUT) / "tests" / "torch" / "tiny_clef.py"
    spec = importlib.util.spec_from_file_location("tiny_clef", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def release(tiny, tmp_path_factory):
    return tiny.build_tiny_release(tmp_path_factory.mktemp("tiny-release"))


@pytest.fixture(scope="module")
def adapter(tiny, release, tmp_path_factory):
    """A trained tiny adapter whose `outage` temperature is 2.0."""
    import torch
    from clef_tune.train import train

    project = tmp_path_factory.mktemp("adapter")
    config = tiny.write_tiny_project(project, release, {"train": {"epochs": 2}})
    result = train(config, device=torch.device("cpu"))
    (result.adapter_dir / "calibration.json").write_text(
        json.dumps({"outage": 2.0}), encoding="utf-8"
    )
    return result.adapter_dir


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("CLEF_TUNE_DEVICE", "cpu")


def write_input(tmp_path, rows=5, name="malt_test"):
    inputs = tmp_path / "input"
    inputs.mkdir(exist_ok=True)
    body = "".join(f"t{i},s,the site is down {'again ' * i}\n" for i in range(rows))
    (inputs / f"{name}.csv").write_text("id,subject,body\n" + body, encoding="utf-8")
    return inputs


def run(monkeypatch, adapter, tmp_path, rows=5, limit="512"):
    monkeypatch.setattr(score_job, "MODEL_MOUNT", None)  # the adapter's own base model
    monkeypatch.setattr(score_job, "SLICE", 2)
    inputs = write_input(tmp_path, rows)
    score_job.score(str(adapter), "tuned", str(inputs), str(tmp_path / "out"), limit)
    lines = (tmp_path / "out" / "tuned-malt_test.jsonl").read_text(encoding="utf-8").splitlines()
    timing = json.loads((tmp_path / "out" / "tuned-malt_test-timing.json").read_text())
    return [json.loads(line) for line in lines], timing


def test_every_trace_gets_a_line_with_logits_and_the_limit_it_was_read_under(
    monkeypatch, adapter, tmp_path
):
    lines, timing = run(monkeypatch, adapter, tmp_path)
    assert [line["id"] for line in lines] == [f"t{i}" for i in range(5)]
    assert all(line["limit"] == 512 and len(line["logits"]["outage"]) == 2 for line in lines)
    assert timing["rows"] == 5 and timing["tokens"] == sum(line["tokens"] for line in lines)
    assert timing["options"]["outage"] == ["true", "false"] and timing["max_length"] == 512
    assert timing["temperatures"]["outage"] == 2.0


def test_a_rerun_reads_only_what_is_missing(monkeypatch, adapter, tmp_path):
    first, _ = run(monkeypatch, adapter, tmp_path, rows=3)
    again, timing = run(monkeypatch, adapter, tmp_path, rows=5)
    assert again[:3] == first and [line["id"] for line in again[3:]] == ["t3", "t4"]
    assert timing["rows"] == 5


def test_a_trace_that_runs_out_of_memory_is_read_under_half_the_limit(
    monkeypatch, adapter, tmp_path
):
    import clef_tune.evaluation as evaluation
    import torch

    real = evaluation.predict_logits

    def predict_logits(model, items, *args, max_length, **kwargs):
        if len(items) > 1 or max_length > 512:  # no batch fits, and no trace above 512 tokens
            raise torch.OutOfMemoryError("CUDA out of memory")
        return real(model, items, *args, max_length=max_length, **kwargs)

    monkeypatch.setattr(evaluation, "predict_logits", predict_logits)
    monkeypatch.setattr(score_job, "MIN_LENGTH", 512)
    lines, timing = run(monkeypatch, adapter, tmp_path, rows=3, limit="1024")
    assert [line["id"] for line in lines] == ["t0", "t1", "t2"]
    assert all(line["limit"] == 512 for line in lines) and timing["rows"] == 3


def test_only_the_named_test_sets_are_read(monkeypatch, adapter, tmp_path):
    monkeypatch.setattr(score_job, "MODEL_MOUNT", None)
    inputs = write_input(tmp_path, rows=1)
    write_input(tmp_path, rows=1, name="atbench")
    out = tmp_path / "out"
    score_job.score(str(adapter), "base", str(inputs), str(out), "512", "atbench")
    assert sorted(p.name for p in out.iterdir()) == [
        "base-atbench-timing.json",
        "base-atbench.jsonl",
    ]
    with pytest.raises(SystemExit):
        score_job.score(str(adapter), "base", str(inputs), str(out), "512", "no_such_set")
    with pytest.raises(SystemExit):  # asked for by name, but its file was not uploaded
        score_job.score(str(adapter), "base", str(inputs), str(out), "512", "atbench,controlarena")


def test_a_trace_that_fits_under_no_limit_stops_the_job(monkeypatch, adapter, tmp_path):
    import clef_tune.evaluation as evaluation
    import torch

    def predict_logits(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(evaluation, "predict_logits", predict_logits)
    with pytest.raises(torch.OutOfMemoryError):
        run(monkeypatch, adapter, tmp_path, rows=1, limit="8192")


def test_the_released_model_is_saved_and_read_with_no_training(
    monkeypatch, tiny, release, tmp_path
):
    """The `base` step with a head-only config, as data/prepare.py writes judge.yaml."""
    config = tiny.write_tiny_project(
        tmp_path / "project", release, {"train": {"mode": "head_only"}}
    )
    score_job.base(str(config), str(tmp_path / "base"))
    assert not list((tmp_path / "base").glob("*lora*"))  # no LoRA modules were added
    monkeypatch.setattr(score_job, "MODEL_MOUNT", str(release))
    inputs = write_input(tmp_path, rows=3)
    for attempt in ("first", "again"):  # an untrained read gives the same logits every time
        out = tmp_path / attempt
        score_job.score(str(tmp_path / "base"), "base", str(inputs), str(out), "512")
    first, again = ((tmp_path / a / "base-malt_test.jsonl").read_text() for a in ("first", "again"))
    assert first == again and len(first.splitlines()) == 3
