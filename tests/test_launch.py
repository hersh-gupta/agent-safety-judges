"""judge/launch.py: what a job runs, which model it reads with, and which clef-tune it installs."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from clef_tune.launch.source import Source

from analysis import score
from judge import launch, score_job, views

ROOT = Path(__file__).resolve().parents[1]


def test_the_job_reads_with_the_tuned_model_first_and_deletes_its_input():
    source = Source(launch.CLEF_TUNE, "a" * 40)
    every = list(launch.SETS)
    script = launch.job_script(source, ["tuned", "base"], 65536, "someone/judge", every)
    steps = script.splitlines()
    tuned = next(i for i, s in enumerate(steps) if "score /tmp/tuned tuned" in s)
    base = next(i for i, s in enumerate(steps) if "score /tmp/base base" in s)
    assert tuned < base and steps[tuned].endswith(" 65536 " + ",".join(every))
    assert any("score_job.py fetch someone/judge /tmp/tuned" in step for step in steps)
    assert steps[-2] == "rm -rf /work/input"
    assert "a" * 40 in steps[2]


def test_a_job_with_no_tuned_model_never_fetches_one():
    source = Source(launch.CLEF_TUNE, "a" * 40)
    only = launch.job_script(source, ["base"], 16384, None, ["controlarena"])
    assert "/tmp/tuned" not in only and "fetch" not in only
    assert "score_job.py base /work/base_config.yaml /tmp/base\n" in only
    assert "score /tmp/base base /work/input /work/out 16384 controlarena\n" in only


def test_the_released_model_is_read_unless_a_tuned_one_is_named():
    assert launch.chosen_arms(None, None) == ["base"]
    assert launch.chosen_arms(None, "someone/judge") == ["tuned", "base"]
    assert launch.chosen_arms("tuned", "someone/judge") == ["tuned"]
    assert launch.chosen_arms("base,tuned", "someone/judge") == ["tuned", "base"]
    for arms in ("tuned", "tuned,base", "other"):
        with pytest.raises(SystemExit):
            launch.chosen_arms(arms, None)


def test_the_job_installs_the_clef_tune_that_pyproject_pins():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = [d for d in project["project"]["dependencies"] if d.startswith("clef-tune")]
    assert pins == [f"clef-tune @ git+{launch.CLEF_TUNE}@{launch.CLEF_TUNE_COMMIT}"]
    assert Source(launch.CLEF_TUNE, launch.CLEF_TUNE_COMMIT).version.startswith("0.0.0+g15ee6a6")


def test_every_script_names_the_same_test_sets():
    assert launch.SETS == score_job.SETS == score.SETS


def test_the_views_are_read_only_when_named():
    assert not set(launch.VIEWS) & set(launch.SETS)
    assert set(launch.VIEWS) == set(views.VIEWS) == set(score.VIEWS)
