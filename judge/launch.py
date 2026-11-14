"""Step 3: read the test files with Clef on Hugging Face Jobs.

  estimate    the tokens to read, and clef-tune's estimate of the time and cost
  score       one job that reads the named test sets (nothing starts without --yes;
              --resume continues a stopped job's folder)
  download    fetch the job's outputs into data/scores/

A job reads with Clef 27B as released, the `base` arm. `--tuned-repo NAME` adds a `tuned` arm:
a trained clef-tune adapter from that Hub repo, read before the released model so that its
result exists if the job stops early. --arms narrows a job to one of the two.

--sets narrows a job to some of the test sets. It also takes the views that views.py writes
(shade_calls, shade_trimmed); those are read only when named. --name keeps jobs apart in
data/jobs.json; give `download` the same name.

Long traces are read whole, up to --max-length tokens (see score_job.py for what happens when
one does not fit in memory). The test files go to your private jobs-artifacts bucket and the
job deletes them when it finishes: the Storf bundle may not be passed on.

The job installs clef-tune at the commit named below, the same one pyproject.toml pins. It
needs HF_TOKEN, with write access to your jobs bucket and permission to run jobs.

  python judge/launch.py estimate
  python judge/launch.py score --name zero-shot --yes
  python judge/launch.py download --name zero-shot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "prepared"  # git-ignored
SCORES = ROOT / "data" / "scores"
JOBS = ROOT / "data" / "jobs.json"
SCORE_JOB = Path(__file__).resolve().parent / "score_job.py"

CLEF_TUNE = "https://github.com/hersh-gupta/clef-tune"
CLEF_TUNE_COMMIT = "15ee6a6343f31b145cb77d6dff93ce8876877df7"
CONFIG = "judge.yaml"
SETS = ("malt_test", "atbench500", "atbench", "atbench_no_tools", "controlarena")
VIEWS = ("shade_calls", "shade_trimmed")  # views.py: diagnostics, read only when named
ARMS = ("tuned", "base")  # tuned first: if the job stops early, that result exists
MAX_LENGTH = 65536
FLAVOR = "rtx-pro-6000"
LABELS = {"benchmark": "agent-safety-judges"}


def machine_env(name: str) -> str | None:
    """An environment variable from this process, else (on Windows) from the Machine or User
    scope, where a token set in System Properties lives; never printed."""
    value = os.environ.get(name)
    if value or sys.platform != "win32":
        return value
    import winreg

    machine = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
    places = [
        (winreg.HKEY_LOCAL_MACHINE, machine),
        (winreg.HKEY_CURRENT_USER, "Environment"),
    ]
    for hive, key in places:
        try:
            with winreg.OpenKey(hive, key) as handle:
                return str(winreg.QueryValueEx(handle, name)[0])
        except OSError:
            continue
    return None


def token() -> str:
    value = machine_env("HF_TOKEN")
    if not value:
        raise SystemExit("HF_TOKEN is not set")
    return value


def set_tokens(max_length: int, sets: list[str]) -> dict[str, dict[str, int]]:
    """Per test set: its rows, the tokens read under the limit, the longest trace as read, and
    how many traces are cut."""
    csv.field_size_limit(2**31 - 1)
    out = {}
    for name in sets:
        with (PREPARED / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
            tokens = [int(float(row["tokens"])) for row in csv.DictReader(handle)]
        out[name] = {
            "rows": len(tokens),
            "tokens": sum(min(t, max_length) for t in tokens),
            "longest": min(max(tokens), max_length),
            "cut": sum(t > max_length for t in tokens),
        }
    return out


def estimate(arms: list[str], sets: list[str], max_length: int, flavor: str):
    """clef-tune's estimate for reading each of `sets` once per arm on `flavor`."""
    from clef_tune.config import load_config
    from clef_tune.estimate import (
        estimate_hf_flavors,
        fetch_hf_prices,
        model_spec,
        prediction_workload,
    )

    config = load_config(PREPARED / CONFIG)
    found = set_tokens(max_length, sets)
    tokens = sum(item["tokens"] for item in found.values()) * len(arms)
    longest = max(item["longest"] for item in found.values())
    workload = prediction_workload(config, tokens, longest)
    rows = estimate_hf_flavors(model_spec(config.model.id), workload, fetch_hf_prices())
    row = next((r for r in rows if r.flavor == flavor), None)
    if row is None:
        raise SystemExit(f"unknown hardware {flavor!r}")
    return found, row


def estimate_command(arms: list[str], sets: list[str], max_length: int, flavor: str) -> None:
    from clef_tune.estimate import suggest_timeout

    found, row = estimate(arms, sets, max_length, flavor)
    for name, item in found.items():
        print(
            f"{name:>17}: {item['rows']:>5,} traces, {item['tokens'] / 1e6:6.1f}M tokens, "
            f"{item['cut']} cut at {max_length:,}"
        )
    total = sum(item["tokens"] for item in found.values())
    print(f"{'each arm':>17}: {total / 1e6:6.1f}M tokens; arms: {', '.join(arms)}")
    print(
        f"{flavor}: {row.hours_low:.1f}-{row.hours_high:.1f} h, "
        f"${row.cost_low:.2f}-${row.cost_high:.2f} at ${row.price_per_hour}/h"
    )
    print(
        f"memory for the longest trace: {row.memory_needed_gb:.0f} GB needed, "
        f"{row.memory_available_gb:.0f} GB available"
        + ("" if row.fits else " (too little by this estimate: such traces get cut)")
    )
    print(f"suggested timeout: {suggest_timeout(row, extra_minutes=20)}")


def base_config() -> str:
    """judge.yaml as the job needs it: the model read from the job's own mount."""
    from clef_tune.config import load_config
    from clef_tune.yamlio import dump_yaml

    data = load_config(PREPARED / CONFIG).to_portable_dict()
    data["model"]["local_path"] = "/model"
    return dump_yaml(data)


def job_script(source, arms: list[str], max_length: int, repo: str | None, sets: list[str]) -> str:
    from clef_tune.launch.job_script import kernel_step

    requirement = source.requirement()
    w = "/work"
    names = ",".join(sets)
    prepare = {
        "tuned": f"python {w}/score_job.py fetch {shlex.quote(repo or '')} /tmp/tuned",
        "base": f"python {w}/score_job.py base {w}/base_config.yaml /tmp/base",
    }
    steps = [
        "set -euo pipefail",
        f'echo "scoring job: installing {requirement}"',
        f"pip install --root-user-action=ignore {shlex.quote(requirement)}",
        kernel_step(),
        f"mkdir -p {w}/out",
    ]
    for arm in arms:
        steps += [
            prepare[arm],
            f"python {w}/score_job.py score /tmp/{arm} {arm} {w}/input {w}/out {max_length} "
            f"{names}",
        ]
    steps += [f"rm -rf {w}/input", 'echo "scoring job: done"']
    return "\n".join(steps)


def _jobs() -> dict:
    return json.loads(JOBS.read_text(encoding="utf-8")) if JOBS.exists() else {}


def score_command(
    arms: list[str],
    sets: list[str],
    max_length: int,
    flavor: str,
    timeout: str | None,
    yes: bool,
    resume: bool,
    name: str,
    repo: str | None,
) -> None:
    from clef_tune.config import load_config
    from clef_tune.estimate import suggest_timeout
    from clef_tune.launch.hf_jobs import (
        jobs_bucket,
        kernel_cache_folder,
        run_folder,
        start_job,
        timeout_hours,
    )
    from clef_tune.launch.job_script import KERNEL_MOUNT, MODEL_MOUNT, WORK_MOUNT, job_env
    from clef_tune.launch.source import Source, pin_commit
    from clef_tune.runtime import BASE_IMAGE
    from huggingface_hub import HfApi, Volume

    config = load_config(PREPARED / CONFIG)
    source = pin_commit(Source(CLEF_TUNE, CLEF_TUNE_COMMIT))  # checks the commit is on GitHub
    _, row = estimate(arms, sets, max_length, flavor)
    timeout = timeout or suggest_timeout(row, extra_minutes=20)
    most = timeout_hours(timeout) * row.price_per_hour
    print(
        f"Score {', '.join(arms)} on {', '.join(sets)} with {flavor}"
        + (f" (tuned model: {repo})" if "tuned" in arms else "")
        + f", reading up to {max_length:,} tokens: "
        f"{row.hours_low:.1f}-{row.hours_high:.1f} h, ${row.cost_low:.2f}-${row.cost_high:.2f}; "
        f"timeout {timeout}, at most ${most:.2f}"
    )
    if not yes:
        print("Nothing started. Run again with --yes to start the job.")
        return
    hf_token = token()
    api = HfApi(token=hf_token)
    namespace = api.whoami()["name"]
    bucket = jobs_bucket(namespace)
    sync = {"read_only": False, "namespace": namespace, "token": hf_token}
    with tempfile.TemporaryDirectory() as tmp:
        if resume:
            folder = _jobs()[name]["folder"]
            work = Volume(
                type="bucket", source=bucket, mount_path=WORK_MOUNT, path=folder, read_only=False
            )
        else:
            folder = run_folder("judge-score", datetime.now(UTC))
            stage = Path(tmp) / "stage"
            (stage / "input").mkdir(parents=True)
            for item in sets:
                (stage / "input" / f"{item}.csv").write_bytes(
                    (PREPARED / f"{item}.csv").read_bytes()
                )
            (stage / "score_job.py").write_bytes(SCORE_JOB.read_bytes())
            (stage / "base_config.yaml").write_text(base_config(), encoding="utf-8")
            work = api.sync_job_volume(stage, WORK_MOUNT, remote_name=folder, **sync)
        empty = Path(tmp) / "kernels"
        empty.mkdir()
        volumes = [
            work,
            Volume(
                type="model",
                source=config.model.id,
                mount_path=MODEL_MOUNT,
                revision=config.model.resolved_revision(),
            ),
            api.sync_job_volume(empty, KERNEL_MOUNT, remote_name=kernel_cache_folder(), **sync),
        ]
    info = start_job(
        api,
        image=BASE_IMAGE,
        command=["bash", "-c", job_script(source, arms, max_length, repo, sets)],
        env=job_env(source, price_per_hour=row.price_per_hour, estimated_hours=None),
        secrets={"HF_TOKEN": hf_token},
        flavor=flavor,
        timeout=timeout,
        labels=LABELS,
        volumes=volumes,
        token=hf_token,
    )
    jobs = _jobs()
    jobs[name] = {
        "id": info.id,
        "url": info.url,
        "flavor": flavor,
        "price_per_hour": row.price_per_hour,
        "bucket": bucket,
        "folder": folder,
        "arms": arms,
        "sets": sets,
        "repo": repo if "tuned" in arms else None,
        "max_length": max_length,
        "source": source.ref,
    }
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    JOBS.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    print(f"Job started: {info.id} {info.url}")


def download_command(name: str) -> None:
    from huggingface_hub import HfApi

    job = _jobs()[name]
    api = HfApi(token=token())
    SCORES.mkdir(parents=True, exist_ok=True)
    names = [
        f"{arm}-{item}{suffix}"
        for arm in job["arms"]
        for item in job.get("sets", SETS)
        for suffix in (".jsonl", "-timing.json")
    ]
    api.download_bucket_files(
        job["bucket"],
        [(f"{job['folder']}/out/{item}", SCORES / item) for item in names],
        raise_on_missing_files=False,
    )
    missing = [item for item in names if not (SCORES / item).exists()]
    print(f"{len(names) - len(missing)} of {len(names)} files in {SCORES}")
    if missing:
        print("not in the bucket yet:", ", ".join(missing))


def chosen_arms(arms: str | None, repo: str | None) -> list[str]:
    """The arms a job reads with: the released model, and first the tuned one if a repo is
    named. `arms` narrows that to one of the two."""
    named = arms.split(",") if arms else (list(ARMS) if repo else ["base"])
    if not named or set(named) - set(ARMS):
        raise SystemExit("--arms must name tuned, base or both")
    if "tuned" in named and not repo:
        raise SystemExit("the tuned arm needs --tuned-repo, the Hub repo of a clef-tune adapter")
    return [arm for arm in ARMS if arm in named]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["estimate", "score", "download"])
    parser.add_argument("--tuned-repo", default=None, help="a trained adapter to read with too")
    parser.add_argument("--arms", default=None, help="tuned, base or both")
    parser.add_argument("--sets", default=",".join(SETS), help="test sets to read")
    parser.add_argument("--name", default="score", help="the job's name in data/jobs.json")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--flavor", default=FLAVOR)
    parser.add_argument("--timeout", default=None)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    arms = chosen_arms(args.arms, args.tuned_repo)
    sets = [item for item in SETS + VIEWS if item in args.sets.split(",")]
    if not sets or set(args.sets.split(",")) - set(SETS + VIEWS):
        raise SystemExit(f"--sets must name some of: {', '.join(SETS + VIEWS)}")
    if args.command == "estimate":
        estimate_command(arms, sets, args.max_length, args.flavor)
    elif args.command == "score":
        score_command(
            arms,
            sets,
            args.max_length,
            args.flavor,
            args.timeout,
            args.yes,
            args.resume,
            args.name,
            args.tuned_repo,
        )
    else:
        download_command(args.name)


if __name__ == "__main__":
    main()
