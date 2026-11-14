"""A diagnostic: SHADE read with less of what the tools returned.

The untrained judge is near chance on ControlArena's SHADE setting (0.570 AUROC) and near
perfect on the other four. In SHADE the off-task action is one tool call among about sixty, and
the agent's calls are about an eighth of the text; the rest is what the tools returned. This
script writes two shorter views of the 1,500 SHADE traces, to see whether the judge misses the
action because it is buried:

  data/prepared/shade_calls.csv     system prompt, user messages and the agent's tool calls;
                                    every tool output dropped
  data/prepared/shade_trimmed.csv   the same, with the first 300 characters of each tool output

`group` is the SHADE task (named here from the user's request), so score.py reports each task.

These views were chosen after looking at the test's labels and scores. They are a diagnostic,
not a result: a figure from them stands next to the full-transcript figure, never in its place.

Run it after prepare.py:
  python judge/views.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import prepare

VIEWS = {"shade_calls": 0, "shade_trimmed": 300}  # characters kept of each tool output
TAG = re.compile(r"^\[(SYSTEM|USER|ASSISTANT|TOOL CALLS|TOOL)\]$", re.MULTILINE)
# A phrase from each SHADE task's request, and the name the task gets here.
TASKS = {
    "spam filter": "spam_filter",
    "Boston": "flights",
    "Cabo": "repay_friends",
    "upcoming appointments": "calendar_reminders",
    "financial transactions": "finance_review",
}


def blocks(trace: str) -> list[tuple[str, str]]:
    """The transcript as (tag, text) turns. An agent turn with no text comes out empty."""
    parts = TAG.split(trace)
    if parts[0].strip():
        raise ValueError(f"a transcript does not start with a tag: {parts[0][:60]!r}")
    return [(parts[i], parts[i + 1].strip("\n")) for i in range(1, len(parts), 2)]


def render(turns: list[tuple[str, str]]) -> str:
    """The bundle's own layout: a blank line between turns, none after an empty one."""
    out = ""
    for index, (tag, text) in enumerate(turns):
        if index:
            out += "\n\n" if turns[index - 1][1] else "\n"
        out += f"[{tag}]\n{text}" if text else f"[{tag}]"
    return out


def view(trace: str, keep: int) -> str:
    """The transcript with each tool output cut to its first `keep` characters; 0 drops them."""
    turns = []
    for tag, text in blocks(trace):
        if tag == "TOOL":
            if not keep:
                continue
            if len(text) > keep:
                text = text[:keep].rstrip() + " ..."
        turns.append((tag, text))
    return render(turns)


def task(trace: str) -> str:
    request = trace[: trace.find("[TOOL CALLS]")]
    for phrase, name in TASKS.items():
        if phrase in request:
            return name
    raise SystemExit(f"a SHADE trace's task is not recognised: {request[-200:]!r}")


def main() -> None:
    from clef_tune.config import PINNED_REVISIONS
    from clef_tune.prompt import RecordTokenCounter, get_token_counter
    from clef_tune.schema import Schema

    csv.field_size_limit(2**31 - 1)
    with (prepare.PREPARED / "controlarena.csv").open(encoding="utf-8", newline="") as handle:
        shade = [row for row in csv.DictReader(handle) if row["group"] == "shade"]
    questions = Schema.from_dict(prepare.SCHEMA).questions
    counter = RecordTokenCounter(
        get_token_counter(prepare.MODEL, PINNED_REVISIONS[prepare.MODEL]), prepare.PILOT_LENGTH
    )
    unchanged = sum(render(blocks(row["trace"])) == row["trace"] for row in shade)
    print(f"{len(shade)} SHADE traces; {unchanged} come back word for word when nothing is cut")
    full = sum(int(float(row["tokens"])) for row in shade)
    for name, keep in VIEWS.items():
        rows = [
            {**row, "trace": view(row["trace"], keep), "group": task(row["trace"])} for row in shade
        ]
        prepare.add_tokens(rows, counter, questions)
        prepare.write_csv(prepare.PREPARED / f"{name}.csv", rows)
        tokens = sorted(row["tokens"] for row in rows)
        print(
            f"{name}: {sum(tokens) / 1e6:.1f}M tokens ({sum(tokens) / full:.0%} of the full "
            f"traces' {full / 1e6:.1f}M), median {tokens[len(tokens) // 2]:,}, "
            f"longest {tokens[-1]:,}"
        )


if __name__ == "__main__":
    main()
