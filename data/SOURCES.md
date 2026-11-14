# Data sources

No dataset, trace or per-trace score is stored in this repository. Each source is downloaded
from its own page, under its own terms, into `data/raw/` (git-ignored). The commits below are
the ones the published results were made from (downloaded 2026-11-11).

## Test sets

### ControlArena test set (Storf et al. 2026)

- Page: https://huggingface.co/datasets/Syghmon/blackboxmonitorsMATS (gated; you accept the terms on the page)
- Commit: `ccb77e37589baaee1f0e16ac9ab89ceac8f6cc8a`
- Terms: research use only. The raw data may not be passed on, and the maintainers ask to be
  cited. Releasing a model trained mainly on the bundle needs their written permission.
- Paper: Storf, Barton-Cooper, Peters-Gill and Hobbhahn, *Constitutional Black-Box Monitoring for
  Scheming in LLM Agents*, https://arxiv.org/abs/2603.00829
- The runs were made with ControlArena: https://github.com/UKGovernmentBEIS/control-arena
- Used: `control_arena.parquet` (about 320 MB), columns `id`, `formatted_transcript` and
  `ground_truth`. 7,500 traces: five settings, 750 hidden-goal and 750 honest runs each, from
  three agent models.

```bash
hf download Syghmon/blackboxmonitorsMATS control_arena.parquet --repo-type dataset --revision ccb77e37589baaee1f0e16ac9ab89ceac8f6cc8a --local-dir data/raw/storf_bundle
```

### ATBench and ATBench500

- Page: https://huggingface.co/datasets/AI45Research/ATBench
- Commit: `4476ef92ed8f85c8d58d8a5b9dfdf55aa7893138`
- Licence: Apache 2.0
- Used: `ATBench/test.json` (1,000 traces) and `ATBench500/test.json` (the earlier 500-trace
  release). Each record has a safe or unsafe label, and unsafe records name a risk source and a
  failure mode.

```bash
hf download AI45Research/ATBench ATBench/test.json ATBench500/test.json --repo-type dataset --revision 4476ef92ed8f85c8d58d8a5b9dfdf55aa7893138 --local-dir data/raw/atbench
```

### MALT

- Page: https://huggingface.co/datasets/metr-evals/malt-public (gated; you accept the terms on the page)
- Commit: `6100006c2fd23d81e5afe2b1cc3bf3f3fff48df1`
- Licence: MIT
- Used: the 42 `data/public-*.parquet` files (about 4 GB), columns `samples` and `metadata`.
  Kept: Claude 3.7 Sonnet runs labelled `normal`, `sabotage` or `match_weaker_model` in four
  task families, within 16,384 tokens. That is 172 runs.

```bash
hf download metr-evals/malt-public --repo-type dataset --revision 6100006c2fd23d81e5afe2b1cc3bf3f3fff48df1 --include "data/*.parquet" --local-dir data/raw/malt
```

## Used only to train the tuned model

The tuned model in the README's "Does tuning help?" section was trained with the code in
clef-tune's `benchmark-trace-judge` branch, not with this repository. Its training data:

- ASSEBench, https://huggingface.co/datasets/Atarogic/ASSEBench, commit
  `fbec8aa93e2161d71ff535673939dd8e395d39e3`, Apache 2.0: the `unsafe` and `hijacked` questions.
- The Storf bundle's two synthetic sets (`gemini_synthetic`, `unified_dataset`), same page and
  terms as the ControlArena test set above: the `hidden_goal` question.
- MALT, as above, in the task families outside the four test families: the `sandbagged`
  question.

## What is not here

- The tuned model. It was trained partly on the Storf bundle, whose terms do not allow a
  release without the maintainers' permission.
- The traces and the per-trace scores. `results/metrics.json` holds aggregate numbers only.
