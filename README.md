# asmevo: LLM-guided Evolutionary Assembly Optimization

## Project structure

```
asmevo/
├── pyproject.toml          # uv-managed project: dependencies, dev tools, lint/test settings
├── uv.lock                 # pinned dependency versions (commit it; run `uv sync`)
├── configs/
│   ├── default.yaml        # every setting with its default
│   └── local.yaml          # your overrides on top of default.yaml (gitignored)
├── docker/
│   └── Dockerfile          # fixed Linux x86-64
└── src/asmevo/
    ├── main.py             # entry point
    ├── data/               # test set: `val` split of LLM4Code/llm_superoptimizer_ds (200 programs)
    ├── methods/            # one PocketFlow flow per method (nodes.py, flow.py, prompt.py);
    │                       #   nodes.py at this level holds the nodes they share
    ├── eval/               # build, test, and time candidates against gcc -O3
    └── utils/
        └── llm/            # LLM client interface (OpenAI-compatible endpoint)
```

## Usage

Put `OPENAI_API_KEY` (and optionally `HF_TOKEN`) in `.env`. To change any setting, e.g. the
endpoint, model, or `dataset.limit` for a quick run, add only those keys to `configs/local.yaml`
(per experiment, e.g. `dataset.problem_ids` or a budget, use a file passed as `--config FILE`,
applied on top):

```yaml
llm:
  base_url: https://my-endpoint/v1
  model: my-model
```

```bash
uv run asmevo generate --method single_call          # LLM calls only
uv run asmevo generate --method policy_tree          # tree search
uv run asmevo evaluate runs/single_call-<timestamp>  # build/test/time
uv run asmevo curve runs/<run-a> runs/<run-b>        # speedup of the best in the first k calls
```

All but `single_call` generation need Linux x86-64 (natively or via `docker/`).

`policy_tree` searches a tree of complete programs rooted at the -O3 baseline. After each
candidate is built, tested, and timed, a hand-written policy expands it (refine, or repair if
broken), branches to a sibling with a new strategy hint, backtracks to the best node with room,
or terminates after `policy_tree.patience` calls without a new best. It keeps the fastest
correct candidate.

`generate` stops on the first failed API call; pass `--run-dir` to resume. Each run directory
ends up with `generations.jsonl`, `results.jsonl`, and `summary.json`. Candidates that fail to
build or pass every test count as 1.0x speedup.
