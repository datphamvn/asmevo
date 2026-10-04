# asmevo: LLM-guided Evolutionary Assembly Optimization

## Project structure

```
llm_opt/
├── pyproject.toml          # uv-managed project: dependencies, dev tools, lint/test settings
├── uv.lock                 # pinned dependency versions (commit it; run `uv sync`)
├── configs/
│   ├── default.yaml        # default configuration for the search algorithm
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
endpoint, model, or `dataset.limit` for a quick run, add only those keys to `configs/local.yaml`:

```yaml
llm:
  base_url: https://my-endpoint/v1
  model: my-model
```

```bash
uv run asmevo generate --method single_call          # LLM calls; any host
uv run asmevo generate --method best_of_n            # n samples, keep best; Linux x86-64 only
uv run asmevo generate --method refinement_loop      # LLM calls + feedback; Linux x86-64 only
uv run asmevo generate --method policy_tree          # tree search; Linux x86-64 only
uv run asmevo evaluate runs/single_call-<timestamp>  # build/test/time; Linux x86-64 only
uv run asmevo curve runs/<run-a> runs/<run-b>        # best result within the first k calls
```

`refinement_loop` builds, tests, and times every candidate during `generate` and sends the
result back to the model for `refinement_loop.rounds` LLM calls, keeping the fastest correct
candidate. Its per-round results are stored in `generations.jsonl` under `rounds`.

`best_of_n` sends the `single_call` prompt `best_of_n.n` times independently, builds, tests,
and times every sample during `generate`, and keeps the fastest correct one. Its per-sample
results are stored in `generations.jsonl` under `rounds`.

`--config FILE` applies on top of `configs/local.yaml`, e.g. to set `dataset.problem_ids` or a
method's budget for one experiment.

`policy_tree` searches a tree of complete programs rooted at the -O3 baseline. After each
candidate is built, tested, and timed, a hand-written policy expands it (refine, or repair if
broken), branches to a sibling with a new strategy hint, backtracks to the best node with room,
or terminates after `policy_tree.patience` calls without a new best. It keeps the fastest
correct candidate. `curve` re-scores, for each k, what each run would have returned after its
first k calls, and plots speedup against calls.

`generate` stops on the first failed API call; pass `--run-dir` to resume. Each run directory
ends up with `generations.jsonl`, `results.jsonl`, and `summary.json`. Candidates that fail to
build or pass every test count as 1.0x speedup.
