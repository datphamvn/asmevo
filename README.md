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
    ├── methods/            # one folder per method, holding its flow and prompt
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
uv run asmevo evaluate runs/single_call-<timestamp>  # build/test/time; Linux x86-64 only
```

`generate` stops on the first failed API call; pass `--run-dir` to resume. Each run directory
ends up with `generations.jsonl`, `results.jsonl`, and `summary.json`. Candidates that fail to
build or pass every test count as 1.0x speedup.
