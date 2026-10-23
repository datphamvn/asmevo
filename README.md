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
    ├── methods/            # one folder per method, holding its flow and prompt; nodes.py and
    │                       #   prompt.py at this level hold the PocketFlow nodes they share
    ├── eval/               # build, test, and time candidates against gcc -O3
    └── utils/
        └── llm/            # LLM clients: OpenAI Responses API, FPT Cloud (Chat Completions)
```

## Usage

Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` (and optionally `HF_TOKEN`). To change any setting, e.g. the
endpoint, model, or `dataset.limit` for a quick run, add only those keys to `configs/local.yaml`.
The token prices are required (`0.0` if the endpoint is free), so a run never reports a silent $0:

```yaml
llm:
  base_url: https://my-endpoint/v1
  model: my-model
  cost_per_1k_input: 0.00125   # USD per 1k tokens
  cost_per_1k_output: 0.01
```

`llm.provider` selects the API format: `openai` (Responses API, the default) or `fptcloud`
([FPT AI Marketplace](https://marketplace.fptcloud.com), OpenAI Chat Completions format). For
FPT Cloud, put `FPTCLOUD_API_KEY` in `.env`, fill in the prices in `configs/fptcloud.yaml`, and
pass it with `--config` (it replaces `configs/local.yaml` for that run):

```bash
uv run asmevo --config configs/fptcloud.yaml generate --method single_call
```

```bash
uv run asmevo generate --method single_call          # LLM calls; any host
uv run asmevo evaluate runs/single_call-<timestamp>  # build/test/time; Linux x86-64 only
```

### Docker (Linux x86-64)

Building, testing, and timing candidates needs Linux x86-64; the image pins the toolchain
(gcc-12, binutils, hyperfine). Run from the repo root; the bind mount keeps runs and the
dataset cache on the host:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile -t asmevo .
docker run --rm --platform linux/amd64 -v "$PWD":/work asmevo uv run asmevo evaluate runs/<run>
docker run --rm --platform linux/amd64 -v "$PWD":/work asmevo uv run pytest  # incl. runner tests
```

On other hosts (e.g. Apple Silicon) Docker emulates x86-64: good for checking that the
pipeline works, but timings and speedups measured there are meaningless.

`generate` stops on the first failed API call; pass `--run-dir` to resume. Each run directory
ends up with `generations.jsonl`, `results.jsonl`, and `summary.json`; every LLM call in
`generations.jsonl` keeps its full prompt (`messages`) and `reply`. Candidates that fail to
build or pass every test count as 1.0x speedup.
