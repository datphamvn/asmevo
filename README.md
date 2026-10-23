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
    │                       #   at this level, nodes.py holds the nodes the generate methods
    │                       #   share, and search_nodes.py and prompt.py those of the searches
    ├── eval/               # build, test, and time candidates against gcc -O3
    └── utils/
        └── llm/            # LLM clients: OpenAI Responses API, FPT Cloud (Chat Completions)
```

## Usage

Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` (and optionally `HF_TOKEN`). To change any setting, e.g. the
endpoint, model, or `dataset.limit` for a quick run, add only those keys to `configs/local.yaml`
(per experiment, e.g. `dataset.problem_ids` or a budget, use a file passed as `--config FILE`,
applied on top). The token prices are required (`0.0` if the endpoint is free), so a run never
reports a silent $0:

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
pass it with `--config` (applied on top of `configs/local.yaml`):

```bash
uv run asmevo --config configs/fptcloud.yaml generate --method single_call
```

```bash
uv run asmevo generate --method single_call          # LLM calls only
uv run asmevo generate --method policy_tree          # tree search
uv run asmevo generate --method refined_best_of_n    # iterated best-of-n (ACCLAIM)
uv run asmevo evaluate runs/single_call-<timestamp>  # build/test/time
uv run asmevo curve runs/<run-a> runs/<run-b>        # speedup of the best in the first k calls
uv run asmevo plots runs/<run>                     # status, speedup, and token figures
```

All but `single_call` generation need Linux x86-64 (natively or via Docker, below).
`uv run asmevo gcc-baseline --limit 5` checks the dataset's -O3 baseline against our own
`gcc -O3 -S` ([docs/gcc-baseline.md](docs/gcc-baseline.md)).

`policy_tree` searches a tree of complete programs rooted at the -O3 baseline. After each
candidate is built, tested, and timed, a hand-written policy expands it (refine, or repair if
broken), branches to a sibling with a new strategy hint, backtracks to the best node with room,
or terminates after `policy_tree.patience` calls without a new best. It keeps the fastest
correct candidate.

`refined_best_of_n` reads "refined best-of-n" as ACCLAIM's iterated best-of-n
(arXiv:2604.04238, Sec. 3.4): it samples in rounds of `refined_best_of_n.samples_per_round`,
all with the same prompt. A round's fastest correct sample replaces the incumbent (at first the
-O3 baseline) if it is more than `refined_best_of_n.min_gain` faster; the next round sees the
incumbent with its feedback and a note on the last round's samples (each one's speedup or
error, and the input and expected output of the tests they failed) or, while nothing beats -O3
yet, the SuperCoder prompt with that note. Its first round is the SuperCoder prompt, and with
`samples_per_round` at least `max_calls` its one round is plain best-of-N. It evaluates each
distinct program once and scores a copy of -O3 as 1.0x without timing it. Read the 12-call
point of a 32-call run with `curve --ks 12`.

### Docker (Linux x86-64)

Building, testing, and timing candidates needs Linux x86-64; the image pins the toolchain
(gcc 11.4, binutils, hyperfine). Run from the repo root; the bind mount keeps runs and the
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
