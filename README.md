# asmevo: LLM-guided Evolutionary Assembly Optimization

## Project structure

```
llm_opt/
├── pyproject.toml          # uv-managed project: dependencies, dev tools, lint/test settings
├── uv.lock                 # pinned dependency versions (commit it; run `uv sync`)
├── configs/
│   └── default.yaml        # default configuration for the search algorithm
├── docker/
│   └── Dockerfile          # fixed Linux x86-64
└── src/asmevo/
    ├── main.py             # entry point
    └── utils/
        └── llm/
            └── client.py   # LLM client interface (OpenAI-compatible endpoint)
```
