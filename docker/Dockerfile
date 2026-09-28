# Fixed Linux x86-64 toolchain for building, testing, and timing candidates.
FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc-12 binutils libc6-dev make ca-certificates time \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/gcc-12 /usr/local/bin/gcc

COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uv /uvx /bin/

# Keep the Linux venv out of the bind-mounted repo so it never clashes with the host .venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /work

COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-install-project

COPY src ./src
RUN uv sync --frozen

CMD ["bash"]
